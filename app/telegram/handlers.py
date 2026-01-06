import asyncio
from datetime import datetime, timezone
import json
import logging
import mimetypes
from pathlib import Path
import random
import re

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from app.agent.intent_router import RouterDecision, RouterIntent, route_intent
from app.config import get_settings
from app.session_manager import (
    clear_session,
    get_session_context,
    record_image_result,
    record_user_message,
    should_reset_context,
)
from app.storage.database import get_database
from app.telegram.streaming import AsyncStreamHandler, MessageManager

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
IMAGE_OUTPUT_DIR = ROOT_DIR / "artifacts" / "images"
TEMP_IMAGE_DIR = ROOT_DIR / "artifacts" / "tmp_images"
_IMAGE_EDIT_TOKENS = (
    "edit",
    "duzenle",
    "degistir",
    "modify",
    "change",
    "remove",
    "add",
    "replace",
    "sil",
    "ekle",
    "kaldir",
    "renk",
    "color",
    "background",
    "arka plan",
)
_EXPLICIT_IMAGE_TOKENS = (
    "resim",
    "gorsel",
    "image",
    "photo",
    "picture",
    "foto",
    "fotograf",
)

_ANALYSIS_TOKENS = (
    "analyze",
    "analyse",
    "describe",
    "identify",
    "explain",
    "suitable",
    "appropriate",
    "good for",
    "is this good",
    "is this suitable",
    "is it good",
    "is it ok",
    "is it okay",
    "is it professional",
    "how is it",
    "how does it look",
    "does it look",
    "what is",
    "whats",
    "nedir",
    "tanimla",
    "acikla",
    "incele",
    "ne goruyorsun",
    "uygun mu",
    "sence",
    "degerlendir",
    "degerlendirir misin",
    "yorumla",
    "fikir ver",
    "nasil duruyor",
    "nasil gorunuyor",
    "bu hali nasil",
    "bu halin nasil",
    "simdi nasil",
)
_CHAT_CONTEXT_LIMIT = 20
_SUGGESTION_PREFIX = "SUGG:"
_TELEGRAM_TEXT_CHUNK = 3500
_TELEGRAM_CAPTION_LIMIT = 1024
_STREAM_RETRY_ATTEMPTS = 3
_STREAM_RETRY_BASE_DELAY = 0.6
_STREAM_RETRY_MAX_DELAY = 3.0
_SUGGESTION_RETRY_ATTEMPTS = 3
_SUGGESTION_RETRY_BASE_DELAY = 0.4
_SUGGESTION_RETRY_MAX_DELAY = 2.0

_ANALYSIS_CLASSIFIER_PROMPT = """
You classify the user's request about an image.
Return ONLY JSON with this shape:
{{"intent": "ANALYZE|EDIT|UNKNOWN"}}

ANALYZE = user asks for evaluation, description, suitability, or feedback.
EDIT = user asks to change, modify, add/remove elements, or transform the image.
UNKNOWN = unclear.

User message: {message}
""".strip()

_BUTTONS = [
    ("Job Search", "job_search"),
]


def _resolve_user_id(update: Update) -> str | None:
    if update.effective_user and update.effective_user.id:
        return str(update.effective_user.id)
    if update.effective_chat and update.effective_chat.id:
        return str(update.effective_chat.id)
    return None


def _memory_text(message_text: str) -> str:
    cleaned = message_text.strip()
    lowered = cleaned.lower()
    if lowered.startswith("pc:"):
        return cleaned.split(":", 1)[1].strip()
    if lowered.startswith("/pc"):
        parts = cleaned.split(maxsplit=1)
        if len(parts) > 1:
            return parts[1].strip()
    return cleaned


def _resolve_settings(context: ContextTypes.DEFAULT_TYPE):
    settings = context.application.bot_data.get("settings")
    return settings or get_settings()


def build_main_keyboard() -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton(
        text=label, callback_data=value)] for label, value in _BUTTONS]
    return InlineKeyboardMarkup(keyboard)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "ApplyWise bot is online",
        reply_markup=build_main_keyboard(),
    )


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    try:
        await query.answer()
    except Exception as exc:
        logger.warning("Callback query answer failed: %s", exc)
    logger.info("Telegram button selected: %s", query.data)

    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        return

    pending = agent_context.pending_request
    callback_text = str(query.data or "")
    suggestion_text = _strip_suggestion_prefix(callback_text)
    is_suggestion = suggestion_text is not None

    # Check if we should treat this as a pending reply
    # - If pending is NOT startup: always reply to it
    # - If pending IS startup: only reply if button is "job_search"
    should_reply_to_pending = False
    if pending is not None and not is_suggestion:
        if pending.category != "startup":
            should_reply_to_pending = True
        elif str(query.data) in {"job_search", "job search"}:
            should_reply_to_pending = True

    if should_reply_to_pending:
        # If there's a pending request, treat the button click as a reply to it
        from app.agent.state import HumanReply

        reply = HumanReply(
            question_id=pending.question_id,
            text=str(query.data),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="telegram",
        )
        await agent_context.human_reply_queue.put(reply)
    else:
        # If no pending request (or ignoring startup pending), treat it as a new message/command
        message_text = suggestion_text if is_suggestion else callback_text
        if not message_text:
            return
        user_id = _resolve_user_id(update)
        memory_text = _memory_text(message_text)
        if user_id:
            if should_reset_context(user_id, memory_text):
                clear_session(user_id)
            record_user_message(user_id, memory_text)

        # We need a message object to reply to. The query.message is the message with the buttons.
        # We can simulate a new message handling or call route_and_handle directly.
        # Since we want to show it as if the user typed it, let's just use the existing message context
        # but with the new text.

        await _route_and_handle_message(
            update=update,
            context=context,
            agent_context=agent_context,
            user_id=user_id,
            message_text=message_text,
            has_image=False,
            photo=None,
            # Force ignore pending (e.g. startup) for dynamic buttons
            pending=None,
        )


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    user_id = _resolve_user_id(update)
    memory_text = _memory_text(message.text)
    if user_id:
        if should_reset_context(user_id, memory_text):
            clear_session(user_id)
        record_user_message(user_id, memory_text)

    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        await message.reply_text("Agent is not ready. Please try again later.")
        return

    if message.text.strip().upper() == "STOP":
        agent_context.request_stop("telegram_stop")
        await message.reply_text("Stopping all actions. You can send RESUME to continue later.")
        return

    pending = agent_context.pending_request
    if pending is not None and pending.category != "startup":
        handled = await _handle_pending_reply(message, agent_context, message.text)
        if handled:
            return

    await _route_and_handle_message(
        update=update,
        context=context,
        agent_context=agent_context,
        user_id=user_id,
        message_text=message.text,
        has_image=False,
        photo=None,
        pending=pending,
    )


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not (message.photo or message.document):
        return
    media = None
    if message.photo:
        media = message.photo[-1]
    elif message.document and (message.document.mime_type or "").startswith("image/"):
        media = message.document
    if media is None:
        return
    user_id = _resolve_user_id(update)
    caption = message.caption or ""
    if caption:
        memory_text = _memory_text(caption)
        if user_id:
            if should_reset_context(user_id, memory_text):
                clear_session(user_id)
            record_user_message(user_id, memory_text)

    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        await message.reply_text("Agent is not ready. Please try again later.")
        return

    pending = agent_context.pending_request
    if pending is not None and pending.category != "startup":
        handled = await _handle_pending_reply(message, agent_context, caption)
        if handled:
            return

    await _route_and_handle_message(
        update=update,
        context=context,
        agent_context=agent_context,
        user_id=user_id,
        message_text=caption,
        has_image=True,
        photo=media,
        pending=pending,
    )


async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.document:
        return
    document = message.document
    if (document.mime_type or "").startswith("image/"):
        return
    user_id = _resolve_user_id(update)
    caption = message.caption or ""
    if caption:
        memory_text = _memory_text(caption)
        if user_id:
            if should_reset_context(user_id, memory_text):
                clear_session(user_id)
            record_user_message(user_id, memory_text)

    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        await message.reply_text("Agent is not ready. Please try again later.")
        return

    pending = agent_context.pending_request
    if pending is not None and pending.category != "startup":
        handled = await _handle_pending_reply(message, agent_context, caption)
        if handled:
            return

    if pending is not None and pending.category == "startup" and _is_startup_reply(caption):
        await _handle_pending_reply(message, agent_context, caption)
        return

    await _handle_document_request(
        message=message,
        context=context,
        caption=caption,
        document=document,
        user_id=user_id,
        session_context=get_session_context(user_id),
    )


async def handle_pc_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    task = message.text.split(maxsplit=1)
    if len(task) < 2:
        await message.reply_text("Usage: /pc <task>. Example: /pc Steam uygulamasini ac")
        return
    user_id = _resolve_user_id(update)
    memory_text = _memory_text(message.text)
    if user_id:
        if should_reset_context(user_id, memory_text):
            clear_session(user_id)
        record_user_message(user_id, memory_text)
    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        await message.reply_text("Agent is not ready. Please try again later.")
        return
    pending = agent_context.pending_request
    await _route_and_handle_message(
        update=update,
        context=context,
        agent_context=agent_context,
        user_id=user_id,
        message_text=message.text,
        has_image=False,
        photo=None,
        pending=pending,
    )


async def _route_and_handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    agent_context,
    user_id: str | None,
    message_text: str,
    *,
    has_image: bool,
    photo,
    pending,
) -> None:
    message = update.message
    if message is None and update.callback_query is not None:
        message = update.callback_query.message
    if message is None:
        return
    llm_client = context.application.bot_data.get("llm_client")
    image_client = context.application.bot_data.get("image_client")
    session_context = get_session_context(user_id)
    settings = _resolve_settings(context)

    decision = await asyncio.to_thread(
        route_intent,
        llm_client,
        message_text,
        has_image=has_image,
        session_context=session_context,
    )
    task_text, has_pc_prefix = _extract_pc_task(message_text)
    if has_pc_prefix:
        decision = RouterDecision(
            intent=RouterIntent.COMPUTER_USE, reason="forced:pc_prefix")
    if (
        not has_image
        and session_context
        and session_context.get("last_image_path")
        and await _is_analysis_request_with_llm(llm_client, message_text)
        and _mentions_explicit_image_reference(message_text)
    ):
        decision = RouterDecision(intent=RouterIntent.IMAGE_EDIT, reason="override:analysis_last_image")

    if pending is not None and pending.category == "startup" and _is_startup_reply(message_text):
        await _handle_pending_reply(message, agent_context, message_text)
        return

    if decision.intent == RouterIntent.COMPUTER_USE:
        await _start_pc_task(
            task_text,
            update,
            context,
            user_id=user_id,
            require_prefix=has_pc_prefix,
            silent_fail=False,
        )
        return
    if decision.intent == RouterIntent.IMAGE_GEN:
        await _handle_image_generation(
            message,
            image_client,
            llm_client,
            message_text,
            user_id=user_id,
            session_context=session_context,
            settings=settings,
            telegram_app=context.application,
        )
        return
    if decision.intent == RouterIntent.IMAGE_EDIT:
        await _handle_image_edit(
            message,
            image_client,
            llm_client,
            message_text,
            user_id=user_id,
            photo=photo,
            session_context=session_context,
            settings=settings,
            telegram_app=context.application,
        )
        return

    await _handle_chat(
        message,
        llm_client,
        message_text,
        user_id,
        session_context,
        settings=settings,
        telegram_app=context.application,
    )


async def _reply_with_suggestions(
    message,
    text: str,
    llm_client,
    history_text: str | None = None,
    photo=None,
    caption: str | None = None,
) -> None:
    """Sends a message with dynamic suggestions generated by LLM."""
    reply_markup = await _build_suggestion_markup(llm_client, history_text)

    message_text = (text or "").strip()
    if photo:
        photo_caption = (caption or message_text).strip()
        formatted_caption = _format_telegram_html(photo_caption) if photo_caption else ""
        if formatted_caption and len(formatted_caption) > _TELEGRAM_CAPTION_LIMIT:
            await message.reply_photo(photo=photo, caption="Result")
            if message_text:
                await _send_text_in_chunks(
                    message,
                    message_text,
                    chunk_size=_TELEGRAM_TEXT_CHUNK,
                    reply_markup=reply_markup,
                )
            return
        await message.reply_photo(
            photo=photo,
            caption=formatted_caption or None,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
        )
        return
    if message_text:
        if len(message_text) > _TELEGRAM_TEXT_CHUNK:
            await _send_text_in_chunks(
                message,
                message_text,
                chunk_size=_TELEGRAM_TEXT_CHUNK,
                reply_markup=reply_markup,
            )
            return
        await message.reply_text(
            _format_telegram_html(message_text),
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
        )


async def _handle_pending_reply(message, agent_context, reply_text: str | None) -> bool:
    pending = agent_context.pending_request
    if pending is None:
        return False
    if not reply_text or not reply_text.strip():
        await message.reply_text(
            "There is a pending request already. Please reply to it first or send STOP.",
        )
        return True

    from app.agent.state import HumanReply

    message_time = message.date.astimezone(timezone.utc)
    if message_time < pending.asked_at:
        logger.info("Ignored Telegram message (sent before question was asked)")
        return True

    reply = HumanReply(
        question_id=pending.question_id,
        text=reply_text,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="telegram",
    )
    await agent_context.human_reply_queue.put(reply)
    logger.info("Human reply received for question_id=%s", pending.question_id)
    await message.reply_text(f"Received reply for question_id={pending.question_id}. Thank you!")
    return True


async def _handle_chat(
    message,
    llm_client,
    prompt_text: str,
    user_id: str | None,
    session_context: dict | None,
    *,
    settings,
    telegram_app,
) -> None:
    prompt_text = prompt_text.strip()
    if not prompt_text:
        await message.reply_text("Please send a message.")
        return
    if llm_client is None:
        await message.reply_text("Gemini is not configured. Please set GEMINI_API_KEY.")
        return
    db = get_database() if user_id else None
    history = _safe_get_history(
        db, user_id, _CHAT_CONTEXT_LIMIT) if db and user_id else []
    timestamp = _timestamp_from_message(message)
    prompt = _build_chat_prompt(prompt_text, history)
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=prompt_text,
            timestamp=timestamp,
        )
    streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
    show_thoughts = bool(getattr(settings, "show_thoughts", True))
    if streaming_enabled and hasattr(llm_client, "stream_text"):
        async def _stream_runner() -> None:
            try:
                live_message = await message.reply_text("⏳ Thinking...", parse_mode=ParseMode.HTML)
                manager = MessageManager(
                    bot=telegram_app.bot,
                    chat_id=str(message.chat_id),
                    message_id=live_message.message_id,
                )
                handler = None
                result = None
                for attempt in range(1, _STREAM_RETRY_ATTEMPTS + 1):
                    handler = AsyncStreamHandler(
                        manager,
                        show_thoughts=show_thoughts,
                        response_label="",
                    )
                    result = await handler.stream(
                        llm_client,
                        prompt,
                        include_thoughts=show_thoughts,
                    )
                    if result.error is None:
                        break
                    if attempt < _STREAM_RETRY_ATTEMPTS:
                        await manager.update("⏳ Retrying...", force=True)
                        await asyncio.sleep(_retry_delay(attempt, _STREAM_RETRY_BASE_DELAY, _STREAM_RETRY_MAX_DELAY))
                if handler is None or result is None:
                    raise RuntimeError("Streaming did not produce a result.")
                response_text = result.response_text.strip()
                if response_text and result.completed and db and user_id:
                    _safe_add_message(
                        db,
                        user_id=user_id,
                        role="assistant",
                        content=response_text,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                if response_text and result.completed:
                    suggestion_context = _build_suggestion_context(
                        user_id=user_id,
                        db=db,
                        session_context=session_context,
                        user_message=prompt_text,
                        assistant_message=response_text,
                        task_type="chat",
                    )
                    reply_markup = await _build_suggestion_markup(llm_client, suggestion_context)
                    response_limit = 3000 if not handler.thought_text else 2200
                    if len(response_text) > response_limit:
                        await manager.finalize("Response sent below.")
                        await _send_text_in_chunks(
                            message,
                            response_text,
                            chunk_size=_TELEGRAM_TEXT_CHUNK,
                            reply_markup=reply_markup,
                        )
                    else:
                        await manager.finalize(handler.format_message(), reply_markup=reply_markup)
                else:
                    await manager.finalize(handler.format_message())
                if not response_text and not handler.thought_text:
                    await message.reply_text("No response generated.")
            except Exception as exc:  # pragma: no cover - network/telegram errors
                logger.error("Streaming chat failed: %s", exc)
                await message.reply_text("Chat response failed. Please try again.")

        telegram_app.create_task(_stream_runner())
        return

    try:
        response_text = await asyncio.to_thread(llm_client.generate_text, prompt)
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.error("Chat response failed: %s", exc)
        await message.reply_text("Chat response failed. Please try again.")
        return
    if not response_text:
        await message.reply_text("No response generated.")
        return
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="assistant",
            content=response_text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    suggestion_context = _build_suggestion_context(
        user_id=user_id,
        db=db,
        session_context=session_context,
        user_message=prompt_text,
        assistant_message=response_text,
        task_type="chat",
    )
    await _reply_with_suggestions(
        message,
        response_text,
        llm_client,
        history_text=suggestion_context,
    )


async def _handle_image_generation(
    message,
    image_client,
    llm_client,
    prompt_text: str,
    user_id: str | None,
    session_context: dict | None,
    *,
    settings,
    telegram_app,
) -> None:
    if image_client is None:
        await message.reply_text("Gemini image model is not configured. Please set GEMINI_API_KEY.")
        return
    prompt_text = prompt_text.strip()
    if not prompt_text:
        await message.reply_text("Please describe the image you want to generate.")
        return
    db = get_database() if user_id else None
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=prompt_text,
            timestamp=_timestamp_from_message(message),
        )
    streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
    show_thoughts = bool(getattr(settings, "show_thoughts", True))
    if streaming_enabled and show_thoughts and hasattr(image_client, "stream_generate_image"):
        async def _stream_runner() -> None:
            async def _finalize_generation(image_bytes: bytes) -> None:
                output_path = _save_image_bytes(image_bytes, "generated")
                record_image_result(user_id, output_path, prompt_text, source="generated")
                session_context = get_session_context(user_id) or session_context
                if db and user_id:
                    _safe_add_message(
                        db,
                        user_id=user_id,
                        role="assistant",
                        content="Image generated.",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        file_path=output_path,
                        file_type="image/png",
                    )

                suggestion_context = _build_suggestion_context(
                    user_id=user_id,
                    db=db,
                    session_context=session_context,
                    user_message=prompt_text,
                    assistant_message="Image generated.",
                    task_type="image_generation",
                    extra_context="User asked to generate an image.",
                )
                with open(output_path, "rb") as handle:
                    await _reply_with_suggestions(
                        message,
                        "Image generated.",
                        llm_client,
                        history_text=suggestion_context,
                        photo=handle,
                        caption="Image generated.",
                    )

            try:
                live_message = await message.reply_text(
                    "⏳ Generating image...",
                    parse_mode=ParseMode.HTML,
                )
                manager = MessageManager(
                    bot=telegram_app.bot,
                    chat_id=str(message.chat_id),
                    message_id=live_message.message_id,
                )
                handler = None
                image_bytes = None
                last_error: Exception | None = None
                for attempt in range(1, _STREAM_RETRY_ATTEMPTS + 1):
                    handler = AsyncStreamHandler(
                        manager,
                        show_thoughts=show_thoughts,
                        response_label="Decision",
                    )
                    stream_state = await image_client.stream_generate_image(
                        prompt_text,
                        include_thoughts=show_thoughts,
                    )
                    result = await handler.stream_chunks(stream_state.chunks)
                    image_bytes = stream_state.get_image_bytes()
                    if result.error is None and image_bytes:
                        break
                    last_error = result.error or ValueError("No image bytes received from stream.")
                    if attempt < _STREAM_RETRY_ATTEMPTS:
                        await manager.update("⏳ Retrying...", force=True)
                        await asyncio.sleep(_retry_delay(attempt, _STREAM_RETRY_BASE_DELAY, _STREAM_RETRY_MAX_DELAY))
                if handler is None:
                    raise RuntimeError("Stream did not initialize.")
                if not image_bytes or (last_error is not None and image_bytes is None):
                    raise last_error or ValueError("No image bytes received from stream.")
                await _finalize_generation(image_bytes)
                await manager.finalize(handler.format_message())
            except Exception as exc:  # pragma: no cover - network/telegram errors
                logger.error("Image generation stream failed: %s", exc)
                try:
                    image_bytes = await _retry_async(
                        lambda: asyncio.to_thread(image_client.generate_image, prompt_text),
                        label="Image generation",
                        attempts=_STREAM_RETRY_ATTEMPTS,
                        base_delay=_STREAM_RETRY_BASE_DELAY,
                        max_delay=_STREAM_RETRY_MAX_DELAY,
                    )
                except Exception as fallback_exc:
                    logger.error("Image generation failed: %s", fallback_exc)
                    await message.reply_text("Image generation failed. Please try again.")
                    return
                if not image_bytes:
                    logger.warning("Image generation returned empty bytes.")
                    await message.reply_text("Image generation failed. Please try again.")
                    return
                await _finalize_generation(image_bytes)

        telegram_app.create_task(_stream_runner())
        return
    try:
        image_bytes = await asyncio.to_thread(image_client.generate_image, prompt_text)
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.error("Image generation failed: %s", exc)
        await message.reply_text("Image generation failed. Please try again.")
        return
    if not image_bytes:
        logger.warning("Image generation returned empty bytes.")
        await message.reply_text("Image generation failed. Please try again.")
        return
    output_path = _save_image_bytes(image_bytes, "generated")
    record_image_result(user_id, output_path, prompt_text, source="generated")
    session_context = get_session_context(user_id) or session_context
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="assistant",
            content="Image generated.",
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_path=output_path,
            file_type="image/png",
        )

    suggestion_context = _build_suggestion_context(
        user_id=user_id,
        db=db,
        session_context=session_context,
        user_message=prompt_text,
        assistant_message="Image generated.",
        task_type="image_generation",
        extra_context="User asked to generate an image.",
    )
    with open(output_path, "rb") as handle:
        await _reply_with_suggestions(
            message,
            "Image generated.",
            llm_client,
            history_text=suggestion_context,
            photo=handle,
            caption="Image generated."
        )


async def _handle_image_edit(
    message,
    image_client,
    llm_client,
    prompt_text: str,
    *,
    user_id: str | None,
    photo,
    session_context: dict | None,
    settings,
    telegram_app,
) -> None:
    if image_client is None:
        await message.reply_text("Gemini image model is not configured. Please set GEMINI_API_KEY.")
        return
    prompt_text = prompt_text.strip()
    if not prompt_text:
        await message.reply_text("Please describe the edit or analysis you want.")
        return

    reference_path = None
    if photo is not None:
        try:
            reference_path = await _download_photo_to_temp(photo, message)
        except Exception as exc:  # pragma: no cover - network/IO errors
            logger.error("Photo download failed: %s", exc)
            await message.reply_text("Photo download failed. Please try again.")
            return
        record_image_result(
            user_id,
            reference_path,
            prompt_text or None,
            source="uploaded",
        )
    else:
        reference_path = session_context.get(
            "last_image_path") if session_context else None
        if not reference_path or not Path(reference_path).exists():
            await message.reply_text("Please send the image you want to edit.")
            return

    db = get_database() if user_id else None
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=prompt_text,
            timestamp=_timestamp_from_message(message),
            file_path=reference_path,
            file_type=mimetypes.guess_type(reference_path)[
                0] if reference_path else None,
        )

    if await _is_analysis_request_with_llm(llm_client, prompt_text):
        streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
        show_thoughts = bool(getattr(settings, "show_thoughts", True))
        if streaming_enabled and hasattr(llm_client, "stream_with_image"):
            async def _stream_runner() -> None:
                try:
                    live_message = await message.reply_text(
                        "⏳ Analyzing image...",
                        parse_mode=ParseMode.HTML,
                    )
                    manager = MessageManager(
                        bot=telegram_app.bot,
                        chat_id=str(message.chat_id),
                        message_id=live_message.message_id,
                    )
                    handler = AsyncStreamHandler(
                        manager,
                        show_thoughts=show_thoughts,
                        response_label="Decision",
                    )

                    async def _recover() -> str:
                        return await asyncio.to_thread(
                            image_client.analyze_image,
                            prompt_text,
                            reference_path,
                        )

                    result = await handler.stream_chunks(
                        llm_client.stream_with_image(
                            prompt_text,
                            reference_path,
                            include_thoughts=show_thoughts,
                        ),
                        recovery=_recover,
                    )
                    response_text = result.response_text.strip()
                    if response_text and result.completed and db and user_id:
                        _safe_add_message(
                            db,
                            user_id=user_id,
                            role="assistant",
                            content=response_text,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                    if response_text and result.completed:
                        suggestion_context = _build_suggestion_context(
                            user_id=user_id,
                            db=db,
                            session_context=session_context,
                            user_message=prompt_text,
                            assistant_message=response_text,
                            task_type="image_analysis",
                            extra_context="User asked to analyze an image.",
                        )
                        reply_markup = await _build_suggestion_markup(llm_client, suggestion_context)
                        await manager.finalize(handler.format_message(), reply_markup=reply_markup)
                    else:
                        await manager.finalize(handler.format_message())
                    if not response_text and not handler.thought_text:
                        await message.reply_text("No analysis result.")
                except Exception as exc:  # pragma: no cover - network/telegram errors
                    logger.error("Image analysis stream failed: %s", exc)
                    await message.reply_text("Image analysis failed. Please try again.")

            telegram_app.create_task(_stream_runner())
            return
        try:
            response_text = await asyncio.to_thread(
                image_client.analyze_image,
                prompt_text,
                reference_path,
            )
        except Exception as exc:  # pragma: no cover - network/proxy errors
            logger.error("Image analysis failed: %s", exc)
            await message.reply_text("Image analysis failed. Please try again.")
            return
        if response_text:
            if db and user_id:
                _safe_add_message(
                    db,
                    user_id=user_id,
                    role="assistant",
                    content=response_text,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

            suggestion_context = _build_suggestion_context(
                user_id=user_id,
                db=db,
                session_context=session_context,
                user_message=prompt_text,
                assistant_message=response_text,
                task_type="image_analysis",
                extra_context="User asked to analyze an image.",
            )
            await _reply_with_suggestions(
                message,
                response_text,
                llm_client,
                history_text=suggestion_context,
            )
        else:
            await message.reply_text("No analysis result.")
        return
    async def _finalize_edit(image_bytes: bytes) -> None:
        output_path = _save_image_bytes(image_bytes, "edited")
        record_image_result(user_id, output_path, prompt_text, source="edited")
        session_context = get_session_context(user_id) or session_context
        response_text = "Image updated."
        if llm_client is not None:
            summary_prompt = (
                "Write a short, friendly confirmation sentence in the user's language describing the image edit. "
                f"User request: {prompt_text}"
            )
            try:
                generated = await asyncio.to_thread(llm_client.generate_text, summary_prompt)
                if generated:
                    response_text = generated.strip()
            except Exception as exc:  # pragma: no cover - network/proxy errors
                logger.warning("Image edit summary failed: %s", exc)
        if db and user_id:
            _safe_add_message(
                db,
                user_id=user_id,
                role="assistant",
                content=response_text,
                timestamp=datetime.now(timezone.utc).isoformat(),
                file_path=output_path,
                file_type="image/png",
            )

        suggestion_context = _build_suggestion_context(
            user_id=user_id,
            db=db,
            session_context=session_context,
            user_message=prompt_text,
            assistant_message=response_text,
            task_type="image_edit",
            extra_context="User asked to edit an image.",
        )
        with open(output_path, "rb") as handle:
            await _reply_with_suggestions(
                message,
                response_text,
                llm_client,
                history_text=suggestion_context,
                photo=handle,
                caption=response_text,
            )

    streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
    show_thoughts = bool(getattr(settings, "show_thoughts", True))
    if streaming_enabled and show_thoughts and hasattr(image_client, "stream_edit_image"):
        async def _stream_runner() -> None:
            try:
                live_message = await message.reply_text(
                    "⏳ Editing image...",
                    parse_mode=ParseMode.HTML,
                )
                manager = MessageManager(
                    bot=telegram_app.bot,
                    chat_id=str(message.chat_id),
                    message_id=live_message.message_id,
                )
                handler = None
                image_bytes = None
                last_error: Exception | None = None
                for attempt in range(1, _STREAM_RETRY_ATTEMPTS + 1):
                    handler = AsyncStreamHandler(
                        manager,
                        show_thoughts=show_thoughts,
                        response_label="Decision",
                    )
                    stream_state = await image_client.stream_edit_image(
                        prompt_text,
                        reference_path,
                        include_thoughts=show_thoughts,
                    )
                    result = await handler.stream_chunks(stream_state.chunks)
                    image_bytes = stream_state.get_image_bytes()
                    if result.error is None and image_bytes:
                        break
                    last_error = result.error or ValueError("No image bytes received from stream.")
                    if attempt < _STREAM_RETRY_ATTEMPTS:
                        await manager.update("⏳ Retrying...", force=True)
                        await asyncio.sleep(_retry_delay(attempt, _STREAM_RETRY_BASE_DELAY, _STREAM_RETRY_MAX_DELAY))
                if handler is None:
                    raise RuntimeError("Stream did not initialize.")
                if not image_bytes or (last_error is not None and image_bytes is None):
                    raise last_error or ValueError("No image bytes received from stream.")
                await _finalize_edit(image_bytes)
                await manager.finalize(handler.format_message())
            except Exception as exc:  # pragma: no cover - network/telegram errors
                logger.error("Image edit stream failed: %s", exc)
                try:
                    image_bytes = await _retry_async(
                        lambda: asyncio.to_thread(
                            image_client.edit_image,
                            prompt_text,
                            reference_path,
                        ),
                        label="Image edit",
                        attempts=_STREAM_RETRY_ATTEMPTS,
                        base_delay=_STREAM_RETRY_BASE_DELAY,
                        max_delay=_STREAM_RETRY_MAX_DELAY,
                    )
                except Exception as fallback_exc:
                    logger.error("Image edit failed: %s", fallback_exc)
                    await message.reply_text("Image edit failed. Please try again.")
                    return
                if not image_bytes:
                    logger.warning("Image edit returned empty bytes.")
                    await message.reply_text("Image edit failed. Please try again.")
                    return
                await _finalize_edit(image_bytes)

        telegram_app.create_task(_stream_runner())
        return

    try:
        image_bytes = await asyncio.to_thread(
            image_client.edit_image,
            prompt_text,
            reference_path,
        )
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.error("Image edit failed: %s", exc)
        await message.reply_text("Image edit failed. Please try again.")
        return
    if not image_bytes:
        logger.warning("Image edit returned empty bytes.")
        await message.reply_text("Image edit failed. Please try again.")
        return
    await _finalize_edit(image_bytes)


async def _handle_document_request(
    message,
    context,
    caption: str,
    document,
    user_id: str | None,
    session_context: dict | None,
) -> None:
    if not caption.strip():
        await message.reply_text("Please add a caption describing what to do with the file.")
        return
    llm_client = context.application.bot_data.get("llm_client")
    if llm_client is None:
        await message.reply_text("Gemini is not configured. Please set GEMINI_API_KEY.")
        return
    temp_path = None
    db = get_database() if user_id else None
    try:
        temp_path = await _download_document_to_temp(document, message)
    except Exception as exc:  # pragma: no cover - network/IO errors
        logger.error("Document download failed: %s", exc)
        await message.reply_text("Document download failed. Please try again.")
        return

    mime_type = getattr(document, "mime_type", None)
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=caption.strip(),
            timestamp=_timestamp_from_message(message),
            file_path=temp_path,
            file_type=mime_type,
        )
    result = await asyncio.to_thread(
        llm_client.generate_with_file,
        caption.strip(),
        temp_path,
        mime_type,
    )
    if result.error:
        await message.reply_text(result.error)
        return
    if not result.text:
        await message.reply_text("No response generated for the document.")
        return
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="assistant",
            content=result.text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    suggestion_context = _build_suggestion_context(
        user_id=user_id,
        db=db,
        session_context=session_context,
        user_message=caption.strip(),
        assistant_message=result.text,
        task_type="document",
        extra_context="User asked to process a document.",
    )
    reply_markup = await _build_suggestion_markup(llm_client, suggestion_context)
    await _send_text_in_chunks(message, result.text, chunk_size=3500, reply_markup=reply_markup)


def _save_image_bytes(image_bytes: bytes, prefix: str) -> str:
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = IMAGE_OUTPUT_DIR / f"{prefix}_{timestamp}.png"
    path.write_bytes(image_bytes)
    return str(path)


async def _download_document_to_temp(document, message) -> str:
    TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    telegram_file = await document.get_file()
    suffix = ""
    if telegram_file.file_path:
        suffix = Path(telegram_file.file_path).suffix
    if not suffix and getattr(document, "file_name", None):
        suffix = Path(document.file_name).suffix
    if not suffix and getattr(document, "mime_type", None):
        suffix = mimetypes.guess_extension(document.mime_type) or ""
    if not suffix:
        suffix = ".bin"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = TEMP_IMAGE_DIR / \
        f"telegram_doc_{message.message_id}_{timestamp}{suffix}"
    await telegram_file.download_to_drive(custom_path=str(path))
    return str(path)


async def _download_photo_to_temp(photo, message) -> str:
    TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    telegram_file = await photo.get_file()
    suffix = ""
    if telegram_file.file_path:
        suffix = Path(telegram_file.file_path).suffix
    if not suffix and getattr(photo, "file_name", None):
        suffix = Path(photo.file_name).suffix
    if not suffix and getattr(photo, "mime_type", None):
        suffix = mimetypes.guess_extension(photo.mime_type) or ""
    if not suffix:
        suffix = ".jpg"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = TEMP_IMAGE_DIR / \
        f"telegram_{message.message_id}_{timestamp}{suffix}"
    await telegram_file.download_to_drive(custom_path=str(path))
    return str(path)


def _extract_pc_task(message_text: str) -> tuple[str, bool]:
    cleaned = message_text.strip()
    lowered = cleaned.lower()
    if lowered.startswith("pc:"):
        return cleaned.split(":", 1)[1].strip(), True
    if lowered.startswith("/pc"):
        parts = cleaned.split(maxsplit=1)
        if len(parts) > 1:
            return parts[1].strip(), True
        return "", True
    return cleaned, False


def _is_startup_reply(message_text: str) -> bool:
    normalized = message_text.strip().lower()
    return normalized in {"job_search", "job search"}


async def _send_text_in_chunks(
    message,
    text: str,
    chunk_size: int = 3500,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    text = text.strip()
    if not text:
        return
    for start in range(0, len(text), chunk_size):
        chunk = text[start: start + chunk_size]
        is_last = start + chunk_size >= len(text)
        await message.reply_text(
            _format_telegram_html(chunk),
            reply_markup=reply_markup if is_last else None,
            parse_mode=ParseMode.HTML,
        )


def _format_telegram_html(text: str) -> str:
    if not text:
        return text
    parts = text.split("```")
    if len(parts) == 1:
        return _format_inline_html(text)
    formatted_parts: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            code = _escape_html(part.strip("\n"))
            if code:
                formatted_parts.append(f"<pre>{code}</pre>")
        else:
            formatted_parts.append(_format_inline_html(part))
    return "".join(formatted_parts)


def _format_inline_html(text: str) -> str:
    escaped = _escape_html(text)
    lines = escaped.splitlines(keepends=True)
    formatted: list[str] = []
    for line in lines:
        newline = "\n" if line.endswith("\n") else ""
        content = line[:-1] if newline else line
        heading = re.match(r"^(#{1,6})\s+(.*)$", content)
        if heading:
            heading_text = heading.group(2).strip()
            heading_text = _apply_inline_markdown(heading_text)
            content = f"<b>{heading_text}</b>" if heading_text else ""
        else:
            content = _apply_inline_markdown(content)
        formatted.append(content + newline)
    return "".join(formatted)


def _apply_inline_markdown(text: str) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    formatted_parts: list[str] = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) >= 2:
            inner = part[1:-1]
            formatted_parts.append(f"<code>{inner}</code>")
            continue
        styled = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", part)
        styled = re.sub(r"(?<!\*)\*([^\s][^*]*?[^\s])\*(?!\*)", r"<i>\1</i>", styled)
        formatted_parts.append(styled)
    return "".join(formatted_parts)


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _timestamp_from_message(message) -> str:
    message_time = getattr(message, "date", None)
    if message_time:
        return message_time.astimezone(timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _safe_get_history(db, user_id: str, limit: int):
    try:
        return db.get_recent_messages(user_id, limit)
    except Exception as exc:
        logger.warning("Message history load failed: %s", exc)
        return []


def _safe_add_message(db, **kwargs) -> None:
    try:
        db.add_message(**kwargs)
    except Exception as exc:
        logger.warning("Message persistence failed: %s", exc)


def _build_chat_prompt(prompt_text: str, history: list) -> str:
    if not history:
        return prompt_text
    lines = [_format_history_line(entry) for entry in reversed(history)]
    history_block = "\n".join(line for line in lines if line)
    if not history_block:
        return prompt_text
    return f"Conversation so far:\n{history_block}\n\nUser: {prompt_text}"


def _format_history_line(entry) -> str:
    content = (entry.content or "").strip()
    if not content and entry.file_path:
        content = "[attachment]"
    file_note = ""
    if entry.file_path:
        file_note = f" [file: {entry.file_path}"
        if entry.file_type:
            file_note += f", type: {entry.file_type}"
        file_note += "]"
    return f"{entry.role}: {content}{file_note}".strip()


def _build_suggestion_context(
    *,
    user_id: str | None,
    db,
    session_context: dict | None,
    user_message: str | None,
    assistant_message: str | None,
    task_type: str,
    extra_context: str | None = None,
) -> str:
    parts = [f"Task type: {task_type}"]
    session_lines = _format_session_context(session_context)
    if session_lines:
        parts.append("Session context:")
        parts.extend(session_lines)
    history_text = _format_recent_history(db, user_id, _CHAT_CONTEXT_LIMIT)
    if history_text:
        parts.append("Recent conversation history:")
        parts.append(history_text)
    if user_message:
        parts.append(f"User: {user_message}")
    if assistant_message:
        parts.append(f"Assistant: {assistant_message}")
    if extra_context:
        parts.append(f"Extra context: {extra_context}")
    return "\n".join(parts)


def _format_session_context(session_context: dict | None) -> list[str]:
    if not session_context:
        return []
    lines: list[str] = []
    for key in ("active_app", "last_action", "last_summary", "last_image_summary", "last_image_source"):
        value = session_context.get(key)
        if value:
            lines.append(f"- {key}: {value}")
    pending_details = session_context.get("pending_details")
    if isinstance(pending_details, dict) and pending_details:
        pending_items = ", ".join(
            f"{key}={value}" for key, value in pending_details.items() if value
        )
        if pending_items:
            lines.append(f"- pending_details: {pending_items}")
    short_term_memory = session_context.get("short_term_memory")
    if isinstance(short_term_memory, list) and short_term_memory:
        joined = " | ".join(str(item) for item in short_term_memory if item)
        if joined:
            lines.append(f"- session_history: {joined}")
    return lines


def _format_recent_history(db, user_id: str | None, limit: int) -> str:
    if not db or not user_id:
        return ""
    history = _safe_get_history(db, user_id, limit)
    if not history:
        return ""
    lines = [_format_history_line(entry) for entry in reversed(history)]
    history_block = "\n".join(line for line in lines if line)
    return history_block.strip()


def _truncate_callback_data(value: str, max_bytes: int = 64) -> str:
    data = value.strip()
    if not data:
        return ""
    encoded = data.encode("utf-8")
    if len(encoded) <= max_bytes:
        return data
    truncated = encoded[:max_bytes]
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    return ""


def _build_suggestion_callback_data(text: str) -> str:
    prefix_bytes = _SUGGESTION_PREFIX.encode("utf-8")
    remaining = 64 - len(prefix_bytes)
    if remaining <= 0:
        return ""
    truncated = _truncate_callback_data(text, max_bytes=remaining)
    if not truncated:
        return ""
    return f"{_SUGGESTION_PREFIX}{truncated}"


def _strip_suggestion_prefix(value: str) -> str | None:
    if not value.startswith(_SUGGESTION_PREFIX):
        return None
    stripped = value[len(_SUGGESTION_PREFIX) :].strip()
    return stripped if stripped else None


async def _build_suggestion_markup(llm_client, history_text: str | None) -> InlineKeyboardMarkup | None:
    if llm_client is None or not history_text:
        return None
    try:
        suggestions = await _retry_async(
            lambda: asyncio.to_thread(llm_client.generate_suggestions, history_text),
            label="Suggestion generation",
            attempts=_SUGGESTION_RETRY_ATTEMPTS,
            base_delay=_SUGGESTION_RETRY_BASE_DELAY,
            max_delay=_SUGGESTION_RETRY_MAX_DELAY,
        )
        if not suggestions:
            return None
        keyboard = []
        row = []
        for suggestion in suggestions:
            text = suggestion.strip()
            if not text:
                continue
            callback_data = _build_suggestion_callback_data(text)
            if not callback_data:
                continue
            row.append(InlineKeyboardButton(text=text[:30], callback_data=callback_data))
            if len(row) == 2:
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)
        if not keyboard:
            return None
        return InlineKeyboardMarkup(keyboard)
    except Exception as exc:
        logger.warning("Failed to generate suggestions: %s", exc)
        return None


def _augment_session_context_for_image(
    task: str,
    session_context: dict[str, object] | None,
) -> dict[str, object] | None:
    if not session_context:
        return session_context
    last_image_path = session_context.get("last_image_path")
    if not last_image_path or not _mentions_image_reference(task):
        return session_context
    enriched = dict(session_context)
    enriched["intent_hints"] = {
        "image_reference_path": last_image_path,
        "image_reference_summary": session_context.get("last_image_summary"),
        "note": "The user references the last image with a pronoun.",
    }
    return enriched


def _mentions_image_reference(text: str) -> bool:
    normalized = _normalize_text(text)
    return _contains_any(
        normalized,
        ("bunu", "sunu", "su", "this", "that", "o", "gorsel", "resim"),
    )


def _mentions_explicit_image_reference(text: str) -> bool:
    normalized = _normalize_text(text)
    return _contains_any(normalized, _EXPLICIT_IMAGE_TOKENS)


def _is_analysis_request(text: str) -> bool:
    normalized = _normalize_text(text)
    if _contains_any(normalized, _IMAGE_EDIT_TOKENS):
        return False
    return _contains_any(normalized, _ANALYSIS_TOKENS)


async def _is_analysis_request_with_llm(llm_client, text: str) -> bool:
    if llm_client is not None and text.strip():
        intent = await _classify_image_request(llm_client, text)
        if intent == "ANALYZE":
            return True
        if intent == "EDIT":
            return False
    return _is_analysis_request(text)


async def _classify_image_request(llm_client, text: str) -> str | None:
    prompt = _ANALYSIS_CLASSIFIER_PROMPT.format(message=text.strip())
    try:
        response_text = await asyncio.to_thread(llm_client.generate_text, prompt)
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.warning("Image request classification failed: %s", exc)
        return None
    cleaned = _extract_json_block(response_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Image request classification parse failed: %s", cleaned[:200])
        return None
    intent = str(data.get("intent", "")).strip().upper()
    if intent in {"ANALYZE", "EDIT", "UNKNOWN"}:
        return intent
    return None


def _extract_json_block(text: str) -> str:
    cleaned = (text or "").strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            if "{" in part:
                cleaned = part
                break
    cleaned = cleaned.strip()
    start = cleaned.find("{")
    if start != -1:
        cleaned = cleaned[start:]
    end = cleaned.rfind("}")
    if end != -1:
        cleaned = cleaned[: end + 1]
    return cleaned


def _retry_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
    return delay + random.uniform(0, base_delay / 2)


async def _retry_async(
    action,
    *,
    label: str,
    attempts: int,
    base_delay: float,
    max_delay: float,
):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await action()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            logger.warning("%s failed (attempt %s/%s): %s", label, attempt, attempts, exc)
            await asyncio.sleep(_retry_delay(attempt, base_delay, max_delay))
    if last_exc:
        raise last_exc
    raise RuntimeError(f"{label} failed without exception.")


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _normalize_text(value: str) -> str:
    normalized = value.lower()
    return normalized.translate(
        {
            ord("\u00e7"): "c",
            ord("\u011f"): "g",
            ord("\u0131"): "i",
            ord("\u00f6"): "o",
            ord("\u015f"): "s",
            ord("\u00fc"): "u",
        }
    )


async def _start_pc_task(
    task: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: str | None = None,
    require_prefix: bool = True,
    silent_fail: bool = False,
) -> bool:
    message = update.message
    if message is None:
        message = update.effective_message
    if message is None and update.callback_query is not None:
        message = update.callback_query.message
    if message is None:
        return False
    task = task.strip()
    if not task:
        if not silent_fail:
            await message.reply_text("Please provide a task after /pc or pc:.")
        return False

    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        if not silent_fail:
            await message.reply_text("Agent is not ready. Please try again later.")
        return False

    pending = agent_context.pending_request
    if pending is not None and pending.category != "startup":
        if not silent_fail:
            await message.reply_text(
                "There is a pending request already. Please reply to it first or send STOP.",
            )
        return False

    llm_client = context.application.bot_data.get("llm_client")
    if llm_client is None:
        if require_prefix and not silent_fail:
            await message.reply_text("Gemini is not configured. Please set GEMINI_API_KEY.")
        return False

    settings = _resolve_settings(context)
    streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
    show_thoughts = bool(getattr(settings, "show_thoughts", True))
    live_message = None
    if streaming_enabled and hasattr(llm_client, "stream_text"):
        live_message = await message.reply_text("⏳ Planning PC actions...", parse_mode=ParseMode.HTML)
    else:
        await message.reply_text(f"Planning PC actions for: {task}")
    session_context = get_session_context(user_id)
    planner_context = _augment_session_context_for_image(task, session_context)

    async def _runner() -> None:
        from app.pc_agent.controller import run_task_with_approval
        from app.pc_agent.llm_planner import plan_pc_actions

        stream_handler = None
        manager = None
        handler = None
        if live_message is not None:
            manager = MessageManager(
                bot=context.application.bot,
                chat_id=str(message.chat_id),
                message_id=live_message.message_id,
            )
            handler = AsyncStreamHandler(
                manager,
                show_thoughts=show_thoughts,
                response_label="Plan",
                response_as_code=True,
            )

            async def _stream_prompt(prompt: str) -> str:
                result = await handler.stream(
                    llm_client,
                    prompt,
                    include_thoughts=show_thoughts,
                )
                await manager.finalize(handler.format_message())
                return result.response_text

            stream_handler = _stream_prompt

        steps = await plan_pc_actions(
            llm_client,
            task,
            session_context=planner_context,
            stream_handler=stream_handler,
        )
        if not steps:
            if require_prefix and not silent_fail:
                await context.application.bot.send_message(
                    chat_id=message.chat_id,
                    text="Could not build a PC action plan. Try rephrasing the task.",
                )
            return
        result = await run_task_with_approval(
            agent_context=agent_context,
            telegram_app=context.application,
            chat_id=str(message.chat_id),
            task=task,
            context={"steps": steps},
            user_id=user_id,
            llm_client=llm_client,
        )
        final_text = f"PC task finished: {result.reason}."

        suggestion_context = _build_suggestion_context(
            user_id=user_id,
            db=get_database() if user_id else None,
            session_context=get_session_context(user_id),
            user_message=task,
            assistant_message=final_text,
            task_type="computer_use",
            extra_context=f"Result: {result.reason}",
        )
        reply_markup = await _build_suggestion_markup(llm_client, suggestion_context)

        await context.application.bot.send_message(
            chat_id=message.chat_id,
            text=final_text,
            reply_markup=reply_markup,
        )

    context.application.create_task(_runner())
    return True
