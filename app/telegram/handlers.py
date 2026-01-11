import asyncio
import inspect
import io
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import logging
import mimetypes
from pathlib import Path
import random
import re
from uuid import uuid4

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from PIL import Image, ImageDraw, ImageFont

from app.agent.intent_router import RouterDecision, RouterIntent, route_intent
from app.agent.llm_client import (
    GEMINI_FLASH_MAX_DOCUMENT_BYTES,
    GEMINI_FLASH_MAX_INLINE_IMAGE_BYTES,
    GEMINI_IMAGE_MIME_TYPES,
)
from app.config import get_settings
from app.memory.long_term import ConversationMemoryManager, migrate_profile_json
from app.session_manager import (
    clear_session,
    get_session_context,
    record_image_result,
    record_user_message,
    should_reset_context,
)
from app.services.gmail_service import GMAIL_LABELS, DraftSpec, build_gmail_service
from app.storage.database import get_database
from app.storage.user_settings import is_anonymous_mode, set_anonymous_mode
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
    "retouch",
    "adjust",
)
_EXPLICIT_IMAGE_TOKENS = (
    "image",
    "photo",
    "picture",
    "visual",
    "pic",
    "resim",
    "gorsel",
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
    "what do you see",
    "is it suitable",
    "evaluate",
    "review it",
    "comment",
    "give feedback",
    "how does it look now",
    "nedir",
    "tanimla",
    "acikla",
    "incele",
    "ne goruyorsun",
    "uygun mu",
    "sence",
    "degerlendir",
    "yorumla",
    "fikir ver",
    "nasil duruyor",
    "nasil gorunuyor",
    "bu hali nasil",
    "simdi nasil",
)
_IMAGE_COMPARE_TOKENS = (
    "compare",
    "which",
    "prefer",
    "better",
    "best",
    "karsilastir",
    "hangisi",
    "hangisini",
    "tercih",
)
_CHAT_CONTEXT_LIMIT = 20
_SUGGESTION_PREFIX = "SUGG:"
_TELEGRAM_TEXT_CHUNK = 3500
_TELEGRAM_CAPTION_LIMIT = 1024
_DRAFT_PREVIEW_MAX_BODY = _TELEGRAM_TEXT_CHUNK - 200
_STREAM_RETRY_ATTEMPTS = 3
_STREAM_RETRY_BASE_DELAY = 0.6
_STREAM_RETRY_MAX_DELAY = 3.0
_SUGGESTION_RETRY_ATTEMPTS = 3
_SUGGESTION_RETRY_BASE_DELAY = 0.4
_SUGGESTION_RETRY_MAX_DELAY = 2.0
_TELEGRAM_REPLY_RETRY_ATTEMPTS = 2
_TELEGRAM_REPLY_RETRY_BASE_DELAY = 0.4
_TELEGRAM_REPLY_RETRY_MAX_DELAY = 2.0
_MAX_SOURCE_COUNT = 5
_PENDING_DOCUMENT_TTL_SECONDS = 20 * 60
_CHART_TOKENS = (
    "chart",
    "plot",
    "bar",
    "bar chart",
    "candlestick",
    "kandil",
    "ohlc",
    "line",
    "cizgi",
    "moving average",
    "hareketli ortalama",
    "column",
    "grafik",
    "sutun",
)
_CANDLE_TOKENS = (
    "candlestick",
    "ohlc",
    "kandil",
)
_LINE_TOKENS = (
    "line",
    "moving average",
    "cizgi",
    "hareketli ortalama",
)
_MA_TOKENS = (
    "moving average",
    "average",
    "ma",
    "ortalama",
)
_CHART_MAX_ITEMS = 8
_CHART_WIDTH = 900
_CHART_HEIGHT = 600
_BYTES_PER_MB = 1024 * 1024
_DEFAULT_AUDIO_PROMPT = "Transcribe the audio and provide a concise summary."
_DEFAULT_VIDEO_PROMPT = "Summarize the video and transcribe any spoken audio."
_CLEAR_HISTORY_COMMANDS = {
    "clear_history",
    "clear history",
    "/clear_history",
    "gecmis temizle",
    "sohbet gecmisi temizle",
    "sohbeti temizle",
    "gecmisi sil",
}
_CLEAR_HISTORY_CONFIRM_TOKENS = {"yes", "y", "confirm", "evet", "onay", "onayla"}
_CLEAR_HISTORY_CANCEL_TOKENS = {"no", "n", "cancel", "hayir", "iptal", "vazgec"}
_CLEAR_HISTORY_FLAG = "awaiting_clear_history_confirmation"
_THINKING_ON_COMMAND = "thinking_on"
_THINKING_OFF_COMMAND = "thinking_off"
_SCREENSHOT_ON_COMMAND = "screenshot on"
_SCREENSHOT_OFF_COMMAND = "screenshot off"
_BROWSER_USE_ON_COMMANDS = {"browser on", "browser use on"}
_BROWSER_USE_OFF_COMMANDS = {"browser off", "browser use off"}
_COMPUTER_USE_ON_COMMANDS = {"pc on", "computer on", "computer use on"}
_COMPUTER_USE_OFF_COMMANDS = {"pc off", "computer off", "computer use off"}
_ANON_ON_COMMANDS = {"anonymous on", "anonim on"}
_ANON_OFF_COMMANDS = {"anonymous off", "anonim off", "anonim kapat"}
_COMMAND_LIST_COMMANDS = {"commands", "/commands", "/help", "help", "komutlar", "/komutlar"}
_CLEAR_HISTORY_CONFIRM_DATA = "clear_history_yes"
_CLEAR_HISTORY_CANCEL_DATA = "clear_history_no"
_GMAIL_SEND_CONFIRM_DATA = "gmail_send_yes"
_GMAIL_SEND_CANCEL_DATA = "gmail_send_no"
_APPROVAL_ID_PATTERN = re.compile(
    r"approval_id=([A-Za-z0-9-]+)", re.IGNORECASE)
_GMAIL_APPROVE_TOKENS = {"yes", "y", "ok", "confirm", "approve", "evet", "onay", "onayla", "gonder"}
_GMAIL_REJECT_TOKENS = {"no", "n", "cancel", "reject", "hayir", "iptal", "vazgec"}

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


@dataclass(frozen=True)
class PendingDocument:
    path: str
    mime_type: str | None
    created_at: float


_PENDING_DOCUMENTS: dict[str, PendingDocument] = {}


def _resolve_user_id(update: Update) -> str | None:
    if update.effective_user and update.effective_user.id:
        return str(update.effective_user.id)
    if update.effective_chat and update.effective_chat.id:
        return str(update.effective_chat.id)
    return None


def _resolve_user_id_from_message(message) -> str | None:
    if message is None:
        return None
    user = getattr(message, "from_user", None)
    if user is not None and getattr(user, "id", None):
        return str(user.id)
    chat_id = getattr(message, "chat_id", None)
    if chat_id:
        return str(chat_id)
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


def _extract_approval_id(text: str | None) -> str | None:
    if not text:
        return None
    match = _APPROVAL_ID_PATTERN.search(text)
    if not match:
        return None
    return match.group(1)


def _extract_approval_id_from_reply(message) -> str | None:
    reply = getattr(message, "reply_to_message", None)
    if reply is None:
        return None
    reply_text = getattr(reply, "text", None) or getattr(
        reply, "caption", None) or ""
    return _extract_approval_id(reply_text)


def _build_reply_context_note(message, user_id: str | None = None) -> str | None:
    reply = getattr(message, "reply_to_message", None)
    if reply is None:
        return None
    entry = _lookup_reply_entry(reply, user_id)
    if entry is not None:
        note = _format_reply_entry(entry)
        if note:
            return note
    content = _extract_reply_content(reply)
    if not content:
        return None
    cleaned = content.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace('"', "'")
    return f'System Note: "The user is replying to this message: {cleaned}"'


def _lookup_reply_entry(reply, user_id: str | None):
    if not user_id:
        return None
    source_message_id = getattr(reply, "message_id", None)
    source_chat_id = getattr(reply, "chat_id", None)
    try:
        db = get_database()
    except Exception as exc:
        logger.warning("Reply context lookup failed: %s", exc)
        return None
    if source_message_id is not None:
        for source in ("telegram_callback", "telegram"):
            entry = db.get_message_by_source_id(
                user_id,
                source,
                str(source_message_id),
                str(source_chat_id) if source_chat_id is not None else None,
            )
            if entry is not None:
                return entry
    content = (getattr(reply, "text", None) or getattr(
        reply, "caption", None) or "").strip()
    if not content:
        return None
    reply_from = getattr(reply, "from_user", None)
    role = None
    if reply_from is not None:
        role = "assistant" if getattr(reply_from, "is_bot", False) else "user"
    return db.find_message_by_content(user_id, content, role=role)


def _format_reply_entry(entry) -> str | None:
    line = _format_history_line(entry)
    cleaned = line.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace('"', "'")
    return f'System Note: "The user is replying to this message: {cleaned}"'


def _extract_reply_content(reply) -> str | None:
    text = (getattr(reply, "text", None) or getattr(
        reply, "caption", None) or "").strip()
    if text:
        return text
    return _format_reply_metadata(reply)


def _format_reply_metadata(reply) -> str | None:
    parts: list[str] = []
    message_id = getattr(reply, "message_id", None)
    if message_id:
        parts.append(f"message_id={message_id}")
    photo = getattr(reply, "photo", None)
    if photo:
        photo_item = photo[-1]
        photo_id = getattr(photo_item, "file_unique_id", None) or getattr(
            photo_item, "file_id", None)
        if photo_id:
            parts.append(f"photo_id={photo_id}")
    document = getattr(reply, "document", None)
    if document:
        doc_name = getattr(document, "file_name", None)
        doc_id = getattr(document, "file_unique_id", None) or getattr(
            document, "file_id", None)
        if doc_name:
            parts.append(f"document_name={doc_name}")
        if doc_id:
            parts.append(f"document_id={doc_id}")
    video = getattr(reply, "video", None)
    if video:
        video_id = getattr(video, "file_unique_id", None) or getattr(
            video, "file_id", None)
        if video_id:
            parts.append(f"video_id={video_id}")
    video_note = getattr(reply, "video_note", None)
    if video_note:
        video_note_id = getattr(video_note, "file_unique_id", None) or getattr(
            video_note, "file_id", None)
        if video_note_id:
            parts.append(f"video_note_id={video_note_id}")
    audio = getattr(reply, "audio", None)
    if audio:
        audio_id = getattr(audio, "file_unique_id", None) or getattr(
            audio, "file_id", None)
        if audio_id:
            parts.append(f"audio_id={audio_id}")
    voice = getattr(reply, "voice", None)
    if voice:
        voice_id = getattr(voice, "file_unique_id", None) or getattr(
            voice, "file_id", None)
        if voice_id:
            parts.append(f"voice_id={voice_id}")
    sticker = getattr(reply, "sticker", None)
    if sticker:
        sticker_id = getattr(sticker, "file_unique_id",
                             None) or getattr(sticker, "file_id", None)
        if sticker_id:
            parts.append(f"sticker_id={sticker_id}")
    animation = getattr(reply, "animation", None)
    if animation:
        animation_id = getattr(animation, "file_unique_id", None) or getattr(
            animation, "file_id", None)
        if animation_id:
            parts.append(f"animation_id={animation_id}")
    if not parts:
        return None
    return ", ".join(parts)


def _extract_reply_photo(message):
    reply = getattr(message, "reply_to_message", None)
    if reply is None:
        return None
    photo = getattr(reply, "photo", None)
    if photo:
        return photo[-1]
    document = getattr(reply, "document", None)
    if document and (document.mime_type or "").startswith("image/"):
        return document
    return None


def _build_callback_source_meta(query) -> dict[str, str | None] | None:
    if query is None:
        return None
    message = getattr(query, "message", None)
    if message is None:
        return None
    message_id = getattr(message, "message_id", None)
    if message_id is None:
        return None
    chat_id = getattr(message, "chat_id", None)
    return {
        "source": "telegram_callback",
        "source_message_id": str(message_id),
        "source_chat_id": str(chat_id) if chat_id is not None else None,
    }


def _user_source_kwargs(message, source_meta: dict[str, str | None] | None) -> dict[str, str | None]:
    if source_meta:
        return source_meta
    return {"message": message}


def _inject_reply_context(prompt_text: str, reply_note: str | None) -> str:
    if not reply_note:
        return prompt_text
    if not prompt_text:
        return reply_note
    return f"{reply_note}\n\n{prompt_text}"


def _resolve_settings(context: ContextTypes.DEFAULT_TYPE):
    settings = context.application.bot_data.get("settings")
    return settings or get_settings()


def _get_memory_manager(telegram_app, settings):
    manager = telegram_app.bot_data.get("memory_manager")
    if manager is None:
        manager = ConversationMemoryManager(settings)
        telegram_app.bot_data["memory_manager"] = manager
    return manager


def _get_gmail_service(context: ContextTypes.DEFAULT_TYPE):
    service = context.application.bot_data.get("gmail_service")
    if service is not None:
        return service
    settings = _resolve_settings(context)
    service = build_gmail_service(settings)
    context.application.bot_data["gmail_service"] = service
    return service


def _parse_limit_arg(value: str | None, default: int = 5, max_value: int = 25) -> int:
    if not value:
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return max(1, min(max_value, parsed))


def _extract_first_email(text: str) -> str:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else ""


def _extract_limit_from_text(text: str, default: int = 5, max_value: int = 25) -> int:
    match = re.search(r"\b(\d{1,3})\b", text)
    if not match:
        return default
    try:
        return _parse_limit_arg(match.group(1), default=default, max_value=max_value)
    except ValueError:
        return default


def _resolve_gmail_time_hint(text: str) -> str | None:
    normalized = _normalize_text(text or "")
    if "today" in normalized or "bugun" in normalized or "bu gun" in normalized:
        return "newer_than:1d"
    if "dun" in normalized or "yesterday" in normalized:
        return "newer_than:2d"
    if "bu hafta" in normalized or "this week" in normalized:
        return "newer_than:7d"
    if "son hafta" in normalized or "last week" in normalized:
        return "newer_than:14d"
    if "bu ay" in normalized or "this month" in normalized:
        return "newer_than:30d"
    if "son ay" in normalized or "last month" in normalized:
        return "newer_than:60d"
    return None


def _should_send_email(text: str) -> bool:
    normalized = _normalize_text(text or "")
    return any(
        token in normalized
        for token in (
            "send",
            "email",
            "mail",
            "deliver",
            "gonder",
            "yolla",
            "eposta at",
            "mail at",
            "gonderir misin",
            "gonder",
            "gondersene",
        )
    )


def _should_summarize_email(text: str) -> bool:
    normalized = _normalize_text(text or "")
    return any(token in normalized for token in ("summary", "summarize", "ozet", "ozetle", "kisa ozet"))


def _store_gmail_pending_send(context: ContextTypes.DEFAULT_TYPE, question_id: str, payload: dict) -> None:
    pending = context.application.bot_data.get("gmail_pending_send")
    if not isinstance(pending, dict):
        pending = {}
    pending[question_id] = payload
    context.application.bot_data["gmail_pending_send"] = pending


def _pop_gmail_pending_send(context: ContextTypes.DEFAULT_TYPE, question_id: str) -> dict | None:
    pending = context.application.bot_data.get("gmail_pending_send")
    if not isinstance(pending, dict):
        return None
    return pending.pop(question_id, None)


def _parse_json_block(text: str) -> dict | None:
    if not text:
        return None
    match = re.search(r"{.*}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _parse_subject_body(text: str) -> tuple[str, str]:
    data = _parse_json_block(text) or {}
    subject = str(data.get("subject") or "").strip()
    body = str(data.get("body") or "").strip()
    return subject, body


def _contains_word_token(text: str, tokens: set[str]) -> bool:
    for token in tokens:
        if not token:
            continue
        if re.search(rf"\b{re.escape(token)}\b", text):
            return True
    return False


def _extract_draft_request(llm_client, message_text: str) -> dict:
    if llm_client is None:
        email = _extract_first_email(message_text)
        return {
            "to": email,
            "subject": "",
            "prompt": message_text.strip(),
            "needs_clarification": not bool(email),
            "clarification_question": "Who should I send it to? Please provide an email address.",
        }
        prompt = (
            "Extract email draft details from the user message.\n"
            "Return ONLY JSON with keys: to, subject, prompt, needs_clarification, clarification_question.\n"
            "- 'to' should be an email address if present.\n"
            "- 'subject' can be empty.\n"
            "- 'prompt' is the body request.\n"
            "- If recipient missing, set needs_clarification=true and provide a short English question.\n\n"
            f"Message: {message_text}\n"
        )
    response = llm_client.generate_text(prompt)
    data = _parse_json_block(response) or {}
    return {
        "to": str(data.get("to") or "").strip(),
        "subject": str(data.get("subject") or "").strip(),
        "prompt": str(data.get("prompt") or "").strip() or message_text.strip(),
        "needs_clarification": bool(data.get("needs_clarification")),
        "clarification_question": str(data.get("clarification_question") or "").strip()
        or "Who should I send it to? Please provide an email address.",
    }


def _build_draft_preview_text(spec: DraftSpec, max_body_chars: int = _DRAFT_PREVIEW_MAX_BODY) -> str:
    body = (spec.body or "").strip()
    if len(body) > max_body_chars:
        body = f"{body[:max_body_chars].rstrip()}..."
    return f"To: {spec.to}\nSubject: {spec.subject}\n\n{body}"


async def _maybe_send_gmail_status(
    context: ContextTypes.DEFAULT_TYPE,
    message,
    text: str,
) -> None:
    settings = _resolve_settings(context)
    if not bool(getattr(settings, "show_thoughts", True)):
        return
    try:
        await message.reply_text(text)
    except Exception as exc:
        logger.debug("Gmail status message failed: %s", exc)


def _extract_signature_name(user_id: str | None) -> str | None:
    if not user_id:
        return None
    db = get_database()
    profile_json = db.get_user_profile(user_id)
    if not profile_json:
        return None
    migrated_json, migrated = migrate_profile_json(profile_json)
    if migrated:
        profile_json = migrated_json
        db.set_user_profile(user_id, profile_json)
    try:
        profile = json.loads(profile_json)
    except Exception:
        return None
    if not isinstance(profile, dict):
        return None
    facts = profile.get("facts")
    if not isinstance(facts, list) or not facts:
        return None
    normalized_map: dict[str, str] = {}
    for item in facts:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        if not key or not value:
            continue
        normalized_map[_normalize_text(key)] = value
    full_name_keys = (
        "ad soyad",
        "isim soyisim",
        "full name",
        "name",
        "isim",
    )
    for key in full_name_keys:
        if key in normalized_map:
            return normalized_map[key]
    first = normalized_map.get("ad") or normalized_map.get("first name") or normalized_map.get("first_name")
    last = normalized_map.get("soyad") or normalized_map.get("last name") or normalized_map.get("last_name")
    if first and last:
        return f"{first} {last}".strip()
    return first or last


def _apply_signature(spec: DraftSpec, signature_name: str | None) -> DraftSpec:
    if not signature_name:
        return spec
    body = (spec.body or "").rstrip()
    if body.endswith(signature_name):
        return spec
    updated_body = f"{body}\n\n{signature_name}" if body else signature_name
    return DraftSpec(to=spec.to, subject=spec.subject, body=updated_body)


def _build_revision_prompt(subject: str, body: str, instruction: str) -> str:
    return (
        "You are revising an email draft.\n"
        "Return ONLY JSON with keys: subject, body.\n"
        "Keep the recipient context the same. Apply the user's instruction.\n\n"
        f"Current subject: {subject}\n"
        f"Current body:\n{body}\n\n"
        f"Instruction: {instruction}\n"
    )


def _parse_gmail_action_plan(intent: RouterIntent, payload: dict) -> dict:
    operation = str(payload.get("operation") or "").strip().lower()
    if not operation:
        operation = {
            RouterIntent.GMAIL_INBOX: "inbox",
            RouterIntent.GMAIL_SUMMARY: "summary",
            RouterIntent.GMAIL_SEARCH: "search",
            RouterIntent.GMAIL_DRAFT: "draft",
            RouterIntent.GMAIL_SEND: "send",
            RouterIntent.GMAIL_QUESTION: "question",
        }.get(intent, "question")
    limit = payload.get("limit")
    try:
        limit_value = int(limit) if limit is not None else None
    except (TypeError, ValueError):
        limit_value = None
    post_actions = payload.get("post_actions") if isinstance(payload.get("post_actions"), list) else []
    vip_senders = payload.get("vip_senders") if isinstance(payload.get("vip_senders"), list) else []
    return {
        "operation": operation,
        "query": str(payload.get("query") or "").strip(),
        "question": str(payload.get("question") or "").strip(),
        "limit": limit_value,
        "draft": payload.get("draft") if isinstance(payload.get("draft"), dict) else {},
        "post_actions": [str(item).strip() for item in post_actions if str(item).strip()],
        "vip_senders": [str(item).strip() for item in vip_senders if str(item).strip()],
        "focus_mode": payload.get("focus_mode"),
        "vip_only": payload.get("vip_only"),
    }


async def _handle_gmail_intelligent_request(
    context: ContextTypes.DEFAULT_TYPE,
    message,
    *,
    intent: RouterIntent,
    message_text: str,
) -> bool:
    llm_client = context.application.bot_data.get("llm_client")
    if llm_client is None:
        return False
    await _maybe_send_gmail_status(context, message, "Gmail: processing request...")
    try:
        gmail_service = _get_gmail_service(context)
    except Exception as exc:
        logger.warning("Gmail service unavailable: %s", exc)
        await message.reply_text("Gmail service is not configured.")
        return True
    prompt = (
        "You are a Gmail action planner. Decide the best action for the user request.\n"
        "Return ONLY JSON with keys:\n"
        "operation: inbox|summary|search|question|thread_summary|extract_tasks|extract_schedule|sentiment|draft|send|count|newsletter_digest|prioritize\n"
        "query: optional Gmail-style query or natural language\n"
        "limit: optional integer\n"
        "question: optional question to answer\n"
        "draft: {to, subject, prompt} when drafting/sending\n"
        "post_actions: optional list from [archive, label_later, label_newsletter]\n"
        "vip_senders: optional list of emails\n"
        "focus_mode: optional true/false\n"
        "vip_only: optional true/false\n\n"
        f"User message: {message_text}\n"
    )
    response = llm_client.generate_text(prompt)
    payload = _parse_json_block(response) or {}
    plan = _parse_gmail_action_plan(intent, payload)

    settings = _resolve_settings(context)
    if plan.get("focus_mode") is not None or plan.get("vip_only") is not None or plan.get("vip_senders"):
        updated = replace(
            settings,
            gmail_focus_mode=bool(plan.get("focus_mode")) if plan.get("focus_mode") is not None else settings.gmail_focus_mode,
            gmail_vip_only=bool(plan.get("vip_only")) if plan.get("vip_only") is not None else settings.gmail_vip_only,
            gmail_vip_senders=",".join(plan.get("vip_senders")) if plan.get("vip_senders") else settings.gmail_vip_senders,
        )
        context.application.bot_data["settings"] = updated

    operation = plan.get("operation")
    limit = plan.get("limit") or _extract_limit_from_text(message_text, default=5, max_value=25)
    query_text = plan.get("query") or message_text

    if operation in {"draft", "send"}:
        draft_data = plan.get("draft") or {}
        recipient = str(draft_data.get("to") or "").strip() or _extract_first_email(message_text)
        subject = str(draft_data.get("subject") or "").strip()
        prompt_text = str(draft_data.get("prompt") or "").strip() or message_text
        if not recipient:
            await message.reply_text("Who should I send it to? Please provide an email address.")
            return True
        try:
            spec = await asyncio.to_thread(
                gmail_service.build_draft_from_prompt, llm_client, recipient, prompt_text
            )
            if subject:
                spec = DraftSpec(to=spec.to, subject=subject, body=spec.body)
            signature_name = _extract_signature_name(_resolve_user_id_from_message(message))
            spec = _apply_signature(spec, signature_name)
            draft_id = await asyncio.to_thread(gmail_service.create_draft, spec)
        except Exception as exc:
            logger.warning("Gmail draft failed: %s", exc)
            await message.reply_text("Failed to create Gmail draft.")
            return True
        agent_context = context.application.bot_data.get("agent_context")
        if operation == "send" or _should_send_email(message_text):
            if agent_context is None:
                await message.reply_text("Draft created, but approval flow is unavailable.")
                return True
            question_id = str(uuid4())
            agent_context.create_pending_request(
                question_id=question_id,
                intent="gmail_send",
                question="Gmail draft is ready. Send it?",
                category="gmail_send",
            )
            _store_gmail_pending_send(
                context,
                question_id,
                {
                    "draft_id": draft_id,
                    "to": spec.to,
                    "subject": spec.subject,
                    "body": spec.body,
                },
            )
            preview_text = _build_draft_preview_text(spec)
            await message.reply_text(
                f"Draft ready:\n{preview_text}\n\nSend it?",
                reply_markup=_build_gmail_send_keyboard(),
            )
            return True
        await message.reply_text(f"Gmail draft created: {draft_id}")
        return True

    if operation in {"inbox", "summary"}:
        query_hint = _resolve_gmail_time_hint(message_text)
        if operation == "inbox":
            await _handle_gmail_inbox(
                context,
                message,
                limit=limit,
                message_text=message_text,
                query_hint=query_hint,
            )
            return True
        await _handle_gmail_summary(
            context,
            message,
            limit=limit,
            message_text=message_text,
            query_hint=query_hint,
        )
        return True

    gmail_query = await asyncio.to_thread(gmail_service.build_search_query, llm_client, query_text)
    try:
        results = await asyncio.to_thread(gmail_service.search, gmail_query, max(10, limit))
    except Exception as exc:
        logger.warning("Gmail search failed: %s", exc)
        await message.reply_text("Gmail search failed.")
        return True
    if not results:
        await message.reply_text("No emails matched your request.")
        return True

    if operation == "count":
        await message.reply_text(f"Matched emails: {len(results)}")
        return True

    if operation == "thread_summary":
        thread_id = results[0].thread_id if results else None
        thread_messages = await asyncio.to_thread(gmail_service.get_thread_messages, thread_id or "")
        if not thread_messages:
            await message.reply_text("Thread not found.")
            return True
        summary = await asyncio.to_thread(gmail_service.summarize_messages, llm_client, thread_messages)
        await message.reply_text(summary)
        return True

    if operation == "search":
        lines = [f"Search results ({gmail_query}):"]
        for mail in results[:limit]:
            sender = mail.sender or "Unknown sender"
            subject = mail.subject or "(no subject)"
            lines.append(f"- {sender} | {subject}")
        await message.reply_text("\n".join(lines))
        return True

    items = []
    for mail in results[:limit]:
        body = (mail.body or mail.snippet or "").strip()
        trimmed = body[:1200]
        items.append(
            f"- From: {mail.sender}\n"
            f"  Subject: {mail.subject}\n"
            f"  Snippet: {mail.snippet or ''}\n"
            f"  Body: {trimmed}"
        )
    if operation == "summary":
        summary = await asyncio.to_thread(gmail_service.summarize_messages, llm_client, results[:limit])
        await message.reply_text(summary)
        return True
    if operation == "newsletter_digest":
        prompt = (
            "Summarize the top 3 newsletter highlights from the emails below. "
            "Keep it concise in Turkish.\n\n"
            + "\n\n".join(items)
        )
        response = await asyncio.to_thread(llm_client.generate_text, prompt)
        await message.reply_text(response.strip())
    elif operation == "extract_tasks":
        prompt = (
            "Extract concrete action items (to-do list) from the emails below. "
            "Return bullet points in Turkish.\n\n"
            + "\n\n".join(items)
        )
        response = await asyncio.to_thread(llm_client.generate_text, prompt)
        await message.reply_text(response.strip())
    elif operation == "extract_schedule":
        prompt = (
            "Extract meeting times, dates, and locations from the emails below. "
            "Return a concise list ready to add to a calendar.\n\n"
            + "\n\n".join(items)
        )
        response = await asyncio.to_thread(llm_client.generate_text, prompt)
        await message.reply_text(response.strip())
    elif operation == "sentiment":
        prompt = (
            "Analyze the tone of the emails below. "
            "Say if the sender is angry, urgent, or neutral, and why.\n\n"
            + "\n\n".join(items)
        )
        response = await asyncio.to_thread(llm_client.generate_text, prompt)
        await message.reply_text(response.strip())
    elif operation == "prioritize":
        prompt = (
            "From these emails, pick the top 3 that require action. "
            "Explain briefly why each needs action.\n\n"
            + "\n\n".join(items)
        )
        response = await asyncio.to_thread(llm_client.generate_text, prompt)
        await message.reply_text(response.strip())
    else:
        question = plan.get("question") or message_text
        prompt = (
            "Answer the user's question using the emails below. "
            "If the question asks for a count, provide the count.\n\n"
            f"Question: {question}\n\n"
            + "\n\n".join(items)
        )
        response = await asyncio.to_thread(llm_client.generate_text, prompt)
        await message.reply_text(response.strip())

    post_actions = plan.get("post_actions") or []
    if post_actions:
        label_later = getattr(settings, "gmail_later_label", "Later") or "Later"
        for mail in results[:limit]:
            if "label_later" in post_actions:
                await asyncio.to_thread(gmail_service.label_message, mail.message_id, label_later)
            if "archive" in post_actions:
                await asyncio.to_thread(gmail_service.archive_message, mail.message_id)
            if "label_newsletter" in post_actions:
                await asyncio.to_thread(gmail_service.label_message, mail.message_id, "Newsletter")
    return True


def build_main_keyboard() -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton(
        text=label, callback_data=value)] for label, value in _BUTTONS]
    return InlineKeyboardMarkup(keyboard)


def _normalize_command_text(value: str) -> str:
    normalized = _normalize_text(value or "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _resolve_anonymous_command(normalized: str) -> bool | None:
    if any(normalized.startswith(token) for token in _ANON_ON_COMMANDS):
        return True
    if any(normalized.startswith(token) for token in _ANON_OFF_COMMANDS):
        return False
    return None


def _store_pending_document(user_id: str, path: str, mime_type: str | None) -> None:
    _PENDING_DOCUMENTS[user_id] = PendingDocument(
        path=path,
        mime_type=mime_type,
        created_at=time.monotonic(),
    )


def _pop_pending_document(user_id: str | None) -> PendingDocument | None:
    if not user_id:
        return None
    pending = _PENDING_DOCUMENTS.pop(user_id, None)
    if not pending:
        return None
    if time.monotonic() - pending.created_at > _PENDING_DOCUMENT_TTL_SECONDS:
        return None
    if not pending.path or not Path(pending.path).exists():
        return None
    return pending


def _is_clear_history_command(normalized: str) -> bool:
    return normalized in _CLEAR_HISTORY_COMMANDS


def _build_clear_history_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(
                text="YES", callback_data=_CLEAR_HISTORY_CONFIRM_DATA),
            InlineKeyboardButton(
                text="NO", callback_data=_CLEAR_HISTORY_CANCEL_DATA),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def _build_gmail_send_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(text="Yes", callback_data=_GMAIL_SEND_CONFIRM_DATA),
            InlineKeyboardButton(text="No", callback_data=_GMAIL_SEND_CANCEL_DATA),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def _is_thinking_command(normalized: str) -> bool:
    return normalized in {_THINKING_ON_COMMAND, _THINKING_OFF_COMMAND}


def _set_show_thoughts(context: ContextTypes.DEFAULT_TYPE, enabled: bool) -> bool:
    settings = _resolve_settings(context)
    if bool(getattr(settings, "show_thoughts", True)) == enabled:
        return False
    context.application.bot_data["settings"] = replace(
        settings, show_thoughts=enabled)
    return True


def _is_screenshot_command(normalized: str) -> bool:
    return normalized in {_SCREENSHOT_ON_COMMAND, _SCREENSHOT_OFF_COMMAND}


def _set_screenshot_enabled(context: ContextTypes.DEFAULT_TYPE, enabled: bool) -> bool:
    settings = _resolve_settings(context)
    if bool(getattr(settings, "screenshot_enabled", True)) == enabled:
        return False
    context.application.bot_data["settings"] = replace(
        settings, screenshot_enabled=enabled)
    return True


def _resolve_browser_use_command(normalized: str) -> bool | None:
    if normalized in _BROWSER_USE_ON_COMMANDS:
        return True
    if normalized in _BROWSER_USE_OFF_COMMANDS:
        return False
    return None


def _resolve_computer_use_command(normalized: str) -> bool | None:
    if normalized in _COMPUTER_USE_ON_COMMANDS:
        return True
    if normalized in _COMPUTER_USE_OFF_COMMANDS:
        return False
    return None


def _set_browser_use_enabled(context: ContextTypes.DEFAULT_TYPE, enabled: bool) -> bool:
    settings = _resolve_settings(context)
    if bool(getattr(settings, "browser_use_enabled", True)) == enabled:
        return False
    context.application.bot_data["settings"] = replace(
        settings, browser_use_enabled=enabled)
    return True


def _set_computer_use_enabled(context: ContextTypes.DEFAULT_TYPE, enabled: bool) -> bool:
    settings = _resolve_settings(context)
    if bool(getattr(settings, "computer_use_enabled", True)) == enabled:
        return False
    context.application.bot_data["settings"] = replace(
        settings, computer_use_enabled=enabled)
    return True


def _is_command_list_command(normalized: str) -> bool:
    return normalized in _COMMAND_LIST_COMMANDS


def _build_command_list_text() -> str:
    return "\n".join(
        (
            "Available commands:",
            "- clear_history or /clear_history: clears chat history.",
            "- thinking_on / thinking_off: toggles thought streaming.",
            "- screenshot on / screenshot off: toggles post-step screenshots.",
            "- browser on / browser off: toggles browser automation.",
            "- pc on / pc off: toggles computer control.",
            "- anonymous on / anonymous off: disables/enables chat logging.",
            "- /memory: lists stored memory summaries.",
            "- /forget <id>: deletes the selected memory item.",
            "- /profile: shows stored profile facts.",
            "- /profile_forget <id>: deletes the selected profile fact.",
            "- /gmail_auth: authorizes Gmail access.",
            "- /inbox [n]: lists unread emails.",
            "- /summarize_last [n]: summarizes unread emails.",
            "- /search_mail <query>: searches Gmail.",
            "- /draft_mail <to> | <subject> | <prompt>: creates a Gmail draft.",
            "- job_search: starts job search (button).",
            "- job_search stop: stops job search.",
            "- stop: stops all actions.",
            "- /pc <task> or pc: <task>: starts a PC control task.",
        )
    )


def _resolve_clear_history_action(value: str) -> str | None:
    if value == _CLEAR_HISTORY_CONFIRM_DATA:
        return "confirm"
    if value == _CLEAR_HISTORY_CANCEL_DATA:
        return "cancel"
    return None


def _is_job_search_stop_command(normalized: str) -> bool:
    return normalized in {"job_search stop", "job search stop"}


def _has_clear_history_pending(context: ContextTypes.DEFAULT_TYPE) -> bool:
    return bool(context.user_data.get(_CLEAR_HISTORY_FLAG))


def _set_clear_history_pending(context: ContextTypes.DEFAULT_TYPE, pending: bool) -> None:
    if pending:
        context.user_data[_CLEAR_HISTORY_FLAG] = True
        return
    context.user_data.pop(_CLEAR_HISTORY_FLAG, None)


def _confirmation_token(normalized: str) -> str:
    return normalized.split()[0] if normalized else ""


async def _request_clear_history_confirmation(message, context: ContextTypes.DEFAULT_TYPE) -> None:
    _set_clear_history_pending(context, True)
    await message.reply_text(
        "Clear chat history? Tap YES to confirm or NO to cancel.",
        reply_markup=_build_clear_history_keyboard(),
    )


async def _handle_clear_history_confirmation(
    message,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: str | None,
    normalized: str,
) -> bool:
    if not _has_clear_history_pending(context):
        return False
    token = _confirmation_token(normalized)
    if token in _CLEAR_HISTORY_CONFIRM_TOKENS:
        _set_clear_history_pending(context, False)
        await _clear_user_history(message, context, user_id)
        return True
    if token in _CLEAR_HISTORY_CANCEL_TOKENS:
        _set_clear_history_pending(context, False)
        await message.reply_text("Clear history canceled.")
        return True
    await message.reply_text("Please reply YES or NO to confirm clear history.")
    return True


async def _clear_user_history(
    message,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: str | None,
) -> None:
    if not user_id:
        await message.reply_text("Unable to clear history (missing user id).")
        return
    db = get_database()
    session_context = get_session_context(user_id)
    extra_paths: list[str] = []
    if session_context:
        extra_paths.extend(_collect_recent_image_paths(session_context))
    file_paths = _safe_delete_messages(db, user_id)
    if extra_paths:
        file_paths = list({*file_paths, *extra_paths})
    deleted_files = _delete_local_files(file_paths)
    clear_session(user_id)
    _clear_legacy_session_file(user_id)
    if deleted_files:
        await message.reply_text("History cleared. Stored messages and local files were removed.")
        return
    await message.reply_text("History cleared. Stored messages were removed.")


def _safe_delete_messages(db, user_id: str) -> list[str]:
    try:
        return db.delete_messages(user_id)
    except Exception as exc:
        logger.warning("Message history delete failed: %s", exc)
        return []


def _delete_local_files(file_paths: list[str]) -> int:
    if not file_paths:
        return 0
    deleted = 0
    root = ROOT_DIR.resolve()
    for value in file_paths:
        if not value:
            continue
        try:
            path = Path(value).expanduser().resolve()
        except Exception as exc:
            logger.warning("Failed to resolve path for deletion: %s", exc)
            continue
        if root not in path.parents and path != root:
            logger.warning("Skipping delete for non-project path: %s", path)
            continue
        if not path.exists() or not path.is_file():
            continue
        try:
            path.unlink()
            deleted += 1
        except Exception as exc:
            logger.warning("File delete failed: %s", exc)
    return deleted


def _clear_legacy_session_file(user_id: str | None) -> bool:
    if not user_id:
        return False
    path = ROOT_DIR / "sessions.json"
    if not path.exists():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read sessions.json: %s", exc)
        return False
    if not isinstance(raw, dict):
        logger.warning("sessions.json is not a dict; skipping cleanup.")
        return False
    if str(user_id) not in raw:
        return False
    raw.pop(str(user_id), None)
    try:
        path.write_text(json.dumps(raw, ensure_ascii=True,
                        indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to update sessions.json: %s", exc)
        return False
    return True


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "Atlas is online. System integrity confirmed.\n\n"
        "Gmail layers are synced, and the Gemini core is standing by. I've indexed your latest "
        "communications and prepared your digital workspace. Your Chief of Staff is ready to orchestrate.\n\n"
        "How shall we direct our focus today?",
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

    clear_action = _resolve_clear_history_action(str(query.data or ""))
    if clear_action:
        message = query.message
        if message is None:
            return
        if not _has_clear_history_pending(context):
            await message.reply_text("Clear history request expired. Send clear_history again.")
            return
        _set_clear_history_pending(context, False)
        if clear_action == "confirm":
            user_id = _resolve_user_id(update)
            await _clear_user_history(message, context, user_id)
        else:
            await message.reply_text("Clear history canceled.")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception as exc:
            logger.debug("Failed to clear history buttons: %s", exc)
        return

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
        if pending.category == "gmail_send":
            message = query.message
            if message is not None:
                handled = await _handle_gmail_send_approval(
                    context,
                    message,
                    agent_context=agent_context,
                    reply_text=str(query.data or ""),
                )
                if handled:
                    try:
                        await query.edit_message_reply_markup(reply_markup=None)
                    except Exception as exc:
                        logger.debug("Failed to clear Gmail send buttons: %s", exc)
                    return
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
        source_meta = _build_callback_source_meta(query)
        user_id = _resolve_user_id(update)
        memory_text = _memory_text(message_text)
        if user_id:
            if should_reset_context(user_id, memory_text):
                clear_session(user_id)
            if not is_anonymous_mode(user_id):
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
            source_meta=source_meta,
            # Force ignore pending (e.g. startup) for dynamic buttons
            pending=None,
        )


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    user_id = _resolve_user_id(update)
    normalized = _normalize_command_text(message.text)
    approval_id = _extract_approval_id_from_reply(message)
    reply_photo = _extract_reply_photo(message)
    reply_doc = None
    reply_msg = getattr(message, "reply_to_message", None)
    if reply_msg is not None:
        reply_doc = getattr(reply_msg, "document", None)
        if reply_doc is not None and (getattr(reply_doc, "mime_type", "") or "").startswith("image/"):
            reply_doc = None

    anon_action = _resolve_anonymous_command(normalized)
    if anon_action is not None:
        if user_id:
            set_anonymous_mode(user_id, anon_action)
            status = "enabled" if anon_action else "disabled"
            await message.reply_text(f"Anonymous mode {status}.")
        else:
            await message.reply_text("Anonymous mode requires a user id.")
        return

    if not approval_id:
        if _is_clear_history_command(normalized):
            await _request_clear_history_confirmation(message, context)
            return
        if _has_clear_history_pending(context):
            handled = await _handle_clear_history_confirmation(message, context, user_id, normalized)
            if handled:
                return
        if _is_thinking_command(normalized):
            enabled = normalized == _THINKING_ON_COMMAND
            changed = _set_show_thoughts(context, enabled)
            status = "enabled" if enabled else "disabled"
            suffix = " now" if changed else " already"
            await message.reply_text(f"Thinking is{suffix} {status}.")
            return
        if _is_screenshot_command(normalized):
            enabled = normalized == _SCREENSHOT_ON_COMMAND
            changed = _set_screenshot_enabled(context, enabled)
            status = "enabled" if enabled else "disabled"
            suffix = " now" if changed else " already"
            await message.reply_text(f"Screenshots are{suffix} {status}.")
            return
        browser_toggle = _resolve_browser_use_command(normalized)
        if browser_toggle is not None:
            changed = _set_browser_use_enabled(context, browser_toggle)
            status = "enabled" if browser_toggle else "disabled"
            suffix = " now" if changed else " already"
            await message.reply_text(f"Browser use is{suffix} {status}.")
            return
        computer_toggle = _resolve_computer_use_command(normalized)
        if computer_toggle is not None:
            changed = _set_computer_use_enabled(context, computer_toggle)
            status = "enabled" if computer_toggle else "disabled"
            suffix = " now" if changed else " already"
            await message.reply_text(f"Computer use is{suffix} {status}.")
            return
        if _is_command_list_command(normalized):
            await message.reply_text(_build_command_list_text())
            return

    memory_text = _memory_text(message.text)
    if user_id:
        if should_reset_context(user_id, memory_text):
            clear_session(user_id)
        if not is_anonymous_mode(user_id):
            record_user_message(user_id, memory_text)

    agent_context = context.application.bot_data.get("agent_context")
    if agent_context is None:
        await message.reply_text("Agent is not ready. Please try again later.")
        return

    if approval_id:
        handled = await _handle_approval_reply(message, agent_context, approval_id)
        if handled:
            return

    if _is_job_search_stop_command(normalized):
        agent_context.request_stop("telegram_job_search_stop")
        await message.reply_text("Job search stopped.")
        return
    if normalized == "stop":
        agent_context.request_stop("telegram_stop")
        await message.reply_text("Stopping all actions. You can send RESUME to continue later.")
        return

    pending = agent_context.pending_request
    if pending is not None and pending.category == "gmail_send":
        handled = await _handle_gmail_send_approval(
            context,
            message,
            agent_context=agent_context,
            reply_text=message.text,
        )
        if handled:
            return
    if pending is not None and pending.category != "startup":
        handled = await _handle_pending_reply(message, agent_context, message.text)
        if handled:
            return

    if reply_doc is not None:
        await _handle_document_request(
            message=message,
            context=context,
            caption=message.text,
            document=reply_doc,
            user_id=user_id,
            session_context=get_session_context(user_id),
        )
        return

    pending_doc = _pop_pending_document(user_id)
    if pending_doc is not None:
        await _handle_document_request_from_path(
            message=message,
            context=context,
            caption=message.text,
            file_path=pending_doc.path,
            mime_type=pending_doc.mime_type,
            user_id=user_id,
            session_context=get_session_context(user_id),
        )
        return

    await _route_and_handle_message(
        update=update,
        context=context,
        agent_context=agent_context,
        user_id=user_id,
        message_text=message.text,
        has_image=bool(reply_photo),
        photo=reply_photo,
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
            if not is_anonymous_mode(user_id):
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
    if not caption.strip() and user_id:
        document_size = getattr(document, "file_size", None)
        if document_size and document_size > GEMINI_FLASH_MAX_DOCUMENT_BYTES:
            await message.reply_text(
                _file_too_large_message(
                    "Document", GEMINI_FLASH_MAX_DOCUMENT_BYTES)
            )
            return
        try:
            temp_path = await _download_document_to_temp(document, message)
        except Exception as exc:  # pragma: no cover - network/IO errors
            logger.error("Document download failed: %s", exc)
            await message.reply_text("Document download failed. Please try again.")
            return
        if document_size is None and temp_path:
            try:
                downloaded_size = Path(temp_path).stat().st_size
            except Exception as exc:
                logger.warning("Document size check failed: %s", exc)
                downloaded_size = None
            if downloaded_size and downloaded_size > GEMINI_FLASH_MAX_DOCUMENT_BYTES:
                await message.reply_text(
                    _file_too_large_message(
                        "Document", GEMINI_FLASH_MAX_DOCUMENT_BYTES)
                )
                return
        _store_pending_document(user_id, temp_path, getattr(document, "mime_type", None))
        await message.reply_text("Please add a caption describing what to do with the file.")
        return
    if caption:
        memory_text = _memory_text(caption)
        if user_id:
            if should_reset_context(user_id, memory_text):
                clear_session(user_id)
            if not is_anonymous_mode(user_id):
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


async def handle_media_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    media, media_kind, default_prompt = _extract_media_payload(message)
    if media is None:
        return
    user_id = _resolve_user_id(update)
    caption = message.caption or ""
    if caption:
        memory_text = _memory_text(caption)
        if user_id:
            if should_reset_context(user_id, memory_text):
                clear_session(user_id)
            if not is_anonymous_mode(user_id):
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

    await _handle_media_request(
        message=message,
        context=context,
        caption=caption,
        media=media,
        user_id=user_id,
        session_context=get_session_context(user_id),
        media_kind=media_kind,
        default_prompt=default_prompt,
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
        if not is_anonymous_mode(user_id):
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


async def handle_memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    user_id = _resolve_user_id(update)
    if not user_id:
        await message.reply_text("No user id available for memory lookup.")
        return
    db = get_database()
    items = db.get_memory_items(user_id, limit=10)
    if not items:
        await message.reply_text("No memory items found.")
        return
    lines = ["Memory items:"]
    for item in items:
        date_label = _format_memory_date(item.created_at)
        title = item.title or "Memory"
        lines.append(f"- {item.id}: {title} ({date_label})")
    await message.reply_text("\n".join(lines))


async def handle_forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    user_id = _resolve_user_id(update)
    if not user_id:
        await message.reply_text("No user id available for memory deletion.")
        return
    parts = message.text.strip().split(maxsplit=1)
    if len(parts) < 2:
        await message.reply_text("Usage: /forget <memory_id>")
        return
    try:
        memory_id = int(parts[1])
    except ValueError:
        await message.reply_text("Memory id must be a number.")
        return
    db = get_database()
    deleted = db.delete_memory_item(user_id, memory_id)
    if deleted:
        await message.reply_text(f"Memory item {memory_id} deleted.")
    else:
        await message.reply_text("Memory item not found.")


async def handle_profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    user_id = _resolve_user_id(update)
    if not user_id:
        await message.reply_text("No user id available for profile lookup.")
        return
    db = get_database()
    profile_json = db.get_user_profile(user_id)
    if not profile_json:
        await message.reply_text("No profile details stored yet.")
        return
    migrated_json, migrated = migrate_profile_json(profile_json)
    if migrated:
        profile_json = migrated_json
        db.set_user_profile(user_id, profile_json)
    try:
        profile = json.loads(profile_json)
    except Exception:
        await message.reply_text("Profile data is not readable.")
        return
    facts = profile.get("facts") if isinstance(profile, dict) else None
    if not isinstance(facts, list) or not facts:
        await message.reply_text("No profile details stored yet.")
        return
    lines = ["Profile facts:"]
    for item in facts[:20]:
        if not isinstance(item, dict):
            continue
        fact_id = str(item.get("id") or "").strip()
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        if not fact_id or not key or not value:
            continue
        lines.append(f"- {fact_id}: {key} = {value}")
    await message.reply_text("\n".join(lines))


async def handle_profile_forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    user_id = _resolve_user_id(update)
    if not user_id:
        await message.reply_text("No user id available for profile updates.")
        return
    parts = message.text.strip().split(maxsplit=1)
    if len(parts) < 2:
        await message.reply_text("Usage: /profile_forget <id>")
        return
    target_id = parts[1].strip()
    db = get_database()
    profile_json = db.get_user_profile(user_id)
    if not profile_json:
        await message.reply_text("No profile details stored yet.")
        return
    migrated_json, migrated = migrate_profile_json(profile_json)
    if migrated:
        profile_json = migrated_json
        db.set_user_profile(user_id, profile_json)
    try:
        profile = json.loads(profile_json)
    except Exception:
        await message.reply_text("Profile data is not readable.")
        return
    if not isinstance(profile, dict):
        await message.reply_text("Profile data is not valid.")
        return
    facts = profile.get("facts")
    if not isinstance(facts, list) or not facts:
        await message.reply_text("No profile details stored yet.")
        return
    remaining = []
    deleted = False
    for item in facts:
        if not isinstance(item, dict):
            continue
        if str(item.get("id") or "").strip() == target_id:
            deleted = True
            continue
        remaining.append(item)
    if not deleted:
        await message.reply_text("Profile item not found.")
        return
    profile["facts"] = remaining
    db.set_user_profile(user_id, json.dumps(profile, ensure_ascii=True))
    await message.reply_text(f"Profile item deleted: {target_id}")


async def _handle_gmail_inbox(
    context: ContextTypes.DEFAULT_TYPE,
    message,
    *,
    limit: int,
    message_text: str,
    query_hint: str | None = None,
) -> None:
    await _maybe_send_gmail_status(context, message, "Gmail: scanning inbox...")
    try:
        gmail_service = _get_gmail_service(context)
    except Exception as exc:
        logger.warning("Gmail service unavailable: %s", exc)
        await message.reply_text("Gmail service is not configured.")
        return
    llm_client = context.application.bot_data.get("llm_client")
    if query_hint is None:
        query_hint = _resolve_gmail_time_hint(message_text)
    try:
        messages = await asyncio.to_thread(gmail_service.list_unread, limit, query_hint)
    except Exception as exc:
        logger.warning("Gmail inbox fetch failed: %s", exc)
        await message.reply_text("Failed to fetch Gmail inbox.")
        return
    if not messages:
        await message.reply_text("No unread emails found.")
        return
    lines = ["Unread emails:"]
    for mail in messages:
        label = "Action Required"
        if llm_client is not None:
            label = await asyncio.to_thread(gmail_service.categorize_message, llm_client, mail)
        sender = mail.sender or "Unknown sender"
        subject = mail.subject or "(no subject)"
        lines.append(f"- [{label}] {sender} | {subject}")
        if label in GMAIL_LABELS:
            try:
                await asyncio.to_thread(gmail_service.label_message, mail.message_id, label)
            except Exception as exc:
                logger.warning("Gmail label apply failed: %s", exc)
    await message.reply_text("\n".join(lines))


async def _handle_gmail_summary(
    context: ContextTypes.DEFAULT_TYPE,
    message,
    *,
    limit: int,
    message_text: str,
    query_hint: str | None = None,
) -> None:
    await _maybe_send_gmail_status(context, message, "Gmail: preparing summary...")
    llm_client = context.application.bot_data.get("llm_client")
    if llm_client is None:
        await message.reply_text("LLM is not configured for summaries.")
        return
    try:
        gmail_service = _get_gmail_service(context)
        messages = await asyncio.to_thread(gmail_service.list_unread, limit, query_hint)
    except Exception as exc:
        logger.warning("Gmail summary fetch failed: %s", exc)
        await message.reply_text("Failed to fetch Gmail inbox.")
        return
    if not messages:
        await message.reply_text("No unread emails found.")
        return
    summary = await asyncio.to_thread(gmail_service.summarize_messages, llm_client, messages)
    await message.reply_text(summary)


async def _handle_gmail_search(
    context: ContextTypes.DEFAULT_TYPE,
    message,
    *,
    query_text: str,
) -> None:
    await _maybe_send_gmail_status(context, message, "Gmail: searching mail...")
    llm_client = context.application.bot_data.get("llm_client")
    try:
        gmail_service = _get_gmail_service(context)
    except Exception as exc:
        logger.warning("Gmail service unavailable: %s", exc)
        await message.reply_text("Gmail service is not configured.")
        return
    gmail_query = query_text
    if llm_client is not None:
        gmail_query = await asyncio.to_thread(
            gmail_service.build_search_query, llm_client, query_text
        )
    try:
        results = await asyncio.to_thread(gmail_service.search, gmail_query, 15)
    except Exception as exc:
        logger.warning("Gmail search failed: %s", exc)
        await message.reply_text("Gmail search failed.")
        return
    if not results:
        await message.reply_text("No emails matched your search.")
        return
    lines = [f"Search results ({gmail_query}):"]
    for mail in results:
        sender = mail.sender or "Unknown sender"
        subject = mail.subject or "(no subject)"
        lines.append(f"- {sender} | {subject}")
    await message.reply_text("\n".join(lines))


async def _handle_gmail_question(
    context: ContextTypes.DEFAULT_TYPE,
    message,
    *,
    question_text: str,
) -> None:
    await _maybe_send_gmail_status(context, message, "Gmail: analyzing messages...")
    llm_client = context.application.bot_data.get("llm_client")
    if llm_client is None:
        await message.reply_text("LLM is not configured for Gmail questions.")
        return
    try:
        gmail_service = _get_gmail_service(context)
    except Exception as exc:
        logger.warning("Gmail service unavailable: %s", exc)
        await message.reply_text("Gmail service is not configured.")
        return
    gmail_query = await asyncio.to_thread(
        gmail_service.build_search_query, llm_client, question_text
    )
    try:
        results = await asyncio.to_thread(gmail_service.search, gmail_query, 25)
    except Exception as exc:
        logger.warning("Gmail question search failed: %s", exc)
        await message.reply_text("Gmail search failed.")
        return
    if not results:
        await message.reply_text("No emails matched your request.")
        return
    items = []
    for mail in results:
        body = (mail.body or mail.snippet or "").strip()
        trimmed = body[:1200]
        items.append(
            f"- From: {mail.sender}\n"
            f"  Subject: {mail.subject}\n"
            f"  Snippet: {mail.snippet or ''}\n"
            f"  Body: {trimmed}"
        )
    prompt = (
        "Answer the user's question using the emails below. "
        "If the question asks for a count, provide the count. "
        "If you cannot find an answer, say so.\n\n"
        f"Question: {question_text}\n\n"
        + "\n\n".join(items)
    )
    response = await asyncio.to_thread(llm_client.generate_text, prompt)
    await message.reply_text(response.strip())


async def _handle_gmail_draft(
    context: ContextTypes.DEFAULT_TYPE,
    message,
    *,
    request_text: str,
) -> None:
    await _maybe_send_gmail_status(context, message, "Gmail: drafting email...")
    llm_client = context.application.bot_data.get("llm_client")
    user_id = _resolve_user_id_from_message(message)
    try:
        gmail_service = _get_gmail_service(context)
    except Exception as exc:
        logger.warning("Gmail service unavailable: %s", exc)
        await message.reply_text("Gmail service is not configured.")
        return
    details = _extract_draft_request(llm_client, request_text)
    if details.get("needs_clarification"):
        await message.reply_text(details.get("clarification_question"))
        return
    recipient = details.get("to") or _extract_first_email(request_text)
    if not recipient:
        await message.reply_text("Who should I send it to? Please provide an email address.")
        return
    prompt = details.get("prompt") or request_text
    try:
        if llm_client is not None:
            spec = await asyncio.to_thread(
                gmail_service.build_draft_from_prompt, llm_client, recipient, prompt
            )
        else:
            spec = DraftSpec(to=recipient, subject=details.get("subject") or "Quick Update", body=prompt)
        if details.get("subject"):
            spec = DraftSpec(to=spec.to, subject=details["subject"], body=spec.body)
        signature_name = _extract_signature_name(user_id)
        spec = _apply_signature(spec, signature_name)
        draft_id = await asyncio.to_thread(gmail_service.create_draft, spec)
    except Exception as exc:
        logger.warning("Gmail draft failed: %s", exc)
        await message.reply_text("Failed to create Gmail draft.")
        return
    if _should_send_email(request_text):
        agent_context = context.application.bot_data.get("agent_context")
        if agent_context is None:
            await message.reply_text("Draft created, but approval flow is unavailable.")
            return
        question_id = str(uuid4())
        agent_context.create_pending_request(
            question_id=question_id,
            intent="gmail_send",
            question="Gmail draft is ready. Send it?",
            category="gmail_send",
        )
        _store_gmail_pending_send(
            context,
            question_id,
            {
                "draft_id": draft_id,
                "to": spec.to,
                "subject": spec.subject,
                "body": spec.body,
            },
        )
        preview_text = _build_draft_preview_text(spec)
        await message.reply_text(
            f"Draft ready:\n{preview_text}\n\nSend it?",
            reply_markup=_build_gmail_send_keyboard(),
        )
        return
    await message.reply_text(f"Gmail draft created: {draft_id}")


async def _handle_gmail_send_approval(
    context: ContextTypes.DEFAULT_TYPE,
    message,
    *,
    agent_context,
    reply_text: str,
) -> bool:
    pending = agent_context.pending_request
    if pending is None or pending.category != "gmail_send":
        return False
    if reply_text in {_GMAIL_SEND_CONFIRM_DATA, _GMAIL_SEND_CANCEL_DATA}:
        normalized = reply_text
    else:
        normalized = _normalize_text(reply_text)
    if normalized == _GMAIL_SEND_CONFIRM_DATA or _contains_word_token(
        normalized, _GMAIL_APPROVE_TOKENS
    ):
        pending_payload = _pop_gmail_pending_send(context, pending.question_id)
        if not pending_payload:
            agent_context.pending_request = None
            await message.reply_text("Pending Gmail draft not found.")
            return True
        try:
            gmail_service = _get_gmail_service(context)
            sent_id = await asyncio.to_thread(
                gmail_service.send_draft, pending_payload.get("draft_id", "")
            )
        except Exception as exc:
            logger.warning("Gmail send failed: %s", exc)
            await message.reply_text(
                "Failed to send email. You may need to re-authenticate Gmail with send permissions."
            )
            agent_context.pending_request = None
            return True
        agent_context.pending_request = None
        await message.reply_text(f"Email sent. Message id: {sent_id}")
        return True
    if normalized == _GMAIL_SEND_CANCEL_DATA or _contains_word_token(
        normalized, _GMAIL_REJECT_TOKENS
    ):
        _pop_gmail_pending_send(context, pending.question_id)
        agent_context.pending_request = None
        await message.reply_text("Email send canceled.")
        return True
    pending_payload = _pop_gmail_pending_send(context, pending.question_id)
    if not pending_payload:
        agent_context.pending_request = None
        await message.reply_text("Pending Gmail draft not found.")
        return True
    llm_client = context.application.bot_data.get("llm_client")
    if llm_client is None:
        await message.reply_text("I need the LLM to revise the draft. Please reply Yes or No.")
        _store_gmail_pending_send(context, pending.question_id, pending_payload)
        return True
    draft_subject = str(pending_payload.get("subject") or "").strip()
    draft_body = str(pending_payload.get("body") or "").strip()
    instruction = reply_text.strip()
    prompt = _build_revision_prompt(draft_subject, draft_body, instruction)
    response = await asyncio.to_thread(llm_client.generate_text, prompt)
    updated_subject, updated_body = _parse_subject_body(response)
    if not updated_subject:
        updated_subject = draft_subject
    if not updated_body:
        updated_body = draft_body
    spec = DraftSpec(
        to=str(pending_payload.get("to") or ""),
        subject=updated_subject,
        body=updated_body,
    )
    signature_name = _extract_signature_name(_resolve_user_id_from_message(message))
    spec = _apply_signature(spec, signature_name)
    try:
        gmail_service = _get_gmail_service(context)
        new_draft_id = await asyncio.to_thread(gmail_service.create_draft, spec)
    except Exception as exc:
        logger.warning("Gmail draft revision failed: %s", exc)
        await message.reply_text("Failed to revise the draft. Please reply Yes or No.")
        _store_gmail_pending_send(context, pending.question_id, pending_payload)
        return True
    _store_gmail_pending_send(
        context,
        pending.question_id,
        {
            "draft_id": new_draft_id,
            "to": spec.to,
            "subject": spec.subject,
            "body": spec.body,
        },
    )
    preview_text = _build_draft_preview_text(spec)
    await message.reply_text(
        f"Revised draft:\n{preview_text}\n\nSend it?",
        reply_markup=_build_gmail_send_keyboard(),
    )
    return True


async def handle_gmail_auth_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    settings = _resolve_settings(context)
    if not getattr(settings, "gmail_credentials_path", ""):
        await message.reply_text("Set GMAIL_OAUTH_CREDENTIALS_PATH to your OAuth JSON first.")
        return
    try:
        gmail_service = _get_gmail_service(context)
        await asyncio.to_thread(gmail_service.ensure_credentials)
    except Exception as exc:
        logger.warning("Gmail auth failed: %s", exc)
        await message.reply_text("Gmail authentication failed. Check logs for details.")
        return
    await message.reply_text("Gmail authentication completed.")


async def handle_inbox_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    parts = message.text.split(maxsplit=1)
    limit = _parse_limit_arg(parts[1] if len(parts) > 1 else None, default=5, max_value=20)
    await _handle_gmail_inbox(context, message, limit=limit, message_text=message.text)


async def handle_summarize_last_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    parts = message.text.split(maxsplit=1)
    limit = _parse_limit_arg(parts[1] if len(parts) > 1 else None, default=5, max_value=15)
    await _handle_gmail_summary(context, message, limit=limit, message_text=message.text)


async def handle_search_mail_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.reply_text("Usage: /search_mail <query>")
        return
    query_text = parts[1].strip()
    if not query_text:
        await message.reply_text("Usage: /search_mail <query>")
        return
    await _handle_gmail_search(context, message, query_text=query_text)


async def handle_draft_mail_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.reply_text("Usage: /draft_mail <to> | <subject> | <prompt>")
        return
    payload = parts[1].strip()
    recipient = ""
    subject_override = ""
    prompt = ""
    if "|" in payload:
        segments = [item.strip() for item in payload.split("|")]
        recipient = segments[0] if len(segments) > 0 else ""
        subject_override = segments[1] if len(segments) > 1 else ""
        prompt = segments[2] if len(segments) > 2 else ""
    else:
        tokens = payload.split(maxsplit=1)
        recipient = tokens[0] if tokens else ""
        prompt = tokens[1] if len(tokens) > 1 else ""
    if not recipient or not prompt:
        await message.reply_text("Usage: /draft_mail <to> | <subject> | <prompt>")
        return
    await _handle_gmail_draft(context, message, request_text=message.text)


async def _route_and_handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    agent_context,
    user_id: str | None,
    message_text: str,
    *,
    has_image: bool,
    photo,
    source_meta: dict[str, str | None] | None = None,
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
        and _has_recent_images(session_context)
        and await _is_analysis_request_with_llm(llm_client, message_text)
        and _mentions_explicit_image_reference(message_text)
    ):
        decision = RouterDecision(
            intent=RouterIntent.IMAGE_EDIT, reason="override:analysis_last_image")

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
            source_meta=source_meta,
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
            source_meta=source_meta,
            settings=settings,
            telegram_app=context.application,
        )
        return
    if decision.intent == RouterIntent.GMAIL_INBOX:
        handled = await _handle_gmail_intelligent_request(
            context, message, intent=decision.intent, message_text=message_text
        )
        if handled:
            return
        limit = _extract_limit_from_text(message_text, default=5, max_value=20)
        await _handle_gmail_inbox(context, message, limit=limit, message_text=message_text)
        return
    if decision.intent == RouterIntent.GMAIL_SUMMARY:
        handled = await _handle_gmail_intelligent_request(
            context, message, intent=decision.intent, message_text=message_text
        )
        if handled:
            return
        limit = _extract_limit_from_text(message_text, default=5, max_value=15)
        await _handle_gmail_summary(context, message, limit=limit, message_text=message_text)
        return
    if decision.intent == RouterIntent.GMAIL_SEARCH:
        handled = await _handle_gmail_intelligent_request(
            context, message, intent=decision.intent, message_text=message_text
        )
        if handled:
            return
        await _handle_gmail_search(context, message, query_text=message_text)
        return
    if decision.intent == RouterIntent.GMAIL_QUESTION:
        handled = await _handle_gmail_intelligent_request(
            context, message, intent=decision.intent, message_text=message_text
        )
        if handled:
            return
        await _handle_gmail_question(context, message, question_text=message_text)
        return
    if decision.intent == RouterIntent.GMAIL_DRAFT:
        handled = await _handle_gmail_intelligent_request(
            context, message, intent=decision.intent, message_text=message_text
        )
        if handled:
            return
        await _handle_gmail_draft(context, message, request_text=message_text)
        return
    if decision.intent == RouterIntent.GMAIL_SEND:
        handled = await _handle_gmail_intelligent_request(
            context, message, intent=decision.intent, message_text=message_text
        )
        if handled:
            return
        await _handle_gmail_draft(context, message, request_text=message_text)
        return

    await _handle_chat(
        message,
        llm_client,
        message_text,
        user_id,
        session_context,
        source_meta=source_meta,
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
) -> object | None:
    """Sends a message with dynamic suggestions generated by LLM."""
    reply_markup = await _build_suggestion_markup(llm_client, history_text)

    message_text = (text or "").strip()
    if photo:
        photo_caption = (caption or message_text).strip()
        formatted_caption = _format_telegram_html(
            photo_caption) if photo_caption else ""
        if formatted_caption and len(formatted_caption) > _TELEGRAM_CAPTION_LIMIT:
            photo_message = await message.reply_photo(photo=photo, caption="Result")
            if message_text:
                await _send_text_in_chunks(
                    message,
                    message_text,
                    chunk_size=_TELEGRAM_TEXT_CHUNK,
                    reply_markup=reply_markup,
                )
            return photo_message
        return await message.reply_photo(
            photo=photo,
            caption=formatted_caption or None,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
        )
    if message_text:
        if len(message_text) > _TELEGRAM_TEXT_CHUNK:
            return await _send_text_in_chunks(
                message,
                message_text,
                chunk_size=_TELEGRAM_TEXT_CHUNK,
                reply_markup=reply_markup,
            )
        return await message.reply_text(
            _format_telegram_html(message_text),
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
        )
    return None


async def _handle_approval_reply(message, agent_context, approval_id: str) -> bool:
    reply_text = (message.text or message.caption or "").strip()
    if not reply_text:
        await message.reply_text(
            "Please reply YES or NO to approve the plan, or send a change request.",
        )
        return True
    pending = agent_context.pending_request
    if pending is not None and pending.question_id == approval_id:
        return await _handle_pending_reply(message, agent_context, reply_text)

    from app.agent.state import HumanReply

    reply = HumanReply(
        question_id=approval_id,
        text=reply_text,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="telegram",
    )
    await agent_context.human_reply_queue.put(reply)
    logger.info("Approval reply received for approval_id=%s", approval_id)
    await message.reply_text(f"Received reply for approval_id={approval_id}. Thank you!")
    return True


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


async def _handle_chart_request(
    message,
    llm_client,
    prompt_text: str,
    user_id: str | None,
    session_context: dict | None,
    *,
    source_meta: dict[str, str | None] | None = None,
    telegram_app,
) -> None:
    db = get_database() if user_id else None
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=prompt_text,
            timestamp=_timestamp_from_message(message),
            **_user_source_kwargs(message, source_meta),
        )
    reply_note = _build_reply_context_note(message, user_id)
    llm_prompt_text = _inject_reply_context(prompt_text, reply_note)
    chart_type = _resolve_chart_type(prompt_text)
    show_ma = _should_show_moving_average(prompt_text)
    payload = None
    if chart_type in {"line", "candlestick"}:
        payload = _generate_time_series_payload(
            prompt_text,
            chart_type=chart_type,
            show_ma=show_ma,
        )
    else:
        try:
            chart_prompt = _build_chart_data_prompt(llm_prompt_text)
            payload_text = await asyncio.to_thread(llm_client.generate_text, chart_prompt)
            payload = _parse_chart_payload(payload_text)
        except Exception as exc:  # pragma: no cover - network/proxy errors
            logger.warning("Chart data generation failed: %s", exc)
        if payload is None:
            payload = _fallback_chart_payload(prompt_text)
    if payload is None:
        await message.reply_text("Could not build chart data. Please try again.")
        return

    image_bytes = None
    if payload.get("type") == "candlestick":
        image_bytes = _render_candlestick_chart(payload)
    elif payload.get("type") == "line":
        image_bytes = _render_line_chart(payload)
    else:
        image_bytes = _render_bar_chart(payload)
    if not image_bytes:
        await message.reply_text("Chart rendering failed. Please try again.")
        return

    output_path = _save_image_bytes(image_bytes, "chart")
    record_image_result(user_id, output_path, prompt_text, source="chart")
    assistant_row_id = None
    if db and user_id:
        assistant_row_id = _safe_add_message(
            db,
            user_id=user_id,
            role="assistant",
            content="Chart generated.",
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_path=output_path,
            file_type="image/png",
        )

    lines = []
    title = payload.get("title")
    if title:
        lines.append(title)
    lines.append("Data:")
    if payload.get("type") == "candlestick":
        for label, point in zip(payload.get("labels", []), payload.get("points", [])):
            close_price = point.get("close")
            lines.append(f"- {label}: {close_price}")
    else:
        for item in payload.get("items", payload.get("points", [])):
            label = item.get("label")
            value = item.get("value")
            if label is None or value is None:
                continue
            lines.append(f"- {label}: {value}")
    summary_text = "\n".join(lines).strip()

    suggestion_context = _build_suggestion_context(
        user_id=user_id,
        db=db,
        session_context=session_context,
        user_message=prompt_text,
        assistant_message=summary_text or "Chart generated.",
        task_type="chart",
        extra_context="Generated a bar chart from fictional data.",
    )
    with open(output_path, "rb") as handle:
        sent_message = await _reply_with_suggestions(
            message,
            summary_text or "Chart generated.",
            llm_client,
            history_text=suggestion_context,
            photo=handle,
            caption=summary_text or "Chart generated.",
        )
    _safe_update_message_source(db, assistant_row_id, sent_message)


async def _handle_chat(
    message,
    llm_client,
    prompt_text: str,
    user_id: str | None,
    session_context: dict | None,
    *,
    source_meta: dict[str, str | None] | None = None,
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
    reply_note = _build_reply_context_note(message, user_id)
    llm_prompt_text = _inject_reply_context(prompt_text, reply_note)
    if _is_chart_request(prompt_text):
        await _handle_chart_request(
            message,
            llm_client,
            prompt_text,
            user_id,
            session_context,
            source_meta=source_meta,
            telegram_app=telegram_app,
        )
        return
    db = get_database() if user_id else None
    anon_enabled = bool(user_id and is_anonymous_mode(user_id))
    history = _safe_get_history(
        db, user_id, _CHAT_CONTEXT_LIMIT) if db and user_id else []
    timestamp = _timestamp_from_message(message)
    memory_manager = None
    memory_block = ""
    profile_block = ""
    memory_item_ids: list[int] = []
    should_summarize = False
    if db and user_id:
        memory_manager = _get_memory_manager(telegram_app, settings)
        if memory_manager.enabled and not anon_enabled:
            profile_block = memory_manager.build_profile_snippet(db.get_user_profile(user_id))
            items = db.get_memory_items(user_id, limit=500)
            memory_block, memory_item_ids = memory_manager.retrieve_memory_block(
                items,
                prompt_text,
            )
            if memory_item_ids:
                db.update_memory_last_used(memory_item_ids)
            should_summarize = memory_manager.record_user_message(user_id)
    prompt = _build_chat_prompt(llm_prompt_text, history, profile_block, memory_block)
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=prompt_text,
            timestamp=timestamp,
            **_user_source_kwargs(message, source_meta),
        )

    def _schedule_memory_updates() -> None:
        if (
            not memory_manager
            or not memory_manager.enabled
            or anon_enabled
            or not db
            or not user_id
        ):
            return

        def _runner() -> None:
            if should_summarize:
                success = memory_manager.summarize_and_store(
                    db=db,
                    llm_client=llm_client,
                    user_key=user_id,
                    chat_id=str(message.chat_id),
                    history_limit=memory_manager.summary_window,
                )
                if success:
                    memory_manager.reset_user_counter(user_id)
            memory_manager.maybe_update_profile(
                db=db,
                llm_client=llm_client,
                user_key=user_id,
                message_text=prompt_text,
            )

        async def _async_runner() -> None:
            await asyncio.to_thread(_runner)

        telegram_app.create_task(_async_runner())
    streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
    show_thoughts = bool(getattr(settings, "show_thoughts", True))
    if streaming_enabled and hasattr(llm_client, "stream_text"):
        async def _stream_runner() -> None:
            try:
                live_message = await message.reply_text("Thinking...", parse_mode=ParseMode.HTML)
                manager = MessageManager(
                    bot=telegram_app.bot,
                    chat_id=str(message.chat_id),
                    message_id=live_message.message_id,
                )
                handler = None
                result = None
                sources = []
                for attempt in range(1, _STREAM_RETRY_ATTEMPTS + 1):
                    handler = AsyncStreamHandler(
                        manager,
                        show_thoughts=show_thoughts,
                        response_label="",
                    )
                    if hasattr(llm_client, "stream_text_with_sources"):
                        stream_state = await llm_client.stream_text_with_sources(
                            prompt,
                            include_thoughts=show_thoughts,
                        )
                        result = await handler.stream_chunks(
                            stream_state.chunks,
                            recovery=lambda: handler._attempt_recovery(
                                llm_client, prompt),
                        )
                        sources = stream_state.get_sources()
                    else:
                        result = await handler.stream(
                            llm_client,
                            prompt,
                            include_thoughts=show_thoughts,
                        )
                    if result.error is None:
                        break
                    if attempt < _STREAM_RETRY_ATTEMPTS:
                        await manager.update("Retrying...", force=True)
                        await asyncio.sleep(_retry_delay(attempt, _STREAM_RETRY_BASE_DELAY, _STREAM_RETRY_MAX_DELAY))
                if handler is None or result is None:
                    raise RuntimeError("Streaming did not produce a result.")
                response_text = result.response_text.strip()
                sources_text = _format_sources_text(sources)
                display_text = response_text
                if sources_text:
                    display_text = f"{response_text}\n\n{sources_text}"
                assistant_row_id = None
                if response_text and result.completed and db and user_id:
                    assistant_row_id = _safe_add_message(
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
                    if len(display_text) > response_limit:
                        await manager.finalize("Response sent below.")
                        sent_message = await _send_text_in_chunks(
                            message,
                            display_text,
                            chunk_size=_TELEGRAM_TEXT_CHUNK,
                            reply_markup=reply_markup,
                        )
                        _safe_update_message_source(
                            db, assistant_row_id, sent_message)
                    else:
                        final_message = handler.format_message(
                            include_thoughts=False)
                        if sources_text:
                            final_message = f"{final_message}\n\n{_format_telegram_html(sources_text)}"
                        await manager.finalize(final_message, reply_markup=reply_markup)
                        _safe_update_message_source(
                            db, assistant_row_id, live_message)
                    _schedule_memory_updates()
                else:
                    await manager.finalize(handler.format_message(include_thoughts=False))
                if not response_text and not handler.thought_text:
                    await message.reply_text("No response generated.")
            except Exception as exc:  # pragma: no cover - network/telegram errors
                logger.error("Streaming chat failed: %s", exc)
                await message.reply_text("Chat response failed. Please try again.")

        telegram_app.create_task(_stream_runner())
        return

    sources = []
    try:
        if hasattr(llm_client, "generate_text_with_sources"):
            response_text, sources = await asyncio.to_thread(
                llm_client.generate_text_with_sources,
                prompt,
            )
        else:
            response_text = await asyncio.to_thread(llm_client.generate_text, prompt)
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.error("Chat response failed: %s", exc)
        await message.reply_text("Chat response failed. Please try again.")
        return
    if not response_text:
        await message.reply_text("No response generated.")
        return
    sources_text = _format_sources_text(sources)
    display_text = response_text
    if sources_text:
        display_text = f"{response_text}\n\n{sources_text}"
    assistant_row_id = None
    if db and user_id:
        assistant_row_id = _safe_add_message(
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
    sent_message = await _reply_with_suggestions(
        message,
        display_text,
        llm_client,
        history_text=suggestion_context,
    )
    _safe_update_message_source(db, assistant_row_id, sent_message)
    _schedule_memory_updates()


async def _handle_image_generation(
    message,
    image_client,
    llm_client,
    prompt_text: str,
    user_id: str | None,
    session_context: dict | None,
    *,
    source_meta: dict[str, str | None] | None = None,
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
    reply_note = _build_reply_context_note(message, user_id)
    llm_prompt_text = _inject_reply_context(prompt_text, reply_note)
    db = get_database() if user_id else None
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=prompt_text,
            timestamp=_timestamp_from_message(message),
            **_user_source_kwargs(message, source_meta),
        )
    streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
    show_thoughts = bool(getattr(settings, "show_thoughts", True))
    if streaming_enabled and show_thoughts and hasattr(image_client, "stream_generate_image"):
        async def _stream_runner() -> None:
            async def _finalize_generation(image_bytes: bytes) -> None:
                output_path = _save_image_bytes(image_bytes, "generated")
                record_image_result(user_id, output_path,
                                    prompt_text, source="generated")
                session_context = get_session_context(
                    user_id) or session_context
                assistant_row_id = None
                if db and user_id:
                    assistant_row_id = _safe_add_message(
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
                    sent_message = await _reply_with_suggestions(
                        message,
                        "Image generated.",
                        llm_client,
                        history_text=suggestion_context,
                        photo=handle,
                        caption="Image generated.",
                    )
                _safe_update_message_source(db, assistant_row_id, sent_message)

            try:
                live_message = await message.reply_text(
                    "Generating image...",
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
                        llm_prompt_text,
                        include_thoughts=show_thoughts,
                    )
                    result = await handler.stream_chunks(stream_state.chunks)
                    image_bytes = stream_state.get_image_bytes()
                    if result.error is None and image_bytes:
                        break
                    last_error = result.error or ValueError(
                        "No image bytes received from stream.")
                    if attempt < _STREAM_RETRY_ATTEMPTS:
                        await manager.update("Retrying...", force=True)
                        await asyncio.sleep(_retry_delay(attempt, _STREAM_RETRY_BASE_DELAY, _STREAM_RETRY_MAX_DELAY))
                if handler is None:
                    raise RuntimeError("Stream did not initialize.")
                if not image_bytes or (last_error is not None and image_bytes is None):
                    raise last_error or ValueError(
                        "No image bytes received from stream.")
                await _finalize_generation(image_bytes)
                await manager.finalize(handler.format_message(include_thoughts=False))
            except Exception as exc:  # pragma: no cover - network/telegram errors
                logger.error("Image generation stream failed: %s", exc)
                try:
                    image_bytes = await _retry_image_bytes(
                        lambda: asyncio.to_thread(
                            image_client.generate_image, llm_prompt_text),
                        label="Image generation",
                        message=message,
                        attempts=_STREAM_RETRY_ATTEMPTS,
                        base_delay=_STREAM_RETRY_BASE_DELAY,
                        max_delay=_STREAM_RETRY_MAX_DELAY,
                    )
                except Exception as fallback_exc:
                    logger.error("Image generation failed: %s", fallback_exc)
                    await _safe_reply_text(
                        message,
                        "Image generation failed. Please try again.",
                        label="Image generation failure reply",
                    )
                    return
                await _finalize_generation(image_bytes)

        telegram_app.create_task(_stream_runner())
        return
    try:
        image_bytes = await _retry_image_bytes(
            lambda: asyncio.to_thread(
                image_client.generate_image, llm_prompt_text),
            label="Image generation",
            message=message,
            attempts=_STREAM_RETRY_ATTEMPTS,
            base_delay=_STREAM_RETRY_BASE_DELAY,
            max_delay=_STREAM_RETRY_MAX_DELAY,
        )
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.error("Image generation failed: %s", exc)
        await _safe_reply_text(
            message,
            "Image generation failed. Please try again.",
            label="Image generation failure reply",
        )
        return
    output_path = _save_image_bytes(image_bytes, "generated")
    record_image_result(user_id, output_path, prompt_text, source="generated")
    session_context = get_session_context(user_id) or session_context
    assistant_row_id = None
    if db and user_id:
        assistant_row_id = _safe_add_message(
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
        sent_message = await _reply_with_suggestions(
            message,
            "Image generated.",
            llm_client,
            history_text=suggestion_context,
            photo=handle,
            caption="Image generated."
        )
    _safe_update_message_source(db, assistant_row_id, sent_message)


async def _handle_image_edit(
    message,
    image_client,
    llm_client,
    prompt_text: str,
    *,
    user_id: str | None,
    photo,
    session_context: dict | None,
    source_meta: dict[str, str | None] | None = None,
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
    reply_note = _build_reply_context_note(message, user_id)
    llm_prompt_text = _inject_reply_context(prompt_text, reply_note)

    reference_path = None
    if photo is not None:
        photo_size = getattr(photo, "file_size", None)
        if photo_size and photo_size > GEMINI_FLASH_MAX_INLINE_IMAGE_BYTES:
            await message.reply_text(
                _file_too_large_message(
                    "Image", GEMINI_FLASH_MAX_INLINE_IMAGE_BYTES)
            )
            return
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
        reference_path, _, ambiguous = _resolve_image_reference(
            prompt_text, session_context
        )
        if ambiguous:
            await message.reply_text(
                _build_image_disambiguation_message(session_context)
            )
            return
        if not reference_path or not Path(reference_path).exists():
            await message.reply_text("Please send the image you want to edit.")
            return

    validation_error = await _validate_image_for_gemini(
        llm_client,
        prompt_text,
        reference_path,
        getattr(photo, "mime_type", None) if photo is not None else None,
    )
    if validation_error:
        await message.reply_text(validation_error)
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
            **_user_source_kwargs(message, source_meta),
        )

    if await _is_analysis_request_with_llm(llm_client, llm_prompt_text):
        streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
        show_thoughts = bool(getattr(settings, "show_thoughts", True))
        if streaming_enabled and hasattr(llm_client, "stream_with_image"):
            async def _stream_runner() -> None:
                try:
                    live_message = await message.reply_text(
                        "Analyzing image...",
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
                            llm_prompt_text,
                            reference_path,
                        )

                    result = await handler.stream_chunks(
                        llm_client.stream_with_image(
                            llm_prompt_text,
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
                            message=live_message,
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
                        await manager.finalize(
                            handler.format_message(include_thoughts=False),
                            reply_markup=reply_markup,
                        )
                    else:
                        await manager.finalize(handler.format_message(include_thoughts=False))
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
                llm_prompt_text,
                reference_path,
            )
        except Exception as exc:  # pragma: no cover - network/proxy errors
            logger.error("Image analysis failed: %s", exc)
            await message.reply_text("Image analysis failed. Please try again.")
            return
        if response_text:
            assistant_row_id = None
            if db and user_id:
                assistant_row_id = _safe_add_message(
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
            sent_message = await _reply_with_suggestions(
                message,
                response_text,
                llm_client,
                history_text=suggestion_context,
            )
            _safe_update_message_source(db, assistant_row_id, sent_message)
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
        assistant_row_id = None
        if db and user_id:
            assistant_row_id = _safe_add_message(
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
            sent_message = await _reply_with_suggestions(
                message,
                response_text,
                llm_client,
                history_text=suggestion_context,
                photo=handle,
                caption=response_text,
            )
        _safe_update_message_source(db, assistant_row_id, sent_message)

    streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
    show_thoughts = bool(getattr(settings, "show_thoughts", True))
    if streaming_enabled and show_thoughts and hasattr(image_client, "stream_edit_image"):
        async def _stream_runner() -> None:
            try:
                live_message = await message.reply_text(
                    "Editing image...",
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
                        llm_prompt_text,
                        reference_path,
                        include_thoughts=show_thoughts,
                    )
                    result = await handler.stream_chunks(stream_state.chunks)
                    image_bytes = stream_state.get_image_bytes()
                    if result.error is None and image_bytes:
                        break
                    last_error = result.error or ValueError(
                        "No image bytes received from stream.")
                    if attempt < _STREAM_RETRY_ATTEMPTS:
                        await manager.update("Retrying...", force=True)
                        await asyncio.sleep(_retry_delay(attempt, _STREAM_RETRY_BASE_DELAY, _STREAM_RETRY_MAX_DELAY))
                if handler is None:
                    raise RuntimeError("Stream did not initialize.")
                if not image_bytes or (last_error is not None and image_bytes is None):
                    raise last_error or ValueError(
                        "No image bytes received from stream.")
                await _finalize_edit(image_bytes)
                await manager.finalize(handler.format_message(include_thoughts=False))
            except Exception as exc:  # pragma: no cover - network/telegram errors
                logger.error("Image edit stream failed: %s", exc)
                try:
                    image_bytes = await _retry_image_bytes(
                        lambda: asyncio.to_thread(
                            image_client.edit_image,
                            llm_prompt_text,
                            reference_path,
                        ),
                        label="Image edit",
                        message=message,
                        attempts=_STREAM_RETRY_ATTEMPTS,
                        base_delay=_STREAM_RETRY_BASE_DELAY,
                        max_delay=_STREAM_RETRY_MAX_DELAY,
                    )
                except Exception as fallback_exc:
                    logger.error("Image edit failed: %s", fallback_exc)
                    await _safe_reply_text(
                        message,
                        "Image edit failed. Please try again.",
                        label="Image edit failure reply",
                    )
                    return
                await _finalize_edit(image_bytes)

        telegram_app.create_task(_stream_runner())
        return

    try:
        image_bytes = await _retry_image_bytes(
            lambda: asyncio.to_thread(
                image_client.edit_image,
                llm_prompt_text,
                reference_path,
            ),
            label="Image edit",
            message=message,
            attempts=_STREAM_RETRY_ATTEMPTS,
            base_delay=_STREAM_RETRY_BASE_DELAY,
            max_delay=_STREAM_RETRY_MAX_DELAY,
        )
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.error("Image edit failed: %s", exc)
        await _safe_reply_text(
            message,
            "Image edit failed. Please try again.",
            label="Image edit failure reply",
        )
        return
    await _finalize_edit(image_bytes)


async def _handle_document_request(
    message,
    context,
    caption: str,
    document,
    user_id: str | None,
    session_context: dict | None,
    source_meta: dict[str, str | None] | None = None,
) -> None:
    caption_text = caption.strip()
    if not caption_text:
        await message.reply_text("Please add a caption describing what to do with the file.")
        return
    llm_client = context.application.bot_data.get("llm_client")
    if llm_client is None:
        await message.reply_text("Gemini is not configured. Please set GEMINI_API_KEY.")
        return
    reply_note = _build_reply_context_note(message, user_id)
    llm_prompt_text = _inject_reply_context(caption_text, reply_note)
    document_size = getattr(document, "file_size", None)
    if document_size and document_size > GEMINI_FLASH_MAX_DOCUMENT_BYTES:
        await message.reply_text(
            _file_too_large_message(
                "Document", GEMINI_FLASH_MAX_DOCUMENT_BYTES)
        )
        return
    temp_path = None
    db = get_database() if user_id else None
    try:
        temp_path = await _download_document_to_temp(document, message)
    except Exception as exc:  # pragma: no cover - network/IO errors
        logger.error("Document download failed: %s", exc)
        await message.reply_text("Document download failed. Please try again.")
        return
    if document_size is None and temp_path:
        try:
            downloaded_size = Path(temp_path).stat().st_size
        except Exception as exc:
            logger.warning("Document size check failed: %s", exc)
            downloaded_size = None
        if downloaded_size and downloaded_size > GEMINI_FLASH_MAX_DOCUMENT_BYTES:
            await message.reply_text(
                _file_too_large_message(
                    "Document", GEMINI_FLASH_MAX_DOCUMENT_BYTES)
            )
            return

    mime_type = getattr(document, "mime_type", None)
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=caption_text,
            timestamp=_timestamp_from_message(message),
            file_path=temp_path,
            file_type=mime_type,
            **_user_source_kwargs(message, source_meta),
        )
    result = await asyncio.to_thread(
        llm_client.generate_with_file,
        llm_prompt_text,
        temp_path,
        mime_type,
    )
    if result.error:
        await message.reply_text(result.error)
        return
    if not result.text:
        await message.reply_text("No response generated for the document.")
        return
    assistant_row_id = None
    if db and user_id:
        assistant_row_id = _safe_add_message(
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
        user_message=caption_text,
        assistant_message=result.text,
        task_type="document",
        extra_context="User asked to process a document.",
    )
    reply_markup = await _build_suggestion_markup(llm_client, suggestion_context)
    sent_message = await _send_text_in_chunks(
        message,
        result.text,
        chunk_size=3500,
        reply_markup=reply_markup,
    )
    _safe_update_message_source(db, assistant_row_id, sent_message)


async def _handle_document_request_from_path(
    *,
    message,
    context,
    caption: str,
    file_path: str,
    mime_type: str | None,
    user_id: str | None,
    session_context: dict | None,
    source_meta: dict[str, str | None] | None = None,
) -> None:
    caption_text = caption.strip()
    if not caption_text:
        await message.reply_text("Please add a caption describing what to do with the file.")
        return
    llm_client = context.application.bot_data.get("llm_client")
    if llm_client is None:
        await message.reply_text("Gemini is not configured. Please set GEMINI_API_KEY.")
        return
    if not file_path or not Path(file_path).exists():
        await message.reply_text("I can't find the file anymore. Please re-send it.")
        return
    reply_note = _build_reply_context_note(message, user_id)
    llm_prompt_text = _inject_reply_context(caption_text, reply_note)
    db = get_database() if user_id else None
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=caption_text,
            timestamp=_timestamp_from_message(message),
            file_path=file_path,
            file_type=mime_type,
            **_user_source_kwargs(message, source_meta),
        )
    result = await asyncio.to_thread(
        llm_client.generate_with_file,
        llm_prompt_text,
        file_path,
        mime_type,
    )
    if result.error:
        await message.reply_text(result.error)
        return
    if not result.text:
        await message.reply_text("No response generated for the document.")
        return
    assistant_row_id = None
    if db and user_id:
        assistant_row_id = _safe_add_message(
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
        user_message=caption_text,
        assistant_message=result.text,
        task_type="document",
        extra_context="User asked to process a document.",
    )
    reply_markup = await _build_suggestion_markup(llm_client, suggestion_context)
    sent_message = await _send_text_in_chunks(
        message,
        result.text,
        chunk_size=3500,
        reply_markup=reply_markup,
    )
    _safe_update_message_source(db, assistant_row_id, sent_message)


async def _handle_media_request(
    message,
    context,
    caption: str,
    media,
    user_id: str | None,
    session_context: dict | None,
    *,
    source_meta: dict[str, str | None] | None = None,
    media_kind: str,
    default_prompt: str,
) -> None:
    llm_client = context.application.bot_data.get("llm_client")
    if llm_client is None:
        await message.reply_text("Gemini is not configured. Please set GEMINI_API_KEY.")
        return
    caption_text = (caption or "").strip()
    prompt_text = caption_text or default_prompt
    if not prompt_text:
        await message.reply_text("Please add a caption describing what to do with the file.")
        return
    reply_note = _build_reply_context_note(message, user_id)
    llm_prompt_text = _inject_reply_context(prompt_text, reply_note)
    temp_path = None
    db = get_database() if user_id else None
    try:
        temp_path = await _download_media_to_temp(media, message, media_kind)
    except Exception as exc:  # pragma: no cover - network/IO errors
        logger.error("%s download failed: %s", media_kind.capitalize(), exc)
        await message.reply_text("Media download failed. Please try again.")
        return
    mime_type = getattr(media, "mime_type", None)
    if db and user_id:
        _safe_add_message(
            db,
            user_id=user_id,
            role="user",
            content=caption_text,
            timestamp=_timestamp_from_message(message),
            file_path=temp_path,
            file_type=mime_type,
            **_user_source_kwargs(message, source_meta),
        )
    result = await asyncio.to_thread(
        llm_client.generate_with_file,
        llm_prompt_text,
        temp_path,
        mime_type,
    )
    if result.error:
        await message.reply_text(result.error)
        return
    if not result.text:
        await message.reply_text("No response generated for the media.")
        return
    assistant_row_id = None
    if db and user_id:
        assistant_row_id = _safe_add_message(
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
        user_message=caption_text or None,
        assistant_message=result.text,
        task_type=media_kind,
        extra_context=f"User asked to process a {media_kind} file.",
    )
    reply_markup = await _build_suggestion_markup(llm_client, suggestion_context)
    sent_message = await _send_text_in_chunks(
        message,
        result.text,
        chunk_size=3500,
        reply_markup=reply_markup,
    )
    _safe_update_message_source(db, assistant_row_id, sent_message)


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


async def _download_media_to_temp(media, message, media_kind: str) -> str:
    TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    telegram_file = await media.get_file()
    suffix = ""
    if telegram_file.file_path:
        suffix = Path(telegram_file.file_path).suffix
    if not suffix and getattr(media, "file_name", None):
        suffix = Path(media.file_name).suffix
    if not suffix and getattr(media, "mime_type", None):
        suffix = mimetypes.guess_extension(media.mime_type) or ""
    if not suffix:
        suffix = ".bin"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = TEMP_IMAGE_DIR / \
        f"telegram_{media_kind}_{message.message_id}_{timestamp}{suffix}"
    await telegram_file.download_to_drive(custom_path=str(path))
    return str(path)


def _extract_media_payload(message) -> tuple[object | None, str, str]:
    if message.voice:
        return message.voice, "audio", _DEFAULT_AUDIO_PROMPT
    if message.audio:
        return message.audio, "audio", _DEFAULT_AUDIO_PROMPT
    if message.video_note:
        return message.video_note, "video", _DEFAULT_VIDEO_PROMPT
    if message.video:
        return message.video, "video", _DEFAULT_VIDEO_PROMPT
    return None, "", ""


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
) -> object | None:
    text = text.strip()
    if not text:
        return None
    last_message = None
    for start in range(0, len(text), chunk_size):
        chunk = text[start: start + chunk_size]
        is_last = start + chunk_size >= len(text)
        last_message = await message.reply_text(
            _format_telegram_html(chunk),
            reply_markup=reply_markup if is_last else None,
            parse_mode=ParseMode.HTML,
        )
    return last_message


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
        styled = re.sub(
            r"(?<!\*)\*([^\s][^*]*?[^\s])\*(?!\*)", r"<i>\1</i>", styled)
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


def _format_memory_date(value: str | None) -> str:
    if not value:
        return "unknown"
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return "unknown"
    return parsed.date().isoformat()


def _safe_get_history(db, user_id: str, limit: int):
    try:
        return db.get_recent_messages(user_id, limit)
    except Exception as exc:
        logger.warning("Message history load failed: %s", exc)
        return []


def _safe_add_message(db, **kwargs) -> int | None:
    user_id = kwargs.get("user_id")
    if user_id and is_anonymous_mode(str(user_id)):
        return None
    message = kwargs.pop("message", None)
    if message is not None:
        role = kwargs.get("role")
        sender = getattr(message, "from_user", None)
        if role == "user" and sender is not None and getattr(sender, "is_bot", False):
            message = None
    if message is not None:
        kwargs.setdefault("source", "telegram")
        message_id = getattr(message, "message_id", None)
        if message_id is not None:
            kwargs.setdefault("source_message_id", str(message_id))
        chat_id = getattr(message, "chat_id", None)
        if chat_id is not None:
            kwargs.setdefault("source_chat_id", str(chat_id))
    try:
        return db.add_message(**kwargs)
    except Exception as exc:
        logger.warning("Message persistence failed: %s", exc)
        return None


def _safe_update_message_source(db, message_row_id: int | None, message) -> None:
    if not db or not message_row_id or message is None:
        return
    message_id = getattr(message, "message_id", None)
    if message_id is None:
        return
    chat_id = getattr(message, "chat_id", None)
    try:
        db.update_message_source(
            message_row_id,
            source="telegram",
            source_message_id=str(message_id),
            source_chat_id=str(chat_id) if chat_id is not None else None,
        )
    except Exception as exc:
        logger.warning("Message source update failed: %s", exc)


def _build_chat_prompt(
    prompt_text: str,
    history: list,
    profile_block: str = "",
    memory_block: str = "",
) -> str:
    parts: list[str] = []
    if profile_block:
        parts.append(f"User profile:\n{profile_block}")
    if memory_block:
        parts.append(memory_block)
    history_block = ""
    if history:
        lines = [_format_history_line(entry) for entry in reversed(history)]
        history_block = "\n".join(line for line in lines if line)
    if history_block:
        parts.append(f"Conversation so far:\n{history_block}")
    if parts:
        parts.append(f"User: {prompt_text}")
        return "\n\n".join(parts)
    return prompt_text


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
    recent_images = _get_recent_images(session_context)
    if recent_images:
        summary = _format_recent_image_summary(recent_images[-1])
        lines.append(
            f"- recent_images: {len(recent_images)}"
            + (f" (latest: {summary})" if summary else "")
        )
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


def _format_sources_text(sources) -> str:
    if not sources:
        return ""
    cleaned: list[tuple[str, str]] = []
    seen = set()
    for source in sources:
        uri = str(getattr(source, "uri", "") or "").strip()
        if not uri or uri in seen:
            continue
        title = str(getattr(source, "title", "") or "").strip() or uri
        cleaned.append((title, uri))
        seen.add(uri)
        if len(cleaned) >= _MAX_SOURCE_COUNT:
            break
    if not cleaned:
        return ""
    lines = ["Sources:"]
    for title, uri in cleaned:
        if title == uri:
            lines.append(f"- {uri}")
        else:
            lines.append(f"- {title} ({uri})")
    return "\n".join(lines)


def _load_chart_font(size: int):
    candidates = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _resolve_chart_type(text: str) -> str:
    normalized = _normalize_text(text)
    if _contains_word_token(normalized, set(_CANDLE_TOKENS)):
        return "candlestick"
    if _contains_word_token(normalized, set(_LINE_TOKENS)):
        return "line"
    return "bar"


def _should_show_moving_average(text: str) -> bool:
    normalized = _normalize_text(text)
    return _contains_word_token(normalized, set(_MA_TOKENS))


def _is_chart_request(text: str) -> bool:
    normalized = _normalize_text(text)
    return _contains_word_token(normalized, set(_CHART_TOKENS))


def _build_chart_data_prompt(message: str) -> str:
    return (
        "You are generating data for a bar chart based on the user request.\n"
        "Return ONLY JSON with this shape:\n"
        "{\n"
        '  "title": "<short title>",\n'
        '  "x_label": "<x axis label>",\n'
        '  "y_label": "<y axis label>",\n'
        '  "items": [\n'
        '    {"label": "<short label>", "value": <integer>}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Use 3-8 items.\n"
        "- Values must be integers between 1 and 500.\n"
        "- Labels must be short (max 12 characters).\n"
        "- If the user asks for fictional data, make it up.\n"
        "- Use the user's language.\n"
        f"User request: {message}\n"
    )


def _parse_chart_payload(text: str) -> dict | None:
    if not text:
        return None
    cleaned = _extract_json_block(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Chart JSON parse failed: %s", cleaned[:200])
        return None
    items = data.get("items") or data.get("data")
    if not isinstance(items, list):
        return None
    normalized_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        value = item.get("value")
        try:
            value = int(float(value))
        except (TypeError, ValueError):
            continue
        if not label or value <= 0:
            continue
        normalized_items.append({"label": label[:12], "value": value})
        if len(normalized_items) >= _CHART_MAX_ITEMS:
            break
    if not normalized_items:
        return None
    return {
        "type": "bar",
        "title": str(data.get("title", "")).strip(),
        "x_label": str(data.get("x_label", "")).strip(),
        "y_label": str(data.get("y_label", "")).strip(),
        "items": normalized_items,
    }


def _extract_symbol(message: str) -> str:
    match = re.search(
        r"hisse\s*adi[:\s]*([A-Za-z0-9-]{2,})", message, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"\b([A-Z][A-Z0-9-]{2,})\b", message)
    if match:
        return match.group(1).strip()
    return "TITAN-AS"


def _build_month_labels(count: int = 12) -> list[str]:
    months = [
        "Ocak",
        "Subat",
        "Mart",
        "Nisan",
        "Mayis",
        "Haziran",
        "Temmuz",
        "Agustos",
        "Eylul",
        "Ekim",
        "Kasim",
        "Aralik",
    ]
    return months[:count]


def _generate_time_series_payload(
    message: str,
    *,
    chart_type: str,
    show_ma: bool,
    points: int = 12,
) -> dict:
    import random

    symbol = _extract_symbol(message)
    labels = _build_month_labels(points)
    title = f"{symbol} Hisse Performansi (1 Yillik)"
    x_label = "Ay"
    y_label = "Fiyat"

    if chart_type == "candlestick":
        series = []
        price = random.randint(80, 160)
        for _ in range(points):
            open_price = price
            change = random.randint(-25, 25)
            close_price = max(10, min(500, open_price + change))
            high = min(500, max(open_price, close_price) +
                       random.randint(5, 30))
            low = max(5, min(open_price, close_price) - random.randint(5, 25))
            series.append(
                {
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close_price,
                }
            )
            price = close_price
        spike_up = max(1, points // 3)
        spike_down = min(points - 2, spike_up + 3)
        series[spike_up]["close"] = min(
            500, series[spike_up]["open"] + random.randint(80, 140))
        series[spike_up]["high"] = min(
            500, series[spike_up]["close"] + random.randint(10, 30))
        series[spike_down]["close"] = max(
            10, series[spike_down]["open"] - random.randint(80, 140))
        series[spike_down]["low"] = max(
            5, series[spike_down]["close"] - random.randint(10, 25))
        return {
            "type": "candlestick",
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
            "labels": labels,
            "points": series,
            "show_ma": show_ma,
            "spikes": {"up": spike_up, "down": spike_down},
        }

    values: list[int] = []
    value = random.randint(80, 160)
    for _ in range(points):
        step = random.randint(-20, 20)
        value = max(10, min(500, value + step))
        values.append(value)
    spike_up = max(1, points // 3)
    spike_down = min(points - 2, spike_up + 4)
    values[spike_up] = min(500, values[spike_up - 1] + random.randint(80, 140))
    values[spike_down] = max(
        10, values[spike_down - 1] - random.randint(80, 140))
    points_payload = [{"label": label, "value": value}
                      for label, value in zip(labels, values)]
    return {
        "type": "line",
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "points": points_payload,
        "show_ma": show_ma,
        "spikes": {"up": spike_up, "down": spike_down},
    }


def _fallback_chart_payload(message: str) -> dict | None:
    match = re.search(r"\(([^)]+)\)", message or "")
    labels: list[str] = []
    if match:
        raw = match.group(1)
        labels = [part.strip() for part in raw.split(",") if part.strip()]
    labels = labels[:_CHART_MAX_ITEMS]
    if not labels:
        return None
    values = [50 + (len(labels) - idx) * 20 for idx in range(len(labels))]
    items = [{"label": label[:12], "value": value}
             for label, value in zip(labels, values)]
    return {
        "type": "bar",
        "title": "Bar chart",
        "x_label": "",
        "y_label": "",
        "items": items,
    }


def _render_bar_chart(payload: dict) -> bytes | None:
    items = payload.get("items") or []
    if not items:
        return None
    labels = [item["label"] for item in items]
    values = [item["value"] for item in items]
    if not labels or not values:
        return None

    width = _CHART_WIDTH
    height = _CHART_HEIGHT
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _load_chart_font(12)
    title_font = _load_chart_font(16)
    axis_font = _load_chart_font(12)

    left = 80
    right = width - 40
    top = 70
    bottom = height - 80
    chart_width = right - left
    chart_height = bottom - top

    title = payload.get("title") or ""
    if title:
        title_box = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_box[2] - title_box[0]
        draw.text(((width - title_w) / 2, 20), title,
                  fill="black", font=title_font)

    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)

    max_value = max(values) if max(values) > 0 else 1
    count = len(values)
    gap = max(8, int(chart_width * 0.04))
    bar_width = max(10, int((chart_width - gap * (count + 1)) / count))
    palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
               "#59a14f", "#edc948", "#b07aa1", "#ff9da7"]

    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = left + gap + idx * (bar_width + gap)
        x1 = x0 + bar_width
        bar_height = int((value / max_value) * chart_height)
        y1 = bottom
        y0 = bottom - bar_height
        color = palette[idx % len(palette)]
        draw.rectangle([x0, y0, x1, y1], fill=color, outline="black")

        value_text = str(value)
        value_box = draw.textbbox((0, 0), value_text, font=font)
        value_w = value_box[2] - value_box[0]
        draw.text((x0 + (bar_width - value_w) / 2, max(top, y0 - 15)),
                  value_text, fill="black", font=font)

        label_box = draw.textbbox((0, 0), label, font=font)
        label_w = label_box[2] - label_box[0]
        draw.text((x0 + (bar_width - label_w) / 2, bottom + 6),
                  label, fill="black", font=font)

    x_label = payload.get("x_label") or ""
    y_label = payload.get("y_label") or ""
    if x_label:
        label_box = draw.textbbox((0, 0), x_label, font=axis_font)
        label_w = label_box[2] - label_box[0]
        draw.text(((width - label_w) / 2, height - 30),
                  x_label, fill="black", font=axis_font)
    if y_label:
        label_box = draw.textbbox((0, 0), y_label, font=axis_font)
        label_h = label_box[3] - label_box[1]
        draw.text((10, top - label_h - 5), y_label,
                  fill="black", font=axis_font)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _render_line_chart(payload: dict) -> bytes | None:
    points = payload.get("points") or []
    if not points:
        return None
    labels = [point["label"] for point in points]
    values = [point["value"] for point in points]
    if not labels or not values:
        return None

    width = _CHART_WIDTH
    height = _CHART_HEIGHT
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _load_chart_font(12)
    title_font = _load_chart_font(16)
    axis_font = _load_chart_font(12)

    left = 80
    right = width - 40
    top = 70
    bottom = height - 80
    chart_width = right - left
    chart_height = bottom - top

    title = payload.get("title") or ""
    if title:
        title_box = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_box[2] - title_box[0]
        draw.text(((width - title_w) / 2, 20), title,
                  fill="black", font=title_font)

    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)

    max_value = max(values) if max(values) > 0 else 1
    min_value = min(values) if min(values) > 0 else 0
    span = max(1, max_value - min_value)

    def _point(idx: int, value: int) -> tuple[int, int]:
        x = left + int((idx / max(1, len(values) - 1)) * chart_width)
        y = bottom - int(((value - min_value) / span) * chart_height)
        return x, y

    line_points = [_point(idx, value) for idx, value in enumerate(values)]
    draw.line(line_points, fill="#4e79a7", width=3)
    for x, y in line_points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3),
                     fill="#4e79a7", outline="white")

    if payload.get("show_ma"):
        window = 3
        ma_points: list[tuple[int, int]] = []
        for idx in range(len(values)):
            if idx + 1 < window:
                continue
            window_values = values[idx + 1 - window: idx + 1]
            avg = sum(window_values) / window
            ma_points.append(_point(idx, int(avg)))
        if len(ma_points) >= 2:
            draw.line(ma_points, fill="#e15759", width=2)

    spikes = payload.get("spikes") or {}
    spike_up = spikes.get("up")
    spike_down = spikes.get("down")
    for label, index, color in (
        ("Yukselis", spike_up, "#59a14f"),
        ("Dusus", spike_down, "#e15759"),
    ):
        if index is None or index < 0 or index >= len(values):
            continue
        x, y = _point(index, values[index])
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=2)
        draw.text((x + 6, max(top, y - 18)), label, fill=color, font=font)

    for idx, label in enumerate(labels):
        x, _ = _point(idx, values[idx])
        label_box = draw.textbbox((0, 0), label, font=font)
        label_w = label_box[2] - label_box[0]
        draw.text((x - label_w / 2, bottom + 6),
                  label, fill="black", font=font)

    x_label = payload.get("x_label") or ""
    y_label = payload.get("y_label") or ""
    if x_label:
        label_box = draw.textbbox((0, 0), x_label, font=axis_font)
        label_w = label_box[2] - label_box[0]
        draw.text(((width - label_w) / 2, height - 30),
                  x_label, fill="black", font=axis_font)
    if y_label:
        label_box = draw.textbbox((0, 0), y_label, font=axis_font)
        label_h = label_box[3] - label_box[1]
        draw.text((10, top - label_h - 5), y_label,
                  fill="black", font=axis_font)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _render_candlestick_chart(payload: dict) -> bytes | None:
    points = payload.get("points") or []
    labels = payload.get("labels") or []
    if not points or not labels:
        return None

    width = _CHART_WIDTH
    height = _CHART_HEIGHT
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _load_chart_font(12)
    title_font = _load_chart_font(16)
    axis_font = _load_chart_font(12)

    left = 80
    right = width - 40
    top = 70
    bottom = height - 80
    chart_width = right - left
    chart_height = bottom - top

    title = payload.get("title") or ""
    if title:
        title_box = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_box[2] - title_box[0]
        draw.text(((width - title_w) / 2, 20), title,
                  fill="black", font=title_font)

    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)

    highs = [point["high"] for point in points]
    lows = [point["low"] for point in points]
    max_value = max(highs) if max(highs) > 0 else 1
    min_value = min(lows) if min(lows) > 0 else 0
    span = max(1, max_value - min_value)

    def _y(value: int) -> int:
        return bottom - int(((value - min_value) / span) * chart_height)

    count = len(points)
    gap = max(6, int(chart_width * 0.03))
    candle_width = max(10, int((chart_width - gap * (count + 1)) / count))

    for idx, (label, point) in enumerate(zip(labels, points)):
        x0 = left + gap + idx * (candle_width + gap)
        x1 = x0 + candle_width
        open_price = point["open"]
        close_price = point["close"]
        high = point["high"]
        low = point["low"]
        color = "#59a14f" if close_price >= open_price else "#e15759"
        draw.line((x0 + candle_width / 2, _y(high), x0 +
                  candle_width / 2, _y(low)), fill="black", width=1)
        y_open = _y(open_price)
        y_close = _y(close_price)
        y_top = min(y_open, y_close)
        y_bottom = max(y_open, y_close)
        draw.rectangle([x0, y_top, x1, y_bottom], fill=color, outline="black")

        label_box = draw.textbbox((0, 0), label, font=font)
        label_w = label_box[2] - label_box[0]
        draw.text((x0 + (candle_width - label_w) / 2, bottom + 6),
                  label, fill="black", font=font)

    if payload.get("show_ma"):
        close_values = [point["close"] for point in points]
        ma_points: list[tuple[int, int]] = []
        window = 3
        for idx in range(len(close_values)):
            if idx + 1 < window:
                continue
            avg = sum(close_values[idx + 1 - window: idx + 1]) / window
            x = left + gap + idx * (candle_width + gap) + candle_width / 2
            ma_points.append((int(x), _y(int(avg))))
        if len(ma_points) >= 2:
            draw.line(ma_points, fill="#4e79a7", width=2)

    spikes = payload.get("spikes") or {}
    for label, index, color in (
        ("Yukselis", spikes.get("up"), "#59a14f"),
        ("Dusus", spikes.get("down"), "#e15759"),
    ):
        if index is None or index < 0 or index >= len(points):
            continue
        x = left + gap + index * (candle_width + gap) + candle_width / 2
        y = _y(points[index]["close"])
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=2)
        draw.text((x + 6, max(top, y - 18)), label, fill=color, font=font)

    x_label = payload.get("x_label") or ""
    y_label = payload.get("y_label") or ""
    if x_label:
        label_box = draw.textbbox((0, 0), x_label, font=axis_font)
        label_w = label_box[2] - label_box[0]
        draw.text(((width - label_w) / 2, height - 30),
                  x_label, fill="black", font=axis_font)
    if y_label:
        label_box = draw.textbbox((0, 0), y_label, font=axis_font)
        label_h = label_box[3] - label_box[1]
        draw.text((10, top - label_h - 5), y_label,
                  fill="black", font=axis_font)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


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
    stripped = value[len(_SUGGESTION_PREFIX):].strip()
    return stripped if stripped else None


async def _build_suggestion_markup(llm_client, history_text: str | None) -> InlineKeyboardMarkup | None:
    if llm_client is None or not history_text:
        return None
    try:
        suggestions = await _retry_async(
            lambda: asyncio.to_thread(
                llm_client.generate_suggestions, history_text),
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
            row.append(InlineKeyboardButton(
                text=text[:30], callback_data=callback_data))
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


def _get_recent_images(session_context: dict | None) -> list[dict]:
    if not session_context:
        return []
    recent = session_context.get("recent_images")
    images = []
    if isinstance(recent, list):
        for item in recent:
            if not isinstance(item, dict):
                continue
            if not item.get("path"):
                continue
            images.append(item)
    if images:
        return images
    last_path = session_context.get("last_image_path")
    if last_path:
        return [
            {
                "path": last_path,
                "prompt": session_context.get("last_image_prompt"),
                "summary": session_context.get("last_image_summary"),
                "source": session_context.get("last_image_source"),
            }
        ]
    return []


def _has_recent_images(session_context: dict | None) -> bool:
    return bool(_get_recent_images(session_context))


def _collect_recent_image_paths(session_context: dict | None) -> list[str]:
    images = _get_recent_images(session_context)
    return [str(item["path"]) for item in images if item.get("path")]


def _format_recent_image_summary(image: dict) -> str:
    summary = image.get("summary") or image.get("prompt") or image.get("source") or ""
    summary = str(summary).replace("\n", " ").strip()
    if not summary:
        return ""
    if len(summary) <= 80:
        return summary
    return f"{summary[:77].rstrip()}..."


def _resolve_image_reference(
    message_text: str,
    session_context: dict | None,
) -> tuple[str | None, str | None, bool]:
    images = _get_recent_images(session_context)
    if not images:
        return None, None, False
    normalized = _normalize_text(message_text or "")
    index = _extract_image_reference_index(normalized)
    if index is not None:
        selected = _select_image_by_recency(images, index)
        if selected:
            return selected.get("path"), selected.get("summary"), False
    if _mentions_first_image(normalized):
        selected = images[0]
        return selected.get("path"), selected.get("summary"), False
    if _mentions_previous_image(normalized):
        selected = _select_image_by_recency(images, 2) or images[-1]
        return selected.get("path"), selected.get("summary"), False
    if _mentions_last_image(normalized):
        selected = images[-1]
        return selected.get("path"), selected.get("summary"), False
    if len(images) > 1 and _is_ambiguous_image_request(normalized):
        return None, None, True
    selected = images[-1]
    return selected.get("path"), selected.get("summary"), False


def _select_image_by_recency(images: list[dict], index: int) -> dict | None:
    if index <= 0:
        return None
    if index > len(images):
        return None
    return images[-index]


def _extract_image_reference_index(text: str) -> int | None:
    match = re.search(r"\b(\d+)\s+(?:previous|back|onceki)\b", text)
    if match:
        return int(match.group(1)) + 1
    match = re.search(r"\b(\d+)\s*(?:image|photo|picture)\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(?:image|photo|picture)\s*(\d+)\b", text)
    if match:
        return int(match.group(1))
    for word, index in _ORDINAL_IMAGE_TOKENS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return index
    return None


def _mentions_last_image(text: str) -> bool:
    return _contains_word(text, ("latest", "last", "most recent", "son", "sonuncu", "en son"))


def _mentions_previous_image(text: str) -> bool:
    return _contains_word(text, ("previous", "prev", "prior", "onceki"))


def _mentions_first_image(text: str) -> bool:
    return _contains_word(text, ("first", "oldest", "ilk"))


def _contains_word(text: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def _is_ambiguous_image_request(text: str) -> bool:
    if _contains_any(text, _IMAGE_COMPARE_TOKENS):
        return True
    if "this or that" in text:
        return True
    pronouns = re.findall(r"\b(this|that|it|bunu|sunu)\b", text)
    if len(set(pronouns)) >= 2:
        return True
    if pronouns and (" ya da " in text or " or " in text):
        return True
    return False


def _build_image_disambiguation_message(session_context: dict | None) -> str:
    images = _get_recent_images(session_context)
    if not images:
        return "Please send the image you want to edit."
    lines = [
        "I found multiple images. Which one should I use?",
        "1=latest, 2=previous, 3=two back",
    ]
    for index, image in enumerate(reversed(images[-5:]), start=1):
        summary = _format_recent_image_summary(image)
        label = summary or "image"
        lines.append(f"{index}) {label}")
    lines.append("You can reply with 'latest', 'previous', or a number.")
    return "\n".join(lines)


_ORDINAL_IMAGE_TOKENS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def _augment_session_context_for_image(
    task: str,
    session_context: dict[str, object] | None,
) -> dict[str, object] | None:
    if not session_context:
        return session_context
    if not _mentions_image_reference(task):
        return session_context
    reference_path, reference_summary, ambiguous = _resolve_image_reference(
        task, session_context
    )
    if ambiguous or not reference_path:
        return session_context
    enriched = dict(session_context)
    enriched["intent_hints"] = {
        "image_reference_path": reference_path,
        "image_reference_summary": reference_summary,
        "note": "The user references the last image with a pronoun.",
    }
    return enriched


def _mentions_image_reference(text: str) -> bool:
    normalized = _normalize_text(text)
    return _contains_any(
        normalized,
        ("this", "that", "it", "image", "photo", "picture", "bunu", "sunu", "resim", "gorsel"),
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
        logger.warning(
            "Image request classification parse failed: %s", cleaned[:200])
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
    on_retry=None,
):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await action()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            logger.warning("%s failed (attempt %s/%s): %s",
                           label, attempt, attempts, exc)
            if on_retry is not None:
                try:
                    result = on_retry(attempt, attempts, exc)
                    if inspect.isawaitable(result):
                        await result
                except Exception as notify_exc:
                    logger.warning(
                        "%s retry notification failed: %s", label, notify_exc)
            await asyncio.sleep(_retry_delay(attempt, base_delay, max_delay))
    if last_exc:
        raise last_exc
    raise RuntimeError(f"{label} failed without exception.")


async def _safe_reply_text(message, text: str, *, label: str = "Telegram reply") -> None:
    async def _send():
        return await message.reply_text(text)

    try:
        await _retry_async(
            _send,
            label=label,
            attempts=_TELEGRAM_REPLY_RETRY_ATTEMPTS,
            base_delay=_TELEGRAM_REPLY_RETRY_BASE_DELAY,
            max_delay=_TELEGRAM_REPLY_RETRY_MAX_DELAY,
        )
    except Exception as exc:
        logger.warning("%s failed: %s", label, exc)


async def _retry_image_bytes(
    action,
    *,
    label: str,
    message,
    attempts: int,
    base_delay: float,
    max_delay: float,
) -> bytes:
    async def _wrapped_action():
        image_bytes = await action()
        if not image_bytes:
            raise ValueError("No image bytes received.")
        return image_bytes

    async def _notify_retry(attempt: int, total: int, exc: Exception):
        notice = f"{label} failed. Retrying ({attempt + 1}/{total})..."
        await _safe_reply_text(message, notice, label=f"{label} retry notice")

    return await _retry_async(
        _wrapped_action,
        label=label,
        attempts=attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        on_retry=_notify_retry,
    )


def _format_limit_mb(limit_bytes: int) -> str:
    return f"{int(limit_bytes / _BYTES_PER_MB)} MB"


def _file_too_large_message(file_kind: str, max_bytes: int) -> str:
    return f"{file_kind} file is too large for Gemini. Max size is {_format_limit_mb(max_bytes)}."


async def _build_unsupported_image_message(
    llm_client,
    prompt_text: str,
    mime_label: str,
) -> str:
    fallback = (
        f"Sorry, Gemini doesn't support this image type ({mime_label}). "
        "Please send PNG, JPEG, WEBP, HEIC, or HEIF."
    )
    if llm_client is None or not prompt_text:
        return fallback
    prompt = (
        "The user tried to upload an image format that Gemini cannot process.\n"
        f"User request: {prompt_text}\n"
        f"Image type: {mime_label}\n"
        "Reply in the user's language with a short, friendly message that this image format is not supported. "
        "Mention the supported formats: PNG, JPEG, WEBP, HEIC, HEIF."
    )
    try:
        response = await asyncio.to_thread(llm_client.generate_text, prompt)
    except Exception as exc:
        logger.warning("Unsupported image message generation failed: %s", exc)
        return fallback
    cleaned = response.strip()
    return cleaned or fallback


def _resolve_image_mime(image_path: str, hinted_mime: str | None) -> str | None:
    return hinted_mime or mimetypes.guess_type(image_path)[0]


async def _validate_image_for_gemini(
    llm_client,
    prompt_text: str,
    image_path: str,
    mime_type: str | None,
) -> str | None:
    resolved_mime = _resolve_image_mime(image_path, mime_type)
    if resolved_mime not in GEMINI_IMAGE_MIME_TYPES:
        label = resolved_mime or Path(image_path).suffix.lower() or "unknown"
        return await _build_unsupported_image_message(llm_client, prompt_text, label)
    try:
        size_bytes = Path(image_path).stat().st_size
    except Exception as exc:
        logger.warning("Image size check failed: %s", exc)
        return None
    if size_bytes > GEMINI_FLASH_MAX_INLINE_IMAGE_BYTES:
        return _file_too_large_message("Image", GEMINI_FLASH_MAX_INLINE_IMAGE_BYTES)
    return None


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

    settings = _resolve_settings(context)
    if not bool(getattr(settings, "computer_use_enabled", True)):
        if not silent_fail:
            await message.reply_text("Computer use is disabled. Send 'pc on' to enable.")
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

    streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
    show_thoughts = bool(getattr(settings, "show_thoughts", True))
    live_message = None
    if streaming_enabled and hasattr(llm_client, "stream_text"):
        live_message = await message.reply_text("Planning PC actions...", parse_mode=ParseMode.HTML)
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
                await manager.finalize(handler.format_message(include_thoughts=False))
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
