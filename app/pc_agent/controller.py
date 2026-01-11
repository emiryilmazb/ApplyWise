from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from uuid import uuid4

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from app.agent.state import AgentContext
from app.config import get_settings
from app.pc_agent.executor import execute_action, ExecutionError
from app.pc_agent.planner import build_action_plan, requires_extra_confirmation
from app.pc_agent.vision import capture_screen
from app.session_manager import (
    get_session_context,
    record_handoff_detail,
    record_task_completion,
)
from app.telegram.streaming import AsyncStreamHandler, MessageManager

logger = logging.getLogger(__name__)


@dataclass
class ControlResult:
    approved: bool
    completed: bool
    reason: str


async def run_task_with_approval(
    agent_context: AgentContext,
    telegram_app,
    chat_id: str | None,
    task: str,
    context: dict | None = None,
    user_id: str | None = None,
    llm_client=None,
) -> ControlResult:
    settings = _resolve_runtime_settings(telegram_app)
    if not bool(getattr(settings, "computer_use_enabled", True)):
        if telegram_app and chat_id:
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text="Computer use is disabled. Send 'pc on' to enable.",
            )
        return ControlResult(approved=False, completed=False, reason="computer_use_disabled")
    _configure_browser_context(agent_context, telegram_app)
    actions = build_action_plan(task, context)
    current_steps = context.get("steps", []) if context else []
    revision_notes: list[str] = []
    approval_id = str(uuid4())
    session_user_id = user_id or chat_id

    async def _await_reply() -> HumanReply | None:
        reply = await agent_context.wait_for_human_reply_or_stop()
        if reply is None:
            return None
        while reply.question_id != approval_id:
            logger.info("Ignored reply for unrelated approval_id=%s",
                        reply.question_id)
            reply = await agent_context.wait_for_human_reply_or_stop()
            if reply is None:
                return None
        return reply

    await _send_approval_message(telegram_app, chat_id, approval_id, actions)
    pending = agent_context.create_pending_request(
        question_id=approval_id,
        intent="PC_ACTION_PLAN",
        question=task,
        category="approval",
    )
    logger.info("Awaiting approval for approval_id=%s", pending.question_id)

    reply = await _await_reply()
    if reply is None:
        return ControlResult(approved=False, completed=False, reason="stopped")

    while True:
        if _is_approval(reply.text):
            agent_context.resolve_pending_request(reply)
            break
        if reply.text.strip().upper() == "NO":
            agent_context.resolve_pending_request(reply)
            return ControlResult(approved=False, completed=False, reason="approval_denied")

        revision = reply.text.strip()
        if not revision:
            reply = await _await_reply()
            if reply is None:
                return ControlResult(approved=False, completed=False, reason="stopped")
            continue

        if llm_client is None:
            if telegram_app and chat_id:
                await telegram_app.bot.send_message(
                    chat_id=chat_id,
                    text="Cannot revise the plan without an LLM. Reply YES/NO or try again later.",
                )
            reply = await _await_reply()
            if reply is None:
                return ControlResult(approved=False, completed=False, reason="stopped")
            continue

        from app.pc_agent.llm_planner import plan_pc_actions

        session_context = get_session_context(session_user_id)
        settings = _resolve_runtime_settings(telegram_app)
        streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
        show_thoughts = bool(getattr(settings, "show_thoughts", True))
        stream_handler = None
        if streaming_enabled and hasattr(llm_client, "stream_text") and telegram_app and chat_id:
            live_message = await telegram_app.bot.send_message(
                chat_id=chat_id,
                text="Revising plan...",
                parse_mode=ParseMode.HTML,
            )
            manager = MessageManager(
                bot=telegram_app.bot,
                chat_id=str(chat_id),
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

        revised_steps = await plan_pc_actions(
            llm_client,
            task,
            session_context=session_context,
            previous_plan=current_steps,
            revision_request=revision,
            stream_handler=stream_handler,
        )
        if not revised_steps:
            if telegram_app and chat_id:
                await telegram_app.bot.send_message(
                    chat_id=chat_id,
                    text="Could not update the plan. Reply YES/NO or send a clearer change.",
                )
            reply = await _await_reply()
            if reply is None:
                return ControlResult(approved=False, completed=False, reason="stopped")
            continue

        revision_notes.append(revision)
        current_steps = revised_steps
        actions = build_action_plan(task, {"steps": current_steps})
        await _send_approval_message(telegram_app, chat_id, approval_id, actions, updated=True)
        reply = await _await_reply()
        if reply is None:
            return ControlResult(approved=False, completed=False, reason="stopped")

    allow_run_executable = False
    if requires_extra_confirmation(actions):
        if telegram_app and chat_id:
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text=_format_high_risk_message(actions),
                reply_markup=_build_high_risk_keyboard(),
            )
        pending = agent_context.create_pending_request(
            question_id=approval_id,
            intent="PC_ACTION_HIGH_RISK",
            question=task,
            category="approval_high_risk",
        )
        logger.info("Awaiting high-risk approval for approval_id=%s", pending.question_id)
        high_risk_reply = await agent_context.wait_for_human_reply_or_stop()
        if high_risk_reply is None:
            return ControlResult(approved=False, completed=False, reason="stopped")
        while high_risk_reply.question_id != approval_id:
            logger.info("Ignored reply for unrelated approval_id=%s",
                        high_risk_reply.question_id)
            high_risk_reply = await agent_context.wait_for_human_reply_or_stop()
            if high_risk_reply is None:
                return ControlResult(approved=False, completed=False, reason="stopped")
        if high_risk_reply.text.strip().upper() != "RUN":
            agent_context.resolve_pending_request(high_risk_reply)
            return ControlResult(approved=False, completed=False, reason="high_risk_denied")
        agent_context.resolve_pending_request(high_risk_reply)
        allow_run_executable = True

    _configure_search_context(agent_context, task, actions)
    if telegram_app and chat_id:
        await telegram_app.bot.send_message(
            chat_id=chat_id,
            text="Confirmed. Executing the action plan.",
        )
    pc_executor = _get_pc_executor(agent_context)
    loop = asyncio.get_running_loop()
    replan_attempts_left = 1
    action_index = 0
    while action_index < len(actions):
        action = actions[action_index]
        if agent_context.stop_requested:
            return ControlResult(approved=True, completed=False, reason="stopped")
        try:
            await loop.run_in_executor(
                pc_executor,
                execute_action,
                action,
                agent_context,
                llm_client,
                allow_run_executable,
            )
        except ExecutionError as exc:
            logger.error("Execution halted: %s", exc)
            if llm_client and current_steps and replan_attempts_left > 0:
                replan_attempts_left -= 1
                if telegram_app and chat_id:
                    await telegram_app.bot.send_message(
                        chat_id=chat_id,
                        text="Step failed. Attempting to replan and retry.",
                    )
                from app.pc_agent.llm_planner import plan_pc_actions

                session_context = get_session_context(session_user_id)
                revision_request = (
                    f"Previous step failed: {action.description}. "
                    f"Error: {exc}. Revise the plan to complete the task."
                )
                revised_steps = await plan_pc_actions(
                    llm_client,
                    task,
                    session_context=session_context,
                    previous_plan=current_steps,
                    revision_request=revision_request,
                )
                if revised_steps:
                    current_steps = revised_steps
                    actions = build_action_plan(task, {"steps": current_steps})
                    _configure_search_context(agent_context, task, actions)
                    action_index = 0
                    continue
            return ControlResult(approved=True, completed=False, reason="execution_error")
        except Exception as exc:
            logger.exception("Execution crashed: %s", exc)
            return ControlResult(approved=True, completed=False, reason="execution_error")
        settings = _resolve_runtime_settings(telegram_app)
        screenshot_enabled = bool(
            getattr(settings, "screenshot_enabled", True))
        await _send_action_screenshot(
            telegram_app,
            chat_id,
            action.description,
            screenshot_enabled,
        )
        action_index += 1

    summary_task = task
    if revision_notes:
        summary_task = f"{task} | revised: {revision_notes[-1]}"
    record_task_completion(session_user_id, summary_task, actions)
    handoff_url = agent_context.pc_state.get("handoff_url")
    if handoff_url:
        record_handoff_detail(session_user_id, "last_url", handoff_url)
    return ControlResult(approved=True, completed=True, reason="completed")


def _get_pc_executor(agent_context: AgentContext) -> ThreadPoolExecutor:
    executor = agent_context.pc_state.get("pc_executor")
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pc-agent")
        agent_context.pc_state["pc_executor"] = executor
    return executor


def _configure_browser_context(agent_context: AgentContext, telegram_app) -> None:
    settings = _resolve_runtime_settings(telegram_app)
    agent_context.pc_state["use_system_browser"] = bool(
        getattr(settings, "pc_use_system_browser", False)
    )
    agent_context.pc_state["browser_name"] = getattr(
        settings, "pc_browser_name", "Chrome"
    )
    agent_context.pc_state["browser_window_title"] = getattr(
        settings, "pc_browser_window_title", ".*Chrome.*"
    )
    agent_context.pc_state["browser_user_data_dir"] = getattr(
        settings, "pc_browser_user_data_dir", ""
    )
    agent_context.pc_state["browser_executable_path"] = getattr(
        settings, "pc_browser_executable_path", ""
    )


def _configure_search_context(agent_context: AgentContext, task: str, actions) -> None:
    has_search = False
    allow_auto_play = True
    for action in actions:
        description = _normalize_text(str(action.description or ""))
        payload_text = ""
        if action.name == "click_text":
            payload_text = _normalize_text(str(action.payload.get("text", "")))
        if "search" in description or "search" in payload_text or "arama" in description or "arama" in payload_text:
            has_search = True
        if action.name in {"click_text", "mouse_click"}:
            combined = f"{description} {payload_text}".strip()
            if (
                "play" in combined
                or "result" in combined
                or "song" in combined
                or "track" in combined
                or "sarki" in combined
                or "cal" in combined
                or "oynat" in combined
                or "baslat" in combined
                or "sonuc" in combined
            ):
                allow_auto_play = False
    if not has_search:
        agent_context.pc_state.pop("pending_search_query", None)
        agent_context.pc_state.pop("allow_auto_play_after_enter", None)
        return
    agent_context.pc_state["pending_search_query"] = _derive_search_query(task)
    agent_context.pc_state["allow_auto_play_after_enter"] = allow_auto_play


def _resolve_runtime_settings(telegram_app):
    if telegram_app is None:
        return get_settings()
    bot_data = getattr(telegram_app, "bot_data", None)
    if not bot_data:
        return get_settings()
    return bot_data.get("settings") or get_settings()


def _derive_search_query(task: str) -> str:
    lowered = _normalize_text(task)
    query = task
    if " ve " in lowered:
        query = task[lowered.rfind(" ve ") + 4:]
    elif " and " in lowered:
        query = task[lowered.rfind(" and ") + 5:]
    query = query.strip(" .,-")
    query = _strip_tokens(
        query,
        ("spotify", "open", "launch", "play",
         "song", "track", "start", "sarki", "cal", "oynat", "baslat"),
    )
    return query.strip(" .,-") or task.strip()


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


def _strip_tokens(text: str, tokens: tuple[str, ...]) -> str:
    normalized = _normalize_text(text)
    for token in tokens:
        token_norm = _normalize_text(token)
        while True:
            idx = normalized.find(token_norm)
            if idx == -1:
                break
            text = text[:idx] + text[idx + len(token_norm):]
            normalized = normalized[:idx] + normalized[idx + len(token_norm):]
    return text


def _format_approval_message(actions) -> str:
    lines = ["I will:"]
    for index, action in enumerate(actions, start=1):
        lines.append(f"{index}. {action.description}")
    lines.append(
        "Reply YES to proceed, NO to cancel, or send a change request.")
    return "\n".join(lines)


def _is_approval(text: str) -> bool:
    return text.strip().upper() == "YES"


def _format_high_risk_message(actions) -> str:
    lines = ["This includes high-risk actions:"]
    for action in actions:
        if action.name == "run_executable":
            path = action.payload.get("path", "unknown")
            lines.append(f"This will run {path} on your machine.")
    lines.append("Type RUN to confirm.")
    return "\n".join(lines)


def _build_approval_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(text="YES", callback_data="YES"),
            InlineKeyboardButton(text="NO", callback_data="NO"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def _build_high_risk_keyboard() -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton(text="RUN", callback_data="RUN")]]
    return InlineKeyboardMarkup(keyboard)


async def _send_approval_message(
    telegram_app,
    chat_id: str | None,
    approval_id: str,
    actions,
    updated: bool = False,
) -> None:
    approval_text = _format_approval_message(actions)
    logger.info("Action plan prepared: %s", approval_text)
    if not telegram_app or not chat_id:
        return
    prefix = "Updated plan:\n" if updated else ""
    await telegram_app.bot.send_message(
        chat_id=chat_id,
        text=f"{prefix}{approval_text}\napproval_id={approval_id}",
        reply_markup=_build_approval_keyboard(),
    )


async def _send_action_screenshot(
    telegram_app,
    chat_id: str | None,
    description: str | None,
    enabled: bool,
) -> None:
    if not enabled or not telegram_app or not chat_id:
        return
    try:
        screenshot = await asyncio.to_thread(capture_screen)
        caption = None
        if description:
            clean = str(description).strip().replace("\n", " ")
            if len(clean) > 900:
                clean = f"{clean[:897]}..."
            caption = clean
        with open(screenshot.path, "rb") as img_file:
            await telegram_app.bot.send_photo(
                chat_id=chat_id,
                photo=img_file,
                caption=caption,
            )
    except Exception as exc:
        logger.warning("Failed to send action screenshot: %s", exc)
