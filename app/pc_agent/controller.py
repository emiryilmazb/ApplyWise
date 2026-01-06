from __future__ import annotations

from dataclasses import dataclass
import logging
from uuid import uuid4

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from app.agent.state import AgentContext
from app.config import get_settings
from app.pc_agent.executor import execute_action, ExecutionError
from app.pc_agent.planner import build_action_plan, requires_extra_confirmation
from app.session_manager import get_session_context, record_task_completion
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
            logger.info("Ignored reply for unrelated approval_id=%s", reply.question_id)
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
        settings = get_settings()
        streaming_enabled = bool(getattr(settings, "streaming_enabled", False))
        show_thoughts = bool(getattr(settings, "show_thoughts", True))
        stream_handler = None
        if streaming_enabled and hasattr(llm_client, "stream_text") and telegram_app and chat_id:
            live_message = await telegram_app.bot.send_message(
                chat_id=chat_id,
                text="⏳ Revising plan...",
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
                await manager.finalize(handler.format_message())
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
        high_risk_reply = await agent_context.wait_for_human_reply_or_stop()
        if high_risk_reply is None:
            return ControlResult(approved=False, completed=False, reason="stopped")
        while high_risk_reply.question_id != approval_id:
            logger.info("Ignored reply for unrelated approval_id=%s", high_risk_reply.question_id)
            high_risk_reply = await agent_context.wait_for_human_reply_or_stop()
            if high_risk_reply is None:
                return ControlResult(approved=False, completed=False, reason="stopped")
        if high_risk_reply.text.strip().upper() != "RUN":
            return ControlResult(approved=False, completed=False, reason="high_risk_denied")
        allow_run_executable = True

    _configure_search_context(agent_context, task, actions)
    for action in actions:
        if agent_context.stop_requested:
            return ControlResult(approved=True, completed=False, reason="stopped")
        try:
            execute_action(
                action,
                agent_context,
                llm_client=llm_client,
                allow_run_executable=allow_run_executable,
            )
        except ExecutionError as exc:
            logger.error("Execution halted: %s", exc)
            return ControlResult(approved=True, completed=False, reason="execution_error")

    summary_task = task
    if revision_notes:
        summary_task = f"{task} | revised: {revision_notes[-1]}"
    record_task_completion(session_user_id, summary_task, actions)
    return ControlResult(approved=True, completed=True, reason="completed")


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


def _derive_search_query(task: str) -> str:
    lowered = _normalize_text(task)
    query = task
    if " ve " in lowered:
        query = task[lowered.rfind(" ve ") + 4 :]
    elif " and " in lowered:
        query = task[lowered.rfind(" and ") + 5 :]
    query = query.strip(" .,-")
    query = _strip_tokens(
        query,
        ("spotify", "open", "launch", "play", "song", "sarki", "cal", "oynat", "baslat"),
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
            text = text[:idx] + text[idx + len(token_norm) :]
            normalized = normalized[:idx] + normalized[idx + len(token_norm) :]
    return text


def _format_approval_message(actions) -> str:
    lines = ["I will:"]
    for index, action in enumerate(actions, start=1):
        lines.append(f"{index}. {action.description}")
    lines.append("Reply YES to proceed, NO to cancel, or send a change request.")
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
