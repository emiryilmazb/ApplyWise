from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from uuid import uuid4

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from app.agent.form_intelligence import process_question
from app.agent.state import AgentContext
from app.agent.types import Intent
from app.config import get_settings
from app.application.session import (
    FailureReason,
    JobApplicationSession,
    SessionStatus,
    StepType,
)
from app.storage.memory import append_past_answer, append_personal_fact, load_memory_context
from app.sites.base import SiteAdapter, StepExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    session: JobApplicationSession
    status: SessionStatus


def _resolve_runtime_settings(telegram_app):
    if telegram_app is None:
        return get_settings()
    bot_data = getattr(telegram_app, "bot_data", None)
    if not bot_data:
        return get_settings()
    return bot_data.get("settings") or get_settings()


def _is_computer_use_enabled(telegram_app, pc_agent_enabled: bool) -> bool:
    if not pc_agent_enabled:
        return False
    settings = _resolve_runtime_settings(telegram_app)
    return bool(getattr(settings, "computer_use_enabled", True))


def _should_persist_answer(intent: Intent, submitted: bool) -> bool:
    if not submitted:
        return False
    return intent not in {
        Intent.MOTIVATION_COMPANY,
        Intent.MOTIVATION_ROLE,
        Intent.OPEN_TEXT,
    }


async def run_session(
    session: JobApplicationSession,
    adapter: SiteAdapter,
    agent_context: AgentContext,
    llm_client,
    memory=None,
    telegram_app=None,
    chat_id: str | None = None,
    pc_agent_enabled: bool = True,
) -> PipelineResult:
    memory = memory or load_memory_context()
    session.log_event(f"Session started on site={adapter.site_name}")

    settings = _resolve_runtime_settings(telegram_app)
    if not bool(getattr(settings, "browser_use_enabled", True)):
        session.pause()
        session.log_event("Paused: browser use disabled")
        if telegram_app and chat_id:
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text="Browser automation is disabled. Send 'browser on' to enable.",
            )
        return PipelineResult(session=session, status=session.status)
    pc_agent_enabled = pc_agent_enabled and bool(getattr(settings, "computer_use_enabled", True))

    page = await _ensure_playwright_page(agent_context, settings)
    if not session.steps:
        session.steps = await _safe_get_steps(
            adapter,
            session,
            page,
            agent_context,
            llm_client,
            telegram_app,
            chat_id,
        )

    for _ in range(len(session.steps)):
        if session.status != SessionStatus.RUNNING:
            break
        if agent_context.stop_requested:
            session.pause()
            session.log_event(f"Paused due to STOP request: {agent_context.stop_reason}")
            return PipelineResult(session=session, status=session.status)
        step = session.current_step()
        if step is None:
            break

        if step.step_type != StepType.QUESTION:
            result = await _safe_execute_step(
                adapter,
                session,
                step,
                None,
                page,
                agent_context,
                llm_client,
                telegram_app,
                chat_id,
            )
            if not result.success:
                if _is_computer_use_enabled(telegram_app, pc_agent_enabled) and result.failure_reason == FailureReason.UNEXPECTED_STRUCTURE:
                    recovery = await _attempt_pc_recovery(
                        session,
                        agent_context,
                        llm_client,
                        telegram_app,
                        chat_id,
                    )
                    if recovery:
                        session.mark_step_completed()
                        continue
                    session.pause()
                    break
                session.fail(result.failure_reason or FailureReason.UNKNOWN)
                await _notify_failure(session, telegram_app, chat_id, result.details)
                break
            session.mark_step_completed()
            continue

        question_text = str(step.payload.get("question", "")).strip()
        if not question_text:
            session.fail(FailureReason.UNEXPECTED_STRUCTURE)
            await _notify_failure(session, telegram_app, chat_id, "Missing question text")
            break

        outcome = process_question(
            question_text=question_text,
            llm_client=llm_client,
            memory=memory,
            company_name=session.job_metadata.company,
            role=session.job_metadata.role,
        )

        answer_text = outcome.answer
        if outcome.needs_human:
            question_id = str(uuid4())
            pending = agent_context.create_pending_request(
                question_id=question_id,
                intent=outcome.normalized.normalized_intent.value,
                question=question_text,
                category="personal"
                if outcome.normalized.normalized_intent == Intent.LEGAL_WORK_AUTHORIZATION
                else "answer",
            )
            session.pause()
            session.human_interactions_count += 1
            session.log_event(f"Paused for human input question_id={pending.question_id}")
            logger.info("Session paused: question_id=%s", pending.question_id)

            if telegram_app and chat_id:
                await telegram_app.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"{session.job_metadata.company} ({session.job_metadata.role}) asks: "
                        f"{question_text}\nquestion_id={pending.question_id}"
                    ),
                )

            while True:
                reply = await agent_context.wait_for_human_reply_or_stop()
                if reply is None:
                    session.pause()
                    session.log_event("Paused due to STOP request")
                    return PipelineResult(session=session, status=session.status)
                if reply.question_id != pending.question_id:
                    logger.info("Ignored human reply for stale question_id=%s", reply.question_id)
                    continue
                answer_text = reply.text
                agent_context.register_answer(
                    outcome.normalized.normalized_intent.value,
                    reply.text,
                    reply.source,
                    reply.timestamp,
                )
                agent_context.resolve_pending_request(reply)
                session.resume()
                session.log_event(f"Resumed after human input question_id={pending.question_id}")
                logger.info("Session resumed after question_id=%s", pending.question_id)
                break

        result = await _safe_execute_step(
            adapter,
            session,
            step,
            answer_text,
            page,
            agent_context,
            llm_client,
            telegram_app,
            chat_id,
        )
        if not result.success:
            if _is_computer_use_enabled(telegram_app, pc_agent_enabled) and result.failure_reason == FailureReason.UNEXPECTED_STRUCTURE:
                recovery = await _attempt_pc_recovery(
                    session,
                    agent_context,
                    llm_client,
                    telegram_app,
                    chat_id,
                )
                if recovery:
                    session.mark_step_completed()
                    continue
                session.pause()
                break
            session.fail(result.failure_reason or FailureReason.UNKNOWN)
            await _notify_failure(session, telegram_app, chat_id, result.details)
            break

        if answer_text:
            timestamp = datetime.now(timezone.utc).isoformat()
            if outcome.needs_human and outcome.normalized.normalized_intent == Intent.LEGAL_WORK_AUTHORIZATION:
                append_personal_fact(
                    outcome.normalized.normalized_intent,
                    answer_text,
                    "human",
                    timestamp,
                    country=_extract_country(question_text),
                )
                memory.personal_facts[outcome.normalized.normalized_intent.value] = {
                    "value": answer_text,
                    "confirmed_at": timestamp,
                    "source": "human",
                }
            elif (
                _should_persist_answer(outcome.normalized.normalized_intent, result.submitted)
                and outcome.reason != "memory_has_answer"
            ):
                append_past_answer(
                    outcome.normalized.normalized_intent,
                    question_text,
                    answer_text,
                    source="human" if outcome.needs_human else "llm",
                    submitted_at=timestamp,
                )
                memory.past_answers.setdefault("entries", []).append(
                    {
                        "intent": outcome.normalized.normalized_intent.value,
                        "question": question_text,
                        "answer": answer_text,
                        "source": "human" if outcome.needs_human else "llm",
                        "submitted_at": timestamp,
                    }
                )

        session.mark_step_completed()

    if session.status == SessionStatus.RUNNING and session.current_step() is None:
        session.complete()

    _log_summary(session, adapter.site_name)
    return PipelineResult(session=session, status=session.status)


async def _notify_failure(
    session: JobApplicationSession,
    telegram_app,
    chat_id: str | None,
    details: str | None,
) -> None:
    logger.error("Session failed: %s", details or "Unknown failure")
    if telegram_app and chat_id:
        await telegram_app.bot.send_message(
            chat_id=chat_id,
            text=(
                f"Application failed for {session.job_metadata.company} "
                f"{session.job_metadata.role}. Reason: {details or 'unknown'}"
            ),
        )


def _log_summary(session: JobApplicationSession, site_name: str) -> None:
    completed_steps = len([step for step in session.steps if step.status.name == "COMPLETED"])
    logger.info(
        "Application summary: site=%s company=%s role=%s steps_completed=%s "
        "human_interactions=%s status=%s",
        site_name,
        session.job_metadata.company,
        session.job_metadata.role,
        completed_steps,
        session.human_interactions_count,
        session.status.value,
    )


async def _attempt_pc_recovery(
    session: JobApplicationSession,
    agent_context: AgentContext,
    llm_client,
    telegram_app,
    chat_id: str | None,
) -> bool:
    from app.pc_agent.controller import run_task_with_approval

    logger.warning("Attempting PC recovery for session=%s", session.job_metadata.company)
    result = await run_task_with_approval(
        agent_context=agent_context,
        telegram_app=telegram_app,
        chat_id=chat_id,
        task="Complete the current application step using PC control",
        context={"url": session.job_metadata.extra.get("url") if session.job_metadata.extra else None},
        llm_client=llm_client,
    )
    session.log_event(f"PC recovery result: {result.reason}")
    return result.completed


def _extract_country(question: str) -> str | None:
    lowered = question.lower()
    token = "work in "
    if token in lowered:
        country = question[lowered.index(token) + len(token) :].strip(" ?.")
        return country or None
    return None


async def _ensure_playwright_page(agent_context: AgentContext, settings):
    if "playwright_async" in agent_context.pc_state:
        return agent_context.pc_state["page_async"]
    playwright = await async_playwright().start()
    headless = bool(getattr(settings, "playwright_headless", False))
    browser = await playwright.chromium.launch(headless=headless)
    page = await browser.new_page()
    agent_context.pc_state["playwright_async"] = playwright
    agent_context.pc_state["browser_async"] = browser
    agent_context.pc_state["page_async"] = page
    return page


async def _safe_get_steps(
    adapter: SiteAdapter,
    session: JobApplicationSession,
    page,
    agent_context,
    llm_client,
    telegram_app,
    chat_id: str | None,
):
    try:
        return await adapter.get_application_steps(
            session,
            page,
            agent_context=agent_context,
            llm_client=llm_client,
            telegram_app=telegram_app,
            chat_id=chat_id,
        )
    except PlaywrightTimeoutError as exc:
        session.fail(FailureReason.TIMEOUT)
        logger.error("Adapter timeout: %s", exc)
        return []
    except Exception as exc:
        session.fail(FailureReason.UNEXPECTED_STRUCTURE)
        logger.error("Adapter get_application_steps error: %s", exc)
        return []


async def _safe_execute_step(
    adapter: SiteAdapter,
    session: JobApplicationSession,
    step,
    answer: str | None,
    page,
    agent_context,
    llm_client,
    telegram_app,
    chat_id: str | None,
) -> StepExecutionResult:
    try:
        return await adapter.execute_step(
            session,
            step,
            answer=answer,
            page=page,
            agent_context=agent_context,
            llm_client=llm_client,
            telegram_app=telegram_app,
            chat_id=chat_id,
        )
    except PlaywrightTimeoutError as exc:
        logger.error("Adapter execute_step timeout: %s", exc)
        return StepExecutionResult(
            success=False,
            failure_reason=FailureReason.TIMEOUT,
            details=str(exc),
        )
    except Exception as exc:
        logger.error("Adapter execute_step error: %s", exc)
        return StepExecutionResult(
            success=False,
            failure_reason=FailureReason.UNEXPECTED_STRUCTURE,
            details=str(exc),
        )
