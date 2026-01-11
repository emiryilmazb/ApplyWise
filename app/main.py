from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path = [path for path in sys.path if path != str(APP_DIR)]

import asyncio
import logging
from uuid import uuid4
from app.agent.model_factory import ModelFactory
from app.agent.state import AgentContext
from app.application.pipeline import run_session
from app.application.session import JobApplicationSession, JobMetadata, SiteType
from app.config import get_settings
from app.logger import init_logging
from app.storage.database import get_database
from app.storage.memory import load_memory_context
from app.sites.external_adapter import ExternalAdapter
from app.sites.kariyer_adapter import KariyerAdapter
from app.sites.linkedin_adapter import LinkedInAdapter
from app.telegram.bot import build_application
from app.telegram.handlers import build_main_keyboard


async def _await_start_command(application, context: AgentContext, chat_id: str | None) -> None:
    question_id = str(uuid4())
    pending = context.create_pending_request(
        question_id=question_id,
        intent="START_COMMAND",
        question=(
            "Atlas is online. System integrity confirmed.\n\n"
            "Gmail layers are synced, and the Gemini core is standing by. I've indexed your latest "
            "communications and prepared your digital workspace. Your Chief of Staff is ready to orchestrate.\n\n"
            "How shall we direct our focus today?"
        ),
        category="startup",
    )
    await application.bot.send_message(
        chat_id=chat_id,
        text=(
            "Atlas is online. System integrity confirmed.\n\n"
            "Gmail layers are synced, and the Gemini core is standing by. I've indexed your latest "
            "communications and prepared your digital workspace. Your Chief of Staff is ready to orchestrate.\n\n"
            "How shall we direct our focus today?"
        ),
        reply_markup=build_main_keyboard(),
    )
    while True:
        reply = await context.wait_for_human_reply_or_stop()
        if reply is None:
            return
        if reply.question_id != pending.question_id:
            await context.human_reply_queue.put(reply)
            await asyncio.sleep(0)
            continue
        if reply.text.strip().lower() in {"job_search", "job search"}:
            context.resolve_pending_request(reply)
            return
        await application.bot.send_message(
            chat_id=chat_id,
            text="Please choose Job Search to continue.",
            reply_markup=build_main_keyboard(),
        )


async def _run_kariyer_flow(
    context: AgentContext,
    settings,
    adapter,
    llm_client,
    memory,
    application,
) -> None:
    search_query = _derive_kariyer_search_query(memory)
    await application.bot.send_message(
        chat_id=settings.telegram_chat_id,
        text="Starting job search on Kariyer.net using your CV profile.",
    )

    session = JobApplicationSession(
        job_metadata=JobMetadata(
            company="Kariyer.net",
            role="Job Application",
            location="Unknown",
            job_description="Kariyer application flow",
            extra={
                "search_query": search_query,
                "kariyernet_username": settings.kariyernet_username,
                "kariyernet_password": settings.kariyernet_password,
            },
        ),
        site_type=SiteType.KARIYER,
    )
    await run_session(
        session=session,
        adapter=adapter,
        agent_context=context,
        llm_client=llm_client,
        memory=memory,
        telegram_app=application,
        chat_id=settings.telegram_chat_id,
    )


def _derive_kariyer_search_query(memory) -> str:
    experience = memory.cv_profile.get("experience", [])
    if experience:
        role = experience[0].get("role")
        if role:
            return str(role)
    summary = str(memory.cv_profile.get("summary", "")).strip()
    if "python" in summary.lower():
        return "Python Developer"
    skills = memory.cv_profile.get("skills", [])
    if skills:
        return f"{skills[0]} Developer"
    return "Software Engineer"


async def _shutdown_playwright(context: AgentContext) -> None:
    browser = context.pc_state.get("browser_async")
    playwright = context.pc_state.get("playwright_async")
    if browser:
        try:
            await browser.close()
        except Exception:
            pass
    if playwright:
        try:
            await playwright.stop()
        except Exception:
            pass


async def main() -> None:
    init_logging()
    logger = logging.getLogger(__name__)
    context = AgentContext()
    settings = get_settings()
    db = get_database()
    model_factory = ModelFactory(
        api_key=settings.gemini_api_key,
        text_model=settings.gemini_model,
        image_model=settings.gemini_image_model,
        enable_google_search=settings.gemini_enable_google_search,
        enable_code_execution=settings.gemini_enable_code_execution,
    )
    llm_client = model_factory.get_text_client()
    image_client = model_factory.get_image_client()
    memory = load_memory_context()
    application = build_application()
    application.bot_data["agent_context"] = context
    application.bot_data["llm_client"] = llm_client
    application.bot_data["image_client"] = image_client
    application.bot_data["model_factory"] = model_factory
    application.bot_data["db"] = db
    application.bot_data["settings"] = settings

    logger.info("Agent starting")
    context.start()

    adapters = {
        SiteType.LINKEDIN: LinkedInAdapter(),
        SiteType.KARIYER: KariyerAdapter(),
        SiteType.EXTERNAL: ExternalAdapter(),
    }
    await application.initialize()
    await application.start()
    await application.updater.start_polling()

    try:
        await _await_start_command(application, context, settings.telegram_chat_id)
        if not context.stop_requested:
            await _run_kariyer_flow(
                context=context,
                settings=settings,
                adapter=adapters[SiteType.KARIYER],
                llm_client=llm_client,
                memory=memory,
                application=application,
            )
        while True:
            await context.stop_event.wait()
            if context.stop_reason == "telegram_job_search_stop":
                context.clear_stop()
                continue
            break
        context.complete()
        logger.info("Agent completed")
    finally:
        await _shutdown_playwright(context)
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
