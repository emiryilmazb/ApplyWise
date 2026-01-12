from __future__ import annotations

import logging
from pathlib import Path
import re
from uuid import uuid4

from app.agent.state import AgentContext
from app.agent.llm_client import LLMClient
from app.application.session import ApplicationStep, FailureReason, JobApplicationSession, StepType
from app.config import get_settings
from app.sites.base import StepExecutionResult

logger = logging.getLogger(__name__)

SCREENSHOT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "screenshots"

_APPLY_TEXTS = (
    "Apply",
    "Apply Now",
    "Submit Application",
)
_SUBMIT_TEXTS = ("Submit", "Save", "Continue", "Apply")
_LOGIN_LINK_TEXTS = ("Sign In", "Login")
_JOB_LISTINGS_TEXTS = (
    "Job Listings",
    "Job Listings and Application",
    "Job Search",
    "Kariyer.net Job Listings",
)


class KariyerAdapter:
    site_name = "kariyer"

    async def get_application_steps(
        self,
        session: JobApplicationSession,
        page,
        agent_context: AgentContext | None = None,
        llm_client: LLMClient | None = None,
        telegram_app=None,
        chat_id: str | None = None,
    ) -> list[ApplicationStep]:
        if page is None:
            raise RuntimeError("Playwright page is required")
        url = session.job_metadata.extra.get("url") if session.job_metadata.extra else None
        search_query = session.job_metadata.extra.get("search_query") if session.job_metadata.extra else None
        username = session.job_metadata.extra.get("kariyernet_username") if session.job_metadata.extra else None
        password = session.job_metadata.extra.get("kariyernet_password") if session.job_metadata.extra else None
        if url:
            await page.goto(url, wait_until="domcontentloaded")
            await _handle_security_check(page, llm_client, agent_context, telegram_app, chat_id)
            await _ensure_logged_in(page, username, password)
        else:
            await _prepare_home_and_login(page, llm_client, agent_context, telegram_app, chat_id, username, password)
            await _go_to_job_listings(page)
            await _search_and_open_listing(page, search_query, telegram_app, chat_id)
        await _click_apply_button(page)
        await page.wait_for_timeout(1000)

        fields = page.locator("input, textarea, select")
        count = await fields.count()
        steps: list[ApplicationStep] = []

        for index in range(count):
            field = fields.nth(index)
            if not await field.is_visible():
                continue

            tag = await field.evaluate("el => el.tagName.toLowerCase()")
            input_type = await field.get_attribute("type") if tag == "input" else None
            if input_type in {"hidden", "submit", "button", "image"}:
                continue

            selector = await _build_selector(field, tag)
            label = await _extract_label(page, field)
            placeholder = await field.get_attribute("placeholder")
            name = await field.get_attribute("name")

            question = label or placeholder or name or f"Field {len(steps) + 1}"
            steps.append(
                ApplicationStep(
                    step_id=str(uuid4()),
                    step_type=StepType.QUESTION,
                    payload={
                        "question": question.strip(),
                        "selector": selector,
                        "index": index,
                        "field_type": tag,
                        "input_type": input_type,
                    },
                )
            )

        submit_selector = await _find_submit_selector(page)
        if steps and submit_selector:
            steps[-1].payload["submit_selector"] = submit_selector
            steps[-1].payload["submit_answer"] = True

        if not steps:
            return [
                ApplicationStep(
                    step_id=str(uuid4()),
                    step_type=StepType.FORM_FILL,
                    payload={"section": "application_form"},
                )
            ]

        return steps

    async def execute_step(
        self,
        session: JobApplicationSession,
        step: ApplicationStep,
        answer: str | None = None,
        page=None,
        agent_context: AgentContext | None = None,
        llm_client: LLMClient | None = None,
        telegram_app=None,
        chat_id: str | None = None,
    ) -> StepExecutionResult:
        if page is None:
            return StepExecutionResult(
                success=False,
                failure_reason=FailureReason.UNEXPECTED_STRUCTURE,
                details="Playwright page is missing",
            )

        logger.info("Kariyer step executed: %s", step.step_type.value)
        if step.step_type != StepType.QUESTION:
            return StepExecutionResult(success=True, submitted=False)

        if answer is None or answer == "":
            return StepExecutionResult(
                success=False,
                failure_reason=FailureReason.UNEXPECTED_STRUCTURE,
                details="Missing answer",
            )

        selector = step.payload.get("selector")
        index = step.payload.get("index", 0)
        field_type = step.payload.get("field_type")
        input_type = step.payload.get("input_type")

        if selector:
            locator = page.locator(selector)
        else:
            locator = page.locator("input, textarea, select").nth(index)

        if field_type == "select":
            await locator.select_option(label=answer)
        elif input_type in {"checkbox", "radio"}:
            if _is_truthy(answer):
                await locator.check()
            else:
                await locator.uncheck()
        elif input_type == "file":
            await locator.set_input_files(answer)
        else:
            await locator.fill(answer)

        submit_selector = step.payload.get("submit_selector")
        if submit_selector:
            await page.locator(submit_selector).first.click()

        return StepExecutionResult(success=True, submitted=bool(step.payload.get("submit_answer")))


async def _click_apply_button(page) -> None:
    for text in _APPLY_TEXTS:
        button = page.get_by_role("button", name=text)
        if await button.count() > 0 and await button.first.is_visible():
            await button.first.click()
            return
    link = page.get_by_role("link", name=_APPLY_TEXTS[0])
    if await link.count() > 0 and await link.first.is_visible():
        await link.first.click()


async def _search_and_open_listing(
    page,
    search_query: str | None,
    telegram_app=None,
    chat_id: str | None = None,
) -> None:
    if not search_query:
        raise RuntimeError("Kariyer search query is missing")

    await page.wait_for_load_state("domcontentloaded")
    await _dismiss_overlays(page)
    search_box = await _find_search_box(page)
    if search_box is None:
        await _notify_search_box_missing(page, telegram_app, chat_id)
        raise RuntimeError("Search box not found on Kariyer.net")

    await search_box.fill(search_query)
    await search_box.press("Enter")
    await page.wait_for_load_state("domcontentloaded")
    await page.wait_for_timeout(1000)

    listing = page.locator("a[href*='is-ilani']").first
    if await listing.count() == 0:
        raise RuntimeError("No job listings found for query")
    await listing.click()
    await page.wait_for_load_state("domcontentloaded")


async def _find_submit_selector(page) -> str | None:
    for text in _SUBMIT_TEXTS:
        button = page.get_by_role("button", name=text)
        if await button.count() > 0 and await button.first.is_visible():
            return f"button:has-text('{text}')"
    submit_input = page.locator("button[type='submit'], input[type='submit']")
    if await submit_input.count() > 0:
        return "button[type='submit'], input[type='submit']"
    return None


async def _extract_label(page, field) -> str | None:
    field_id = await field.get_attribute("id")
    if field_id:
        label = page.locator(f"label[for='{field_id}']")
        if await label.count() > 0:
            text = await label.first.inner_text()
            return text.strip()
    aria = await field.get_attribute("aria-label")
    return aria.strip() if aria else None


async def _build_selector(field, tag: str) -> str | None:
    field_id = await field.get_attribute("id")
    if field_id:
        return f"#{field_id}"
    name = await field.get_attribute("name")
    if name:
        return f"{tag}[name='{name}']"
    data_test = await field.get_attribute("data-testid")
    if data_test:
        return f"{tag}[data-testid='{data_test}']"
    return None


async def _find_search_box(page):
    generic = page.locator(
        "input[type='search'], input[name*='query'], input[name*='keyword'], "
        "input[name*='position'], input[placeholder*='Position'], "
        "input[placeholder*='Job'], input[placeholder*='Search']"
    )
    if await generic.count() > 0 and await generic.first.is_visible():
        return generic.first

    candidates = [
        page.get_by_placeholder("Position"),
        page.get_by_placeholder("Keyword"),
        page.get_by_placeholder("Job"),
        page.get_by_placeholder("Job search"),
        page.get_by_placeholder("Search"),
        page.get_by_role("textbox", name="Position"),
        page.get_by_role("textbox", name="Keyword"),
    ]
    for candidate in candidates:
        if await candidate.count() > 0 and await candidate.first.is_visible():
            return candidate.first
    textbox = page.get_by_role("textbox")
    if await textbox.count() > 0 and await textbox.first.is_visible():
        return textbox.first
    return None


async def _notify_search_box_missing(page, telegram_app, chat_id: str | None) -> None:
    if telegram_app is None or chat_id is None:
        return
    await _send_screenshot_or_message(
        page,
        telegram_app,
        chat_id,
        caption="Search box not found on Kariyer.net. Please advise.",
        fallback_text="Search box not found on Kariyer.net. Please advise.",
    )


async def _dismiss_overlays(page) -> None:
    candidates = ["Kabul", "Onayla", "Accept", "Accept All", "Tamam"]
    for text in candidates:
        button = page.get_by_role("button", name=text)
        if await button.count() > 0 and await button.first.is_visible():
            await button.first.click()
            return


async def _prepare_home_and_login(
    page,
    llm_client,
    agent_context,
    telegram_app,
    chat_id,
    username: str | None,
    password: str | None,
) -> None:
    await page.goto("https://www.kariyer.net", wait_until="domcontentloaded")
    await _handle_security_check(page, llm_client, agent_context, telegram_app, chat_id)
    await _dismiss_overlays(page)
    await _ensure_logged_in(page, username, password)


async def _handle_security_check(page, llm_client, agent_context, telegram_app, chat_id) -> None:
    if await page.get_by_text("Security verification", exact=False).count() == 0:
        return

    button = await _find_hold_button(page)
    if button is not None:
        await _hold_button(page, button)
        return

    decision = await _ask_gemini_for_security_action(page, llm_client)
    if decision["action"] == "HOLD":
        button = await _find_hold_button(page, decision.get("button_text"))
        if button is not None:
            await _hold_button(page, button)
            return

    if agent_context is None or telegram_app is None or chat_id is None:
        raise RuntimeError("Security check requires human input but Telegram context is missing")

    if decision.get("send_screenshot"):
        await _send_screenshot_or_message(
            page,
            telegram_app,
            chat_id,
            caption="Security verification detected. What should I do?",
            fallback_text="Security verification detected. What should I do?",
        )
    else:
        await telegram_app.bot.send_message(
            chat_id=chat_id,
            text="Security verification detected. What should I do?",
        )

    question_id = str(uuid4())
    pending = agent_context.create_pending_request(
        question_id=question_id,
        intent="SECURITY_CHECK",
        question="Security verification requires a hold action. Proceed?",
        category="security",
    )
    while True:
        reply = await agent_context.wait_for_human_reply_or_stop()
        if reply is None:
            raise RuntimeError("Stopped while waiting for security confirmation")
        if reply.question_id != pending.question_id:
            continue
        agent_context.resolve_pending_request(reply)
        if reply.text.strip().upper() == "HOLD":
            button = await _find_hold_button(page)
            if button is None:
                raise RuntimeError("Hold button not found during security confirmation")
            await _hold_button(page, button)
        break


async def _find_hold_button(page, text: str | None = None):
    if text:
        locator = page.get_by_text(text, exact=False)
        if await locator.count() > 0 and await locator.first.is_visible():
            return locator.first

    pattern = re.compile(r"press\\s+and\\s+hold", re.IGNORECASE)
    locator = page.get_by_text(pattern)
    if await locator.count() > 0 and await locator.first.is_visible():
        return locator.first

    locator = page.get_by_role("button", name=pattern)
    if await locator.count() > 0 and await locator.first.is_visible():
        return locator.first

    return None


async def _hold_button(page, button) -> None:
    box = await button.bounding_box()
    if not box:
        raise RuntimeError("Hold button bounding box missing")
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    await page.mouse.move(x, y)
    await page.mouse.down()
    await page.wait_for_timeout(10000)
    await page.mouse.up()


async def _ask_gemini_for_security_action(page, llm_client) -> dict:
    if llm_client is None:
        return {"action": "ASK_HUMAN", "send_screenshot": True}

    prompt = (
        "A Turkish website shows a security verification prompt: "
        "'A security verification is required... Press and hold the button below to continue.' "
        "Decide the next action. Reply in this format:\n"
        "ACTION=HOLD|ASK_HUMAN\n"
        "SEND_SCREENSHOT=YES|NO\n"
        "BUTTON_TEXT=<text or empty>\n"
        f"Current URL: {page.url}\n"
    )
    response = llm_client.generate_text(prompt)
    action_match = re.search(r"ACTION=(HOLD|ASK_HUMAN)", response, re.IGNORECASE)
    screenshot_match = re.search(r"SEND_SCREENSHOT=(YES|NO)", response, re.IGNORECASE)
    button_match = re.search(r"BUTTON_TEXT=(.*)", response)
    return {
        "action": (action_match.group(1).upper() if action_match else "ASK_HUMAN"),
        "send_screenshot": (screenshot_match.group(1).upper() == "YES") if screenshot_match else True,
        "button_text": button_match.group(1).strip() if button_match else None,
    }


async def _ensure_logged_in(page, username: str | None, password: str | None) -> None:
    login_link = await _find_login_link(page)
    if login_link is None:
        return
    if not username or not password:
        raise RuntimeError("Kariyer.net credentials are missing")

    await login_link.click()
    await page.wait_for_load_state("domcontentloaded")

    email_input = page.locator(
        "input[type='email'], input[name*='email'], input[placeholder*='Email']"
    ).first
    password_input = page.locator(
        "input[type='password'], input[name*='password'], input[placeholder*='Password']"
    ).first

    if await email_input.count() == 0 or await password_input.count() == 0:
        raise RuntimeError("Login form fields not found on Kariyer.net")

    await email_input.fill(username)
    await password_input.fill(password)

    submit = page.get_by_role("button", name="Sign In")
    if await submit.count() > 0:
        await submit.first.click()
    else:
        await password_input.press("Enter")

    await page.wait_for_load_state("domcontentloaded")


async def _find_login_link(page):
    for text in _LOGIN_LINK_TEXTS:
        link = page.get_by_role("link", name=text)
        if await link.count() > 0 and await link.first.is_visible():
            return link.first
    return None


async def _ask_human_next_action(
    page,
    agent_context,
    telegram_app,
    chat_id,
    send_screenshot: bool,
) -> dict:
    if agent_context is None or telegram_app is None or chat_id is None:
        raise RuntimeError("Telegram context is required for human decision")

    if send_screenshot:
        await _send_screenshot_or_message(
            page,
            telegram_app,
            chat_id,
            caption="Login or search next? Reply LOGIN or SEARCH.",
            fallback_text="Login or search next? Reply LOGIN or SEARCH.",
        )
    else:
        await telegram_app.bot.send_message(
            chat_id=chat_id,
            text="Login or search next? Reply LOGIN or SEARCH.",
        )

    question_id = str(uuid4())
    pending = agent_context.create_pending_request(
        question_id=question_id,
        intent="KARIYER_NEXT_ACTION",
        question="Choose next action: LOGIN or SEARCH",
        category="navigation",
    )
    while True:
        reply = await agent_context.wait_for_human_reply_or_stop()
        if reply is None:
            raise RuntimeError("Stopped while waiting for next action")
        if reply.question_id != pending.question_id:
            continue
        agent_context.resolve_pending_request(reply)
        normalized = reply.text.strip().upper()
        if normalized in {"LOGIN", "SEARCH"}:
            return {"action": normalized, "send_screenshot": send_screenshot}


async def _go_to_job_listings(page) -> None:
    for text in _JOB_LISTINGS_TEXTS:
        link = page.get_by_role("link", name=text)
        if await link.count() > 0 and await link.first.is_visible():
            await link.first.click()
            await page.wait_for_load_state("domcontentloaded")
            return

    if "is-ilanlari" not in page.url:
        await page.goto("https://www.kariyer.net/is-ilanlari", wait_until="domcontentloaded")


def _resolve_runtime_settings(telegram_app):
    if telegram_app is None:
        return get_settings()
    bot_data = getattr(telegram_app, "bot_data", None)
    if not bot_data:
        return get_settings()
    return bot_data.get("settings") or get_settings()


async def _capture_playwright_screenshot(page) -> Path | None:
    if page is None:
        return None
    try:
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = SCREENSHOT_DIR / f"page_{uuid4().hex}.png"
        await page.screenshot(path=str(path), full_page=True)
        return path
    except Exception as exc:
        logger.warning("Playwright screenshot failed: %s", exc)
        return None


def _capture_desktop_screenshot() -> str | None:
    try:
        from app.pc_agent.vision import capture_screen
    except Exception as exc:
        logger.warning("Desktop screenshot unavailable: %s", exc)
        return None
    try:
        screenshot = capture_screen()
        return screenshot.path
    except Exception as exc:
        logger.warning("Desktop screenshot failed: %s", exc)
        return None


async def _send_screenshot_or_message(
    page,
    telegram_app,
    chat_id: str | None,
    caption: str,
    fallback_text: str,
) -> None:
    if telegram_app is None or chat_id is None:
        return
    settings = _resolve_runtime_settings(telegram_app)
    if not bool(getattr(settings, "screenshot_enabled", True)):
        await telegram_app.bot.send_message(chat_id=chat_id, text=fallback_text)
        return
    screenshot_path = await _capture_playwright_screenshot(page)
    if screenshot_path is None:
        screenshot_path = _capture_desktop_screenshot()
    if screenshot_path is None:
        await telegram_app.bot.send_message(chat_id=chat_id, text=fallback_text)
        return
    with open(screenshot_path, "rb") as handle:
        await telegram_app.bot.send_photo(
            chat_id=chat_id,
            photo=handle,
            caption=caption,
        )
