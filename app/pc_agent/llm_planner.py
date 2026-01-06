from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Callable

from app.agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

ALLOWED_ACTIONS = {
    "open_app",
    "open_browser",
    "navigate",
    "click_text",
    "vision_click",
    "type_text",
    "vision_type",
    "keypress",
    "mouse_click",
    "mouse_move",
    "focus_window",
    "wait_for_window",
    "run_executable",
}


async def plan_pc_actions(
    llm_client: LLMClient,
    task: str,
    session_context: dict[str, Any] | None = None,
    previous_plan: list[dict[str, Any]] | None = None,
    revision_request: str | None = None,
    stream_handler: Callable[[str], Awaitable[str]] | None = None,
) -> list[dict[str, Any]] | None:
    if llm_client is None:
        return None
    prompt = _build_prompt(
        task,
        session_context=session_context,
        previous_plan=previous_plan,
        revision_request=revision_request,
    )
    try:
        if stream_handler:
            response_text = await stream_handler(prompt)
        else:
            response_text = await asyncio.to_thread(llm_client.generate_text, prompt)
    except Exception as exc:  # pragma: no cover - network/proxy errors
        logger.error("LLM planning failed: %s", exc)
        return None
    steps = _parse_steps(response_text)
    if not steps:
        return None
    return _ensure_message_steps(
        steps,
        task,
        session_context=session_context,
        revision_request=revision_request,
    )


def _build_prompt(
    task: str,
    session_context: dict[str, Any] | None = None,
    previous_plan: list[dict[str, Any]] | None = None,
    revision_request: str | None = None,
) -> str:
    context_block = ""
    if session_context:
        context_block = (
            "\nSystem Context (JSON):\n"
            f"{json.dumps(session_context, ensure_ascii=True, indent=2)}\n"
            "Instruction:\n"
            "- If the new request continues the existing context (active_app/last_action), "
            "modify the existing plan instead of creating a new one.\n"
        )
    if previous_plan:
        revision_line = f"Plan Revision Request: {revision_request}\n" if revision_request else ""
        context_block += (
            "\nExisting Plan (JSON):\n"
            f"{json.dumps({'steps': previous_plan}, ensure_ascii=True, indent=2)}\n"
            f"{revision_line}"
            "Instruction:\n"
            "- You are revising the existing plan. Update it minimally to satisfy the revision.\n"
        )

    return f"""
You are a Windows PC control planner. Convert the user task into a minimal JSON plan.

Rules:
- Output ONLY JSON. No markdown. No explanations.
- Use this schema:
  {{
    "steps": [
      {{
        "name": "open_app|open_browser|navigate|click_text|vision_click|type_text|vision_type|keypress|mouse_click|mouse_move|focus_window|wait_for_window|run_executable",
        "description": "short description",
        "risk": "LOW|MEDIUM|HIGH",
        "reversible": true|false,
        "payload": {{ ... }}
      }}
    ]
  }}
- Prefer "open_app" for launching apps by name (Start menu search).
- Use "run_executable" only when an explicit file path is provided by the user.
- If you need to click a button or label, use "click_text" with the visible text.
- If the visible text is unknown or the app is a native desktop UI, prefer "vision_click".
- For text input in native desktop apps, prefer "vision_type" or provide an "instruction".
- For messaging apps, always include steps to: focus the search bar, search the contact, click the matching chat result, focus the message input, then type and send (press_enter true).
- Keep steps short and minimal.
- After opening a desktop app, ensure the app window is ready (use "wait_for_window" or include window_title in "open_app").

Payload fields:
- open_app: {{ "name": "Steam", "window_title": ".*Steam.*", "timeout_seconds": 45 }}
- navigate: {{ "url": "https://example.com" }}
  - click_text: {{ "text": "Sign in", "instruction": "click the Sign in button" }}
  - vision_click: {{ "instruction": "click the search box in WhatsApp (top left)" }}
  - type_text: {{ "value": "hello", "selector": "<css>" }} or {{ "value": "hello", "instruction": "click the message input box" }}
  - vision_type: {{ "value": "hello", "instruction": "click the message input box", "press_enter": false }}
- keypress: {{ "key": "enter" }}
- mouse_click: {{ "x": 100, "y": 200 }}
- mouse_move: {{ "x": 100, "y": 200 }}
- focus_window: {{ "title": ".*Chrome.*" }}
- wait_for_window: {{ "title": ".*Spotify.*", "timeout_seconds": 30, "focus": true }}
- run_executable: {{ "path": "C:\\\\Program Files\\\\App\\\\app.exe" }}
{context_block}

User task: {task}
""".strip()


def _parse_steps(text: str) -> list[dict[str, Any]] | None:
    if not text:
        return None
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = ""
        for part in parts:
            if "{" in part:
                cleaned = part
                break
    cleaned = cleaned.strip()
    if cleaned:
        brace_index = cleaned.find("{")
        bracket_index = cleaned.find("[")
        indices = [i for i in (brace_index, bracket_index) if i != -1]
        if indices:
            start = min(indices)
            cleaned = cleaned[start:]
        if cleaned.startswith("{"):
            end = cleaned.rfind("}")
            if end != -1:
                cleaned = cleaned[: end + 1]
        elif cleaned.startswith("["):
            end = cleaned.rfind("]")
            if end != -1:
                cleaned = cleaned[: end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("LLM plan JSON parse failed: %s", cleaned[:200])
        return None
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    normalized: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        name = step.get("name")
        if name not in ALLOWED_ACTIONS:
            continue
        payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
        risk_value = str(step.get("risk", "MEDIUM")).upper()
        if risk_value not in {"LOW", "MEDIUM", "HIGH"}:
            risk_value = "MEDIUM"
        normalized.append(
            {
                "name": name,
                "description": step.get("description", name),
                "risk": risk_value,
                "reversible": bool(step.get("reversible", True)),
                "payload": payload,
            }
        )
    return normalized or None


def _ensure_message_steps(
    steps: list[dict[str, Any]],
    task: str,
    session_context: dict[str, Any] | None,
    revision_request: str | None,
) -> list[dict[str, Any]]:
    message_text = _extract_message_text(revision_request or task, session_context=session_context)
    if not message_text and revision_request:
        message_text = _extract_message_text(task, session_context=session_context)
    if not message_text:
        return steps
    if not _is_messaging_plan(
        steps,
        task,
        revision_request=revision_request,
        session_context=session_context,
    ):
        return steps

    has_message_step = False
    for step in steps:
        if step.get("name") in {"type_text", "vision_type"} and _is_message_step(step):
            has_message_step = True
            payload = step.setdefault("payload", {})
            if not str(payload.get("value", "")).strip():
                payload["value"] = message_text
            if payload.get("press_enter") is None:
                payload["press_enter"] = True

    if has_message_step:
        return steps

    send_index = _find_send_step_index(steps)
    insert_at = _find_message_insert_index(steps)
    message_step = {
        "name": "type_text",
        "description": "Type and send the message",
        "risk": "LOW",
        "reversible": True,
        "payload": {
            "value": message_text,
            "instruction": "click the message input box",
            "press_enter": True,
        },
    }
    if send_index is not None:
        steps.insert(send_index, message_step)
        return steps
    if insert_at is None:
        steps.append(message_step)
    else:
        steps.insert(insert_at + 1, message_step)
    return steps


def _is_messaging_plan(
    steps: list[dict[str, Any]],
    task: str,
    revision_request: str | None,
    session_context: dict[str, Any] | None,
) -> bool:
    plan_text = " ".join(_step_text(step) for step in steps)
    combined = f"{task} {revision_request or ''}".strip()
    if not (_contains_message_intent(combined) or _contains_message_intent(plan_text)):
        return False
    if _mentions_messaging_app(combined) or _mentions_messaging_app(plan_text):
        return True
    if session_context:
        active_app = str(session_context.get("active_app", "") or "")
        if _mentions_messaging_app(active_app):
            return True
    return False


def _contains_message_intent(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(
        token in normalized
        for token in ("message", "mesaj", "send", "yaz", "gonder", "yolla", "ileti")
    )


def _mentions_messaging_app(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(
        token in normalized
        for token in ("whatsapp", "telegram", "discord", "slack", "teams", "sms")
    )


def _find_message_insert_index(steps: list[dict[str, Any]]) -> int | None:
    for idx in range(len(steps) - 1, -1, -1):
        normalized = _normalize_text(_step_text(steps[idx]))
        if any(token in normalized for token in ("chat", "contact", "sohbet", "message", "mesaj", "search", "ara")):
            return idx
    return None


def _find_send_step_index(steps: list[dict[str, Any]]) -> int | None:
    for idx, step in enumerate(steps):
        normalized = _normalize_text(_step_text(step))
        if any(token in normalized for token in ("send", "gonder", "yolla", "ileti gonder")):
            return idx
    return None


def _is_message_step(step: dict[str, Any]) -> bool:
    normalized = _normalize_text(_step_text(step))
    return any(
        token in normalized
        for token in ("message", "mesaj", "input", "yazi", "send", "yaz", "gonder", "yolla")
    )


def _step_text(step: dict[str, Any]) -> str:
    payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
    return " ".join(
        str(value)
        for value in (
            step.get("description", ""),
            payload.get("instruction", ""),
            payload.get("text", ""),
        )
        if value
    )


def _extract_message_text(task: str, session_context: dict[str, Any] | None) -> str | None:
    message = _extract_quoted_message(task)
    if message:
        return message

    tokens = task.split()
    if not tokens:
        return _last_message_from_context(session_context)

    normalized_tokens = [_normalize_text(_strip_token(token)) for token in tokens]
    for idx in range(len(tokens) - 1):
        if normalized_tokens[idx] == "mesaj" and normalized_tokens[idx + 1] == "at":
            candidate = _clean_message_tokens(tokens[idx + 2 :])
            if candidate:
                return candidate
            candidate = _clean_message_tokens(tokens[:idx])
            if candidate:
                return candidate
        if normalized_tokens[idx] == "mesaj" and _is_send_token(normalized_tokens[idx + 1]):
            candidate = _clean_message_tokens(tokens[:idx])
            if candidate:
                return candidate

    send_index = None
    for idx, token in enumerate(normalized_tokens):
        if _is_send_token(token):
            send_index = idx
    if send_index is not None:
        candidate = _clean_message_tokens(tokens[:send_index])
        if candidate:
            return candidate
        candidate = _clean_message_tokens(tokens[send_index + 1 :])
        if candidate:
            return candidate

    return _last_message_from_context(session_context)


def _extract_quoted_message(text: str) -> str | None:
    for pattern in (r'"([^"]+)"', r"'([^']+)'"):
        match = re.search(pattern, text)
        if match:
            content = match.group(1).strip()
            if content:
                return content
    return None


def _last_message_from_context(session_context: dict[str, Any] | None) -> str | None:
    if not session_context:
        return None
    pending = session_context.get("pending_details", {})
    if isinstance(pending, dict):
        last_message = pending.get("last_message")
        if isinstance(last_message, str) and last_message.strip():
            return last_message.strip()
    return None


def _is_send_token(token: str) -> bool:
    verbs = ("yaz", "gonder", "yolla", "mesajla", "send", "text", "reply")
    return any(token == verb or token.startswith(verb) for verb in verbs)


def _clean_message_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    normalized = [_normalize_text(_strip_token(token)) for token in tokens]
    stopwords = {
        "ona",
        "onlara",
        "onun",
        "bana",
        "sana",
        "bunu",
        "su",
        "sunu",
        "lutfen",
        "please",
        "to",
        "for",
        "the",
        "mesaj",
        "message",
        "text",
    }
    if tokens and _looks_like_recipient_token(tokens[0]):
        normalized.pop(0)
        tokens.pop(0)
    while normalized and normalized[0] in stopwords:
        normalized.pop(0)
        tokens.pop(0)
    while normalized and normalized[-1] in stopwords:
        normalized.pop()
        tokens.pop()
    cleaned = [token for token in tokens if _normalize_text(_strip_token(token)) not in {"mesaj", "message", "text"}]
    return " ".join(cleaned).strip()


def _looks_like_recipient_token(token: str) -> bool:
    token = token.replace("\u2019", "'")
    if "'" not in token:
        return False
    parts = token.split("'")
    if len(parts) < 2:
        return False
    suffix = parts[-1].lower()
    return suffix in {"a", "e", "ya", "ye"}


def _strip_token(value: str) -> str:
    return re.sub(r"[^\w]+", "", value)


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
