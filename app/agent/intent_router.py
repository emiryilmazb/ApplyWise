from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import logging
import random
import time
from typing import Any

from app.agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

_ROUTER_RETRY_ATTEMPTS = 3
_ROUTER_RETRY_BASE_DELAY = 0.2
_ROUTER_RETRY_MAX_DELAY = 2.0


class RouterIntent(Enum):
    CHAT = "CHAT"
    COMPUTER_USE = "COMPUTER_USE"
    IMAGE_GEN = "IMAGE_GEN"
    IMAGE_EDIT = "IMAGE_EDIT"
    GMAIL_INBOX = "GMAIL_INBOX"
    GMAIL_SUMMARY = "GMAIL_SUMMARY"
    GMAIL_SEARCH = "GMAIL_SEARCH"
    GMAIL_DRAFT = "GMAIL_DRAFT"
    GMAIL_SEND = "GMAIL_SEND"
    GMAIL_QUESTION = "GMAIL_QUESTION"


@dataclass(frozen=True)
class RouterDecision:
    intent: RouterIntent
    reason: str | None = None


_IMAGE_GEN_TOKENS = (
    "image",
    "resim",
    "gorsel",
    "draw",
    "generate",
    "create",
    "ciz",
    "uret",
    "tasarla",
    "illustrate",
    "render",
)
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
_COMPUTER_USE_TOKENS = (
    "pc",
    "bilgisayar",
    "site",
    "siteye",
    "web",
    "open",
    "ac",
    "launch",
    "click",
    "tikla",
    "type",
    "yaz",
    "send",
    "message",
    "browser",
    "chrome",
    "spotify",
    "whatsapp",
    "steam",
    "masaustu",
    "desktop",
    "wallpaper",
    "linkedin",
    "ilan",
    "basvur",
    "basvuru",
    "apply",
    "application",
    "form",
    "upload",
    "cv",
    "resume",
)
_IMAGE_PRONOUN_TOKENS = (
    "that",
    "this",
    "it",
    "image",
    "photo",
    "picture",
    "bunu",
    "sunu",
)
_GMAIL_TOKENS = (
    "gmail",
    "mail",
    "email",
    "eposta",
    "inbox",
    "inbox mail",
    "gelen kutu",
    "gelen kutusu",
    "gelen kutum",
    "mail kutusu",
    "mail kutum",
    "mailler",
    "maillerim",
)
_GMAIL_SUMMARY_TOKENS = (
    "summarize",
    "summary",
    "brief summary",
    "ozet",
    "kisa ozet",
    "ozetle",
    "ozet cikar",
    "mail ozet",
)
_GMAIL_SEARCH_TOKENS = ("ara", "search", "bul", "find", "tara", "listele")
_GMAIL_DRAFT_TOKENS = ("taslak", "draft", "yaz", "olustur", "hazirla", "mail yaz", "eposta yaz")
_GMAIL_SEND_TOKENS = (
    "send",
    "deliver",
    "email",
    "mail",
    "gonder",
    "yolla",
    "mail at",
    "eposta gonder",
    "mail gonder",
)
_GMAIL_QUESTION_TOKENS = (
    "otp",
    "code",
    "how many",
    "count",
    "number of",
    "onay kodu",
    "dogrulama kodu",
    "kac tane",
    "sayi",
    "ne kadar",
)


def route_intent(
    llm_client: LLMClient | None,
    message_text: str,
    *,
    has_image: bool,
    session_context: dict[str, Any] | None = None,
) -> RouterDecision:
    if not message_text:
        message_text = ""
    if llm_client is None:
        return _fallback_intent(message_text, has_image=has_image, session_context=session_context)
    prompt = _build_prompt(message_text, has_image=has_image,
                           session_context=session_context)
    last_error: Exception | None = None
    for attempt in range(1, _ROUTER_RETRY_ATTEMPTS + 1):
        try:
            response_text = llm_client.generate_text(prompt)
        except Exception as exc:  # pragma: no cover - network/proxy errors
            last_error = exc
            if attempt < _ROUTER_RETRY_ATTEMPTS:
                logger.warning("Intent router failed (attempt %s/%s): %s",
                               attempt, _ROUTER_RETRY_ATTEMPTS, exc)
                _sleep_with_backoff(attempt)
                continue
            logger.warning("Intent router failed after retries: %s", exc)
            return _fallback_intent(message_text, has_image=has_image, session_context=session_context)
        parsed = _parse_router_response(response_text)
        if parsed is not None:
            return parsed
        last_error = None
        if attempt < _ROUTER_RETRY_ATTEMPTS:
            logger.warning(
                "Intent router parse failed (attempt %s/%s).", attempt, _ROUTER_RETRY_ATTEMPTS)
            _sleep_with_backoff(attempt)
            continue
        logger.warning("Intent router parse failed after retries.")
        return _fallback_intent(message_text, has_image=has_image, session_context=session_context)
    if last_error:
        logger.warning("Intent router failed: %s", last_error)
    return _fallback_intent(message_text, has_image=has_image, session_context=session_context)


def _build_prompt(
    message_text: str,
    *,
    has_image: bool,
    session_context: dict[str, Any] | None,
) -> str:
    context_block = json.dumps(
        session_context or {}, ensure_ascii=True, indent=2)
    return f"""
You are a Telegram intent router for a multimodal assistant.
Choose exactly one intent:
- CHAT: general conversation or questions.
- COMPUTER_USE: requests to operate the computer, apps, or browser.
- IMAGE_GEN: requests to generate a new image from text.
- IMAGE_EDIT: requests to edit, transform, or analyze an image (image-to-image or vision).
- GMAIL_INBOX: requests to check inbox or list recent emails.
- GMAIL_SUMMARY: requests to summarize recent emails.
- GMAIL_SEARCH: requests to search emails with a query.
- GMAIL_DRAFT: requests to draft an email (may ask to send after approval).
- GMAIL_SEND: explicit request to send an email (requires approval).
- GMAIL_QUESTION: questions that require reading emails (counts, codes, status).

Guidance:
- If an image is attached, and the user asks to change or analyze it, choose IMAGE_EDIT.
- If the user wants a new image from a description, choose IMAGE_GEN.
- If the user wants to take an action on the computer (including setting wallpaper), choose COMPUTER_USE.
- If the user asks for time-sensitive factual info (exchange rates, news, weather), choose CHAT. The assistant can use built-in Google Search tools, so do not use COMPUTER_USE just to browse.
- If the user asks for charts, graphs, or plots, choose CHAT. The assistant can generate charts without computer control.
- If the user refers to "this/that" and context has a last_image, assume the reference is the last image.

Context JSON:
{context_block}

Has image: {str(has_image).lower()}
User message: {message_text}

Return ONLY JSON with this shape:
{{ "intent": "CHAT|COMPUTER_USE|IMAGE_GEN|IMAGE_EDIT|GMAIL_INBOX|GMAIL_SUMMARY|GMAIL_SEARCH|GMAIL_DRAFT|GMAIL_SEND|GMAIL_QUESTION", "reason": "<short reason>" }}
""".strip()


def _parse_router_response(text: str) -> RouterDecision | None:
    if not text:
        return None
    cleaned = _extract_json(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Router JSON parse failed: %s", cleaned[:200])
        return None
    intent_value = str(data.get("intent", "")).strip().upper()
    if intent_value not in {intent.value for intent in RouterIntent}:
        return None
    reason = str(data.get("reason", "")).strip() or None
    return RouterDecision(intent=RouterIntent(intent_value), reason=reason)


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            if "{" in part:
                cleaned = part
                break
    cleaned = cleaned.strip()
    brace_index = cleaned.find("{")
    if brace_index != -1:
        cleaned = cleaned[brace_index:]
    end_index = cleaned.rfind("}")
    if end_index != -1:
        cleaned = cleaned[: end_index + 1]
    return cleaned


def _fallback_intent(
    message_text: str,
    *,
    has_image: bool,
    session_context: dict[str, Any] | None,
) -> RouterDecision:
    normalized = _normalize_text(message_text)
    if has_image:
        return RouterDecision(intent=RouterIntent.IMAGE_EDIT, reason="fallback:image")
    if _contains_any(normalized, _IMAGE_EDIT_TOKENS):
        return RouterDecision(intent=RouterIntent.IMAGE_EDIT, reason="fallback:edit")
    if _contains_any(normalized, _IMAGE_GEN_TOKENS):
        return RouterDecision(intent=RouterIntent.IMAGE_GEN, reason="fallback:gen")
    if _contains_any(normalized, _GMAIL_TOKENS):
        return _fallback_gmail_intent(normalized)
    if _contains_any(normalized, _COMPUTER_USE_TOKENS):
        return RouterDecision(intent=RouterIntent.COMPUTER_USE, reason="fallback:pc")
    if _has_recent_images(session_context) and _contains_any(normalized, _IMAGE_PRONOUN_TOKENS):
        if _contains_any(normalized, ("masaustu", "desktop", "wallpaper")):
            return RouterDecision(intent=RouterIntent.COMPUTER_USE, reason="fallback:pronoun_pc")
        return RouterDecision(intent=RouterIntent.IMAGE_EDIT, reason="fallback:pronoun_edit")
    return RouterDecision(intent=RouterIntent.CHAT, reason="fallback:chat")


def _fallback_gmail_intent(normalized: str) -> RouterDecision:
    if _contains_any(normalized, _GMAIL_SEND_TOKENS):
        return RouterDecision(intent=RouterIntent.GMAIL_SEND, reason="fallback:gmail_send")
    if _contains_any(normalized, _GMAIL_SUMMARY_TOKENS):
        return RouterDecision(intent=RouterIntent.GMAIL_SUMMARY, reason="fallback:gmail_summary")
    if _contains_any(normalized, _GMAIL_SEARCH_TOKENS):
        return RouterDecision(intent=RouterIntent.GMAIL_SEARCH, reason="fallback:gmail_search")
    if _contains_any(normalized, _GMAIL_DRAFT_TOKENS):
        return RouterDecision(intent=RouterIntent.GMAIL_DRAFT, reason="fallback:gmail_draft")
    if _contains_any(normalized, _GMAIL_QUESTION_TOKENS):
        return RouterDecision(intent=RouterIntent.GMAIL_QUESTION, reason="fallback:gmail_question")
    return RouterDecision(intent=RouterIntent.GMAIL_INBOX, reason="fallback:gmail_inbox")


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _has_recent_images(session_context: dict[str, Any] | None) -> bool:
    if not session_context:
        return False
    if session_context.get("last_image_path"):
        return True
    recent = session_context.get("recent_images")
    return isinstance(recent, list) and bool(recent)


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


def _sleep_with_backoff(attempt: int) -> None:
    delay = min(_ROUTER_RETRY_MAX_DELAY,
                _ROUTER_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    delay += random.uniform(0, _ROUTER_RETRY_BASE_DELAY / 2)
    time.sleep(delay)
