from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import threading
from typing import Any

from app.storage.database import get_database

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_LOADED = False
_SESSIONS: dict[str, "SessionState"] = {}

_SHORT_TERM_LIMIT = 5
_RECENT_IMAGE_LIMIT = 10
_RESET_TOKENS = ("reset", "cancel", "stop", "iptal", "vazgec")

_APP_ALIASES: dict[str, tuple[str, ...]] = {
    "whatsapp": ("whatsapp", "whats app"),
    "spotify": ("spotify",),
    "browser": ("browser", "chrome", "edge", "firefox", "brave", "opera"),
    "telegram": ("telegram",),
    "discord": ("discord",),
    "slack": ("slack",),
    "teams": ("teams", "microsoft teams"),
    "zoom": ("zoom",),
    "outlook": ("outlook",),
    "gmail": ("gmail",),
    "steam": ("steam",),
    "notepad": ("notepad", "not defteri", "notdefteri"),
    "word": ("word", "microsoft word", "ms word"),
    "excel": ("excel", "microsoft excel", "ms excel"),
    "powerpoint": ("powerpoint", "ppt", "microsoft powerpoint"),
}


@dataclass
class SessionState:
    user_id: str
    active_app: str | None = None
    last_action: str | None = None
    pending_details: dict[str, Any] = field(default_factory=dict)
    short_term_memory: list[str] = field(default_factory=list)
    last_summary: str | None = None
    last_image_path: str | None = None
    last_image_prompt: str | None = None
    last_image_summary: str | None = None
    last_image_source: str | None = None
    recent_images: list[dict[str, Any]] = field(default_factory=list)


def get_session_context(user_id: str | None) -> dict[str, Any] | None:
    if not user_id:
        return None
    with _LOCK:
        session = _get_or_create_session(user_id)
    context = {
        "active_app": session.active_app,
        "last_action": session.last_action,
        "pending_details": session.pending_details,
        "short_term_memory": session.short_term_memory,
        "last_summary": session.last_summary,
        "last_image_path": session.last_image_path,
        "last_image_prompt": session.last_image_prompt,
        "last_image_summary": session.last_image_summary,
        "last_image_source": session.last_image_source,
        "recent_images": session.recent_images,
    }
    if not any(context.values()):
        return None
    return context


def record_user_message(user_id: str | None, message: str) -> None:
    if not user_id or not message:
        return
    with _LOCK:
        session = _get_or_create_session(user_id)
        session.short_term_memory.append(message)
        session.short_term_memory = session.short_term_memory[-_SHORT_TERM_LIMIT:]
        _SESSIONS[user_id] = session
        _save_sessions()


def clear_session(user_id: str | None) -> None:
    if not user_id:
        return
    with _LOCK:
        _load_sessions()
        _SESSIONS.pop(user_id, None)
        try:
            db = get_database()
            db.delete_session(user_id)
        except Exception as exc:
            logger.warning("Session store delete failed: %s", exc)
        _save_sessions()


def should_reset_context(user_id: str | None, message: str) -> bool:
    if not user_id or not message:
        return False
    normalized = _normalize_text(message)
    if any(token in normalized for token in _RESET_TOKENS):
        return True
    with _LOCK:
        _load_sessions()
        session = _SESSIONS.get(user_id)
    if session is None or not session.active_app:
        return False
    mentioned = _detect_app_mentions(normalized)
    if not mentioned:
        return False
    active = _canonicalize_app(session.active_app)
    if active is None:
        return False
    return active not in mentioned


def record_task_completion(
    user_id: str | None,
    task: str,
    actions: list[Any],
) -> None:
    if not user_id:
        return
    with _LOCK:
        session = _get_or_create_session(user_id)
        extracted = _extract_action_details(actions)
        if extracted.get("active_app"):
            new_app = extracted["active_app"]
            if session.active_app and _canonicalize_app(session.active_app) != _canonicalize_app(new_app):
                session.pending_details = {}
            session.active_app = new_app
        if extracted.get("last_action"):
            session.last_action = extracted["last_action"]
        if extracted.get("pending_details"):
            session.pending_details.update(extracted["pending_details"])
        summary = _summarize_task(task, actions)
        if summary:
            session.last_summary = summary
        _SESSIONS[user_id] = session
        _save_sessions()


def record_handoff_detail(user_id: str | None, key: str, value: Any) -> None:
    if not user_id or not key or value is None:
        return
    with _LOCK:
        session = _get_or_create_session(user_id)
        session.pending_details[str(key)] = value
        _SESSIONS[user_id] = session
        _save_sessions()


def record_image_result(
    user_id: str | None,
    image_path: str,
    prompt: str | None = None,
    source: str | None = None,
) -> None:
    if not user_id or not image_path:
        return
    with _LOCK:
        session = _get_or_create_session(user_id)
        summary = _build_image_summary(prompt, source)
        session.last_image_path = image_path
        session.last_image_prompt = prompt
        session.last_image_summary = summary
        session.last_image_source = source
        session.recent_images.append(
            {
                "path": image_path,
                "prompt": prompt,
                "summary": summary,
                "source": source,
            }
        )
        session.recent_images = _trim_recent_images(session.recent_images)
        _SESSIONS[user_id] = session
        _save_sessions()


def _summarize_task(task: str, actions: list[Any]) -> str:
    if not task:
        return ""
    if not actions:
        return task
    previews = [str(action.description) for action in actions[:3] if getattr(action, "description", None)]
    summary = "; ".join(previews)
    if len(actions) > 3:
        summary = f"{summary}; +{len(actions) - 3} more" if summary else f"{len(actions)} steps"
    return f"{task} | {summary}" if summary else task


def _extract_action_details(actions: list[Any]) -> dict[str, Any]:
    active_app = None
    last_action = None
    pending_details: dict[str, Any] = {}
    for action in actions:
        name = getattr(action, "name", None)
        payload = getattr(action, "payload", {}) or {}
        description = str(getattr(action, "description", "") or "")
        if name:
            last_action = str(name)
        if name == "open_app":
            app_name = payload.get("name")
            if app_name:
                active_app = str(app_name)
        elif name in {"open_browser", "navigate"}:
            active_app = "browser"
        elif name in {"focus_window", "wait_for_window"}:
            title = payload.get("title")
            cleaned = _clean_window_title(str(title)) if title else ""
            if cleaned:
                active_app = cleaned
        _extract_pending_details_from_action(name, description, payload, pending_details)
    return {
        "active_app": active_app,
        "last_action": last_action,
        "pending_details": pending_details,
    }


def _extract_pending_details_from_action(
    name: str | None,
    description: str,
    payload: dict[str, Any],
    pending_details: dict[str, Any],
) -> None:
    if not name:
        return
    description_norm = _normalize_text(description)
    instruction_norm = _normalize_text(str(payload.get("instruction", "")))
    text_value = str(payload.get("text", "")).strip()
    typed_value = str(payload.get("value", "")).strip()
    is_search = "search" in description_norm or "search" in instruction_norm or "ara" in description_norm
    is_message = any(
        token in description_norm
        for token in ("message", "send", "reply", "mesaj", "yaz", "gonder", "yolla", "ileti")
    ) or any(token in instruction_norm for token in ("message", "send", "reply", "mesaj", "yaz", "gonder", "yolla", "ileti"))
    is_contact = (
        "contact" in description_norm
        or "chat" in description_norm
        or "message" in description_norm
        or "chat" in description_norm
        or "sohbet" in description_norm
    )
    if name == "click_text" and is_contact and text_value:
        pending_details["last_contact"] = text_value
    if name in {"type_text", "vision_type"} and is_search:
        if typed_value and len(typed_value.split()) <= 4:
            pending_details["last_contact"] = typed_value
        elif text_value and len(text_value.split()) <= 4:
            pending_details["last_contact"] = text_value
    if is_search:
        if typed_value:
            pending_details.setdefault("last_search", typed_value)
        elif text_value:
            pending_details.setdefault("last_search", text_value)
    if is_message and typed_value:
        pending_details["last_message"] = typed_value


def _clean_window_title(title: str) -> str:
    if not title:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _canonicalize_app(value: str) -> str | None:
    normalized = _normalize_text(value)
    for canonical, aliases in _APP_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                return canonical
    return normalized or None


def _detect_app_mentions(text: str) -> set[str]:
    matches: set[str] = set()
    for canonical, aliases in _APP_ALIASES.items():
        for alias in aliases:
            if alias in text:
                matches.add(canonical)
                break
    return matches


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


def _load_sessions() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    try:
        db = get_database()
        raw = db.load_sessions()
    except Exception as exc:
        logger.warning("Session store load failed; running stateless: %s", exc)
        _SESSIONS.clear()
        return
    if not isinstance(raw, dict):
        logger.warning("Session store invalid; running stateless.")
        _SESSIONS.clear()
        return
    for user_id, data in raw.items():
        if not isinstance(data, dict):
            continue
        _SESSIONS[str(user_id)] = SessionState(
            user_id=str(user_id),
            active_app=data.get("active_app"),
            last_action=data.get("last_action"),
            pending_details=data.get("pending_details")
            if isinstance(data.get("pending_details"), dict)
            else {},
            short_term_memory=[
                str(item)
                for item in data.get("short_term_memory", [])
                if isinstance(item, str)
            ],
            last_summary=data.get("last_summary"),
            last_image_path=data.get("last_image_path"),
            last_image_prompt=data.get("last_image_prompt"),
            last_image_summary=data.get("last_image_summary"),
            last_image_source=data.get("last_image_source"),
            recent_images=_normalize_recent_images(data.get("recent_images")),
        )
        _hydrate_last_image_from_recent(_SESSIONS[str(user_id)])


def _save_sessions() -> None:
    try:
        db = get_database()
        for user_id, state in _SESSIONS.items():
            db.save_session_state(
                user_id,
                active_app=state.active_app,
                last_action=state.last_action,
                pending_details=state.pending_details,
                short_term_memory=state.short_term_memory,
                last_summary=state.last_summary,
                last_image_path=state.last_image_path,
                last_image_prompt=state.last_image_prompt,
                last_image_summary=state.last_image_summary,
                last_image_source=state.last_image_source,
                recent_images=state.recent_images,
            )
    except Exception as exc:
        logger.warning("Session store save failed: %s", exc)


def _get_or_create_session(user_id: str) -> SessionState:
    _load_sessions()
    session = _SESSIONS.get(user_id)
    if session is None:
        session = SessionState(user_id=user_id)
        _SESSIONS[user_id] = session
    return session


def _build_image_summary(prompt: str | None, source: str | None) -> str | None:
    summary_parts = []
    if source:
        summary_parts.append(source)
    if prompt:
        summary_parts.append(prompt)
    return ": ".join(summary_parts) if summary_parts else None


def _normalize_recent_images(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if not path:
            continue
        normalized.append(
            {
                "path": str(path),
                "prompt": item.get("prompt"),
                "summary": item.get("summary"),
                "source": item.get("source"),
            }
        )
    return _trim_recent_images(normalized)


def _hydrate_last_image_from_recent(session: SessionState) -> None:
    if session.recent_images:
        last = session.recent_images[-1]
        session.last_image_path = last.get("path") or session.last_image_path
        session.last_image_prompt = last.get("prompt") or session.last_image_prompt
        session.last_image_summary = last.get("summary") or session.last_image_summary
        session.last_image_source = last.get("source") or session.last_image_source
        return
    if session.last_image_path:
        summary = session.last_image_summary or _build_image_summary(
            session.last_image_prompt, session.last_image_source
        )
        session.recent_images = [
            {
                "path": session.last_image_path,
                "prompt": session.last_image_prompt,
                "summary": summary,
                "source": session.last_image_source,
            }
        ]


def _trim_recent_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(images) <= _RECENT_IMAGE_LIMIT:
        return images
    return images[-_RECENT_IMAGE_LIMIT:]
