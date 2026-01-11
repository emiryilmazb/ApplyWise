from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import logging
import re
from email.message import EmailMessage
from html import unescape
from typing import Iterable

from googleapiclient.discovery import build

from app.services.gmail_auth_service import GmailAuthConfig, GmailAuthService

logger = logging.getLogger(__name__)

GMAIL_LABELS = ("Critical", "Newsletter", "Action Required")


@dataclass(frozen=True)
class GmailMessage:
    message_id: str
    thread_id: str | None
    subject: str
    sender: str
    recipient: str | None
    date: str | None
    snippet: str | None
    body: str | None
    label_ids: list[str]


@dataclass(frozen=True)
class DraftSpec:
    to: str
    subject: str
    body: str


class GmailService:
    def __init__(self, auth_service: GmailAuthService, user_id: str = "me") -> None:
        self._auth = auth_service
        self._user_id = user_id
        self._service = None
        self._label_cache: dict[str, str] = {}

    @property
    def user_id(self) -> str:
        return self._user_id

    def ensure_credentials(self) -> None:
        self._auth.get_credentials()

    def delete_token(self) -> bool:
        return self._auth.delete_token()

    def start_reauth_flow(self):
        return self._auth.start_reauth_flow()

    def reset(self) -> None:
        self._service = None
        self._label_cache = {}

    def list_unread(self, max_results: int = 10, query: str | None = None) -> list[GmailMessage]:
        search_query = "is:unread"
        if query:
            search_query = f"{search_query} {query}"
        message_ids = self._list_messages(search_query, max_results)
        return [self.get_message(message_id) for message_id in message_ids]

    def search(self, query: str, max_results: int = 10) -> list[GmailMessage]:
        message_ids = self._list_messages(query, max_results)
        return [self.get_message(message_id) for message_id in message_ids]

    def summarize_messages(self, llm_client, messages: Iterable[GmailMessage]) -> str:
        items = []
        for message in messages:
            body = (message.body or message.snippet or "").strip()
            trimmed = body[:1200]
            items.append(
                f"- From: {message.sender}\n"
                f"  Subject: {message.subject}\n"
                f"  Snippet: {message.snippet or ''}\n"
                f"  Body: {trimmed}"
            )
        if not items:
            return "No messages to summarize."
        prompt = (
            "You are an executive assistant. Summarize the following emails in bullet points, "
            "highlighting urgent actions and owners. Keep it concise.\n\n"
            + "\n\n".join(items)
        )
        return llm_client.generate_text(prompt).strip()

    def build_search_query(self, llm_client, query: str) -> str:
        cleaned = (query or "").strip()
        if not cleaned:
            return cleaned
        if re.search(r"\\b(from|to|subject|is|label|has|after|before):", cleaned):
            return cleaned
        prompt = (
            "Convert the following request into a Gmail search query. "
            "Return ONLY the query string.\n\n"
            f"Request: {cleaned}\n"
        )
        try:
            response = llm_client.generate_text(prompt).strip()
        except Exception as exc:
            logger.warning("Gmail search query generation failed: %s", exc)
            return cleaned
        response = response.strip().strip('"').strip("'")
        return response or cleaned

    def categorize_message(self, llm_client, message: GmailMessage) -> str:
        fallback = self._heuristic_label(message)
        body = (message.body or message.snippet or "").strip()
        trimmed = body[:1200]
        prompt = (
            "Classify the email into one label: Critical, Newsletter, Action Required.\n"
            "Return ONLY the label name.\n\n"
            f"From: {message.sender}\n"
            f"Subject: {message.subject}\n"
            f"Snippet: {message.snippet or ''}\n"
            f"Body: {trimmed}\n"
        )
        try:
            label = llm_client.generate_text(prompt).strip()
        except Exception as exc:
            logger.warning("Gmail categorization failed: %s", exc)
            return fallback
        cleaned = label.strip().strip('"').strip("'")
        if cleaned not in GMAIL_LABELS:
            return fallback
        return cleaned

    def label_message(self, message_id: str, label_name: str) -> None:
        label_id = self._ensure_label(label_name)
        if not label_id:
            return
        service = self._get_service()
        service.users().messages().modify(
            userId=self._user_id,
            id=message_id,
            body={"addLabelIds": [label_id]},
        ).execute()

    def create_draft(self, spec: DraftSpec) -> str:
        message = EmailMessage()
        message["To"] = spec.to
        message["Subject"] = spec.subject
        message.set_content(spec.body)
        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        service = self._get_service()
        result = (
            service.users()
            .drafts()
            .create(userId=self._user_id, body={"message": {"raw": encoded}})
            .execute()
        )
        return result.get("id", "")

    def send_draft(self, draft_id: str) -> str:
        if not draft_id:
            return ""
        service = self._get_service()
        result = (
            service.users()
            .drafts()
            .send(userId=self._user_id, body={"id": draft_id})
            .execute()
        )
        return result.get("id", "")

    def build_draft_from_prompt(self, llm_client, recipient: str, prompt: str) -> DraftSpec:
        guidance = (
            "You are drafting an email. Return ONLY JSON with keys subject and body.\n"
            "Be professional, concise, and action-oriented.\n\n"
            f"Recipient: {recipient}\n"
            f"Prompt: {prompt}\n"
        )
        response = llm_client.generate_text(guidance).strip()
        subject, body = _parse_subject_body_json(response)
        if not subject:
            subject = "Quick Update"
        if not body:
            body = prompt.strip()
        return DraftSpec(to=recipient, subject=subject, body=body)

    def _list_messages(self, query: str, max_results: int) -> list[str]:
        service = self._get_service()
        response = (
            service.users()
            .messages()
            .list(userId=self._user_id, q=query, maxResults=max_results)
            .execute()
        )
        return [item.get("id") for item in response.get("messages", []) if item.get("id")]

    def get_message(self, message_id: str) -> GmailMessage:
        service = self._get_service()
        message = (
            service.users()
            .messages()
            .get(userId=self._user_id, id=message_id, format="full")
            .execute()
        )
        return _build_message(message, fallback_id=message_id)

    def get_thread_messages(self, thread_id: str) -> list[GmailMessage]:
        if not thread_id:
            return []
        service = self._get_service()
        thread = (
            service.users()
            .threads()
            .get(userId=self._user_id, id=thread_id, format="full")
            .execute()
        )
        messages = thread.get("messages", []) or []
        return [_build_message(item) for item in messages if isinstance(item, dict)]

    def _ensure_label(self, label_name: str) -> str | None:
        if label_name in self._label_cache:
            return self._label_cache[label_name]
        service = self._get_service()
        labels = service.users().labels().list(userId=self._user_id).execute()
        for label in labels.get("labels", []) or []:
            if label.get("name") == label_name:
                label_id = label.get("id")
                if label_id:
                    self._label_cache[label_name] = label_id
                    return label_id
        created = (
            service.users()
            .labels()
            .create(
                userId=self._user_id,
                body={"name": label_name, "labelListVisibility": "labelShow"},
            )
            .execute()
        )
        label_id = created.get("id")
        if label_id:
            self._label_cache[label_name] = label_id
        return label_id

    def archive_message(self, message_id: str) -> None:
        if not message_id:
            return
        service = self._get_service()
        service.users().messages().modify(
            userId=self._user_id,
            id=message_id,
            body={"removeLabelIds": ["INBOX"]},
        ).execute()

    def _get_service(self):
        if self._service is None:
            creds = self._auth.get_credentials()
            self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return self._service

    def _heuristic_label(self, message: GmailMessage) -> str:
        subject = (message.subject or "").lower()
        sender = (message.sender or "").lower()
        body = (message.body or message.snippet or "").lower()
        newsletter_markers = ("newsletter", "unsubscribe", "digest")
        if any(marker in subject or marker in body for marker in newsletter_markers):
            return "Newsletter"
        urgent_markers = ("urgent", "asap", "action required", "deadline")
        if any(marker in subject or marker in body for marker in urgent_markers):
            return "Action Required"
        critical_markers = ("invoice", "payment", "security", "breach", "password")
        if any(marker in subject or marker in body or marker in sender for marker in critical_markers):
            return "Critical"
        return "Action Required"


def build_gmail_service(settings) -> GmailService:
    config = GmailAuthConfig(
        credentials_path=getattr(settings, "gmail_credentials_path", ""),
        token_path=getattr(settings, "gmail_token_path", None),
        scopes=getattr(settings, "gmail_scopes", None),
        oauth_flow=getattr(settings, "gmail_oauth_flow", "local"),
        open_browser=bool(getattr(settings, "gmail_oauth_open_browser", True)),
    )
    auth_service = GmailAuthService(config)
    user_id = getattr(settings, "gmail_user_id", "me") or "me"
    return GmailService(auth_service, user_id=user_id)


def _extract_headers(headers: Iterable[dict]) -> dict[str, str]:
    extracted: dict[str, str] = {}
    for header in headers:
        name = (header.get("name") or "").strip().lower()
        value = (header.get("value") or "").strip()
        if name:
            extracted[name] = value
    return extracted


def _extract_body(payload: dict) -> str | None:
    parts = _flatten_parts(payload)
    plain = _extract_part_body(parts, "text/plain")
    if plain:
        return plain
    html = _extract_part_body(parts, "text/html")
    if html:
        return _strip_html(html)
    return None


def _flatten_parts(payload: dict) -> list[dict]:
    if not payload:
        return []
    stack = [payload]
    parts: list[dict] = []
    while stack:
        part = stack.pop()
        inner = part.get("parts")
        if inner:
            stack.extend(inner)
        else:
            parts.append(part)
    return parts


def _extract_part_body(parts: Iterable[dict], mime_type: str) -> str | None:
    for part in parts:
        if part.get("mimeType") != mime_type:
            continue
        body = part.get("body", {}) or {}
        data = body.get("data")
        if not data:
            continue
        try:
            decoded = base64.urlsafe_b64decode(data.encode("utf-8"))
            return decoded.decode("utf-8", errors="replace")
        except Exception:
            continue
    return None


def _strip_html(value: str) -> str:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", value)
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\\s+", " ", cleaned)
    return cleaned.strip()


def _parse_subject_body_json(value: str) -> tuple[str, str]:
    match = re.search(r"{.*}", value, re.DOTALL)
    if not match:
        return "", ""
    try:
        data = json.loads(match.group(0))
    except Exception:
        return "", ""
    subject = str(data.get("subject") or "").strip()
    body = str(data.get("body") or "").strip()
    return subject, body


def _build_message(message: dict, fallback_id: str | None = None) -> GmailMessage:
    payload = message.get("payload", {}) if isinstance(message, dict) else {}
    headers = _extract_headers(payload.get("headers", [])) if isinstance(payload, dict) else {}
    body = _extract_body(payload) if isinstance(payload, dict) else None
    return GmailMessage(
        message_id=message.get("id", fallback_id or ""),
        thread_id=message.get("threadId"),
        subject=headers.get("subject", ""),
        sender=headers.get("from", ""),
        recipient=headers.get("to"),
        date=headers.get("date"),
        snippet=message.get("snippet"),
        body=body,
        label_ids=list(message.get("labelIds") or []),
    )
