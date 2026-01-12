from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.services.gmail_auth_service import GmailAuthConfig, GmailAuthService

logger = logging.getLogger(__name__)

DEFAULT_DOCS_SCOPES = (
    "https://www.googleapis.com/auth/documents",
)


@dataclass(frozen=True)
class DocInfo:
    document_id: str
    title: str
    url: str
    revision_id: str | None


class DocsService:
    def __init__(self, auth_service: GmailAuthService) -> None:
        self._auth = auth_service
        self._service = None

    def ensure_credentials(self) -> None:
        self._auth.get_credentials()

    def reset(self) -> None:
        self._service = None

    def delete_token(self) -> bool:
        return self._auth.delete_token()

    def start_reauth_flow(self):
        return self._auth.start_reauth_flow()

    def get_document(self, *, document_id: str) -> DocInfo:
        doc = self._get_document_raw(document_id)
        return _normalize_doc(doc)

    def get_document_text(
        self,
        *,
        document_id: str,
        max_chars: int = 12000,
    ) -> dict[str, object]:
        doc = self._get_document_raw(document_id)
        text, truncated = _extract_doc_text(doc, max_chars)
        return {
            "document": _normalize_doc(doc),
            "text": text,
            "truncated": truncated,
        }

    def create_document(self, *, title: str) -> DocInfo:
        cleaned = str(title or "").strip()
        if not cleaned:
            raise ValueError("Missing document title.")
        try:
            doc = self._get_service().documents().create(body={"title": cleaned}).execute()
        except HttpError as exc:
            raise _docs_http_error(exc) from exc
        return _normalize_doc(doc or {})

    def append_text(self, *, document_id: str, text: str) -> dict[str, object]:
        cleaned = str(text or "")
        if not cleaned:
            raise ValueError("Missing text to append.")
        doc = self._get_document_raw(document_id)
        insert_index = _resolve_insert_index(doc)
        requests = [
            {
                "insertText": {
                    "location": {"index": insert_index},
                    "text": cleaned,
                }
            }
        ]
        try:
            self._get_service().documents().batchUpdate(
                documentId=document_id, body={"requests": requests}
            ).execute()
        except HttpError as exc:
            raise _docs_http_error(exc) from exc
        return {
            "document_id": document_id,
            "inserted": len(cleaned),
            "insert_index": insert_index,
        }

    def replace_text(
        self,
        *,
        document_id: str,
        find_text: str,
        replace_text: str,
        match_case: bool = False,
    ) -> dict[str, object]:
        find_value = str(find_text or "")
        if not find_value:
            raise ValueError("Missing text to replace.")
        requests = [
            {
                "replaceAllText": {
                    "containsText": {"text": find_value, "matchCase": bool(match_case)},
                    "replaceText": str(replace_text or ""),
                }
            }
        ]
        try:
            response = self._get_service().documents().batchUpdate(
                documentId=document_id, body={"requests": requests}
            ).execute()
        except HttpError as exc:
            raise _docs_http_error(exc) from exc
        replies = response.get("replies", []) if isinstance(response, dict) else []
        changed = 0
        if replies and isinstance(replies[0], dict):
            changed = int(replies[0].get("replaceAllText", {}).get("occurrencesChanged") or 0)
        return {
            "document_id": document_id,
            "occurrences_changed": changed,
        }

    def _get_document_raw(self, document_id: str) -> dict:
        cleaned = str(document_id or "").strip()
        if not cleaned:
            raise ValueError("Missing document id.")
        try:
            return self._get_service().documents().get(documentId=cleaned).execute()
        except HttpError as exc:
            raise _docs_http_error(exc) from exc

    def _get_service(self):
        if self._service is None:
            creds = self._auth.get_credentials()
            self._service = build("docs", "v1", credentials=creds, cache_discovery=False)
        return self._service


def build_docs_service(settings) -> DocsService:
    token_path = getattr(settings, "docs_token_path", "") or str(_default_docs_token_path())
    config = GmailAuthConfig(
        credentials_path=getattr(settings, "gmail_credentials_path", ""),
        token_path=token_path,
        scopes=getattr(settings, "docs_scopes", None),
        oauth_flow=getattr(settings, "gmail_oauth_flow", "local"),
        open_browser=bool(getattr(settings, "gmail_oauth_open_browser", True)),
        default_scopes=DEFAULT_DOCS_SCOPES,
    )
    auth_service = GmailAuthService(config)
    return DocsService(auth_service)


def _normalize_doc(doc: dict) -> DocInfo:
    doc_id = str(doc.get("documentId") or "")
    title = str(doc.get("title") or "")
    revision_id = str(doc.get("revisionId") or "") or None
    url = f"https://docs.google.com/document/d/{doc_id}/edit" if doc_id else ""
    return DocInfo(
        document_id=doc_id,
        title=title,
        url=url,
        revision_id=revision_id,
    )


def _extract_doc_text(doc: dict, max_chars: int) -> tuple[str, bool]:
    if not isinstance(doc, dict):
        return "", False
    content = doc.get("body", {}).get("content", [])
    if not isinstance(content, list):
        return "", False
    limit = max_chars if max_chars and max_chars > 0 else None
    parts: list[str] = []
    length = 0
    for block in content:
        paragraph = block.get("paragraph") if isinstance(block, dict) else None
        if not isinstance(paragraph, dict):
            continue
        elements = paragraph.get("elements", [])
        if not isinstance(elements, list):
            continue
        for element in elements:
            text_run = element.get("textRun") if isinstance(element, dict) else None
            if not isinstance(text_run, dict):
                continue
            segment = text_run.get("content")
            if not segment:
                continue
            if limit is not None and length >= limit:
                return "".join(parts), True
            if limit is not None and length + len(segment) > limit:
                remaining = max(0, limit - length)
                parts.append(segment[:remaining])
                return "".join(parts), True
            parts.append(segment)
            length += len(segment)
    return "".join(parts).strip(), False


def _resolve_insert_index(doc: dict) -> int:
    content = doc.get("body", {}).get("content", [])
    if not isinstance(content, list) or not content:
        return 1
    last = content[-1] if isinstance(content[-1], dict) else {}
    end_index = int(last.get("endIndex") or 1)
    return max(1, end_index - 1)


def _docs_http_error(exc: HttpError) -> Exception:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "resp", None), "status", None)
    if status == 404:
        return ValueError("Document not found.")
    if status == 403:
        return ValueError("Permission denied for this document.")
    return ValueError(f"Docs request failed (status {status}).")


def _default_docs_token_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "memory" / "docs_token.json"
