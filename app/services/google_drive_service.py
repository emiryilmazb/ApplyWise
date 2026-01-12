from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from uuid import uuid4

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from app.services.gmail_auth_service import GmailAuthConfig, GmailAuthService


DEFAULT_DRIVE_SCOPES = (
    "https://www.googleapis.com/auth/drive.readonly",
)

_GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
_GOOGLE_SHEET_MIME = "application/vnd.google-apps.spreadsheet"
_GOOGLE_SLIDES_MIME = "application/vnd.google-apps.presentation"
_GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"

_DOC_MIME_TYPES = {
    _GOOGLE_DOC_MIME,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}
_SHEET_MIME_TYPES = {
    _GOOGLE_SHEET_MIME,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
_SLIDES_MIME_TYPES = {
    _GOOGLE_SLIDES_MIME,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

_EXPORT_TEXT_MIME = {
    _GOOGLE_DOC_MIME: "text/plain",
    _GOOGLE_SHEET_MIME: "text/csv",
    _GOOGLE_SLIDES_MIME: "text/plain",
}

_EXPORT_ANALYSIS_MIME = {
    _GOOGLE_DOC_MIME: "application/pdf",
    _GOOGLE_SHEET_MIME: "text/csv",
    _GOOGLE_SLIDES_MIME: "application/pdf",
}


@dataclass(frozen=True)
class DriveFileRecord:
    file_id: str
    name: str
    mime_type: str
    modified_time: str
    owners: list[str]
    web_view_link: str
    size_bytes: int | None
    parents: list[str]


class DriveService:
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

    def search_files(
        self,
        *,
        query: str | None = None,
        content_query: str | None = None,
        file_type: str | None = None,
        folder_id: str | None = None,
        modified_after: str | None = None,
        modified_before: str | None = None,
        owner: str | None = None,
        max_results: int = 10,
        page_token: str | None = None,
    ) -> tuple[list[DriveFileRecord], str | None]:
        service = self._get_service()
        q = _build_drive_query(
            query=query,
            content_query=content_query,
            file_type=file_type,
            folder_id=folder_id,
            modified_after=modified_after,
            modified_before=modified_before,
            owner=owner,
        )
        response = (
            service.files()
            .list(
                q=q,
                pageSize=max(1, min(100, max_results)),
                pageToken=page_token or None,
                fields=_LIST_FIELDS,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files = response.get("files", []) or []
        records = [_normalize_file(item) for item in files if isinstance(item, dict)]
        return records, response.get("nextPageToken")

    def list_folder(
        self,
        *,
        folder_id: str,
        max_results: int = 20,
        page_token: str | None = None,
    ) -> tuple[list[DriveFileRecord], str | None]:
        if not folder_id:
            raise ValueError("📂 Missing folder id.")
        return self.search_files(
            folder_id=folder_id,
            max_results=max_results,
            page_token=page_token,
        )

    def get_metadata(self, *, file_id: str) -> DriveFileRecord:
        if not file_id:
            raise ValueError("📂 Missing file id.")
        try:
            service = self._get_service()
            data = (
                service.files()
                .get(
                    fileId=file_id,
                    fields=_GET_FIELDS,
                    supportsAllDrives=True,
                )
                .execute()
            )
        except HttpError as exc:
            raise _drive_http_error(exc) from exc
        if not isinstance(data, dict):
            raise ValueError("📂 File metadata unavailable.")
        return _normalize_file(data)

    def get_file_text(self, *, file_id: str, max_chars: int = 12000) -> dict[str, object]:
        if not file_id:
            raise ValueError("📂 Missing file id.")
        metadata = self._get_file_metadata(file_id)
        mime_type = str(metadata.get("mimeType") or "")
        text_bytes = self._download_text_bytes(file_id, mime_type)
        if text_bytes is None:
            raise ValueError("📂 This file type is not supported for text extraction.")
        try:
            text = text_bytes.decode("utf-8", errors="replace")
        except Exception:
            text = text_bytes.decode("latin-1", errors="replace")
        truncated = False
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
            truncated = True
        return {
            "file": _normalize_file(metadata),
            "text": text,
            "truncated": truncated,
        }

    def download_for_analysis(self, *, file_id: str) -> tuple[Path, str, DriveFileRecord]:
        if not file_id:
            raise ValueError("📂 Missing file id.")
        metadata = self._get_file_metadata(file_id)
        mime_type = str(metadata.get("mimeType") or "")
        export_mime = _EXPORT_ANALYSIS_MIME.get(mime_type)
        if export_mime:
            request = self._get_service().files().export_media(
                fileId=file_id,
                mimeType=export_mime,
            )
            target_mime = export_mime
        else:
            request = self._get_service().files().get_media(
                fileId=file_id,
                supportsAllDrives=True,
            )
            target_mime = mime_type or "application/octet-stream"
        temp_path = _build_temp_path(metadata.get("name"), target_mime)
        self._download_to_path(request, temp_path)
        return temp_path, target_mime, _normalize_file(metadata)

    def _get_service(self):
        if self._service is None:
            creds = self._auth.get_credentials()
            self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def _download_text_bytes(self, file_id: str, mime_type: str) -> bytes | None:
        export_mime = _EXPORT_TEXT_MIME.get(mime_type)
        if export_mime:
            request = self._get_service().files().export_media(
                fileId=file_id,
                mimeType=export_mime,
            )
            return self._download_bytes(request)
        if mime_type.startswith("text/"):
            request = self._get_service().files().get_media(
                fileId=file_id,
                supportsAllDrives=True,
            )
            return self._download_bytes(request)
        return None

    def _get_file_metadata(self, file_id: str) -> dict:
        try:
            data = (
                self._get_service()
                .files()
                .get(
                    fileId=file_id,
                    fields=_GET_FIELDS,
                    supportsAllDrives=True,
                )
                .execute()
            )
        except HttpError as exc:
            raise _drive_http_error(exc) from exc
        if not isinstance(data, dict):
            raise ValueError("📂 File metadata unavailable.")
        return data

    def _download_bytes(self, request) -> bytes:
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        try:
            while not done:
                _, done = downloader.next_chunk()
        except HttpError as exc:
            raise _drive_http_error(exc) from exc
        return fh.getvalue()

    def _download_to_path(self, request, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            try:
                while not done:
                    _, done = downloader.next_chunk()
            except HttpError as exc:
                raise _drive_http_error(exc) from exc


def build_drive_service(settings) -> DriveService:
    token_path = getattr(settings, "drive_token_path", "") or str(_default_drive_token_path())
    config = GmailAuthConfig(
        credentials_path=getattr(settings, "gmail_credentials_path", ""),
        token_path=token_path,
        scopes=getattr(settings, "drive_scopes", None),
        oauth_flow=getattr(settings, "gmail_oauth_flow", "local"),
        open_browser=bool(getattr(settings, "gmail_oauth_open_browser", True)),
        default_scopes=DEFAULT_DRIVE_SCOPES,
    )
    auth_service = GmailAuthService(config)
    return DriveService(auth_service)


def _build_drive_query(
    *,
    query: str | None,
    content_query: str | None,
    file_type: str | None,
    folder_id: str | None,
    modified_after: str | None,
    modified_before: str | None,
    owner: str | None,
) -> str:
    parts = ["trashed=false"]
    if folder_id:
        parts.append(f"'{_escape_drive_value(folder_id)}' in parents")
    if query:
        escaped = _escape_drive_value(query)
        parts.append(f"name contains '{escaped}'")
    if content_query:
        escaped = _escape_drive_value(content_query)
        parts.append(f"fullText contains '{escaped}'")
    if file_type:
        mime_filters = _mime_types_for_label(file_type)
        if mime_filters:
            if len(mime_filters) == 1:
                parts.append(f"mimeType = '{mime_filters[0]}'")
            else:
                joined = " or ".join(f"mimeType = '{mime}'" for mime in mime_filters)
                parts.append(f"({joined})")
    if modified_after:
        parts.append(f"modifiedTime >= '{_escape_drive_value(modified_after)}'")
    if modified_before:
        parts.append(f"modifiedTime <= '{_escape_drive_value(modified_before)}'")
    if owner:
        parts.append(f"'{_escape_drive_value(owner)}' in owners")
    return " and ".join(parts)


def _mime_types_for_label(value: str) -> list[str]:
    normalized = value.strip().lower()
    if not normalized:
        return []
    if normalized in {"folder", "folders"}:
        return [_GOOGLE_FOLDER_MIME]
    if normalized in {"pdf"}:
        return ["application/pdf"]
    if normalized in {"doc", "docs", "document", "documents", "word"}:
        return sorted(_DOC_MIME_TYPES)
    if normalized in {"sheet", "sheets", "spreadsheet", "spreadsheets", "excel"}:
        return sorted(_SHEET_MIME_TYPES)
    if normalized in {"slides", "presentation", "presentations", "deck"}:
        return sorted(_SLIDES_MIME_TYPES)
    return []


def _escape_drive_value(value: str) -> str:
    return str(value).replace("'", "\\'")


def _normalize_file(data: dict) -> DriveFileRecord:
    owners = _pluck_owners(data.get("owners"))
    size_raw = data.get("size")
    try:
        size_bytes = int(size_raw) if size_raw is not None else None
    except (TypeError, ValueError):
        size_bytes = None
    parents = data.get("parents") if isinstance(data.get("parents"), list) else []
    parents = [str(item) for item in parents if item]
    return DriveFileRecord(
        file_id=str(data.get("id") or ""),
        name=str(data.get("name") or ""),
        mime_type=str(data.get("mimeType") or ""),
        modified_time=str(data.get("modifiedTime") or ""),
        owners=owners,
        web_view_link=str(data.get("webViewLink") or ""),
        size_bytes=size_bytes,
        parents=parents,
    )


def _pluck_owners(items) -> list[str]:
    owners = []
    if not isinstance(items, list):
        return owners
    for item in items:
        if not isinstance(item, dict):
            continue
        email = str(item.get("emailAddress") or "").strip()
        name = str(item.get("displayName") or "").strip()
        if email:
            owners.append(email)
        elif name:
            owners.append(name)
    return owners


def _drive_http_error(exc: HttpError) -> Exception:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "resp", None), "status", None)
    if status == 404:
        return ValueError("📂 File not found.")
    if status == 403:
        return ValueError("📂 Permission denied for this file.")
    return ValueError(f"📂 Drive request failed (status {status}).")


def _build_temp_path(name: str | None, mime_type: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    base = root / "artifacts" / "tmp_drive"
    base.mkdir(parents=True, exist_ok=True)
    suffix = _suffix_for_mime(mime_type)
    safe_name = "".join(ch for ch in (name or "drive_file") if ch.isalnum() or ch in {"-", "_"}).strip("_")
    if not safe_name:
        safe_name = "drive_file"
    unique = uuid4().hex[:8]
    return base / f"{safe_name}_{unique}{suffix}"


def _default_drive_token_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "drive_token.json"


def _suffix_for_mime(mime_type: str) -> str:
    if mime_type == "application/pdf":
        return ".pdf"
    if mime_type == "text/csv":
        return ".csv"
    if mime_type.startswith("text/"):
        return ".txt"
    return ""


_GET_FIELDS = "id,name,mimeType,modifiedTime,owners(displayName,emailAddress),webViewLink,size,parents"
_LIST_FIELDS = f"nextPageToken,files({_GET_FIELDS})"
