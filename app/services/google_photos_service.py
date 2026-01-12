from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.services.gmail_auth_service import GmailAuthConfig, GmailAuthService


DEFAULT_PHOTOS_SCOPES = (
    "https://www.googleapis.com/auth/photoslibrary",
)


@dataclass(frozen=True)
class PhotoMediaItem:
    media_item_id: str
    filename: str
    mime_type: str
    base_url: str
    product_url: str
    creation_time: str
    width: int | None
    height: int | None
    camera_make: str | None
    camera_model: str | None
    location: dict[str, float] | None


@dataclass(frozen=True)
class PhotoAlbum:
    album_id: str
    title: str
    media_items_count: int | None
    cover_photo_base_url: str
    product_url: str


class PhotosService:
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

    def search_media_items(
        self,
        *,
        date_ranges: Iterable[dict[str, str]] | None = None,
        content_categories: Iterable[str] | None = None,
        media_type: str | None = None,
        album_id: str | None = None,
        max_results: int = 25,
        page_token: str | None = None,
    ) -> tuple[list[PhotoMediaItem], str | None]:
        service = self._get_service()
        body: dict[str, object] = {"pageSize": max(1, min(100, max_results))}
        if page_token:
            body["pageToken"] = page_token
        filters = _build_filters(
            date_ranges=date_ranges,
            content_categories=content_categories,
            media_type=media_type,
        )
        if filters:
            body["filters"] = filters
        if album_id:
            body["albumId"] = str(album_id)
        try:
            response = service.mediaItems().search(body=body).execute()
        except HttpError as exc:
            raise _photos_http_error(exc) from exc
        items = response.get("mediaItems", []) or []
        records = [
            _normalize_media_item(item)
            for item in items
            if isinstance(item, dict)
        ]
        return records, response.get("nextPageToken")

    def list_albums(
        self,
        *,
        max_results: int = 25,
        page_token: str | None = None,
    ) -> tuple[list[PhotoAlbum], str | None]:
        service = self._get_service()
        params: dict[str, object] = {"pageSize": max(1, min(50, max_results))}
        if page_token:
            params["pageToken"] = page_token
        try:
            response = service.albums().list(**params).execute()
        except HttpError as exc:
            raise _photos_http_error(exc) from exc
        items = response.get("albums", []) or []
        albums = [
            _normalize_album(item)
            for item in items
            if isinstance(item, dict)
        ]
        return albums, response.get("nextPageToken")

    def create_album(self, *, title: str) -> PhotoAlbum:
        cleaned = str(title or "").strip()
        if not cleaned:
            raise ValueError("Missing album title.")
        body = {"album": {"title": cleaned}}
        try:
            response = self._get_service().albums().create(body=body).execute()
        except HttpError as exc:
            raise _photos_http_error(exc) from exc
        if not isinstance(response, dict):
            raise ValueError("Photo album creation failed.")
        return _normalize_album(response)

    def add_to_album(self, *, album_id: str, media_item_ids: Iterable[str]) -> dict[str, object]:
        if not album_id:
            raise ValueError("Missing album id.")
        ids = [str(item).strip() for item in media_item_ids or [] if str(item).strip()]
        if not ids:
            raise ValueError("Missing media item ids.")
        body = {"mediaItemIds": ids}
        try:
            self._get_service().albums().batchAddMediaItems(
                albumId=str(album_id),
                body=body,
            ).execute()
        except HttpError as exc:
            raise _photos_http_error(exc) from exc
        return {"album_id": str(album_id), "added_count": len(ids)}

    def get_media_metadata(self, *, media_item_id: str) -> PhotoMediaItem:
        if not media_item_id:
            raise ValueError("Missing media item id.")
        try:
            response = self._get_service().mediaItems().get(
                mediaItemId=str(media_item_id)
            ).execute()
        except HttpError as exc:
            raise _photos_http_error(exc) from exc
        if not isinstance(response, dict):
            raise ValueError("Photo metadata unavailable.")
        return _normalize_media_item(response)

    def _get_service(self):
        if self._service is None:
            creds = self._auth.get_credentials()
            self._service = build(
                "photoslibrary", "v1", credentials=creds, cache_discovery=False
            )
        return self._service


def build_photos_service(settings) -> PhotosService:
    token_path = getattr(settings, "photos_token_path", "") or str(_default_photos_token_path())
    config = GmailAuthConfig(
        credentials_path=getattr(settings, "gmail_credentials_path", ""),
        token_path=token_path,
        scopes=getattr(settings, "photos_scopes", None),
        oauth_flow=getattr(settings, "gmail_oauth_flow", "local"),
        open_browser=bool(getattr(settings, "gmail_oauth_open_browser", True)),
        default_scopes=DEFAULT_PHOTOS_SCOPES,
    )
    auth_service = GmailAuthService(config)
    return PhotosService(auth_service)


def _build_filters(
    *,
    date_ranges: Iterable[dict[str, str]] | None,
    content_categories: Iterable[str] | None,
    media_type: str | None,
) -> dict[str, object]:
    filters: dict[str, object] = {}
    ranges = _normalize_date_ranges(date_ranges)
    if ranges:
        filters["dateFilter"] = {"ranges": ranges}
    categories = _normalize_content_categories(content_categories)
    if categories:
        filters["contentFilter"] = {"includedContentCategories": categories}
    media_types = _normalize_media_type(media_type)
    if media_types:
        filters["mediaTypeFilter"] = {"mediaTypes": media_types}
    return filters


def _normalize_date_ranges(date_ranges: Iterable[dict[str, str]] | None) -> list[dict[str, object]]:
    if not date_ranges:
        return []
    ranges = []
    for item in date_ranges:
        if not isinstance(item, dict):
            raise ValueError("Date ranges must be objects with start_date/end_date.")
        start = _parse_date(item.get("start_date"))
        end = _parse_date(item.get("end_date"))
        if not start and not end:
            continue
        range_item: dict[str, object] = {}
        if start:
            range_item["startDate"] = start
        if end:
            range_item["endDate"] = end
        ranges.append(range_item)
    return ranges


def _parse_date(value: str | None) -> dict[str, int] | None:
    if not value:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    try:
        parsed = datetime.strptime(cleaned, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Dates must be formatted as YYYY-MM-DD.") from exc
    return {"year": parsed.year, "month": parsed.month, "day": parsed.day}


def _normalize_content_categories(content_categories: Iterable[str] | None) -> list[str]:
    if content_categories is None:
        return []
    if isinstance(content_categories, str):
        raw_items = content_categories.split(",")
    else:
        raw_items = list(content_categories)
    categories = []
    for item in raw_items:
        cleaned = str(item).strip()
        if cleaned:
            categories.append(cleaned.upper())
    return categories


def _normalize_media_type(media_type: str | None) -> list[str]:
    if not media_type:
        return []
    normalized = str(media_type).strip().upper()
    if normalized in {"PHOTO", "VIDEO"}:
        return [normalized]
    return []


def _normalize_media_item(data: dict) -> PhotoMediaItem:
    metadata = data.get("mediaMetadata") if isinstance(data.get("mediaMetadata"), dict) else {}
    width = _coerce_int(metadata.get("width"))
    height = _coerce_int(metadata.get("height"))
    photo_data = metadata.get("photo") if isinstance(metadata.get("photo"), dict) else {}
    location = metadata.get("location") if isinstance(metadata.get("location"), dict) else None
    location_payload = _normalize_location(location)
    return PhotoMediaItem(
        media_item_id=str(data.get("id") or ""),
        filename=str(data.get("filename") or ""),
        mime_type=str(data.get("mimeType") or ""),
        base_url=str(data.get("baseUrl") or ""),
        product_url=str(data.get("productUrl") or ""),
        creation_time=str(metadata.get("creationTime") or ""),
        width=width,
        height=height,
        camera_make=_clean_optional(photo_data.get("cameraMake")),
        camera_model=_clean_optional(photo_data.get("cameraModel")),
        location=location_payload,
    )


def _normalize_album(data: dict) -> PhotoAlbum:
    count = _coerce_int(data.get("mediaItemsCount"))
    return PhotoAlbum(
        album_id=str(data.get("id") or ""),
        title=str(data.get("title") or ""),
        media_items_count=count,
        cover_photo_base_url=str(data.get("coverPhotoBaseUrl") or ""),
        product_url=str(data.get("productUrl") or ""),
    )


def _normalize_location(data: dict | None) -> dict[str, float] | None:
    if not data:
        return None
    latitude = _coerce_float(data.get("latitude"))
    longitude = _coerce_float(data.get("longitude"))
    altitude = _coerce_float(data.get("altitude"))
    payload = {}
    if latitude is not None:
        payload["latitude"] = latitude
    if longitude is not None:
        payload["longitude"] = longitude
    if altitude is not None:
        payload["altitude"] = altitude
    return payload or None


def _clean_optional(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _photos_http_error(exc: HttpError) -> Exception:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "resp", None), "status", None)
    if status == 404:
        return ValueError("Photo resource not found.")
    if status in {403, 429}:
        return ValueError("Photos API quota exceeded. Please retry later.")
    return ValueError(f"Photos request failed (status {status}).")


def _default_photos_token_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "photos_token.json"
