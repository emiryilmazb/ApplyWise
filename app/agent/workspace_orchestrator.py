from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from typing import Any

from app.agent.llm_client import LLMClient
from app.services.gmail_service import GmailService
from app.services.google_calendar_service import CalendarEvent, CalendarService
from app.services.google_tasks_service import TaskItem, TaskList, TasksService
from app.services.google_people_service import ContactRecord, PeopleService
from app.services.google_drive_service import DriveFileRecord, DriveService
from app.services.google_docs_service import DocInfo, DocsService
from app.services.google_photos_service import PhotoAlbum, PhotoMediaItem, PhotosService
from app.services.google_sheets_service import SheetInfo, SheetsService
from app.services.google_youtube_service import (
    YouTubePlaylist,
    YouTubePlaylistItem,
    YouTubeService,
    YouTubeVideo,
)

logger = logging.getLogger(__name__)

_MAX_TOOL_STEPS = 6
_MAX_TOOL_CALLS = 10
_MAX_GMAIL_BODY_CHARS = 1200
_MAX_DRAFT_BODY_CHARS = 1200
_FOLLOWUP_MAX_IDS = 5
_WORKSPACE_FOLLOWUP_TOKENS = (
    "link",
    "baglanti",
    "yukaridaki",
    "onceki",
    "son sheet",
    "son spreadsheet",
    "son tablo",
    "son dokuman",
    "son document",
    "son doc",
    "son docs",
    "dokuman",
    "belge",
    "docs",
    "son video",
    "son playlist",
    "son dosya",
    "son etkinlik",
    "son gorev",
    "son kisi",
    "son album",
    "son foto",
    "sonuncu",
    "tekrar",
    "yeniden",
    "dogru",
    "emin",
    "kontrol",
    "calismiyor",
    "calismadi",
    "acilmiyor",
    "gecersiz",
)
_WORKSPACE_SEARCH_ACTION_TOKENS = ("ara", "search", "bul", "listele", "tara", "find")

_SYSTEM_PROMPT = """You are Atlas, a corporate executive assistant.
Use the provided tools to manage Google Docs, Google Sheets, YouTube, Drive, Gmail, Google Calendar, Google Tasks, Google Contacts (People API), and Google Photos.
Always use function calls for any calendar, task, Drive, Docs, Sheets, YouTube, or Photos data retrieval or updates.
When sending email, create a draft first and only send after explicit user confirmation.
If a request needs contact details, use People tools to locate names, emails, phone numbers, or birthdays.
Minimize tool calls, respect free-tier limits, and only request data you need.
If required details are missing or multiple matches exist, ask a short clarification question instead of guessing.
Translate vague photo requests into structured filters (date ranges, content categories, media type) before calling Photos tools.
Use professional status emojis: 📊 for Sheets operations, 📁 for Drive/album operations, 🔎 for searches, ✅ for successful logging, 🎥 for YouTube searches, 📸 for photo searches, 🔍 for scanning.
Respond in a concise, executive tone using the user's language.
"""


@dataclass(frozen=True)
class WorkspaceAction:
    name: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True)
class WorkspaceResult:
    handled: bool
    response_text: str
    actions: list[WorkspaceAction]


def orchestrate_workspace_request(
    *,
    llm_client: LLMClient | None,
    gmail_service: GmailService,
    calendar_service: CalendarService,
    tasks_service: TasksService,
    people_service: PeopleService,
    drive_service: DriveService,
    docs_service: DocsService,
    photos_service: PhotosService,
    sheets_service: SheetsService,
    youtube_service: YouTubeService,
    message_text: str,
    session_context: dict[str, Any] | None = None,
) -> WorkspaceResult:
    if llm_client is None or not hasattr(llm_client, "generate_content"):
        return WorkspaceResult(handled=False, response_text="", actions=[])
    tools = _build_tools()
    if not tools:
        return WorkspaceResult(handled=False, response_text="", actions=[])
    pending_details = _get_pending_details(session_context)
    followup_enabled = _should_use_last_workspace_context(message_text, pending_details)
    followup_hint = _build_followup_hint(pending_details) if followup_enabled else None
    contents = [_build_user_content(message_text, session_context, followup_hint=followup_hint)]
    actions: list[WorkspaceAction] = []
    tool_executor = _WorkspaceToolExecutor(
        llm_client=llm_client,
        gmail_service=gmail_service,
        calendar_service=calendar_service,
        tasks_service=tasks_service,
        people_service=people_service,
        drive_service=drive_service,
        docs_service=docs_service,
        photos_service=photos_service,
        sheets_service=sheets_service,
        youtube_service=youtube_service,
        pending_details=pending_details,
        followup_enabled=followup_enabled,
    )
    for _step in range(_MAX_TOOL_STEPS):
        response = llm_client.generate_content(
            contents,
            tools=tools,
            system_instruction=_SYSTEM_PROMPT,
            use_default_tools=False,
        )
        if response is None:
            break
        function_calls = _extract_function_calls(response)
        response_text = (getattr(response, "text", None) or "").strip()
        if not function_calls:
            if response_text:
                return WorkspaceResult(
                    handled=True,
                    response_text=response_text,
                    actions=actions,
                )
            break
        tool_parts = []
        for call in function_calls[:_MAX_TOOL_CALLS]:
            action = tool_executor.execute(call.name, call.args)
            actions.append(action)
            tool_parts.append(_build_function_response_part(call.name, action))
        if not tool_parts:
            break
        contents.append(_build_tool_content(tool_parts))
    fallback = _build_fallback_response(actions)
    return WorkspaceResult(handled=True, response_text=fallback, actions=actions)


@dataclass(frozen=True)
class _FunctionCall:
    name: str
    args: dict[str, Any]


class _WorkspaceToolExecutor:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        gmail_service: GmailService,
        calendar_service: CalendarService,
        tasks_service: TasksService,
        people_service: PeopleService,
        drive_service: DriveService,
        docs_service: DocsService,
        photos_service: PhotosService,
        sheets_service: SheetsService,
        youtube_service: YouTubeService,
        pending_details: dict[str, Any] | None = None,
        followup_enabled: bool = False,
    ) -> None:
        self._llm_client = llm_client
        self._gmail_service = gmail_service
        self._calendar_service = calendar_service
        self._tasks_service = tasks_service
        self._people_service = people_service
        self._drive_service = drive_service
        self._docs_service = docs_service
        self._photos_service = photos_service
        self._sheets_service = sheets_service
        self._youtube_service = youtube_service
        self._followup_enabled = followup_enabled
        pending = pending_details or {}
        self._last_drive_file_ids = _parse_id_list(pending.get("last_drive_file_ids"))
        self._last_drive_folder_id = _clean_text(pending.get("last_drive_folder_id"))
        self._last_calendar_event_ids = _parse_id_list(
            pending.get("last_calendar_event_ids")
        )
        self._last_task_ids = _parse_id_list(pending.get("last_task_ids"))
        self._last_tasklist_ids = _parse_id_list(pending.get("last_tasklist_ids"))
        self._last_tasklist_id = _clean_text(pending.get("last_tasklist_id"))
        if not self._last_tasklist_id and self._last_tasklist_ids:
            self._last_tasklist_id = self._last_tasklist_ids[0]
        self._last_people_resource_names = _parse_id_list(
            pending.get("last_people_resource_names")
        )
        self._last_photo_album_ids = _parse_id_list(pending.get("last_photo_album_ids"))
        self._last_photo_media_item_ids = _parse_id_list(
            pending.get("last_photo_media_item_ids")
        )
        self._last_docs_document_ids = _parse_id_list(
            pending.get("last_docs_document_ids")
        )
        self._last_docs_document_id = _clean_text(
            pending.get("last_docs_document_id")
        )
        if not self._last_docs_document_id and self._last_docs_document_ids:
            self._last_docs_document_id = self._last_docs_document_ids[0]
        self._last_sheet_spreadsheet_ids = _parse_id_list(
            pending.get("last_sheet_spreadsheet_ids")
        )
        self._last_sheet_spreadsheet_id = _clean_text(
            pending.get("last_sheet_spreadsheet_id")
        )
        if not self._last_sheet_spreadsheet_id and self._last_sheet_spreadsheet_ids:
            self._last_sheet_spreadsheet_id = self._last_sheet_spreadsheet_ids[0]
        self._last_youtube_video_ids = _parse_id_list(
            pending.get("last_youtube_video_ids")
        )
        self._last_youtube_playlist_ids = _parse_id_list(
            pending.get("last_youtube_playlist_ids")
        )
        self._last_youtube_playlist_item_ids = _parse_id_list(
            pending.get("last_youtube_playlist_item_ids")
        )
        self._tools = {
            "gmail_search_messages": self._gmail_search_messages,
            "gmail_get_thread_messages": self._gmail_get_thread_messages,
            "gmail_create_draft": self._gmail_create_draft,
            "gmail_send_draft": self._gmail_send_draft,
            "calendar_list_events": self._calendar_list_events,
            "calendar_create_event": self._calendar_create_event,
            "calendar_update_event": self._calendar_update_event,
            "calendar_delete_event": self._calendar_delete_event,
            "tasks_list_tasklists": self._tasks_list_tasklists,
            "tasks_list": self._tasks_list,
            "tasks_create": self._tasks_create,
            "tasks_update": self._tasks_update,
            "tasks_complete": self._tasks_complete,
            "tasks_delete": self._tasks_delete,
            "people_search_contacts": self._people_search_contacts,
            "people_get_contact": self._people_get_contact,
            "people_list_connections": self._people_list_connections,
            "drive_search_files": self._drive_search_files,
            "drive_list_folder": self._drive_list_folder,
            "drive_get_metadata": self._drive_get_metadata,
            "drive_get_file_text": self._drive_get_file_text,
            "drive_analyze_file": self._drive_analyze_file,
            "docs_get_document": self._docs_get_document,
            "docs_get_text": self._docs_get_text,
            "docs_create_document": self._docs_create_document,
            "docs_append_text": self._docs_append_text,
            "docs_replace_text": self._docs_replace_text,
            "photos_search_media_items": self._photos_search_media_items,
            "photos_list_albums": self._photos_list_albums,
            "photos_create_album": self._photos_create_album,
            "photos_add_to_album": self._photos_add_to_album,
            "photos_get_media_metadata": self._photos_get_media_metadata,
            "sheets_create_spreadsheet": self._sheets_create_spreadsheet,
            "sheets_append_row": self._sheets_append_row,
            "sheets_read_range": self._sheets_read_range,
            "sheets_update_range": self._sheets_update_range,
            "sheets_find_spreadsheet": self._sheets_find_spreadsheet,
            "youtube_search_videos": self._youtube_search_videos,
            "youtube_list_video_details": self._youtube_list_video_details,
            "youtube_list_playlists": self._youtube_list_playlists,
            "youtube_list_playlist_items": self._youtube_list_playlist_items,
            "youtube_add_to_playlist": self._youtube_add_to_playlist,
            "youtube_add_to_watch_later": self._youtube_add_to_watch_later,
            "youtube_remove_from_playlist": self._youtube_remove_from_playlist,
            "youtube_fetch_transcript": self._youtube_fetch_transcript,
        }

    def execute(self, name: str, args: dict[str, Any]) -> WorkspaceAction:
        tool = self._tools.get(name)
        if tool is None:
            return WorkspaceAction(
                name=name,
                args=args,
                ok=False,
                result=None,
                error="Unknown tool name.",
            )
        try:
            result = tool(args)
            return WorkspaceAction(
                name=name,
                args=args,
                ok=True,
                result=result,
                error=None,
            )
        except Exception as exc:  # pragma: no cover - runtime/api errors
            logger.warning("Workspace tool %s failed: %s", name, exc)
            return WorkspaceAction(
                name=name,
                args=args,
                ok=False,
                result=None,
                error=str(exc),
            )

    def _resolve_last_id(self, candidates: list[str]) -> str:
        if not self._followup_enabled:
            return ""
        return candidates[0] if candidates else ""

    def _resolve_last_tasklist_id(self) -> str | None:
        if not self._followup_enabled:
            return None
        return self._last_tasklist_id

    def _resolve_last_docs_id(self) -> str:
        if not self._followup_enabled:
            return ""
        return self._last_docs_document_id or self._resolve_last_id(self._last_docs_document_ids)

    def _resolve_last_sheet_id(self) -> str:
        if not self._followup_enabled:
            return ""
        return self._last_sheet_spreadsheet_id or ""

    def _resolve_last_youtube_video_id(self) -> str:
        return self._resolve_last_id(self._last_youtube_video_ids)

    def _resolve_last_youtube_playlist_id(self) -> str:
        return self._resolve_last_id(self._last_youtube_playlist_ids)

    def _resolve_last_youtube_playlist_item_id(self) -> str:
        return self._resolve_last_id(self._last_youtube_playlist_item_ids)

    def _gmail_search_messages(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("Missing Gmail query.")
        max_results = _coerce_int(
            args.get("max_results"), default=5, low=1, high=25)
        if not _looks_like_gmail_query(query):
            query = self._gmail_service.build_search_query(
                self._llm_client, query)
        messages = self._gmail_service.search(query, max_results=max_results)
        return {
            "query": query,
            "messages": [_format_gmail_message(item) for item in messages],
        }

    def _gmail_get_thread_messages(self, args: dict[str, Any]) -> dict[str, Any]:
        thread_id = str(args.get("thread_id") or "").strip()
        if not thread_id:
            raise ValueError("Missing Gmail thread id.")
        messages = self._gmail_service.get_thread_messages(thread_id)
        return {
            "thread_id": thread_id,
            "messages": [_format_gmail_message(item) for item in messages],
        }

    def _gmail_create_draft(self, args: dict[str, Any]) -> dict[str, Any]:
        recipient = str(args.get("to") or "").strip()
        if not recipient:
            raise ValueError("Missing draft recipient.")
        subject = _clean_text(args.get("subject")) or ""
        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Missing draft prompt.")
        spec = self._gmail_service.build_draft_from_prompt(
            self._llm_client,
            recipient,
            prompt,
        )
        if subject:
            spec = spec.__class__(to=spec.to, subject=subject, body=spec.body)
        draft_id = self._gmail_service.create_draft(spec)
        return {
            "draft_id": draft_id,
            "draft": _format_draft(spec),
        }

    def _gmail_send_draft(self, args: dict[str, Any]) -> dict[str, Any]:
        draft_id = str(args.get("draft_id") or "").strip()
        if not draft_id:
            raise ValueError("Missing draft id.")
        message_id = self._gmail_service.send_draft(draft_id)
        return {"draft_id": draft_id, "message_id": message_id}

    def _calendar_list_events(self, args: dict[str, Any]) -> dict[str, Any]:
        events = self._calendar_service.list_events(
            time_min=_clean_text(args.get("time_min")),
            time_max=_clean_text(args.get("time_max")),
            query=_clean_text(args.get("query")),
            max_results=_coerce_int(
                args.get("max_results"), default=10, low=1, high=25),
            calendar_id=_clean_text(args.get("calendar_id")),
            include_cancelled=bool(args.get("include_cancelled", False)),
        )
        return {"events": [_format_calendar_event(event) for event in events]}

    def _calendar_create_event(self, args: dict[str, Any]) -> dict[str, Any]:
        summary = str(args.get("summary") or "").strip()
        if not summary:
            raise ValueError("Missing event summary.")
        event = self._calendar_service.create_event(
            summary=summary,
            start_time=_clean_text(args.get("start_time")),
            end_time=_clean_text(args.get("end_time")),
            start_date=_clean_text(args.get("start_date")),
            end_date=_clean_text(args.get("end_date")),
            description=_clean_text(args.get("description")),
            location=_clean_text(args.get("location")),
            attendees=_coerce_str_list(args.get("attendees")),
            time_zone=_clean_text(args.get("time_zone")),
            calendar_id=_clean_text(args.get("calendar_id")),
        )
        return {"event": _format_calendar_event(event)}

    def _calendar_update_event(self, args: dict[str, Any]) -> dict[str, Any]:
        event_id = str(args.get("event_id") or "").strip()
        if not event_id:
            event_id = self._resolve_last_id(self._last_calendar_event_ids)
        if not event_id:
            raise ValueError("Missing event id.")
        event = self._calendar_service.update_event(
            event_id=event_id,
            summary=_clean_text(args.get("summary")),
            start_time=_clean_text(args.get("start_time")),
            end_time=_clean_text(args.get("end_time")),
            start_date=_clean_text(args.get("start_date")),
            end_date=_clean_text(args.get("end_date")),
            description=_clean_text(args.get("description")),
            location=_clean_text(args.get("location")),
            attendees=_coerce_str_list(args.get("attendees")),
            time_zone=_clean_text(args.get("time_zone")),
            calendar_id=_clean_text(args.get("calendar_id")),
        )
        return {"event": _format_calendar_event(event)}

    def _calendar_delete_event(self, args: dict[str, Any]) -> dict[str, Any]:
        event_id = str(args.get("event_id") or "").strip()
        if not event_id:
            event_id = self._resolve_last_id(self._last_calendar_event_ids)
        if not event_id:
            raise ValueError("Missing event id.")
        self._calendar_service.delete_event(
            event_id=event_id,
            calendar_id=_clean_text(args.get("calendar_id")),
        )
        return {"deleted": True, "event_id": event_id}

    def _tasks_list_tasklists(self, args: dict[str, Any]) -> dict[str, Any]:
        tasklists = self._tasks_service.list_tasklists(
            max_results=_coerce_int(
                args.get("max_results"), default=25, low=1, high=25)
        )
        return {"tasklists": [_format_tasklist(item) for item in tasklists]}

    def _tasks_list(self, args: dict[str, Any]) -> dict[str, Any]:
        tasklist_id = _clean_text(args.get("tasklist_id"))
        tasklist_title = _clean_text(args.get("tasklist_title"))
        if not tasklist_id and not tasklist_title:
            tasklist_id = self._resolve_last_tasklist_id()
        tasks = self._tasks_service.list_tasks(
            tasklist_id=tasklist_id,
            tasklist_title=tasklist_title,
            show_completed=bool(args.get("show_completed", False)),
            due_min=_clean_text(args.get("due_min")),
            due_max=_clean_text(args.get("due_max")),
            max_results=_coerce_int(
                args.get("max_results"), default=20, low=1, high=50),
        )
        return {"tasks": [_format_task(item) for item in tasks]}

    def _tasks_create(self, args: dict[str, Any]) -> dict[str, Any]:
        title = str(args.get("title") or "").strip()
        if not title:
            raise ValueError("Missing task title.")
        tasklist_id = _clean_text(args.get("tasklist_id"))
        tasklist_title = _clean_text(args.get("tasklist_title"))
        if not tasklist_id and not tasklist_title:
            tasklist_id = self._resolve_last_tasklist_id()
        task = self._tasks_service.create_task(
            title=title,
            notes=_clean_text(args.get("notes")),
            due=_clean_text(args.get("due")),
            tasklist_id=tasklist_id,
            tasklist_title=tasklist_title,
        )
        return {"task": _format_task(task)}

    def _tasks_update(self, args: dict[str, Any]) -> dict[str, Any]:
        task_id = str(args.get("task_id") or "").strip()
        if not task_id:
            task_id = self._resolve_last_id(self._last_task_ids)
        if not task_id:
            raise ValueError("Missing task id.")
        tasklist_id = _clean_text(args.get("tasklist_id"))
        tasklist_title = _clean_text(args.get("tasklist_title"))
        if not tasklist_id and not tasklist_title:
            tasklist_id = self._resolve_last_tasklist_id()
        task = self._tasks_service.update_task(
            task_id=task_id,
            title=_clean_text(args.get("title")),
            notes=_clean_text(args.get("notes")),
            due=_clean_text(args.get("due")),
            status=_clean_text(args.get("status")),
            completed=_clean_text(args.get("completed")),
            tasklist_id=tasklist_id,
            tasklist_title=tasklist_title,
        )
        return {"task": _format_task(task)}

    def _tasks_complete(self, args: dict[str, Any]) -> dict[str, Any]:
        task_id = str(args.get("task_id") or "").strip()
        if not task_id:
            task_id = self._resolve_last_id(self._last_task_ids)
        if not task_id:
            raise ValueError("Missing task id.")
        tasklist_id = _clean_text(args.get("tasklist_id"))
        tasklist_title = _clean_text(args.get("tasklist_title"))
        if not tasklist_id and not tasklist_title:
            tasklist_id = self._resolve_last_tasklist_id()
        task = self._tasks_service.complete_task(
            task_id=task_id,
            tasklist_id=tasklist_id,
            tasklist_title=tasklist_title,
        )
        return {"task": _format_task(task)}

    def _tasks_delete(self, args: dict[str, Any]) -> dict[str, Any]:
        task_id = str(args.get("task_id") or "").strip()
        if not task_id:
            task_id = self._resolve_last_id(self._last_task_ids)
        if not task_id:
            raise ValueError("Missing task id.")
        tasklist_id = _clean_text(args.get("tasklist_id"))
        tasklist_title = _clean_text(args.get("tasklist_title"))
        if not tasklist_id and not tasklist_title:
            tasklist_id = self._resolve_last_tasklist_id()
        self._tasks_service.delete_task(
            task_id=task_id,
            tasklist_id=tasklist_id,
            tasklist_title=tasklist_title,
        )
        return {"deleted": True, "task_id": task_id}

    def _people_search_contacts(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("Missing contact search query.")
        max_results = _coerce_int(
            args.get("max_results"), default=10, low=1, high=25)
        fields = _coerce_str_list(args.get("fields"))
        contacts = self._people_service.search_contacts(
            query=query,
            max_results=max_results,
            fields=fields,
        )
        return {"contacts": [_format_contact(contact) for contact in contacts]}

    def _people_get_contact(self, args: dict[str, Any]) -> dict[str, Any]:
        resource_name = str(args.get("resource_name") or "").strip()
        if not resource_name:
            resource_name = self._resolve_last_id(self._last_people_resource_names)
        if not resource_name:
            raise ValueError("Missing contact resource name.")
        fields = _coerce_str_list(args.get("fields"))
        contact = self._people_service.get_contact(
            resource_name=resource_name,
            fields=fields,
        )
        return {"contact": _format_contact(contact)}

    def _people_list_connections(self, args: dict[str, Any]) -> dict[str, Any]:
        max_results = _coerce_int(
            args.get("max_results"), default=10, low=1, high=50)
        fields = _coerce_str_list(args.get("fields"))
        page_token = _clean_text(args.get("page_token"))
        contacts, next_token = self._people_service.list_connections(
            max_results=max_results,
            page_token=page_token,
            fields=fields,
        )
        return {
            "contacts": [_format_contact(contact) for contact in contacts],
            "next_page_token": next_token,
        }

    def _drive_search_files(self, args: dict[str, Any]) -> dict[str, Any]:
        query = _clean_text(args.get("query"))
        content_query = _clean_text(args.get("content_query"))
        file_type = _clean_text(args.get("file_type"))
        folder_id = _clean_text(args.get("folder_id"))
        modified_after = _clean_text(args.get("modified_after"))
        modified_before = _clean_text(args.get("modified_before"))
        owner = _clean_text(args.get("owner"))
        max_results = _coerce_int(
            args.get("max_results"), default=10, low=1, high=100)
        page_token = _clean_text(args.get("page_token"))
        files, next_token = self._drive_service.search_files(
            query=query,
            content_query=content_query,
            file_type=file_type,
            folder_id=folder_id,
            modified_after=modified_after,
            modified_before=modified_before,
            owner=owner,
            max_results=max_results,
            page_token=page_token,
        )
        return {
            "files": [_format_drive_file(item) for item in files],
            "next_page_token": next_token,
        }

    def _drive_list_folder(self, args: dict[str, Any]) -> dict[str, Any]:
        folder_id = _clean_text(args.get("folder_id"))
        if not folder_id and self._followup_enabled:
            folder_id = self._last_drive_folder_id
        max_results = _coerce_int(
            args.get("max_results"), default=20, low=1, high=100)
        page_token = _clean_text(args.get("page_token"))
        files, next_token = self._drive_service.list_folder(
            folder_id=folder_id or "",
            max_results=max_results,
            page_token=page_token,
        )
        return {
            "files": [_format_drive_file(item) for item in files],
            "next_page_token": next_token,
        }

    def _drive_get_metadata(self, args: dict[str, Any]) -> dict[str, Any]:
        file_id = str(args.get("file_id") or "").strip()
        if not file_id:
            file_id = self._resolve_last_id(self._last_drive_file_ids)
        if not file_id:
            raise ValueError("📂 Missing file id.")
        file_record = self._drive_service.get_metadata(file_id=file_id)
        return {"file": _format_drive_file(file_record)}

    def _drive_get_file_text(self, args: dict[str, Any]) -> dict[str, Any]:
        file_id = str(args.get("file_id") or "").strip()
        if not file_id:
            file_id = self._resolve_last_id(self._last_drive_file_ids)
        if not file_id:
            raise ValueError("📂 Missing file id.")
        max_chars = _coerce_int(args.get("max_chars"),
                                default=12000, low=2000, high=60000)
        payload = self._drive_service.get_file_text(
            file_id=file_id, max_chars=max_chars)
        file_record = payload.get("file")
        if isinstance(file_record, DriveFileRecord):
            payload["file"] = _format_drive_file(file_record)
        return payload

    def _drive_analyze_file(self, args: dict[str, Any]) -> dict[str, Any]:
        file_id = str(args.get("file_id") or "").strip()
        if not file_id:
            file_id = self._resolve_last_id(self._last_drive_file_ids)
        prompt = str(args.get("prompt") or args.get("question") or "").strip()
        if not file_id:
            raise ValueError("📂 Missing file id.")
        if not prompt:
            raise ValueError("📂 Missing analysis prompt.")
        if not hasattr(self._llm_client, "generate_with_file"):
            raise ValueError("📂 File analysis is unavailable.")
        temp_path = None
        try:
            temp_path, mime_type, file_record = self._drive_service.download_for_analysis(
                file_id=file_id)
            result = self._llm_client.generate_with_file(
                prompt, str(temp_path), mime_type)
            if result.error:
                raise ValueError(result.error)
            return {
                "analysis": result.text or "",
                "file": _format_drive_file(file_record),
            }
        finally:
            if temp_path:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("Drive temp file cleanup failed.")

    def _docs_get_document(self, args: dict[str, Any]) -> dict[str, Any]:
        document_id = str(args.get("document_id") or "").strip()
        if not document_id:
            document_id = self._resolve_last_docs_id()
        if not document_id:
            raise ValueError("Missing document id.")
        doc = self._docs_service.get_document(document_id=document_id)
        return {"document": _format_doc(doc)}

    def _docs_get_text(self, args: dict[str, Any]) -> dict[str, Any]:
        document_id = str(args.get("document_id") or "").strip()
        if not document_id:
            document_id = self._resolve_last_docs_id()
        if not document_id:
            raise ValueError("Missing document id.")
        max_chars = _coerce_int(args.get("max_chars"), default=12000, low=2000, high=60000)
        payload = self._docs_service.get_document_text(
            document_id=document_id,
            max_chars=max_chars,
        )
        doc_record = payload.get("document")
        if isinstance(doc_record, DocInfo):
            payload["document"] = _format_doc(doc_record)
        return payload

    def _docs_create_document(self, args: dict[str, Any]) -> dict[str, Any]:
        title = str(args.get("title") or "").strip()
        if not title:
            raise ValueError("Missing document title.")
        doc = self._docs_service.create_document(title=title)
        return {"document": _format_doc(doc)}

    def _docs_append_text(self, args: dict[str, Any]) -> dict[str, Any]:
        document_id = str(args.get("document_id") or "").strip()
        if not document_id:
            document_id = self._resolve_last_docs_id()
        text = str(args.get("text") or "")
        if not document_id:
            raise ValueError("Missing document id.")
        if not text.strip():
            raise ValueError("Missing text to append.")
        return self._docs_service.append_text(document_id=document_id, text=text)

    def _docs_replace_text(self, args: dict[str, Any]) -> dict[str, Any]:
        document_id = str(args.get("document_id") or "").strip()
        if not document_id:
            document_id = self._resolve_last_docs_id()
        find_text = str(args.get("find_text") or "").strip()
        replace_text = str(args.get("replace_text") or "")
        match_case = bool(args.get("match_case", False))
        if not document_id:
            raise ValueError("Missing document id.")
        if not find_text:
            raise ValueError("Missing text to replace.")
        return self._docs_service.replace_text(
            document_id=document_id,
            find_text=find_text,
            replace_text=replace_text,
            match_case=match_case,
        )

    def _photos_search_media_items(self, args: dict[str, Any]) -> dict[str, Any]:
        date_ranges = args.get("date_ranges")
        if date_ranges is not None and not isinstance(date_ranges, list):
            raise ValueError("Date ranges must be a list of objects.")
        content_categories = _coerce_str_list(args.get("content_categories"))
        media_type = _clean_text(args.get("media_type"))
        album_id = _clean_text(args.get("album_id"))
        if not album_id and self._followup_enabled and self._last_photo_album_ids:
            album_id = self._last_photo_album_ids[0]
        max_results = _coerce_int(
            args.get("max_results"), default=10, low=1, high=100)
        page_token = _clean_text(args.get("page_token"))
        items, next_token = self._photos_service.search_media_items(
            date_ranges=date_ranges,
            content_categories=content_categories,
            media_type=media_type,
            album_id=album_id,
            max_results=max_results,
            page_token=page_token,
        )
        return {
            "media_items": [_format_photo_item(item) for item in items],
            "next_page_token": next_token,
            "match_count": len(items),
        }

    def _photos_list_albums(self, args: dict[str, Any]) -> dict[str, Any]:
        max_results = _coerce_int(
            args.get("max_results"), default=25, low=1, high=50)
        page_token = _clean_text(args.get("page_token"))
        albums, next_token = self._photos_service.list_albums(
            max_results=max_results,
            page_token=page_token,
        )
        return {
            "albums": [_format_photo_album(album) for album in albums],
            "next_page_token": next_token,
        }

    def _photos_create_album(self, args: dict[str, Any]) -> dict[str, Any]:
        title = str(args.get("title") or "").strip()
        if not title:
            raise ValueError("Missing album title.")
        album = self._photos_service.create_album(title=title)
        return {"album": _format_photo_album(album)}

    def _photos_add_to_album(self, args: dict[str, Any]) -> dict[str, Any]:
        album_id = str(args.get("album_id") or "").strip()
        if not album_id:
            album_id = self._resolve_last_id(self._last_photo_album_ids)
        media_item_ids = args.get("media_item_ids")
        if not isinstance(media_item_ids, list):
            media_item_ids = []
        if not media_item_ids and self._followup_enabled and self._last_photo_media_item_ids:
            media_item_ids = list(self._last_photo_media_item_ids)
        result = self._photos_service.add_to_album(
            album_id=album_id,
            media_item_ids=media_item_ids,
        )
        return result

    def _photos_get_media_metadata(self, args: dict[str, Any]) -> dict[str, Any]:
        media_item_id = str(args.get("media_item_id") or "").strip()
        if not media_item_id:
            media_item_id = self._resolve_last_id(self._last_photo_media_item_ids)
        if not media_item_id:
            raise ValueError("Missing media item id.")
        item = self._photos_service.get_media_metadata(
            media_item_id=media_item_id)
        return {"media_item": _format_photo_item(item)}

    def _sheets_create_spreadsheet(self, args: dict[str, Any]) -> dict[str, Any]:
        title = str(args.get("title") or "").strip()
        if not title:
            raise ValueError("Missing spreadsheet title.")
        sheet_title = _clean_text(args.get("sheet_title"))
        info = self._sheets_service.create_spreadsheet(
            title=title,
            sheet_title=sheet_title,
        )
        return {"spreadsheet": _format_sheet_info(info)}

    def _sheets_append_row(self, args: dict[str, Any]) -> dict[str, Any]:
        spreadsheet_id = str(args.get("spreadsheet_id") or "").strip()
        if not spreadsheet_id:
            spreadsheet_id = self._resolve_last_sheet_id()
        if not spreadsheet_id:
            raise ValueError("Missing spreadsheet id.")
        sheet_name = _clean_text(args.get("sheet_name"))
        row_values = args.get("row_values")
        row_data = args.get("row_data")
        headers = _coerce_str_list(args.get("headers"))
        if isinstance(row_values, list):
            row_values = list(row_values)
        else:
            row_values = None
        if not isinstance(row_data, dict):
            row_data = None
        return self._sheets_service.append_row(
            spreadsheet_id=spreadsheet_id,
            sheet_name=sheet_name,
            row_values=row_values,
            row_data=row_data,
            headers=headers,
        )

    def _sheets_read_range(self, args: dict[str, Any]) -> dict[str, Any]:
        spreadsheet_id = str(args.get("spreadsheet_id") or "").strip()
        range_a1 = str(args.get("range") or "").strip()
        if not spreadsheet_id:
            spreadsheet_id = self._resolve_last_sheet_id()
        if not spreadsheet_id:
            raise ValueError("Missing spreadsheet id.")
        if not range_a1:
            raise ValueError("Missing range.")
        return self._sheets_service.read_range(
            spreadsheet_id=spreadsheet_id,
            range_a1=range_a1,
        )

    def _sheets_update_range(self, args: dict[str, Any]) -> dict[str, Any]:
        spreadsheet_id = str(args.get("spreadsheet_id") or "").strip()
        range_a1 = str(args.get("range") or "").strip()
        values = args.get("values")
        if not spreadsheet_id:
            spreadsheet_id = self._resolve_last_sheet_id()
        if not spreadsheet_id:
            raise ValueError("Missing spreadsheet id.")
        if not range_a1:
            raise ValueError("Missing range.")
        if not isinstance(values, list):
            raise ValueError("Missing values to update.")
        cleaned_values = [list(row) for row in values if isinstance(row, list)]
        if not cleaned_values:
            raise ValueError("Missing values to update.")
        return self._sheets_service.update_range(
            spreadsheet_id=spreadsheet_id,
            range_a1=range_a1,
            values=cleaned_values,
        )

    def _sheets_find_spreadsheet(self, args: dict[str, Any]) -> dict[str, Any]:
        query = _clean_text(args.get("query"))
        if not query:
            raise ValueError("Missing spreadsheet search query.")
        max_results = _coerce_int(
            args.get("max_results"), default=5, low=1, high=25)
        exact_match = bool(args.get("exact_match", False))
        files, _ = self._drive_service.search_files(
            query=query,
            file_type="spreadsheet",
            max_results=max_results,
        )
        if exact_match:
            normalized = query.strip().lower()
            files = [item for item in files if item.name.lower() == normalized]
        return {"spreadsheets": [_format_drive_file(item) for item in files]}

    def _youtube_search_videos(self, args: dict[str, Any]) -> dict[str, Any]:
        query = _clean_text(args.get("query"))
        if not query:
            raise ValueError("Missing YouTube search query.")
        max_results = _coerce_int(
            args.get("max_results"), default=5, low=1, high=5)
        videos = self._youtube_service.search_videos(
            query=query, max_results=max_results)
        return {"videos": [_format_youtube_video(item) for item in videos]}

    def _youtube_list_video_details(self, args: dict[str, Any]) -> dict[str, Any]:
        video_ids = _coerce_str_list(args.get("video_ids")) or []
        if not video_ids and self._followup_enabled:
            video_ids = list(self._last_youtube_video_ids)
        videos = self._youtube_service.list_video_details(video_ids=video_ids)
        return {"videos": [_format_youtube_video(item) for item in videos]}

    def _youtube_list_playlists(self, args: dict[str, Any]) -> dict[str, Any]:
        max_results = _coerce_int(
            args.get("max_results"), default=25, low=1, high=50)
        playlists = self._youtube_service.list_playlists(
            max_results=max_results)
        return {"playlists": [_format_youtube_playlist(item) for item in playlists]}

    def _youtube_list_playlist_items(self, args: dict[str, Any]) -> dict[str, Any]:
        playlist_id = str(args.get("playlist_id") or "").strip()
        if not playlist_id:
            playlist_id = self._resolve_last_youtube_playlist_id()
        if not playlist_id:
            raise ValueError("Missing playlist id.")
        max_results = _coerce_int(
            args.get("max_results"), default=10, low=1, high=50)
        items = self._youtube_service.list_playlist_items(
            playlist_id=playlist_id,
            max_results=max_results,
        )
        return {"items": [_format_youtube_playlist_item(item) for item in items]}

    def _youtube_add_to_playlist(self, args: dict[str, Any]) -> dict[str, Any]:
        playlist_id = str(args.get("playlist_id") or "").strip()
        video_id = str(args.get("video_id") or "").strip()
        if not playlist_id:
            playlist_id = self._resolve_last_youtube_playlist_id()
        if not video_id:
            video_id = self._resolve_last_youtube_video_id()
        if not playlist_id:
            raise ValueError("Missing playlist id.")
        if not video_id:
            raise ValueError("Missing video id.")
        item = self._youtube_service.add_to_playlist(
            playlist_id=playlist_id,
            video_id=video_id,
        )
        return {"item": _format_youtube_playlist_item(item)}

    def _youtube_add_to_watch_later(self, args: dict[str, Any]) -> dict[str, Any]:
        video_id = str(args.get("video_id") or "").strip()
        if not video_id:
            video_id = self._resolve_last_youtube_video_id()
        if not video_id:
            raise ValueError("Missing video id.")
        item = self._youtube_service.add_to_watch_later(video_id=video_id)
        return {"item": _format_youtube_playlist_item(item)}

    def _youtube_remove_from_playlist(self, args: dict[str, Any]) -> dict[str, Any]:
        playlist_item_id = str(args.get("playlist_item_id") or "").strip()
        if not playlist_item_id:
            playlist_item_id = self._resolve_last_youtube_playlist_item_id()
        if not playlist_item_id:
            raise ValueError("Missing playlist item id.")
        self._youtube_service.remove_from_playlist(
            playlist_item_id=playlist_item_id)
        return {"deleted": True, "playlist_item_id": playlist_item_id}

    def _youtube_fetch_transcript(self, args: dict[str, Any]) -> dict[str, Any]:
        video_id = str(args.get("video_id") or "").strip()
        if not video_id:
            video_id = self._resolve_last_youtube_video_id()
        if not video_id:
            raise ValueError("Missing video id.")
        return self._youtube_service.fetch_transcript(video_id=video_id)


def _build_tools() -> list[Any]:
    try:
        from google.genai import types
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Function calling tools unavailable: %s", exc)
        return []
    declarations = [_build_declaration(types, item)
                    for item in _function_specs()]
    try:
        return [types.Tool(function_declarations=declarations)]
    except Exception:
        return [{"function_declarations": declarations}]


def _build_declaration(types, data: dict[str, Any]):
    if hasattr(types, "FunctionDeclaration"):
        try:
            return types.FunctionDeclaration(**data)
        except Exception:
            return data
    return data


def _function_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "gmail_search_messages",
            "description": "Search Gmail and return message metadata with short bodies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Gmail search query or natural language."},
                    "max_results": {"type": "integer", "description": "Max messages to return (1-25)."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "gmail_get_thread_messages",
            "description": "Get all messages from a Gmail thread.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thread_id": {"type": "string", "description": "Gmail thread id."},
                },
                "required": ["thread_id"],
            },
        },
        {
            "name": "gmail_create_draft",
            "description": "Create a Gmail draft from a prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address."},
                    "subject": {"type": "string", "description": "Email subject (optional)."},
                    "prompt": {"type": "string", "description": "Draft instructions or body details."},
                },
                "required": ["to", "prompt"],
            },
        },
        {
            "name": "gmail_send_draft",
            "description": "Send an existing Gmail draft by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_id": {"type": "string", "description": "Draft id to send."},
                },
                "required": ["draft_id"],
            },
        },
        {
            "name": "calendar_list_events",
            "description": "List calendar events within a time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calendar_id": {"type": "string", "description": "Calendar id (default primary)."},
                    "time_min": {"type": "string", "description": "RFC3339 start timestamp."},
                    "time_max": {"type": "string", "description": "RFC3339 end timestamp."},
                    "query": {"type": "string", "description": "Free-text search query."},
                    "max_results": {"type": "integer", "description": "Max events to return (1-25)."},
                    "include_cancelled": {"type": "boolean", "description": "Include cancelled events."},
                },
            },
        },
        {
            "name": "calendar_create_event",
            "description": "Create a calendar event. Provide start_time/end_time (RFC3339) or start_date/end_date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calendar_id": {"type": "string", "description": "Calendar id (default primary)."},
                    "summary": {"type": "string", "description": "Event title."},
                    "description": {"type": "string", "description": "Event description."},
                    "location": {"type": "string", "description": "Event location."},
                    "start_time": {"type": "string", "description": "RFC3339 start timestamp."},
                    "end_time": {"type": "string", "description": "RFC3339 end timestamp."},
                    "start_date": {"type": "string", "description": "All-day start date (YYYY-MM-DD)."},
                    "end_date": {"type": "string", "description": "All-day end date (YYYY-MM-DD)."},
                    "time_zone": {"type": "string", "description": "IANA time zone (optional)."},
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Attendee email addresses.",
                    },
                },
                "required": ["summary"],
            },
        },
        {
            "name": "calendar_update_event",
            "description": "Update a calendar event by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calendar_id": {"type": "string", "description": "Calendar id (default primary)."},
                    "event_id": {"type": "string", "description": "Event id to update."},
                    "summary": {"type": "string", "description": "Event title."},
                    "description": {"type": "string", "description": "Event description."},
                    "location": {"type": "string", "description": "Event location."},
                    "start_time": {"type": "string", "description": "RFC3339 start timestamp."},
                    "end_time": {"type": "string", "description": "RFC3339 end timestamp."},
                    "start_date": {"type": "string", "description": "All-day start date (YYYY-MM-DD)."},
                    "end_date": {"type": "string", "description": "All-day end date (YYYY-MM-DD)."},
                    "time_zone": {"type": "string", "description": "IANA time zone (optional)."},
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Attendee email addresses.",
                    },
                },
                "required": ["event_id"],
            },
        },
        {
            "name": "calendar_delete_event",
            "description": "Delete a calendar event by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calendar_id": {"type": "string", "description": "Calendar id (default primary)."},
                    "event_id": {"type": "string", "description": "Event id to delete."},
                },
                "required": ["event_id"],
            },
        },
        {
            "name": "tasks_list_tasklists",
            "description": "List Google Tasks task lists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "description": "Max task lists to return (1-25)."},
                },
            },
        },
        {
            "name": "tasks_list",
            "description": "List tasks from a task list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasklist_id": {"type": "string", "description": "Task list id."},
                    "tasklist_title": {"type": "string", "description": "Task list title if id unknown."},
                    "show_completed": {"type": "boolean", "description": "Include completed tasks."},
                    "due_min": {"type": "string", "description": "RFC3339 due date minimum."},
                    "due_max": {"type": "string", "description": "RFC3339 due date maximum."},
                    "max_results": {"type": "integer", "description": "Max tasks to return (1-50)."},
                },
            },
        },
        {
            "name": "tasks_create",
            "description": "Create a task in Google Tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasklist_id": {"type": "string", "description": "Task list id."},
                    "tasklist_title": {"type": "string", "description": "Task list title if id unknown."},
                    "title": {"type": "string", "description": "Task title."},
                    "notes": {"type": "string", "description": "Task notes."},
                    "due": {"type": "string", "description": "RFC3339 due date/time."},
                },
                "required": ["title"],
            },
        },
        {
            "name": "tasks_update",
            "description": "Update a task by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasklist_id": {"type": "string", "description": "Task list id."},
                    "tasklist_title": {"type": "string", "description": "Task list title if id unknown."},
                    "task_id": {"type": "string", "description": "Task id."},
                    "title": {"type": "string", "description": "Task title."},
                    "notes": {"type": "string", "description": "Task notes."},
                    "due": {"type": "string", "description": "RFC3339 due date/time."},
                    "status": {"type": "string", "description": "Task status (needsAction/completed)."},
                    "completed": {"type": "string", "description": "RFC3339 completion time."},
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "tasks_complete",
            "description": "Mark a task completed by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasklist_id": {"type": "string", "description": "Task list id."},
                    "tasklist_title": {"type": "string", "description": "Task list title if id unknown."},
                    "task_id": {"type": "string", "description": "Task id."},
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "tasks_delete",
            "description": "Delete a task by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasklist_id": {"type": "string", "description": "Task list id."},
                    "tasklist_title": {"type": "string", "description": "Task list title if id unknown."},
                    "task_id": {"type": "string", "description": "Task id."},
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "people_search_contacts",
            "description": "Search Google Contacts by name, email, or phone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Contact name, email, or phone to search."},
                    "max_results": {"type": "integer", "description": "Max contacts to return (1-25)."},
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional People API fields to return.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "people_get_contact",
            "description": "Retrieve a single contact by People API resource name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_name": {"type": "string", "description": "People API resource name."},
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional People API fields to return.",
                    },
                },
                "required": ["resource_name"],
            },
        },
        {
            "name": "people_list_connections",
            "description": "List connections from Google Contacts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "description": "Max contacts to return (1-50)."},
                    "page_token": {"type": "string", "description": "Page token for pagination."},
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional People API fields to return.",
                    },
                },
            },
        },
        {
            "name": "drive_search_files",
            "description": "Search Google Drive by name, type, or content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "File name search text."},
                    "content_query": {"type": "string", "description": "Full-text search text."},
                    "file_type": {
                        "type": "string",
                        "description": "Filter by type: pdf, doc, spreadsheet, presentation, folder.",
                    },
                    "folder_id": {"type": "string", "description": "Parent folder id filter."},
                    "modified_after": {"type": "string", "description": "RFC3339 modifiedTime >= value."},
                    "modified_before": {"type": "string", "description": "RFC3339 modifiedTime <= value."},
                    "owner": {"type": "string", "description": "Owner email address filter."},
                    "max_results": {"type": "integer", "description": "Max files to return (1-100)."},
                    "page_token": {"type": "string", "description": "Page token for pagination."},
                },
            },
        },
        {
            "name": "drive_list_folder",
            "description": "List files within a specific Drive folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder_id": {"type": "string", "description": "Drive folder id."},
                    "max_results": {"type": "integer", "description": "Max files to return (1-100)."},
                    "page_token": {"type": "string", "description": "Page token for pagination."},
                },
                "required": ["folder_id"],
            },
        },
        {
            "name": "drive_get_metadata",
            "description": "Retrieve metadata for a Drive file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string", "description": "Drive file id."},
                },
                "required": ["file_id"],
            },
        },
        {
            "name": "drive_get_file_text",
            "description": "Extract text from a Drive file (Google Docs or text-based files).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string", "description": "Drive file id."},
                    "max_chars": {"type": "integer", "description": "Max characters to return (2000-60000)."},
                },
                "required": ["file_id"],
            },
        },
        {
            "name": "drive_analyze_file",
            "description": "Analyze a Drive file with a specific question or summary request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string", "description": "Drive file id."},
                    "prompt": {"type": "string", "description": "Analysis request or question."},
                },
                "required": ["file_id", "prompt"],
            },
        },
        {
            "name": "docs_get_document",
            "description": "Retrieve metadata for a Google Doc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Google Docs document id."},
                },
                "required": ["document_id"],
            },
        },
        {
            "name": "docs_get_text",
            "description": "Extract plain text from a Google Doc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Google Docs document id."},
                    "max_chars": {"type": "integer", "description": "Max characters to return (2000-60000)."},
                },
                "required": ["document_id"],
            },
        },
        {
            "name": "docs_create_document",
            "description": "Create a new Google Doc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Document title."},
                },
                "required": ["title"],
            },
        },
        {
            "name": "docs_append_text",
            "description": "Append text to a Google Doc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Google Docs document id."},
                    "text": {"type": "string", "description": "Text to append."},
                },
                "required": ["document_id", "text"],
            },
        },
        {
            "name": "docs_replace_text",
            "description": "Replace text in a Google Doc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Google Docs document id."},
                    "find_text": {"type": "string", "description": "Text to search for."},
                    "replace_text": {"type": "string", "description": "Replacement text."},
                    "match_case": {"type": "boolean", "description": "Match case when searching."},
                },
                "required": ["document_id", "find_text", "replace_text"],
            },
        },
        {
            "name": "photos_search_media_items",
            "description": "Search Google Photos by date ranges, content categories, or media type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_ranges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_date": {"type": "string", "description": "YYYY-MM-DD start date."},
                                "end_date": {"type": "string", "description": "YYYY-MM-DD end date."}
                            }
                        },
                        "description": "Date ranges for filtering (YYYY-MM-DD)."
                    },
                    "content_categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Content categories like RECEIPTS, DOCUMENTS, LANDSCAPES."
                    },
                    "media_type": {"type": "string", "description": "PHOTO or VIDEO."},
                    "album_id": {"type": "string", "description": "Optional album id to search within."},
                    "max_results": {"type": "integer", "description": "Max media items to return (1-100)."},
                    "page_token": {"type": "string", "description": "Page token for pagination."}
                }
            }
        },
        {
            "name": "photos_list_albums",
            "description": "List Google Photos albums.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "description": "Max albums to return (1-50)."},
                    "page_token": {"type": "string", "description": "Page token for pagination."}
                }
            }
        },
        {
            "name": "photos_create_album",
            "description": "Create a new Google Photos album.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Album title."}
                },
                "required": ["title"]
            }
        },
        {
            "name": "photos_add_to_album",
            "description": "Add media items to a Google Photos album.",
            "parameters": {
                "type": "object",
                "properties": {
                    "album_id": {"type": "string", "description": "Album id."},
                    "media_item_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Media item ids to add."
                    }
                },
                "required": ["album_id", "media_item_ids"]
            }
        },
        {
            "name": "photos_get_media_metadata",
            "description": "Retrieve metadata for a Google Photos media item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "media_item_id": {"type": "string", "description": "Media item id."}
                },
                "required": ["media_item_id"]
            }
        },
        {
            "name": "sheets_create_spreadsheet",
            "description": "Create a new Google Spreadsheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Spreadsheet title."},
                    "sheet_title": {"type": "string", "description": "Optional first sheet title."},
                },
                "required": ["title"],
            },
        },
        {
            "name": "sheets_append_row",
            "description": "Append a row to a sheet, creating headers if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string", "description": "Spreadsheet id."},
                    "sheet_name": {"type": "string", "description": "Sheet tab name (optional)."},
                    "row_values": {"type": "array", "items": {"type": "string"}, "description": "Row values in order."},
                    "row_data": {"type": "object", "description": "Row values by header name."},
                    "headers": {"type": "array", "items": {"type": "string"}, "description": "Headers to set when initializing."},
                },
                "required": ["spreadsheet_id"],
            },
        },
        {
            "name": "sheets_read_range",
            "description": "Read values from a sheet range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string", "description": "Spreadsheet id."},
                    "range": {"type": "string", "description": "A1 range (e.g., Sheet1!A1:C10)."},
                },
                "required": ["spreadsheet_id", "range"],
            },
        },
        {
            "name": "sheets_update_range",
            "description": "Update values in a sheet range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string", "description": "Spreadsheet id."},
                    "range": {"type": "string", "description": "A1 range (e.g., Sheet1!B2:D2)."},
                    "values": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}, "description": "2D values array."},
                },
                "required": ["spreadsheet_id", "range", "values"],
            },
        },
        {
            "name": "sheets_find_spreadsheet",
            "description": "Find spreadsheets in Drive by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Spreadsheet name or keyword."},
                    "exact_match": {"type": "boolean", "description": "Require exact title match."},
                    "max_results": {"type": "integer", "description": "Max spreadsheets to return (1-25)."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "youtube_search_videos",
            "description": "Search YouTube videos by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords."},
                    "max_results": {"type": "integer", "description": "Max videos to return (1-5)."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "youtube_list_video_details",
            "description": "Fetch details for specific YouTube videos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_ids": {"type": "array", "items": {"type": "string"}, "description": "Video ids."},
                },
                "required": ["video_ids"],
            },
        },
        {
            "name": "youtube_list_playlists",
            "description": "List the user's YouTube playlists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "description": "Max playlists to return (1-50)."},
                },
            },
        },
        {
            "name": "youtube_list_playlist_items",
            "description": "List items inside a playlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string", "description": "Playlist id."},
                    "max_results": {"type": "integer", "description": "Max items to return (1-50)."},
                },
                "required": ["playlist_id"],
            },
        },
        {
            "name": "youtube_add_to_playlist",
            "description": "Add a video to a playlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string", "description": "Playlist id."},
                    "video_id": {"type": "string", "description": "Video id."},
                },
                "required": ["playlist_id", "video_id"],
            },
        },
        {
            "name": "youtube_add_to_watch_later",
            "description": "Add a video to Watch Later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string", "description": "Video id."},
                },
                "required": ["video_id"],
            },
        },
        {
            "name": "youtube_remove_from_playlist",
            "description": "Remove a playlist item by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_item_id": {"type": "string", "description": "Playlist item id."},
                },
                "required": ["playlist_item_id"],
            },
        },
        {
            "name": "youtube_fetch_transcript",
            "description": "Fetch a YouTube video transcript if available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string", "description": "Video id."},
                },
                "required": ["video_id"],
            },
        },
    ]


def _get_pending_details(session_context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(session_context, dict):
        return {}
    pending = session_context.get("pending_details")
    if isinstance(pending, dict):
        return pending
    return {}


def _parse_id_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


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


def _has_followup_ids(pending_details: dict[str, Any]) -> bool:
    for key in (
        "last_drive_file_ids",
        "last_drive_folder_id",
        "last_calendar_event_ids",
        "last_task_ids",
        "last_tasklist_ids",
        "last_tasklist_id",
        "last_people_resource_names",
        "last_photo_album_ids",
        "last_photo_media_item_ids",
        "last_docs_document_ids",
        "last_docs_document_id",
        "last_sheet_spreadsheet_ids",
        "last_sheet_spreadsheet_id",
        "last_youtube_video_ids",
        "last_youtube_playlist_ids",
        "last_youtube_playlist_item_ids",
    ):
        if pending_details.get(key):
            return True
    return False


def _should_use_last_workspace_context(
    message_text: str,
    pending_details: dict[str, Any],
) -> bool:
    if not pending_details or not _has_followup_ids(pending_details):
        return False
    normalized = _normalize_text(message_text or "")
    if _contains_any(normalized, _WORKSPACE_SEARCH_ACTION_TOKENS):
        return False
    return _contains_any(normalized, _WORKSPACE_FOLLOWUP_TOKENS)


def _build_followup_hint(pending_details: dict[str, Any]) -> str | None:
    hints: list[str] = []
    drive_ids = _parse_id_list(pending_details.get("last_drive_file_ids"))
    if drive_ids:
        hints.append(f"drive_file_ids={','.join(drive_ids[:_FOLLOWUP_MAX_IDS])}")
    folder_id = _clean_text(pending_details.get("last_drive_folder_id"))
    if folder_id:
        hints.append(f"drive_folder_id={folder_id}")
    doc_ids = _parse_id_list(pending_details.get("last_docs_document_ids"))
    doc_id = _clean_text(pending_details.get("last_docs_document_id"))
    doc_preview = doc_id or (doc_ids[0] if doc_ids else "")
    if doc_preview:
        hints.append(f"document_id={doc_preview}")
    event_ids = _parse_id_list(pending_details.get("last_calendar_event_ids"))
    if event_ids:
        hints.append(f"calendar_event_ids={','.join(event_ids[:_FOLLOWUP_MAX_IDS])}")
    task_ids = _parse_id_list(pending_details.get("last_task_ids"))
    if task_ids:
        hints.append(f"task_ids={','.join(task_ids[:_FOLLOWUP_MAX_IDS])}")
    tasklist_ids = _parse_id_list(pending_details.get("last_tasklist_ids"))
    tasklist_id = _clean_text(pending_details.get("last_tasklist_id"))
    tasklist_preview = tasklist_id or (tasklist_ids[0] if tasklist_ids else "")
    if tasklist_preview:
        hints.append(f"tasklist_id={tasklist_preview}")
    people_ids = _parse_id_list(pending_details.get("last_people_resource_names"))
    if people_ids:
        hints.append(f"people_resource_names={','.join(people_ids[:_FOLLOWUP_MAX_IDS])}")
    album_ids = _parse_id_list(pending_details.get("last_photo_album_ids"))
    if album_ids:
        hints.append(f"photo_album_ids={','.join(album_ids[:_FOLLOWUP_MAX_IDS])}")
    media_ids = _parse_id_list(pending_details.get("last_photo_media_item_ids"))
    if media_ids:
        hints.append(f"photo_media_item_ids={','.join(media_ids[:_FOLLOWUP_MAX_IDS])}")
    sheet_ids = _parse_id_list(pending_details.get("last_sheet_spreadsheet_ids"))
    sheet_id = _clean_text(pending_details.get("last_sheet_spreadsheet_id"))
    sheet_preview = sheet_id or (sheet_ids[0] if sheet_ids else "")
    if sheet_preview:
        hints.append(f"spreadsheet_id={sheet_preview}")
    youtube_video_ids = _parse_id_list(pending_details.get("last_youtube_video_ids"))
    if youtube_video_ids:
        hints.append(f"youtube_video_ids={','.join(youtube_video_ids[:_FOLLOWUP_MAX_IDS])}")
    youtube_playlist_ids = _parse_id_list(pending_details.get("last_youtube_playlist_ids"))
    if youtube_playlist_ids:
        hints.append(f"youtube_playlist_ids={','.join(youtube_playlist_ids[:_FOLLOWUP_MAX_IDS])}")
    youtube_item_ids = _parse_id_list(pending_details.get("last_youtube_playlist_item_ids"))
    if youtube_item_ids:
        hints.append(f"youtube_playlist_item_ids={','.join(youtube_item_ids[:_FOLLOWUP_MAX_IDS])}")
    if not hints:
        return None
    return "Follow-up context (reuse if relevant): " + "; ".join(hints)


def _build_user_content(
    message_text: str,
    session_context: dict[str, Any] | None,
    *,
    followup_hint: str | None = None,
):
    from google.genai import types

    now = datetime.now().astimezone().isoformat()
    context_block = json.dumps(
        session_context or {}, ensure_ascii=True, indent=2)
    prompt = (
        "User request:\n"
        f"{message_text.strip()}\n\n"
        f"Current time: {now}\n"
        f"Session context:\n{context_block}\n"
    )
    if followup_hint:
        prompt += f"\n{followup_hint}\n"
    part = _text_part(types, prompt)
    return types.Content(role="user", parts=[part])


def _build_tool_content(parts: list[Any]):
    from google.genai import types

    return types.Content(role="tool", parts=parts)


def _text_part(types, text: str):
    if hasattr(types.Part, "from_text"):
        return types.Part.from_text(text=text)
    return types.Part(text=text)


def _build_function_response_part(tool_name: str, action: WorkspaceAction):
    from google.genai import types

    payload = {
        "ok": action.ok,
        "result": action.result,
        "error": action.error,
    }
    if hasattr(types.Part, "from_function_response"):
        return types.Part.from_function_response(name=tool_name, response=payload)
    response = types.FunctionResponse(name=tool_name, response=payload)
    return types.Part(function_response=response)


def _extract_function_calls(response) -> list[_FunctionCall]:
    calls: list[_FunctionCall] = []
    seen: set[str] = set()
    raw_calls = getattr(response, "function_calls", None) or []
    for call in raw_calls:
        extracted = _coerce_function_call(call)
        if extracted:
            key = _function_call_key(extracted)
            if key in seen:
                continue
            seen.add(key)
            calls.append(extracted)
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            call = getattr(part, "function_call", None)
            extracted = _coerce_function_call(call)
            if extracted:
                key = _function_call_key(extracted)
                if key in seen:
                    continue
                seen.add(key)
                calls.append(extracted)
    return calls


def _function_call_key(call: _FunctionCall) -> str:
    try:
        encoded = json.dumps(
            call.args, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
    except (TypeError, ValueError):
        encoded = str(call.args)
    return f"{call.name}:{encoded}"


def _coerce_function_call(call) -> _FunctionCall | None:
    if call is None:
        return None
    name = getattr(call, "name", None) or getattr(call, "function_name", None)
    if not name and isinstance(call, dict):
        name = call.get("name")
    if not name:
        return None
    raw_args = getattr(call, "args", None)
    if raw_args is None and isinstance(call, dict):
        raw_args = call.get("args")
    args = _normalize_args(raw_args)
    return _FunctionCall(name=str(name), args=args)


def _normalize_args(raw_args: Any) -> dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if hasattr(raw_args, "to_dict"):
        try:
            return raw_args.to_dict()
        except Exception:
            return {}
    if isinstance(raw_args, str):
        try:
            data = json.loads(raw_args)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _format_gmail_message(message) -> dict[str, Any]:
    body = (message.body or message.snippet or "").strip()
    return {
        "message_id": message.message_id,
        "thread_id": message.thread_id,
        "subject": message.subject,
        "sender": message.sender,
        "date": message.date,
        "snippet": message.snippet,
        "body": _truncate(body, _MAX_GMAIL_BODY_CHARS),
    }


def _format_draft(spec) -> dict[str, Any]:
    return {
        "to": spec.to,
        "subject": spec.subject,
        "body": _truncate(spec.body or "", _MAX_DRAFT_BODY_CHARS),
    }


def _format_calendar_event(event: CalendarEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "summary": event.summary,
        "start": event.start,
        "end": event.end,
        "html_link": event.html_link,
        "status": event.status,
        "location": event.location,
    }


def _format_tasklist(tasklist: TaskList) -> dict[str, Any]:
    return {"tasklist_id": tasklist.tasklist_id, "title": tasklist.title}


def _format_task(task: TaskItem) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "title": task.title,
        "notes": task.notes,
        "status": task.status,
        "due": task.due,
    }


def _format_contact(contact: ContactRecord) -> dict[str, Any]:
    return {
        "resource_name": contact.resource_name,
        "display_name": contact.display_name,
        "emails": contact.emails,
        "phones": contact.phones,
        "birthdays": contact.birthdays,
        "organizations": contact.organizations,
        "urls": contact.urls,
        "addresses": contact.addresses,
    }


def _format_drive_file(file: DriveFileRecord) -> dict[str, Any]:
    return {
        "file_id": file.file_id,
        "name": file.name,
        "mime_type": file.mime_type,
        "modified_time": file.modified_time,
        "owners": file.owners,
        "web_view_link": file.web_view_link,
        "size_bytes": file.size_bytes,
        "parents": file.parents,
    }


def _format_doc(doc: DocInfo) -> dict[str, Any]:
    return {
        "document_id": doc.document_id,
        "title": doc.title,
        "url": doc.url,
        "revision_id": doc.revision_id,
    }


def _format_photo_item(item: PhotoMediaItem) -> dict[str, Any]:
    return {
        "media_item_id": item.media_item_id,
        "filename": item.filename,
        "mime_type": item.mime_type,
        "base_url": item.base_url,
        "product_url": item.product_url,
        "creation_time": item.creation_time,
        "width": item.width,
        "height": item.height,
        "camera_make": item.camera_make,
        "camera_model": item.camera_model,
        "location": item.location,
    }


def _format_photo_album(album: PhotoAlbum) -> dict[str, Any]:
    return {
        "album_id": album.album_id,
        "title": album.title,
        "media_items_count": album.media_items_count,
        "cover_photo_base_url": album.cover_photo_base_url,
        "product_url": album.product_url,
    }


def _format_sheet_info(sheet: SheetInfo) -> dict[str, Any]:
    return {
        "spreadsheet_id": sheet.spreadsheet_id,
        "title": sheet.title,
        "url": sheet.url,
    }


def _format_youtube_video(video: YouTubeVideo) -> dict[str, Any]:
    return {
        "video_id": video.video_id,
        "title": video.title,
        "description": video.description,
        "published_at": video.published_at,
        "channel_title": video.channel_title,
        "url": video.url,
    }


def _format_youtube_playlist(playlist: YouTubePlaylist) -> dict[str, Any]:
    return {
        "playlist_id": playlist.playlist_id,
        "title": playlist.title,
        "description": playlist.description,
    }


def _format_youtube_playlist_item(item: YouTubePlaylistItem) -> dict[str, Any]:
    return {
        "playlist_item_id": item.playlist_item_id,
        "video_id": item.video_id,
        "title": item.title,
        "description": item.description,
        "published_at": item.published_at,
    }


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit].rstrip()}..."


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _coerce_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return None


def _coerce_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def _looks_like_gmail_query(value: str) -> bool:
    if ":" in value:
        return True
    return False


def _looks_like_scan_request(args: dict[str, Any]) -> bool:
    categories = args.get("content_categories") if isinstance(
        args, dict) else None
    if categories is None:
        return False
    if isinstance(categories, str):
        items = [item.strip().upper()
                 for item in categories.split(",") if item.strip()]
    elif isinstance(categories, list):
        items = [str(item).strip().upper()
                 for item in categories if str(item).strip()]
    else:
        items = []
    return any(item in {"DOCUMENTS", "RECEIPTS", "TEXT", "WHITEBOARDS"} for item in items)


def _build_fallback_response(actions: list[WorkspaceAction]) -> str:
    if not actions:
        return (
            "Please share the Sheets, Drive, Photos, YouTube, calendar, task, or contact details you'd like me to handle."
        )
    successes = [action for action in actions if action.ok]
    failures = [action for action in actions if not action.ok]
    if not successes and failures:
        first_error = failures[0].error or "Unknown error"
        return f"I was unable to complete the request. {first_error}"
    task_creates = [
        action for action in successes if action.name == "tasks_create"]
    event_creates = [
        action for action in successes if action.name == "calendar_create_event"]
    contact_searches = [
        action for action in successes if action.name == "people_search_contacts"]
    contact_fetches = [
        action for action in successes if action.name == "people_get_contact"]
    drive_searches = [
        action for action in successes if action.name == "drive_search_files"]
    drive_lists = [
        action for action in successes if action.name == "drive_list_folder"]
    photo_searches = [
        action for action in successes if action.name == "photos_search_media_items"]
    photo_album_lists = [
        action for action in successes if action.name == "photos_list_albums"]
    photo_album_creates = [
        action for action in successes if action.name == "photos_create_album"]
    photo_album_adds = [
        action for action in successes if action.name == "photos_add_to_album"]
    sheets_appends = [
        action for action in successes if action.name == "sheets_append_row"]
    sheets_creates = [
        action for action in successes if action.name == "sheets_create_spreadsheet"]
    youtube_searches = [
        action for action in successes if action.name == "youtube_search_videos"]
    youtube_playlist_adds = [
        action
        for action in successes
        if action.name in {"youtube_add_to_playlist", "youtube_add_to_watch_later"}
    ]
    scan_searches = [
        action for action in photo_searches if _looks_like_scan_request(action.args)
    ]
    parts = [""]
    if task_creates:
        parts.append(f"Added {len(task_creates)} task(s).")
    if event_creates:
        event = event_creates[0].result.get(
            "event") if event_creates[0].result else {}
        summary = event.get("summary") if isinstance(event, dict) else None
        start_time = event.get("start") if isinstance(event, dict) else None
        if summary and start_time:
            parts.append(f"Scheduled '{summary}' for {start_time}.")
        else:
            parts.append("Scheduled the calendar event.")
    if contact_searches or contact_fetches:
        parts.append("Contact information has been retrieved.")
    if drive_searches or drive_lists:
        parts.append("📁 Drive results are ready.")
    if scan_searches:
        parts.append("🔍 Scan results are ready.")
    elif photo_searches:
        parts.append("📸 Photo results are ready.")
    if photo_album_lists or photo_album_creates or photo_album_adds:
        parts.append("📁 Album operations completed.")
    if sheets_creates:
        parts.append("📊 Spreadsheet created.")
    if sheets_appends:
        parts.append("✅ Logged the update to Sheets.")
    if youtube_searches:
        parts.append("🎥 YouTube results are ready.")
    if youtube_playlist_adds:
        parts.append("✅ Added the video to the playlist.")
    if len(parts) == 1:
        parts.append("Completed the requested updates.")
    if failures:
        parts.append(
            "Some items could not be completed; let me know what to retry.")
    return " ".join(parts)
