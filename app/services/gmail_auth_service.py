from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Callable, Iterable
import webbrowser
import wsgiref.simple_server
import wsgiref.util

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logger = logging.getLogger(__name__)

DEFAULT_SCOPES = (
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/contacts",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/photoslibrary",
)


def parse_scopes(
    value: str | Iterable[str] | None,
    default_scopes: Iterable[str] | None = None,
) -> list[str]:
    fallback = list(default_scopes or DEFAULT_SCOPES)
    if value is None:
        return fallback
    if isinstance(value, str):
        cleaned = [item.strip() for item in value.split(",") if item.strip()]
        return cleaned or fallback
    return [item for item in value if item] or fallback


def _default_token_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "gmail_token.json"


@dataclass(frozen=True)
class GmailAuthConfig:
    credentials_path: str
    token_path: str | None = None
    scopes: Iterable[str] | None = None
    oauth_flow: str = "local"
    open_browser: bool = True
    default_scopes: Iterable[str] | None = None


class GmailAuthService:
    def __init__(self, config: GmailAuthConfig) -> None:
        self._config = config
        self._scopes = parse_scopes(config.scopes, config.default_scopes)
        token_path = config.token_path or str(_default_token_path())
        self._token_path = Path(token_path)

    @property
    def token_path(self) -> Path:
        return self._token_path

    def get_credentials(self) -> Credentials:
        creds = None
        if self._token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self._token_path))
            except Exception as exc:
                logger.warning("Failed to load Gmail token: %s", exc)
        if creds and not creds.scopes and self._scopes:
            creds = Credentials.from_authorized_user_file(
                str(self._token_path), scopes=self._scopes
            )
        if creds and not creds.has_scopes(self._scopes):
            logger.info("Stored token missing required scopes; reauthorization required.")
            creds = None
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self._save_token(creds)
            return creds
        if creds and creds.valid:
            return creds
        credentials_path = (self._config.credentials_path or "").strip()
        if not credentials_path:
            raise ValueError("Gmail credentials path is not configured.")
        credentials_path_value = Path(credentials_path)
        if not credentials_path_value.is_absolute():
            root = Path(__file__).resolve().parents[2]
            credentials_path_value = root / credentials_path_value
        flow = InstalledAppFlow.from_client_secrets_file(
            str(credentials_path_value), self._scopes
        )
        oauth_flow = (self._config.oauth_flow or "local").strip().lower()
        if oauth_flow == "console":
            creds = flow.run_console()
        else:
            creds = flow.run_local_server(open_browser=self._config.open_browser)
        self._save_token(creds)
        return creds

    def delete_token(self) -> bool:
        if self._token_path.exists():
            self._token_path.unlink()
            return True
        return False

    def start_reauth_flow(self) -> tuple[str, Callable[[], Credentials]]:
        credentials_path = (self._config.credentials_path or "").strip()
        if not credentials_path:
            raise ValueError("Gmail credentials path is not configured.")
        credentials_path_value = Path(credentials_path)
        if not credentials_path_value.is_absolute():
            root = Path(__file__).resolve().parents[2]
            credentials_path_value = root / credentials_path_value
        flow = InstalledAppFlow.from_client_secrets_file(
            str(credentials_path_value), self._scopes
        )
        host = "localhost"
        port = 8080
        wsgi_app = _RedirectWSGIApp(
            "The authentication flow has completed. You may close this window."
        )
        wsgiref.simple_server.WSGIServer.allow_reuse_address = False
        local_server = wsgiref.simple_server.make_server(
            host, port, wsgi_app, handler_class=_WSGIRequestHandler
        )
        flow.redirect_uri = f"http://{host}:{local_server.server_port}/"
        auth_url, _ = flow.authorization_url(prompt="consent")
        if self._config.open_browser:
            webbrowser.get().open(auth_url, new=1, autoraise=True)

        def _complete() -> Credentials:
            try:
                local_server.handle_request()
                if not wsgi_app.last_request_uri:
                    raise RuntimeError("Authorization response was not received.")
                authorization_response = wsgi_app.last_request_uri.replace(
                    "http", "https"
                )
                flow.fetch_token(authorization_response=authorization_response)
                creds = flow.credentials
                self._save_token(creds)
                return creds
            finally:
                local_server.server_close()

        return auth_url, _complete

    def _save_token(self, creds: Credentials) -> None:
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        with self._token_path.open("w", encoding="utf-8") as handle:
            handle.write(creds.to_json())
            handle.write("\n")


class _WSGIRequestHandler(wsgiref.simple_server.WSGIRequestHandler):
    def log_message(self, format, *args):
        logger.info(format, *args)


class _RedirectWSGIApp:
    def __init__(self, success_message: str) -> None:
        self.last_request_uri: str | None = None
        self._success_message = success_message

    def __call__(self, environ, start_response):
        start_response("200 OK", [("Content-type", "text/plain; charset=utf-8")])
        self.last_request_uri = wsgiref.util.request_uri(environ)
        return [self._success_message.encode("utf-8")]
