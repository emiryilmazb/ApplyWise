from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv() -> None:
        return None


@dataclass(frozen=True)
class Settings:
    app_name: str = "ApplyWise"
    log_level: str = "INFO"
    sqlite_db_path: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    gemini_image_model: str = "gemini-3-pro-image-preview"
    streaming_enabled: bool = True
    show_thoughts: bool = True
    kariyer_job_url: str = ""
    kariyernet_username: str = ""
    kariyernet_password: str = ""


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        app_name=os.getenv("APPLYWISE_APP_NAME", "ApplyWise"),
        log_level=os.getenv("APPLYWISE_LOG_LEVEL", "INFO"),
        sqlite_db_path=os.getenv("SQLITE_DB_PATH", ""),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        gemini_image_model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview"),
        streaming_enabled=_env_flag("STREAMING_ENABLED", True),
        show_thoughts=_env_flag("SHOW_THOUGHTS", True),
        kariyer_job_url=os.getenv("KARIYER_JOB_URL", ""),
        kariyernet_username=os.getenv("KARIYERNET_USERNAME", ""),
        kariyernet_password=os.getenv("KARIYERNET_PASSWORD", ""),
    )
