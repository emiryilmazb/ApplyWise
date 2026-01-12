from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv() -> None:
        return None


@dataclass(frozen=True)
class Settings:
    app_name: str = "Atlas"
    log_level: str = "INFO"
    log_style: str = "pretty"
    log_color: bool = True
    httpx_log_path_only: bool = False
    sqlite_db_path: str = ""
    memory_enabled: bool = True
    memory_aggressive_inference: bool = True
    memory_summary_every_n_user_msg: int = 12
    memory_retrieval_top_k: int = 5
    embedding_backend: str = "sentence_transformers"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    anonymity_default_off: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    gemini_image_model: str = "gemini-3-pro-image-preview"
    gemini_enable_google_search: bool = True
    gemini_enable_code_execution: bool = True
    gmail_credentials_path: str = ""
    gmail_token_path: str = ""
    gmail_scopes: str = ""
    gmail_user_id: str = "me"
    gmail_poll_enabled: bool = False
    gmail_poll_interval_seconds: int = 120
    gmail_poll_max_results: int = 10
    gmail_vip_senders: str = ""
    gmail_vip_only: bool = False
    gmail_later_label: str = "Later"
    gmail_focus_mode: bool = False
    gmail_focus_labels: str = "Critical,Action Required"
    gmail_oauth_flow: str = "local"
    gmail_oauth_open_browser: bool = True
    drive_token_path: str = ""
    drive_scopes: str = ""
    docs_token_path: str = ""
    docs_scopes: str = ""
    photos_token_path: str = ""
    photos_scopes: str = ""
    streaming_enabled: bool = True
    show_thoughts: bool = True
    screenshot_enabled: bool = True
    browser_use_enabled: bool = True
    computer_use_enabled: bool = True
    playwright_headless: bool = False
    pc_use_system_browser: bool = True
    pc_browser_name: str = "Chrome"
    pc_browser_window_title: str = ".*Chrome.*"
    pc_browser_user_data_dir: str = ""
    pc_browser_executable_path: str = ""
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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        app_name=os.getenv("ATLAS_APP_NAME", "Atlas"),
        log_level=os.getenv("ATLAS_LOG_LEVEL", "INFO"),
        log_style=os.getenv("ATLAS_LOG_STYLE", "pretty"),
        log_color=_env_flag("ATLAS_LOG_COLOR", True),
        httpx_log_path_only=_env_flag("HTTPX_LOG_PATH_ONLY", False),
        sqlite_db_path=os.getenv("SQLITE_DB_PATH", ""),
        memory_enabled=_env_flag("MEMORY_ENABLED", True),
        memory_aggressive_inference=_env_flag("MEMORY_AGGRESSIVE_INFERENCE", True),
        memory_summary_every_n_user_msg=_env_int("MEMORY_SUMMARY_EVERY_N_USER_MSG", 12),
        memory_retrieval_top_k=_env_int("MEMORY_RETRIEVAL_TOP_K", 5),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "sentence_transformers"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
        anonymity_default_off=_env_flag("ANONYMITY_DEFAULT_OFF", True),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        gemini_image_model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview"),
        gemini_enable_google_search=_env_flag("GEMINI_ENABLE_GOOGLE_SEARCH", True),
        gemini_enable_code_execution=_env_flag("GEMINI_ENABLE_CODE_EXECUTION", True),
        gmail_credentials_path=os.getenv("GMAIL_OAUTH_CREDENTIALS_PATH", ""),
        gmail_token_path=os.getenv("GMAIL_OAUTH_TOKEN_PATH", ""),
        gmail_scopes=os.getenv("GMAIL_SCOPES", ""),
        gmail_user_id=os.getenv("GMAIL_USER_ID", "me"),
        gmail_poll_enabled=_env_flag("GMAIL_POLL_ENABLED", False),
        gmail_poll_interval_seconds=_env_int("GMAIL_POLL_INTERVAL_SECONDS", 120),
        gmail_poll_max_results=_env_int("GMAIL_POLL_MAX_RESULTS", 10),
        gmail_vip_senders=os.getenv("GMAIL_VIP_SENDERS", ""),
        gmail_vip_only=_env_flag("GMAIL_VIP_ONLY", False),
        gmail_later_label=os.getenv("GMAIL_LATER_LABEL", "Later"),
        gmail_focus_mode=_env_flag("GMAIL_FOCUS_MODE", False),
        gmail_focus_labels=os.getenv("GMAIL_FOCUS_LABELS", "Critical,Action Required"),
        gmail_oauth_flow=os.getenv("GMAIL_OAUTH_FLOW", "local"),
        gmail_oauth_open_browser=_env_flag("GMAIL_OAUTH_OPEN_BROWSER", True),
        drive_token_path=os.getenv("DRIVE_OAUTH_TOKEN_PATH", ""),
        drive_scopes=os.getenv("DRIVE_SCOPES", ""),
        docs_token_path=os.getenv("DOCS_OAUTH_TOKEN_PATH", ""),
        docs_scopes=os.getenv("DOCS_SCOPES", ""),
        photos_token_path=os.getenv("PHOTOS_OAUTH_TOKEN_PATH", ""),
        photos_scopes=os.getenv("PHOTOS_SCOPES", ""),
        streaming_enabled=_env_flag("STREAMING_ENABLED", True),
        show_thoughts=_env_flag("SHOW_THOUGHTS", True),
        screenshot_enabled=_env_flag("SCREENSHOT_ENABLED", True),
        browser_use_enabled=_env_flag("BROWSER_USE_ENABLED", True),
        computer_use_enabled=_env_flag("COMPUTER_USE_ENABLED", True),
        playwright_headless=_env_flag("PLAYWRIGHT_HEADLESS", False),
        pc_use_system_browser=_env_flag("PC_USE_SYSTEM_BROWSER", True),
        pc_browser_name=os.getenv("PC_BROWSER_NAME", "Chrome"),
        pc_browser_window_title=os.getenv("PC_BROWSER_WINDOW_TITLE", ".*Chrome.*"),
        pc_browser_user_data_dir=os.getenv("PC_BROWSER_USER_DATA_DIR", ""),
        pc_browser_executable_path=os.getenv("PC_BROWSER_EXECUTABLE_PATH", ""),
        kariyer_job_url=os.getenv("KARIYER_JOB_URL", ""),
        kariyernet_username=os.getenv("KARIYERNET_USERNAME", ""),
        kariyernet_password=os.getenv("KARIYERNET_PASSWORD", ""),
    )
