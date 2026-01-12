from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from app.config import get_settings
from app.services.gmail_polling import schedule_gmail_polling

from .handlers import (
    handle_button,
    handle_document_message,
    handle_forget_command,
    handle_google_auth_command,
    handle_inbox_command,
    handle_memory_command,
    handle_media_message,
    handle_pc_command,
    handle_photo_message,
    handle_profile_command,
    handle_profile_clear_command,
    handle_profile_forget_command,
    handle_profile_forget_key_command,
    handle_profile_set_command,
    handle_restart_command,
    handle_reauth_command,
    handle_search_mail_command,
    handle_summarize_last_command,
    handle_draft_mail_command,
    handle_text_message,
    start,
)


async def _post_init(application: Application) -> None:
    settings = get_settings()
    if not settings.telegram_chat_id:
        raise ValueError("TELEGRAM_CHAT_ID is not set")
    await application.bot.send_message(
        chat_id=settings.telegram_chat_id,
        text="Bot initialized successfully",
    )
    schedule_gmail_polling(application)


def build_application() -> Application:
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set")

    application = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .post_init(_post_init)
        .build()
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("pc", handle_pc_command))
    application.add_handler(CommandHandler("memory", handle_memory_command))
    application.add_handler(CommandHandler("forget", handle_forget_command))
    application.add_handler(CommandHandler("profile", handle_profile_command))
    application.add_handler(CommandHandler("profile_forget", handle_profile_forget_command))
    application.add_handler(CommandHandler("profile_set", handle_profile_set_command))
    application.add_handler(CommandHandler("profile_forget_key", handle_profile_forget_key_command))
    application.add_handler(CommandHandler("profile_clear", handle_profile_clear_command))
    application.add_handler(CommandHandler("restart", handle_restart_command))
    application.add_handler(CommandHandler("google_auth", handle_google_auth_command))
    application.add_handler(CommandHandler("google_reauth", handle_reauth_command))
    application.add_handler(CommandHandler("inbox", handle_inbox_command))
    application.add_handler(CommandHandler("summarize_last", handle_summarize_last_command))
    application.add_handler(CommandHandler("search_mail", handle_search_mail_command))
    application.add_handler(CommandHandler("draft_mail", handle_draft_mail_command))
    application.add_handler(CallbackQueryHandler(handle_button))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo_message))
    application.add_handler(
        MessageHandler(filters.AUDIO | filters.VOICE | filters.VIDEO | filters.VIDEO_NOTE, handle_media_message)
    )
    application.add_handler(
        MessageHandler(filters.Document.ALL & ~filters.Document.IMAGE, handle_document_message)
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    return application
