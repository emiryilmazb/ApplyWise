# Atlas

Atlas is a local, Telegram-first AI agent that automates job applications, desktop workflows, and Google Workspace tasks using Gemini. It runs on your machine, keeps human approvals in the loop, and stores memory in SQLite so it can keep context between sessions.

## What it can do

- Job application automation for computer use with Playwright-driven form parsing, CV-aware answers, and human review for risky or legal questions (LinkedIn adapter and Kariyer.net adapter are a stub/template today).
- PC control workflows with action planning, approvals (YES/NO), optional high-risk confirmation (RUN), and step screenshots.
- Google Workspace assistant capabilities for Gmail, Calendar, Tasks, Drive, Docs, Sheets, People (Contacts), Photos, and YouTube.
- External ATS adapter that can run from a predefined list of fields when you have a custom flow.
- Multimodal Telegram chat: image generation/editing/analysis, document summaries, and audio/video transcription.
- Memory system with summaries and profile facts, plus an anonymous mode to stop new writes.

## Architecture overview

- `app/main.py`: entrypoint; bootstraps models, memory, Telegram, and job flow.
- `app/telegram/`: Telegram bot handlers, commands, streaming, and intent routing.
- `app/pc_agent/`: computer-use planner, executor, and vision capture.
- `app/application/`: job application pipeline and session state machine.
- `app/sites/`: site adapters (Kariyer, LinkedIn stub, external ATS).
- `app/agent/`: Gemini clients, answer generation, routing, and safety.
- `app/services/`: Gmail and Google Workspace integrations.
- `app/storage/`: SQLite database and session persistence.
- `data/`: default SQLite DB and OAuth tokens.
- `artifacts/`: generated/edited images and temporary files (created at runtime).

## Requirements

- Python 3.11+
- Windows (recommended for full PC automation with PyAutoGUI/Pywinauto)
- A Gemini API key (Google AI Studio or Vertex AI)
- Telegram account
- Optional: Google Cloud project with OAuth credentials for Gmail/Drive/Docs/etc.

## Installation

1. Clone and enter the repo.
   ```bash
   git clone <repo-url>
   cd Atlas
   ```

2. Create and activate a virtual environment.
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install Python dependencies.
   ```bash
   pip install -r app/requirements.txt
   ```

4. Install Playwright browsers.
   ```bash
   playwright install
   ```

5. Create your `.env` file from the template.
   ```bash
   copy .env.example .env
   ```

## Docker (headless)

Docker runs Atlas in headless mode. Desktop automation (PyAutoGUI/Pywinauto) is disabled in the
compose file because containers cannot control the host GUI.

1. Copy the environment template and fill in secrets.
   ```bash
   copy .env.example .env
   ```
2. Build and start the container.
   ```bash
   docker compose up --build
   ```

Notes:
- `docker-compose.yml` mounts `data/`, `artifacts/`, and `credentials.json` so tokens and images persist.
- Google OAuth in headless mode uses `GMAIL_OAUTH_FLOW=console` and `GMAIL_OAUTH_OPEN_BROWSER=False`.
- To use full PC automation, run the app directly on Windows instead of in Docker.

## Telegram bot setup (required)

1. Open Telegram and chat with `@BotFather`.
2. Send `/newbot` and follow the prompts to pick a name and username.
3. Copy the bot token provided by BotFather.
4. Start a chat with your new bot and send `/start`.
5. Get your numeric chat id:
   - Option A: Open a chat with `@userinfobot` and copy the id it returns.
   - Option B: After sending a message to your bot, open:
     `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
     and read the `chat.id` value.
6. Set these in `.env`:
   ```ini
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_numeric_chat_id
   ```

Notes:
- For group chats, the chat id is negative. Add the bot to the group, send a message, and read `chat.id` from `getUpdates`.
- The bot will refuse to start without `TELEGRAM_CHAT_ID`.

## Gemini API setup (required)

Set your API key in `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3-flash-preview
GEMINI_IMAGE_MODEL=gemini-3-pro-image-preview
GEMINI_ENABLE_GOOGLE_SEARCH=True
GEMINI_ENABLE_CODE_EXECUTION=True
```

## Google OAuth setup (optional, for Gmail/Drive/Docs/etc.)

1. Create a Google Cloud project and enable the APIs you need:
   - Gmail, Calendar, Tasks, People, Drive, Docs, Sheets, Photos, YouTube Data.
2. Create OAuth credentials (Desktop app) and download the JSON.
3. Place the JSON somewhere in the repo (for example `credentials.json`).
4. Set the path in `.env`:
   ```ini
   GMAIL_OAUTH_CREDENTIALS_PATH=credentials.json
   ```
5. Start the bot and run `/google_auth` in Telegram to complete the OAuth flow.
6. If you change scopes later, run `/google_reauth`.

Token files (defaults):
- `GMAIL_OAUTH_TOKEN_PATH` (default `data/gmail_token.json`)
- `DRIVE_OAUTH_TOKEN_PATH` (default `data/drive_token.json`)
- `DOCS_OAUTH_TOKEN_PATH` (default `data/docs_token.json`)
- `PHOTOS_OAUTH_TOKEN_PATH` (default `data/photos_token.json`)

You can override scopes with `GMAIL_SCOPES`, `DRIVE_SCOPES`, `DOCS_SCOPES`, `PHOTOS_SCOPES`. If you override Gmail scopes, include any extra scopes you want to use (see `.env.example`).
If you run on a headless machine, set `GMAIL_OAUTH_FLOW=console` and `GMAIL_OAUTH_OPEN_BROWSER=False`.

## Configuration tips

Minimal required settings:
```ini
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
GEMINI_API_KEY=
```

Job application settings:
```ini
KARIYER_JOB_URL=
KARIYERNET_USERNAME=
KARIYERNET_PASSWORD=
```

Automation toggles:
```ini
STREAMING_ENABLED=True
SHOW_THOUGHTS=True
SCREENSHOT_ENABLED=True
BROWSER_USE_ENABLED=True
COMPUTER_USE_ENABLED=True
PLAYWRIGHT_HEADLESS=False
```

PC browser control (optional):
```ini
PC_USE_SYSTEM_BROWSER=True
PC_BROWSER_NAME=Chrome
PC_BROWSER_WINDOW_TITLE=.*Chrome.*
PC_BROWSER_USER_DATA_DIR=
PC_BROWSER_EXECUTABLE_PATH=
```

Memory settings:
```ini
MEMORY_ENABLED=True
MEMORY_SUMMARY_EVERY_N_USER_MSG=12
MEMORY_RETRIEVAL_TOP_K=5
EMBEDDING_BACKEND=sentence_transformers
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
ANONYMITY_DEFAULT_OFF=True
SQLITE_DB_PATH=
```

Gmail polling (optional):
```ini
GMAIL_POLL_ENABLED=False
GMAIL_POLL_INTERVAL_SECONDS=120
GMAIL_POLL_MAX_RESULTS=10
GMAIL_VIP_SENDERS=
GMAIL_VIP_ONLY=False
GMAIL_LATER_LABEL=Later
GMAIL_FOCUS_MODE=False
GMAIL_FOCUS_LABELS=Critical,Action Required
```

## Usage

Start the agent:
```bash
python -m app.main
```

On startup, the bot sends a "Job Search" button. Selecting it launches the Kariyer.net flow.

Examples:
- `job_search` starts the job search workflow.
- `job_search stop` stops the job search workflow.
- `/pc open chrome and search for python jobs` runs a PC control task (requires approval).

## Job application flow notes

- The Kariyer.net flow searches using a derived query from your CV profile and then fills visible form fields.
- For better auto-answers, populate the CV profile tables in SQLite (`data/atlas.db`, tables prefixed with `cv_`).
- If a form asks about legal/work authorization or other high-risk fields, the agent will ask you to answer in Telegram.
- `KARIYER_JOB_URL` exists in config but is not used by the current flow.

## PC control approval flow

1. The agent proposes an action plan and asks for approval in Telegram.
2. Reply `YES` to proceed or `NO` to cancel. You can also reply with a revision request to refine the plan.
3. If a step involves running executables, you will be asked to confirm with `RUN`.

## Images and media

- Generate images by sending a prompt (no attachment required).
- Edit or analyze an image by replying to the image with your instructions.
- Supported image types: PNG, JPEG, WEBP, HEIC, HEIF (Gemini limits apply).
- Documents (PDF, TXT, DOCX, PPTX, XLSX, ODT, RTF) and common audio/video formats can be summarized in chat.

## Command reference

- `clear_history` or `/clear_history`: clears chat history.
- `thinking_on` / `thinking_off`: toggles thought streaming.
- `screenshot on` / `screenshot off`: toggles post-step screenshots.
- `browser on` / `browser off`: toggles browser automation.
- `pc on` / `pc off`: toggles computer control.
- `anonymous on` / `anonymous off`: disables/enables chat logging.
- `/memory`: lists stored memory summaries.
- `/forget <id>`: deletes the selected memory item.
- `/profile`: shows stored profile facts.
- `/profile_forget <id>`: deletes the selected profile fact.
- `/profile_set <key> | <value>`: stores a profile fact.
- `/profile_forget_key <key> [| <value>]`: deletes matching profile facts.
- `/profile_clear`: clears all stored profile facts.
- `/google_auth`: authorizes Google access.
- `/google_reauth`: reauthorizes Google access with updated scopes.
- `/inbox [n]`: lists unread emails.
- `/summarize_last [n]`: summarizes unread emails.
- `/search_mail <query>`: searches Gmail.
- `/draft_mail <to> | <subject> | <prompt>`: creates a Gmail draft.
- `job_search`: starts job search (button).
- `job_search stop`: stops job search.
- `stop`: stops all actions.
- `/pc <task>` or `pc: <task>`: starts a PC control task.

## Data and storage

- SQLite DB: `data/atlas.db` by default (override with `SQLITE_DB_PATH`).
- Memory summaries and profile facts live in the SQLite database.
- OAuth tokens live in `data/` unless overridden.
- Generated and edited images are saved under `artifacts/`.

For memory details, see `MEMORY.md`.

## Tests

```bash
pytest
```

## Disclaimer

This tool automates interactions with third-party websites and software. Make sure you comply with each platform's Terms of Service. The developers are not responsible for account restrictions or bans resulting from use.

## License

MIT License
