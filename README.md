# Atlas

**The Next-Generation AI Agent for Autonomous Computer Use**

Atlas is a cutting-edge Python-based AI agent designed to interact with your computer just like a human does. Powered by advanced multimodal Large Language Models (LLMs) like Google Gemini, Atlas "sees" your screen, understands complex user interfaces, and autonomously navigates through desktop and web applications to execute tasks with precision.

While its core architecture is built for **General Computer Use**, Atlas shines brightest in its ability to automate one of the most tedious tasks in modern life: **Job Applications**.

## 🚀 The Vision: True Computer Use

Atlas isn't just a script; it's an intelligent agent that bridges the gap between AI and your operating system. By leveraging **Vision-Language Models (VLMs)**, it breaks free from fragile DOM selectors and API limitations, interacting with applications visually and logically.

### Core Capabilities

- **👀 Visual Perception:** Uses state-of-the-art vision models to analyze screen content, identifying buttons, forms, and context even when underlying code is obfuscated.
- **🖱️ Human-Like Interaction:** Controls mouse and keyboard inputs (PyAutoGUI, Pywinauto) to click, type, scroll, and navigate seamlessly across different applications.
- **🧠 Intelligent Planning:** Deconstructs high-level goals into actionable steps, adapting to unexpected pop-ups, security checks, or UI changes dynamically.
- **🤖 PC Agent Mode:** Extends beyond the browser to manage desktop apps. It can control **Spotify** playback, send messages on **WhatsApp Desktop**, and more, demonstrating true OS-level agency.

## 🌟 Feature Spotlight: Autonomous Job Applications

Atlas leverages its powerful Computer Use foundation to revolutionize the job hunt. It acts as your tireless personal recruiter working 24/7.

- **Smart Search & Match:** Automatically scans platforms like **Kariyer.net** (with architecture ready for LinkedIn) to find roles matching your specific CV profile and experience.
- **Auto-Apply Workflow:** Navigates application forms, fills in details, uploads resumes, and handles multi-step submissions without lifting a finger.
- **Human-in-the-Loop:** Keeps you in control via a **Telegram Bot** interface. The agent reports progress and asks for your decision on critical steps or ambiguity (e.g., specific security verifications), ensuring safety and accuracy.

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- Windows OS (Recommended for full Desktop Automation features)
- A Google Cloud Project with Gemini API access

### Setup

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/atlas.git
    cd atlas
    ```

2.  **Create a Virtual Environment**

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r app/requirements.txt
    ```

4.  **Install Playwright Browsers**

    ```bash
    playwright install
    ```

5.  **Configure Environment**
    Copy `.env.example` to `.env` and fill in your credentials:

    ```ini
    ATLAS_APP_NAME=Atlas

    # Telegram Configuration
    TELEGRAM_BOT_TOKEN=your_telegram_bot_token
    TELEGRAM_CHAT_ID=your_telegram_chat_id

    # Obtaining Telegram Credentials:
    # 1. Start a chat with @BotFather on Telegram.
    # 2. Send /newbot to create a new bot and get your TELEGRAM_BOT_TOKEN.
    # 3. Start a chat with @userinfobot to get your numeric TELEGRAM_CHAT_ID.

    # Google Gemini API
    GEMINI_API_KEY=your_gemini_api_key
    GEMINI_MODEL=gemini-3-flash-preview
    GEMINI_IMAGE_MODEL=gemini-3-pro-image-preview

    # Platform Credentials (Optional)
    KARIYERNET_USERNAME=email@example.com
    KARIYERNET_PASSWORD=your_password
    ```

## ▶️ Usage

Start the agent:

```bash
python -m app.main
```

Once running, the agent will message you on Telegram: _"Atlas is online. System integrity confirmed. Gmail layers are synced, and the Gemini core is standing by. I've indexed your latest communications and prepared your digital workspace. Your Chief of Staff is ready to orchestrate.How shall we direct our focus today?"_

You can command it to start the **"Job Search"** workflow or interact with its other PC agent capabilities directly through the chat interface.

## Memory and Privacy

Atlas can store long-term conversation summaries and a lightweight user profile to improve recall across days. Anonymous mode (`anonim on`) stops new message and memory writes. See `MEMORY.md` for details and configuration.

## 📂 Architecture

- `app/pc_agent/`: The heart of the Computer Use system. Handles vision processing, screen capture, and low-level mouse/keyboard execution.
- `app/agent/`: The brain. Manages state, intent classification, and decision-making logic using LLMs.
- `app/sites/`: Platform-specific adapters that guide the agent on specific websites (e.g., Kariyer.net).
- `app/telegram/`: The remote command center. Allows you to communicate with the agent from anywhere.

## ⚠️ Disclaimer

This tool automates interactions with third-party websites and software. Please ensure you comply with the Terms of Service (ToS) of all platforms you use. The developers are not responsible for any account restrictions or bans resulting from the use of this tool.

## 📄 License

MIT License
