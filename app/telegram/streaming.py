from __future__ import annotations

from dataclasses import dataclass
import asyncio
import logging
import random
import re
import time
from typing import Awaitable, Callable, Optional

from telegram.constants import ParseMode
from telegram.error import BadRequest, TelegramError

from app.agent.llm_client import StreamChunk

logger = logging.getLogger(__name__)

_DEFAULT_MIN_INTERVAL = 0.8
_DEFAULT_MAX_INTERVAL = 1.2
_DEFAULT_MIN_CHARS = 120
_DEFAULT_MAX_LENGTH = 3500


@dataclass
class StreamResult:
    thought_text: str
    response_text: str
    completed: bool
    error: Exception | None = None
    recovered: bool = False


class MessageManager:
    def __init__(
        self,
        bot,
        chat_id: str,
        message_id: int,
        *,
        parse_mode: ParseMode = ParseMode.HTML,
        min_interval: float = _DEFAULT_MIN_INTERVAL,
        max_interval: float = _DEFAULT_MAX_INTERVAL,
        min_chars: int = _DEFAULT_MIN_CHARS,
    ) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._message_id = message_id
        self._parse_mode = parse_mode
        self._min_interval = min_interval
        self._max_interval = max_interval
        self._min_chars = min_chars
        self._lock = asyncio.Lock()
        self._pending_text: str | None = None
        self._last_sent_text = ""
        self._last_sent_at = 0.0
        self._target_interval = self._pick_interval()
        self._flush_task: asyncio.Task | None = None

    async def update(self, text: str, *, force: bool = False) -> None:
        text = text.strip()
        if not text:
            return
        async with self._lock:
            self._pending_text = text
            if force:
                await self._flush_locked()
                return
            now = time.monotonic()
            if self._should_flush(now, text):
                await self._flush_locked()
                return
            if self._flush_task is None or self._flush_task.done():
                delay = max(0.05, self._target_interval - (now - self._last_sent_at))
                self._flush_task = asyncio.create_task(self._delayed_flush(delay))

    async def finalize(self, text: str, *, reply_markup=None) -> None:
        text = text.strip()
        if not text:
            return
        async with self._lock:
            self._pending_text = text
            await self._flush_locked(reply_markup=reply_markup)
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
                self._flush_task = None

    async def _delayed_flush(self, delay: float) -> None:
        await asyncio.sleep(delay)
        async with self._lock:
            await self._flush_locked()

    def _should_flush(self, now: float, text: str) -> bool:
        if self._last_sent_at == 0.0:
            return True
        if now - self._last_sent_at >= self._target_interval:
            return True
        if len(text) - len(self._last_sent_text) >= self._min_chars:
            return True
        return False

    def _pick_interval(self) -> float:
        return random.uniform(self._min_interval, self._max_interval)

    async def _flush_locked(self, *, reply_markup=None) -> None:
        if not self._pending_text:
            return
        text = self._pending_text
        if text == self._last_sent_text and reply_markup is None:
            return
        try:
            await self._bot.edit_message_text(
                chat_id=self._chat_id,
                message_id=self._message_id,
                text=text,
                parse_mode=self._parse_mode,
                reply_markup=reply_markup,
            )
        except BadRequest as exc:
            if "Message is not modified" in str(exc):
                return
            logger.warning("Telegram edit failed: %s", exc)
        except TelegramError as exc:
            logger.warning("Telegram edit failed: %s", exc)
        else:
            self._last_sent_text = text
            self._last_sent_at = time.monotonic()
            self._target_interval = self._pick_interval()


class AsyncStreamHandler:
    def __init__(
        self,
        message_manager: MessageManager,
        *,
        show_thoughts: bool = True,
        thought_label: str = "🧠 Thoughts",
        response_label: str = "Decision",
        response_as_code: bool = False,
        max_length: int = _DEFAULT_MAX_LENGTH,
    ) -> None:
        self._message_manager = message_manager
        self._show_thoughts = show_thoughts
        self._thought_label = thought_label
        self._response_label = response_label
        self._response_as_code = response_as_code
        self._max_length = max_length
        self._thought_text = ""
        self._response_text = ""
        self._notice: str | None = None

    @property
    def thought_text(self) -> str:
        return self._thought_text

    @property
    def response_text(self) -> str:
        return self._response_text

    async def stream(self, llm_client, prompt: str, *, include_thoughts: bool = True) -> StreamResult:
        return await self._stream_from_chunks(
            llm_client.stream_text(prompt, include_thoughts=include_thoughts),
            recovery=lambda: self._attempt_recovery(llm_client, prompt),
        )

    async def stream_chunks(
        self,
        chunks,
        *,
        recovery: Optional[Callable[[], Awaitable[str]]] = None,
    ) -> StreamResult:
        return await self._stream_from_chunks(chunks, recovery=recovery)

    async def _stream_from_chunks(
        self,
        chunks,
        *,
        recovery: Optional[Callable[[], Awaitable[str]]],
    ) -> StreamResult:
        completed = False
        error: Exception | None = None
        recovered = False
        try:
            async for chunk in chunks:
                self._append_chunk(chunk)
                await self._message_manager.update(self.format_message())
            completed = True
        except Exception as exc:
            logger.warning("Stream interrupted: %s", exc)
            error = exc
            self._notice = "Stream interrupted. Showing partial response."
            if recovery is not None:
                recovered_text = await recovery()
                if recovered_text:
                    self._response_text += recovered_text
                    recovered = True
                    self._notice = "Stream interrupted. Recovered with a fallback response."
        await self._message_manager.update(self.format_message(), force=True)
        return StreamResult(
            thought_text=self._thought_text,
            response_text=self._response_text,
            completed=completed or recovered,
            error=error,
            recovered=recovered,
        )

    async def _attempt_recovery(self, llm_client, prompt: str) -> str:
        if not hasattr(llm_client, "generate_text"):
            return ""
        if not prompt:
            return ""
        tail = self._response_text.strip()[-600:]
        recovery_prompt = (
            f"{prompt}\n\n"
            "The response stream was interrupted. Continue the answer from the partial response "
            "below without repeating it.\n\n"
            f"Partial response:\n{tail}"
        )
        try:
            return await asyncio.to_thread(llm_client.generate_text, recovery_prompt)
        except Exception as exc:
            logger.warning("Stream recovery failed: %s", exc)
            return ""

    def _append_chunk(self, chunk: StreamChunk) -> None:
        if chunk.is_thought and self._show_thoughts:
            self._thought_text += chunk.text
            return
        if not chunk.is_thought:
            self._response_text += chunk.text

    def format_message(self) -> str:
        thought_text = self._trim_text(self._thought_text, max_chars=1100)
        response_text = self._trim_text(
            self._response_text,
            max_chars=3000 if not thought_text else 2200,
        )
        blocks: list[str] = []
        if self._show_thoughts and thought_text:
            blocks.append(f"{self._thought_label}\n<pre>{_escape_html(thought_text)}</pre>")
        if response_text:
            if self._response_as_code:
                body = f"<pre>{_escape_html(response_text)}</pre>"
            else:
                body = _format_stream_html(response_text)
            label = _escape_html(self._response_label)
            if label:
                blocks.append(f"<b>{label}</b>\n{body}")
            else:
                blocks.append(body)
        if self._notice:
            blocks.append(f"⚠️ {_escape_html(self._notice)}")
        if not blocks:
            return "⏳ Thinking..."
        formatted = "\n\n".join(blocks)
        if len(formatted) > self._max_length:
            formatted = "\n\n".join(blocks[:2])
        return formatted.strip()

    @staticmethod
    def _trim_text(text: str, *, max_chars: int) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3]}..."


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _format_stream_html(text: str) -> str:
    escaped = _escape_html(text)
    lines = escaped.splitlines(keepends=True)
    formatted: list[str] = []
    for line in lines:
        newline = "\n" if line.endswith("\n") else ""
        content = line[:-1] if newline else line
        heading = re.match(r"^(#{1,6})\s+(.*)$", content)
        if heading:
            heading_text = heading.group(2).strip()
            heading_text = _apply_inline_markdown(heading_text)
            content = f"<b>{heading_text}</b>" if heading_text else ""
        else:
            content = _apply_inline_markdown(content)
        formatted.append(content + newline)
    return "".join(formatted)


def _apply_inline_markdown(text: str) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    formatted_parts: list[str] = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) >= 2:
            inner = part[1:-1]
            formatted_parts.append(f"<code>{inner}</code>")
            continue
        styled = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", part)
        styled = re.sub(r"(?<!\*)\*([^\s][^*]*?[^\s])\*(?!\*)", r"<i>\1</i>", styled)
        formatted_parts.append(styled)
    return "".join(formatted_parts)
