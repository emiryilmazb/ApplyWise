from __future__ import annotations

import json
import logging
import math
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from app.memory.embeddings import (
    NoOpEmbedder,
    TfidfEmbedder,
    batch_cosine_similarity,
    build_embedder,
    cosine_similarity,
    deserialize_embedding,
    normalize_text,
    serialize_embedding,
)
from app.storage.database import MemoryItemRow

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """
Sen bir sohbet ozetleyicisisin. Asagidaki mesajlari ozetle.

Kurallar:
- Sadece verilen mesajlari kullan, yeni bilgi uydurma.
- Kesin olmayan bilgiyi kesinmis gibi yazma.
- JSON disinda hicbir sey yazma.

Mesajlar:
{messages}

JSON formati:
{{
  "title": "...",
  "summary": "5-10 cumle",
  "key_facts": ["...", "..."],
  "open_loops": ["...", "..."],
  "tags": ["...", "..."],
  "importance": 0.0
}}
""".strip()

_PROFILE_PROMPT = """
Kullanicinin mesaji, ileride tekrar referans edilebilecek kalici bir bilgi iceriyorsa kaydet.

Kurallar:
- Sadece mesajda gecen bilgileri kaydet, yeni bilgi uydurma.
- Kanit (evidence) alani, mesajdan birebir alinti olsun.
- Tercihler, ilgi alanlari, begenmedikleri veya kacinmak istedikleri seyler acikca
  soylenecek kadar net olmasa da ima ediliyorsa, dusuk confidence ile kaydet.
  Kimlik/konum gibi hassas dahil kaydet.
- Degisim/iptal ifadesi varsa removals listesine yaz.
- Kaydedilecek bir sey yoksa bos listeler dondur.
- JSON disinda hicbir sey yazma.

Profil anahtarlari (ornekler):
- identity.name | identity.language | identity.timezone | identity.location
- preferences.response_style.format | preferences.response_style.length | preferences.response_style.emojis
- preferences.communication.tone | preferences.communication.language
- preferences.work.type | preferences.work.location | preferences.task_focus
- interests.topic | interests.company | interests.role | interests.industry
- dislikes.topic | dislikes.company | dislikes.role
- constraints.avoid | constraints.never

Mevcut profil:
{profile_json}

Mesaj:
{message}

JSON formati:
{{
  "facts": [
    {{
      "key": "...",
      "value": "...",
      "evidence": "...",
      "importance": 0.0,
      "confidence": 0.0,
      "inferred": false,
      "ttl_days": 0
    }}
  ],
  "removals": [
    {{
      "key": "...",
      "value": "...",
      "reason": "..."
    }}
  ]
}}
""".strip()

_PROFILE_VERSION = 2
_ALLOWED_PROFILE_KEY_PREFIXES = (
    "identity.",
    "preferences.",
    "interests.",
    "dislikes.",
    "constraints.",
)
_EXCLUSIVE_PROFILE_KEY_PREFIXES = (
    "identity.",
    "preferences.response_style.",
    "preferences.communication.",
    "preferences.work.",
    "preferences.task_focus",
)
_INFERRED_ALLOWED_PREFIXES = (
    "preferences.",
    "interests.",
    "dislikes.",
    "constraints.",
)
_INFERRED_CONFIDENCE_THRESHOLD = 0.65
_INFERRED_TTL_DAYS = 45
_PROFILE_OVERRIDE_TOKENS = (
    "now",
    "actually",
    "instead",
    "from now on",
    "no longer",
    "anymore",
    "artik",
    "bundan sonra",
    "aslinda",
    "degisti",
    "degistirdim",
    "vazgectim",
    "tercihim",
)

_PROFILE_PATTERNS = ()

_DEFAULT_MAX_CONTEXT_CHARS = 2000


class ConversationMemoryManager:
    def __init__(self, settings) -> None:
        self._enabled = bool(getattr(settings, "memory_enabled", True))
        self._aggressive_inference = bool(
            getattr(settings, "memory_aggressive_inference", False)
        )
        self._summary_every = int(getattr(settings, "memory_summary_every_n_user_msg", 12))
        self._top_k = int(getattr(settings, "memory_retrieval_top_k", 5))
        backend = getattr(settings, "embedding_backend", "sentence_transformers")
        model_name = getattr(settings, "embedding_model_name", "all-MiniLM-L6-v2")
        self._embedder = build_embedder(backend, model_name)
        self._message_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def summary_window(self) -> int:
        return max(4, self._summary_every * 2)

    def record_user_message(self, user_key: str) -> bool:
        if not self._enabled or not user_key:
            return False
        with self._lock:
            count = self._message_counts.get(user_key, 0) + 1
            self._message_counts[user_key] = count
        return self._summary_every > 0 and count % self._summary_every == 0

    def reset_user_counter(self, user_key: str) -> None:
        if not user_key:
            return
        with self._lock:
            self._message_counts[user_key] = 0

    def build_profile_snippet(self, profile_json: str | None, max_chars: int = 600) -> str:
        if not profile_json:
            return ""
        try:
            profile = json.loads(profile_json)
        except Exception:
            return ""
        if not isinstance(profile, dict):
            return ""
        facts = _prune_expired_facts(profile.get("facts"))
        lines = _format_profile_facts(facts)
        snippet = "\n".join(lines).strip() if lines else ""
        if not snippet:
            return ""
        if len(snippet) > max_chars:
            return snippet[:max_chars].rstrip()
        return snippet

    def retrieve_memory_block(
        self,
        items: list[MemoryItemRow],
        query_text: str,
        max_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
    ) -> tuple[str, list[int]]:
        if not items or not query_text:
            return "", []
        if isinstance(self._embedder, NoOpEmbedder):
            return "", []
        normalized_query = query_text.strip()
        if not normalized_query:
            return "", []
        scores: list[tuple[float, MemoryItemRow]] = []
        if isinstance(self._embedder, TfidfEmbedder):
            corpus = [item.summary for item in items]
            query_vec, corpus_vecs = self._embedder.embed_with_corpus(normalized_query, corpus)
            sims = batch_cosine_similarity(query_vec, corpus_vecs)
            for item, sim in zip(items, sims):
                scores.append((self._score_item(sim, item), item))
        else:
            query_vec = self._embedder.embed_text(normalized_query)
            for item in items:
                if not item.embedding:
                    continue
                item_vec = deserialize_embedding(item.embedding)
                if item_vec.size != query_vec.size:
                    continue
                sim = cosine_similarity(query_vec, item_vec)
                scores.append((self._score_item(sim, item), item))
        if not scores:
            return "", []
        scores.sort(key=lambda pair: pair[0], reverse=True)
        chosen: list[MemoryItemRow] = []
        for _, item in scores[: self._top_k]:
            chosen.append(item)
        block = _format_memory_block(chosen, max_chars=max_chars)
        return block, [item.id for item in chosen]

    def summarize_and_store(
        self,
        *,
        db,
        llm_client,
        user_key: str,
        chat_id: str | None,
        history_limit: int,
    ) -> bool:
        if not self._enabled or not llm_client or not user_key:
            return False
        rows = db.get_recent_messages(user_key, history_limit)
        messages_text = _format_messages_for_summary(rows)
        if not messages_text:
            return False
        prompt = _SUMMARY_PROMPT.format(messages=messages_text)
        try:
            response = llm_client.generate_text(prompt)
        except Exception as exc:
            logger.warning("Memory summary generation failed: %s", exc)
            return False
        payload = extract_json_payload(response)
        if not payload:
            return False
        summary = str(payload.get("summary") or "").strip()
        if not summary:
            return False
        title = str(payload.get("title") or "").strip() or None
        key_facts = _normalize_string_list(payload.get("key_facts"))
        open_loops = _normalize_string_list(payload.get("open_loops"))
        tags = _normalize_string_list(payload.get("tags"))
        importance = _normalize_importance(payload.get("importance"))
        summary_text = _compose_summary_text(summary, key_facts, open_loops)
        embedding_payload = None
        if not isinstance(self._embedder, TfidfEmbedder):
            try:
                embedding_payload = serialize_embedding(
                    self._embedder.embed_text(_compose_embedding_text(title, summary_text, tags))
                )
            except Exception as exc:
                logger.warning("Embedding failed: %s", exc)
                embedding_payload = None
        tags_json = json.dumps(tags, ensure_ascii=True) if tags else None
        db.add_memory_item(
            user_key=user_key,
            chat_id=chat_id,
            kind="summary",
            title=title,
            summary=summary_text,
            tags=tags_json,
            embedding=embedding_payload,
            importance=importance,
        )
        return True

    def maybe_update_profile(
        self,
        *,
        db,
        llm_client,
        user_key: str,
        message_text: str,
    ) -> bool:
        if not self._enabled or not llm_client or not user_key or not message_text:
            return False
        if not _should_update_profile(message_text):
            return False
        existing = db.get_user_profile(user_key) or "{}"
        prompt = _PROFILE_PROMPT.format(
            profile_json=existing,
            message=message_text.strip(),
        )
        try:
            response = llm_client.generate_text(prompt)
        except Exception as exc:
            logger.warning("Profile update generation failed: %s", exc)
            return False
        payload = extract_json_payload(response)
        if not payload or not isinstance(payload, dict):
            return False
        facts = _normalize_profile_facts(payload)
        removals = _normalize_profile_removals(payload)
        if not facts and not removals:
            return False
        return self._apply_profile_update(
            db=db,
            user_key=user_key,
            facts=facts,
            removals=removals,
            message_text=message_text,
            source="llm",
            allow_missing_evidence=self._aggressive_inference,
        )

    def apply_profile_update(
        self,
        *,
        db,
        user_key: str,
        facts: list[dict[str, Any]] | None = None,
        removals: list[dict[str, Any]] | None = None,
        message_text: str = "",
        source: str = "user",
        allow_missing_evidence: bool = True,
    ) -> bool:
        if not user_key:
            return False
        return self._apply_profile_update(
            db=db,
            user_key=user_key,
            facts=facts or [],
            removals=removals or [],
            message_text=message_text,
            source=source,
            allow_missing_evidence=allow_missing_evidence,
        )

    def clear_profile(self, *, db, user_key: str) -> bool:
        if not user_key:
            return False
        payload = {"version": _PROFILE_VERSION, "facts": []}
        db.set_user_profile(user_key, json.dumps(payload, ensure_ascii=True))
        return True

    def _apply_profile_update(
        self,
        *,
        db,
        user_key: str,
        facts: list[dict[str, Any]],
        removals: list[dict[str, Any]],
        message_text: str,
        source: str,
        allow_missing_evidence: bool,
    ) -> bool:
        existing = db.get_user_profile(user_key) or "{}"
        try:
            base = json.loads(existing) if existing else {}
        except Exception:
            base = {}
        if not isinstance(base, dict):
            base = {}
        base, migrated = _migrate_legacy_profile(base)
        pruned = _prune_expired_facts(base.get("facts"))
        updated_facts = _apply_profile_removals(pruned, removals)
        updated_facts = _merge_profile_facts(
            updated_facts,
            facts,
            message_text,
            allow_missing_evidence=allow_missing_evidence,
            source_override=source,
        )
        if updated_facts == pruned and not removals and not migrated:
            return False
        base["facts"] = updated_facts
        base["version"] = _PROFILE_VERSION
        base = _trim_profile_facts(base)
        db.set_user_profile(user_key, json.dumps(base, ensure_ascii=True))
        return True

    def _score_item(self, sim: float, item: MemoryItemRow) -> float:
        importance = float(item.importance or 0.0)
        recency = _recency_boost(item.created_at)
        return 0.75 * sim + 0.15 * importance + 0.10 * recency


def extract_json_payload(raw_text: str | None) -> dict[str, Any] | None:
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
    extracted = _extract_first_json_object(cleaned)
    if not extracted:
        return None
    try:
        parsed = json.loads(extracted)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _extract_first_json_object(text: str) -> str | None:
    start = None
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : idx + 1]
    return None


def _format_messages_for_summary(rows: list) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    for entry in reversed(rows):
        role = str(getattr(entry, "role", "") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        label = "User" if role == "user" else "Assistant"
        content = str(getattr(entry, "content", "") or "").strip()
        if not content and getattr(entry, "file_path", None):
            content = "[attachment]"
        if not content:
            continue
        lines.append(f"{label}: {content}")
    return "\n".join(lines).strip()


def _compose_summary_text(summary: str, key_facts: list[str], open_loops: list[str]) -> str:
    parts = [summary.strip()]
    if key_facts:
        parts.append("Key facts:")
        parts.extend(f"- {item}" for item in key_facts)
    if open_loops:
        parts.append("Open loops:")
        parts.extend(f"- {item}" for item in open_loops)
    return "\n".join(part for part in parts if part).strip()


def _compose_embedding_text(title: str | None, summary: str, tags: list[str]) -> str:
    parts: list[str] = []
    if title:
        parts.append(title)
    if summary:
        parts.append(summary)
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    return "\n".join(parts).strip()


def _normalize_string_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    return []


def _normalize_importance(value: Any) -> float:
    try:
        importance = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, importance))


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.6
    return max(0.0, min(1.0, confidence))


def _is_allowed_profile_key(key: str) -> bool:
    cleaned = key.strip().lower()
    if not cleaned:
        return False
    return any(cleaned.startswith(prefix) for prefix in _ALLOWED_PROFILE_KEY_PREFIXES)


def _is_allowed_profile_removal_key(key: str) -> bool:
    cleaned = key.strip().lower()
    if not cleaned:
        return False
    if _is_allowed_profile_key(cleaned):
        return True
    return any(cleaned == prefix.rstrip(".") for prefix in _ALLOWED_PROFILE_KEY_PREFIXES)


def _is_exclusive_profile_key(key: str) -> bool:
    cleaned = key.strip().lower()
    if not cleaned:
        return False
    return any(cleaned.startswith(prefix) for prefix in _EXCLUSIVE_PROFILE_KEY_PREFIXES)


def _is_inferred_key_allowed(key: str, confidence: float) -> bool:
    cleaned = key.strip().lower()
    if not cleaned:
        return False
    if confidence < _INFERRED_CONFIDENCE_THRESHOLD:
        return False
    return any(cleaned.startswith(prefix) for prefix in _INFERRED_ALLOWED_PREFIXES)


def _is_override_signal(message_text: str) -> bool:
    normalized = normalize_text(message_text or "")
    if not normalized:
        return False
    return any(token in normalized for token in _PROFILE_OVERRIDE_TOKENS)


def _fact_score(item: dict[str, Any]) -> float:
    confidence = _normalize_confidence(item.get("confidence"))
    importance = _normalize_importance(item.get("importance"))
    return 0.7 * confidence + 0.3 * importance


def _should_replace_fact(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    override: bool,
) -> bool:
    if override:
        return True
    return _fact_score(candidate) >= _fact_score(existing) + 0.05


def _recency_boost(created_at: str | None) -> float:
    if not created_at:
        return 0.5
    dt = _parse_datetime(created_at)
    if not dt:
        return 0.5
    delta = datetime.now(timezone.utc) - dt
    days = max(delta.total_seconds() / 86400.0, 0.0)
    return float(math.exp(-days / 30.0))


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _should_update_profile(message_text: str) -> bool:
    normalized = normalize_text(message_text)
    if not normalized:
        return False
    if _PROFILE_PATTERNS:
        for pattern in _PROFILE_PATTERNS:
            if re.search(pattern, normalized):
                return True
        return False
    return bool(re.search(r"[a-zA-Z\u00c7\u011e\u0130\u015e\u00d6\u00dc\u00e7\u011f\u0131\u015f\u00f6\u00fc]", normalized))


def _merge_dicts(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_profile_facts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    facts = payload.get("facts", [])
    if not isinstance(facts, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        if not key or not value:
            continue
        if not _is_allowed_profile_key(key):
            continue
        importance = _normalize_importance(item.get("importance"))
        confidence = _normalize_confidence(item.get("confidence"))
        ttl_days = _normalize_ttl(item.get("ttl_days"))
        inferred = bool(item.get("inferred"))
        normalized.append(
            {
                "key": key,
                "value": value,
                "evidence": evidence,
                "importance": importance,
                "confidence": confidence,
                "inferred": inferred,
                "ttl_days": ttl_days,
            }
        )
    return normalized


def _normalize_profile_removals(payload: dict[str, Any]) -> list[dict[str, str]]:
    removals = payload.get("removals", [])
    if not isinstance(removals, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in removals:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        if not key:
            continue
        if not _is_allowed_profile_removal_key(key):
            continue
        normalized.append({"key": key, "value": value})
    return normalized


def _merge_profile_facts(
    existing: Any,
    new_facts: list[dict[str, Any]],
    message_text: str,
    *,
    allow_missing_evidence: bool = False,
    source_override: str | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(existing, list):
        existing_list: list[dict[str, Any]] = []
    else:
        existing_list = [dict(item) for item in existing if isinstance(item, dict)]
    message_lower = (message_text or "").lower()
    override = _is_override_signal(message_text)
    updated = list(existing_list)
    now = datetime.now(timezone.utc).isoformat()
    for item in new_facts:
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        if not key or not value:
            continue
        if not _is_allowed_profile_key(key):
            continue
        confidence = _normalize_confidence(item.get("confidence"))
        value_lower = value.lower()
        value_in_message = value_lower in message_lower
        evidence_in_message = bool(evidence) and evidence.lower() in message_lower
        allow_inferred = allow_missing_evidence and _is_inferred_key_allowed(
            key, confidence
        )
        inferred = False
        if evidence_in_message:
            pass
        elif value_in_message:
            evidence = value
        else:
            if not allow_inferred:
                continue
            inferred = True
        if not evidence and not inferred:
            continue
        existing_match = _find_fact(updated, key, value)
        if existing_match is not None:
            existing_match["last_seen_at"] = now
            existing_match["importance"] = max(
                _normalize_importance(existing_match.get("importance")),
                _normalize_importance(item.get("importance")),
            )
            existing_match["confidence"] = max(
                _normalize_confidence(existing_match.get("confidence")),
                confidence,
            )
            existing_match["ttl_days"] = max(
                _normalize_ttl(existing_match.get("ttl_days")),
                _normalize_ttl(item.get("ttl_days")),
            )
            continue
        if _is_exclusive_profile_key(key):
            existing_same_key = _find_fact_by_key(updated, key)
            if existing_same_key is not None:
                if not _should_replace_fact(existing_same_key, item, override):
                    continue
                updated = [fact for fact in updated if not _keys_match(fact.get("key"), key)]
        ttl_days = _normalize_ttl(item.get("ttl_days"))
        if inferred and ttl_days == 0:
            ttl_days = _INFERRED_TTL_DAYS
        if inferred:
            confidence = min(confidence, 0.85)
        entry = {
            "id": uuid4().hex,
            "key": key,
            "value": value,
            "evidence": evidence,
            "importance": _normalize_importance(item.get("importance")),
            "confidence": confidence,
            "inferred": inferred,
            "ttl_days": ttl_days,
            "created_at": now,
            "last_seen_at": now,
            "source": source_override or "llm",
        }
        updated.append(entry)
    return updated


def _find_fact(items: list[dict[str, Any]], key: str, value: str) -> dict[str, Any] | None:
    key_lower = key.strip().lower()
    value_lower = value.strip().lower()
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("key") or "").strip().lower() != key_lower:
            continue
        if str(item.get("value") or "").strip().lower() != value_lower:
            continue
        return item
    return None


def _fact_exists(items: list[dict[str, Any]], key: str, value: str) -> bool:
    return _find_fact(items, key, value) is not None


def _find_fact_by_key(items: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    key_lower = key.strip().lower()
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("key") or "").strip().lower() == key_lower:
            return item
    return None


def _keys_match(existing_key: Any, target_key: str) -> bool:
    return str(existing_key or "").strip().lower() == target_key.strip().lower()


def _normalize_ttl(value: Any) -> int:
    try:
        ttl = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(3650, ttl))


def _apply_profile_removals(
    facts: Any,
    removals: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if not isinstance(facts, list):
        return []
    cleaned = [dict(item) for item in facts if isinstance(item, dict)]
    if not removals:
        return cleaned
    remaining: list[dict[str, Any]] = []
    for item in cleaned:
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        if not key or not value:
            continue
        if _should_remove_fact(key, value, removals):
            continue
        remaining.append(item)
    return remaining


def _should_remove_fact(key: str, value: str, removals: list[dict[str, str]]) -> bool:
    key_lower = key.strip().lower()
    value_lower = value.strip().lower()
    for removal in removals:
        removal_key = str(removal.get("key") or "").strip().lower()
        removal_value = str(removal.get("value") or "").strip().lower()
        if not removal_key:
            continue
        if not _matches_removal_key(key_lower, removal_key):
            continue
        if removal_value and removal_value != value_lower:
            continue
        return True
    return False


def _matches_removal_key(item_key: str, removal_key: str) -> bool:
    if item_key == removal_key:
        return True
    if removal_key.endswith("."):
        return item_key.startswith(removal_key)
    return item_key.startswith(removal_key + ".")


def _prune_expired_facts(facts: Any, now: datetime | None = None) -> list[dict[str, Any]]:
    if not isinstance(facts, list):
        return []
    timestamp = now or datetime.now(timezone.utc)
    remaining: list[dict[str, Any]] = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        if _is_fact_expired(item, timestamp):
            continue
        remaining.append(item)
    return remaining


def _is_fact_expired(item: dict[str, Any], now: datetime) -> bool:
    ttl_days = _normalize_ttl(item.get("ttl_days"))
    if ttl_days <= 0:
        return False
    last_seen = _parse_datetime(str(item.get("last_seen_at") or ""))
    if last_seen is not None:
        anchor = last_seen
    else:
        anchor = _parse_datetime(str(item.get("created_at") or ""))
    if anchor is None:
        return False
    return now - anchor >= timedelta(days=ttl_days)


def _format_profile_facts(facts: Any, limit: int = 8) -> list[str]:
    if not isinstance(facts, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        if not key or not value:
            continue
        importance = _normalize_importance(item.get("importance"))
        confidence = _normalize_confidence(item.get("confidence"))
        inferred = bool(item.get("inferred"))
        created_at = str(item.get("created_at") or "")
        cleaned.append(
            {
                "key": key,
                "value": value,
                "importance": importance,
                "confidence": confidence,
                "inferred": inferred,
                "created_at": created_at,
            }
        )
    if not cleaned:
        return []
    cleaned.sort(
        key=lambda item: (_parse_datetime(item["created_at"]) or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )
    cleaned.sort(key=_fact_score, reverse=True)
    buckets: dict[str, list[dict[str, Any]]] = {
        "identity": [],
        "preferences": [],
        "interests": [],
        "dislikes": [],
        "constraints": [],
        "other": [],
    }
    for item in cleaned:
        buckets[_profile_bucket(item["key"])].append(item)
    order = [
        ("identity", "Identity"),
        ("preferences", "Preferences"),
        ("interests", "Interests"),
        ("dislikes", "Dislikes"),
        ("constraints", "Constraints"),
        ("other", "Other"),
    ]
    lines: list[str] = []
    facts_added = 0
    for key, label in order:
        items = buckets.get(key) or []
        if not items:
            continue
        lines.append(f"{label}:")
        for item in items:
            if facts_added >= limit:
                return lines
            suffix = " (inferred)" if item.get("inferred") else ""
            lines.append(f"- {item['key']}: {item['value']}{suffix}")
            facts_added += 1
    return lines


def _profile_bucket(key: str) -> str:
    lowered = key.strip().lower()
    if lowered.startswith("identity."):
        return "identity"
    if lowered.startswith("preferences."):
        return "preferences"
    if lowered.startswith("interests."):
        return "interests"
    if lowered.startswith("dislikes."):
        return "dislikes"
    if lowered.startswith("constraints."):
        return "constraints"
    return "other"


def _migrate_legacy_profile(profile: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    changed = False
    facts: list[dict[str, Any]] = []
    existing = profile.get("facts")
    if isinstance(existing, list):
        facts = [item for item in existing if isinstance(item, dict)]
    def _add_fact(key: str, value: str) -> None:
        nonlocal changed, facts
        if not value:
            return
        if _fact_exists(facts, key, value):
            return
        facts.append(
            {
                "id": uuid4().hex,
                "key": key,
                "value": value,
                "evidence": "legacy_profile",
                "importance": 0.5,
                "confidence": 0.6,
                "ttl_days": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_seen_at": datetime.now(timezone.utc).isoformat(),
                "source": "legacy",
            }
        )
        changed = True
    mappings = {
        "name": "identity.name",
        "preferred_tone": "preferences.preferred_tone",
        "address_form": "preferences.address_form",
        "partner_name": "relationship.partner_name",
        "current_company": "job.current_company",
        "current_role": "job.current_role",
    }
    for legacy_key, fact_key in mappings.items():
        value = str(profile.get(legacy_key) or "").strip()
        if value:
            _add_fact(fact_key, value)
            profile.pop(legacy_key, None)
    response_style = profile.get("response_style")
    if isinstance(response_style, dict):
        fmt = str(response_style.get("format") or "").strip()
        length = str(response_style.get("length") or "").strip()
        if fmt:
            _add_fact("preferences.response_style.format", fmt)
        if length:
            _add_fact("preferences.response_style.length", length)
        profile.pop("response_style", None)
    if facts:
        profile["facts"] = facts
        profile["version"] = _PROFILE_VERSION
    return profile, changed


def _trim_profile_facts(profile: dict[str, Any], max_items: int = 100) -> dict[str, Any]:
    facts = _prune_expired_facts(profile.get("facts"))
    if not facts:
        profile["facts"] = []
        return profile
    if len(facts) <= max_items:
        profile["facts"] = facts
        return profile
    cleaned = [item for item in facts if isinstance(item, dict)]
    cleaned.sort(
        key=lambda item: _parse_datetime(str(item.get("created_at") or "")) or datetime.min.replace(tzinfo=timezone.utc),
    )
    profile["facts"] = cleaned[-max_items:]
    return profile


def migrate_profile_json(profile_json: str) -> tuple[str, bool]:
    if not profile_json:
        return profile_json, False
    try:
        profile = json.loads(profile_json)
    except Exception:
        return profile_json, False
    if not isinstance(profile, dict):
        return profile_json, False
    profile, changed = _migrate_legacy_profile(profile)
    pruned = _prune_expired_facts(profile.get("facts"))
    if pruned != profile.get("facts"):
        profile["facts"] = pruned
        changed = True
    if profile.get("version") != _PROFILE_VERSION:
        profile["version"] = _PROFILE_VERSION
        changed = True
    if not changed:
        return profile_json, False
    return json.dumps(profile, ensure_ascii=True), True


def _format_memory_block(items: list[MemoryItemRow], max_chars: int) -> str:
    if not items:
        return ""
    lines: list[str] = [
        "Relevant past context (use if helpful; prefer current session state if conflict):"
    ]
    total_chars = sum(len(line) for line in lines)
    for item in items:
        date_label = _date_label(item.created_at)
        title = item.title or "Memory"
        summary = _trim_text(item.summary, 360)
        line = f"- [{title}] ({date_label}): {summary}"
        if total_chars + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total_chars += len(line) + 1
    return "\n".join(lines).strip()


def _trim_text(text: str, max_chars: int) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _date_label(value: str | None) -> str:
    dt = _parse_datetime(value) if value else None
    if not dt:
        return "unknown"
    return dt.date().isoformat()
