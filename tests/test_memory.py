from datetime import datetime, timedelta, timezone

import numpy as np

from app.memory.embeddings import cosine_similarity, deserialize_embedding, serialize_embedding
from app.memory.long_term import (
    _apply_profile_removals,
    _merge_profile_facts,
    _prune_expired_facts,
    extract_json_payload,
)
from app.storage.database import DatabaseManager
from app.storage import user_settings


def test_embedding_roundtrip() -> None:
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    payload = serialize_embedding(vector)
    restored = deserialize_embedding(payload)
    assert np.allclose(vector, restored)


def test_cosine_similarity() -> None:
    vec_a = np.array([1.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0], dtype=np.float32)
    vec_c = np.array([1.0, 1.0], dtype=np.float32)
    assert cosine_similarity(vec_a, vec_b) == 0.0
    assert round(cosine_similarity(vec_a, vec_c), 6) == 0.707107


def test_extract_json_payload() -> None:
    raw = "Sure.```json\\n{\\\"title\\\": \\\"Hello\\\", \\\"summary\\\": \\\"World\\\"}\\n```"
    payload = extract_json_payload(raw)
    assert payload is not None
    assert payload["title"] == "Hello"
    assert payload["summary"] == "World"


def test_anonymous_mode_toggle(tmp_path, monkeypatch) -> None:
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    monkeypatch.setattr(user_settings, "get_database", lambda: db)
    user_settings.clear_anonymous_cache()
    user_settings.set_anonymous_mode("user-1", True)
    assert user_settings.is_anonymous_mode("user-1") is True
    user_settings.set_anonymous_mode("user-1", False)
    assert user_settings.is_anonymous_mode("user-1") is False


def test_merge_profile_facts_replaces_exclusive_on_override() -> None:
    existing = [
        {
            "id": "1",
            "key": "preferences.response_style.length",
            "value": "short",
            "importance": 0.5,
            "confidence": 0.8,
            "ttl_days": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    new_facts = [
        {
            "key": "preferences.response_style.length",
            "value": "long",
            "evidence": "long",
            "importance": 0.7,
            "confidence": 0.7,
            "ttl_days": 0,
        }
    ]
    updated = _merge_profile_facts(
        existing,
        new_facts,
        "From now on I want long answers.",
        allow_missing_evidence=False,
        source_override="llm",
    )
    values = [item["value"] for item in updated if item.get("key") == "preferences.response_style.length"]
    assert values == ["long"]


def test_merge_profile_facts_allows_multiple_interests() -> None:
    existing = [
        {
            "id": "1",
            "key": "interests.topic",
            "value": "ai",
            "importance": 0.6,
            "confidence": 0.6,
            "ttl_days": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    new_facts = [
        {
            "key": "interests.topic",
            "value": "robotics",
            "evidence": "robotics",
            "importance": 0.4,
            "confidence": 0.7,
            "ttl_days": 0,
        }
    ]
    updated = _merge_profile_facts(
        existing,
        new_facts,
        "I am interested in robotics.",
        allow_missing_evidence=False,
        source_override="llm",
    )
    values = sorted(
        [item["value"] for item in updated if item.get("key") == "interests.topic"]
    )
    assert values == ["ai", "robotics"]


def test_apply_profile_removals_by_prefix() -> None:
    facts = [
        {"key": "preferences.response_style.length", "value": "short"},
        {"key": "preferences.communication.tone", "value": "direct"},
        {"key": "interests.topic", "value": "ai"},
    ]
    removals = [{"key": "preferences", "value": ""}]
    remaining = _apply_profile_removals(facts, removals)
    keys = [item.get("key") for item in remaining]
    assert keys == ["interests.topic"]


def test_prune_expired_facts() -> None:
    now = datetime.now(timezone.utc)
    facts = [
        {
            "key": "interests.topic",
            "value": "ai",
            "ttl_days": 1,
            "created_at": (now - timedelta(days=2)).isoformat(),
        },
        {
            "key": "interests.topic",
            "value": "ml",
            "ttl_days": 1,
            "created_at": (now - timedelta(hours=12)).isoformat(),
        },
        {
            "key": "preferences.communication.tone",
            "value": "direct",
            "ttl_days": 0,
            "created_at": (now - timedelta(days=10)).isoformat(),
        },
    ]
    remaining = _prune_expired_facts(facts, now=now)
    values = sorted([item["value"] for item in remaining])
    assert values == ["direct", "ml"]
