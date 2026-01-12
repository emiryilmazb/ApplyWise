# Long-Term Memory

Atlas can store compact conversation summaries and retrieve relevant past context in later chats.

## How it works

- Short-term context: the last 20 messages are loaded from SQLite and used as usual.
- Long-term memory: every N user messages (default 12), the bot summarizes recent turns and stores a memory item.
- Retrieval: each new user message is embedded, scored against stored summaries, and the top-k items are injected into the prompt before the chat history.
- Profile memory (lightweight): if the user explicitly states stable preferences (tone, name, formatting), a small profile JSON is updated and injected into the prompt.

## Anonymous mode

Use:

- `anonim on`  -> stop saving new messages and memory items.
- `anonim off` -> resume saving.

When anonymous mode is on, new messages and memory writes are skipped, but the bot still replies using existing stored history.

## Profile cleanup

- `/profile` shows stored profile facts (with ids).
- `/profile_forget <id>` removes a specific profile fact by id.

## Configuration (env vars)

```
MEMORY_ENABLED=True
MEMORY_SUMMARY_EVERY_N_USER_MSG=12
MEMORY_RETRIEVAL_TOP_K=5
EMBEDDING_BACKEND=sentence_transformers   # or: tfidf
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
ANONYMITY_DEFAULT_OFF=True
```

## Privacy

- Summaries are stored in SQLite (see `data/atlas.db` by default).
- `anonim on` stops new writes; use `/forget <id>` to delete specific memory items.

## Disabling memory

Set `MEMORY_ENABLED=False` in `.env`.
