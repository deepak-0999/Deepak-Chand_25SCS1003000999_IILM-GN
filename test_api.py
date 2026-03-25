"""
Tests for the Lore API.

Run with:
    pip install pytest pytest-asyncio httpx
    pytest test_api.py -v
"""

import json
import pytest
import asyncio
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from main import app
from models import Genre, Choice, SessionState
from prompts import build_system_prompt, build_start_prompt, build_choice_prompt
from session_store import SessionStore
from config import settings

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────

MOCK_STORY_RESPONSE = """The iron gate groans open before you. Beyond it, three paths vanish into shadow — the smell of wet stone, the flicker of distant torchlight, and something else, older than either. Above, the dragon's wingbeats fade. You have seconds before it returns.

Your hand finds the hilt of your blade. The darkness waits.

```json
{
  "chapter_title": "The Iron Gate",
  "choices": [
    {"key": "A", "title": "Enter the cave on the left", "subtitle": "Darkness, risk, possible treasure"},
    {"key": "B", "title": "Follow the torchlight", "subtitle": "Someone else is down here"},
    {"key": "C", "title": "Hold position and listen", "subtitle": "Information before action"}
  ]
}
```"""


def make_mock_message(text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=text)]
    return mock_msg


def make_mock_stream(text: str):
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter(list(text))
    return mock_stream


# ── Health ────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "active_sessions" in data
    assert "timestamp" in data


# ── Genres ────────────────────────────────────────────────────────

def test_list_genres():
    r = client.get("/genres")
    assert r.status_code == 200
    genres = r.json()["genres"]
    assert len(genres) == 4
    ids = [g["id"] for g in genres]
    assert set(ids) == {"fantasy", "horror", "scifi", "romance"}
    recommended = [g for g in genres if g["recommended"]]
    assert len(recommended) == 1
    assert recommended[0]["id"] == "fantasy"


# ── Session start ─────────────────────────────────────────────────

@patch("main.client")
def test_start_session_fantasy(mock_client):
    mock_client.messages.create.return_value = make_mock_message(MOCK_STORY_RESPONSE)

    r = client.post("/session/start", json={"genre": "fantasy", "player_name": "Aria"})
    assert r.status_code == 200
    data = r.json()

    assert "session_id" in data
    assert data["genre"] == "fantasy"
    assert data["turn"] == 1
    assert len(data["choices"]) == 3
    assert data["choices"][0]["key"] == "A"
    assert "chapter_title" in data
    assert len(data["story"]) > 10


@patch("main.client")
def test_start_session_all_genres(mock_client):
    mock_client.messages.create.return_value = make_mock_message(MOCK_STORY_RESPONSE)
    for genre in ["fantasy", "horror", "scifi", "romance"]:
        r = client.post("/session/start", json={"genre": genre, "player_name": "Tester"})
        assert r.status_code == 200, f"Genre {genre} failed"


@patch("main.client")
def test_start_session_default_name(mock_client):
    mock_client.messages.create.return_value = make_mock_message(MOCK_STORY_RESPONSE)
    r = client.post("/session/start", json={"genre": "horror"})
    assert r.status_code == 200


def test_start_session_invalid_genre():
    r = client.post("/session/start", json={"genre": "western", "player_name": "Bob"})
    assert r.status_code == 422


# ── Session get ───────────────────────────────────────────────────

@patch("main.client")
def test_get_session(mock_client):
    mock_client.messages.create.return_value = make_mock_message(MOCK_STORY_RESPONSE)
    r = client.post("/session/start", json={"genre": "scifi", "player_name": "Nova"})
    session_id = r.json()["session_id"]

    r2 = client.get(f"/session/{session_id}")
    assert r2.status_code == 200
    assert r2.json()["session_id"] == session_id
    assert r2.json()["turn"] == 1


def test_get_nonexistent_session():
    r = client.get("/session/does-not-exist")
    assert r.status_code == 404


# ── Session delete ────────────────────────────────────────────────

@patch("main.client")
def test_delete_session(mock_client):
    mock_client.messages.create.return_value = make_mock_message(MOCK_STORY_RESPONSE)
    r = client.post("/session/start", json={"genre": "romance", "player_name": "Lena"})
    sid = r.json()["session_id"]

    rd = client.delete(f"/session/{sid}")
    assert rd.status_code == 200

    r2 = client.get(f"/session/{sid}")
    assert r2.status_code == 404


# ── Choice endpoint ───────────────────────────────────────────────

@patch("main.client")
def test_make_choice_streaming(mock_client):
    mock_client.messages.create.return_value = make_mock_message(MOCK_STORY_RESPONSE)
    r = client.post("/session/start", json={"genre": "fantasy", "player_name": "Kael"})
    sid = r.json()["session_id"]

    mock_client.messages.stream.return_value = make_mock_stream(MOCK_STORY_RESPONSE)

    r2 = client.post(f"/session/{sid}/choose", json={"choice_key": "A"})
    assert r2.status_code == 200
    assert "text/event-stream" in r2.headers["content-type"]

    events = _parse_sse(r2.text)
    types = [e["type"] for e in events]
    assert "story_chunk" in types
    assert "choices" in types
    assert "done" in types

    choices_event = next(e for e in events if e["type"] == "choices")
    assert len(choices_event["choices"]) == 3
    assert choices_event["turn"] == 2


@patch("main.client")
def test_make_choice_free_text(mock_client):
    mock_client.messages.create.return_value = make_mock_message(MOCK_STORY_RESPONSE)
    r = client.post("/session/start", json={"genre": "horror", "player_name": "Sam"})
    sid = r.json()["session_id"]

    mock_client.messages.stream.return_value = make_mock_stream(MOCK_STORY_RESPONSE)

    r2 = client.post(f"/session/{sid}/choose", json={"free_text": "I search behind the painting"})
    assert r2.status_code == 200
    events = _parse_sse(r2.text)
    assert any(e["type"] == "choices" for e in events)


@patch("main.client")
def test_make_choice_invalid_key(mock_client):
    mock_client.messages.create.return_value = make_mock_message(MOCK_STORY_RESPONSE)
    r = client.post("/session/start", json={"genre": "fantasy", "player_name": "X"})
    sid = r.json()["session_id"]

    r2 = client.post(f"/session/{sid}/choose", json={"choice_key": "Z"})
    assert r2.status_code == 400


def test_choose_nonexistent_session():
    r = client.post("/session/ghost-id/choose", json={"choice_key": "A"})
    assert r.status_code == 404


# ── Session store ─────────────────────────────────────────────────

def test_session_store_ttl():
    from datetime import timedelta
    store = SessionStore(max_sessions=10)
    store._ttl = timedelta(seconds=0)  # expire immediately

    session = SessionState(
        session_id="x", genre=Genre.fantasy, player_name="P",
        turn=1, history=[], last_choices=[]
    )
    store.save("x", session)
    assert store.get("x") is None  # expired


def test_session_store_eviction():
    store = SessionStore(max_sessions=3)
    for i in range(4):
        s = SessionState(
            session_id=str(i), genre=Genre.fantasy, player_name="P",
            turn=1, history=[], last_choices=[]
        )
        store.save(str(i), s)
    assert store.count() == 3  # oldest evicted


def test_session_store_purge():
    from datetime import timedelta
    store = SessionStore(max_sessions=10)
    s = SessionState(
        session_id="y", genre=Genre.fantasy, player_name="P",
        turn=1, history=[], last_choices=[]
    )
    store.save("y", s)
    store._ttl = timedelta(seconds=0)
    purged = store.purge_expired()
    assert purged == 1
    assert store.count() == 0


# ── Prompt builders ───────────────────────────────────────────────

def test_system_prompt_contains_format():
    for genre in Genre:
        prompt = build_system_prompt(genre)
        assert "```json" in prompt
        assert "chapter_title" in prompt
        assert '"key": "A"' in prompt


def test_start_prompt_includes_name():
    prompt = build_start_prompt(Genre.fantasy, "Aria")
    assert "Aria" in prompt


def test_choice_prompt_includes_key_and_title():
    prompt = build_choice_prompt("B", "Follow the torchlight", None)
    assert "B" in prompt
    assert "Follow the torchlight" in prompt


def test_choice_prompt_with_free_text():
    prompt = build_choice_prompt("A", "Enter the cave", "but I ready my torch first")
    assert "but I ready my torch first" in prompt


# ── Response parser ───────────────────────────────────────────────

def test_parse_story_response_clean():
    from main import _parse_story_response
    raw = """The hall is vast and silent.

```json
{
  "chapter_title": "The Silent Hall",
  "choices": [
    {"key": "A", "title": "Approach the throne", "subtitle": "Risk, but reward"},
    {"key": "B", "title": "Examine the tapestries", "subtitle": "Clues in the weave"},
    {"key": "C", "title": "Listen at the far door", "subtitle": "What lies beyond"}
  ]
}
```"""
    result = _parse_story_response(raw)
    assert result["chapter_title"] == "The Silent Hall"
    assert len(result["choices"]) == 3
    assert result["story"].strip() == "The hall is vast and silent."
    assert result["choices"][0].key == "A"


def test_parse_story_response_fallback():
    from main import _parse_story_response
    raw = "The story continues, but the model forgot to include choices."
    result = _parse_story_response(raw)
    assert len(result["choices"]) == 3  # fallback choices
    assert result["story"] == raw


# ── Helpers ───────────────────────────────────────────────────────

def _parse_sse(raw: str) -> list[dict]:
    """Parse raw SSE text into a list of event dicts."""
    events = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events
