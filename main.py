"""
Lore — AI Storytelling Engine
FastAPI backend with streaming, session management, and choice generation.
"""

import json
import uuid
import asyncio
from datetime import datetime
from typing import AsyncIterator

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from models import (
    Genre,
    StartSessionRequest,
    StartSessionResponse,
    ChoiceRequest,
    ChoiceResponse,
    StoryChunk,
    SessionState,
    Choice,
)
from session_store import SessionStore
from prompts import build_system_prompt, build_choice_prompt, build_start_prompt

# ── App ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Lore API",
    description="Hybrid AI storytelling engine — streaming story + structured choices",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
store = SessionStore(max_sessions=settings.max_sessions)


# ── Routes ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "active_sessions": store.count(),
    }


@app.post("/session/start", response_model=StartSessionResponse)
async def start_session(req: StartSessionRequest):
    """
    Create a new story session for a chosen genre.
    Returns session_id + the opening scene (non-streamed, fast cold start).
    """
    session_id = str(uuid.uuid4())

    system = build_system_prompt(req.genre)
    user_msg = build_start_prompt(req.genre, req.player_name)

    # Non-streaming first call — gets the opening scene + first choices
    response = client.messages.create(
        model=settings.model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text
    parsed = _parse_story_response(raw)

    session = SessionState(
        session_id=session_id,
        genre=req.genre,
        player_name=req.player_name,
        turn=1,
        history=[
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": raw},
        ],
        last_choices=parsed["choices"],
    )
    store.save(session_id, session)

    return StartSessionResponse(
        session_id=session_id,
        genre=req.genre,
        turn=1,
        story=parsed["story"],
        choices=parsed["choices"],
        chapter_title=parsed.get("chapter_title", "Prologue"),
    )


@app.post("/session/{session_id}/choose")
async def make_choice(session_id: str, req: ChoiceRequest, request: Request):
    """
    Player picks a choice (A/B/C) or sends free text.
    Returns a streaming SSE response: story chunks first, then a final
    JSON event with the next set of choices.
    """
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    # Validate choice key if choice-based
    if req.choice_key and req.choice_key not in ["A", "B", "C"]:
        raise HTTPException(status_code=400, detail="choice_key must be A, B, or C")

    # Build the player's message from their selection
    player_message = _build_player_message(req, session)

    return StreamingResponse(
        _stream_story(session, player_message, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Return current session state (for reconnecting clients)."""
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return session


@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """Explicitly end a session and free memory."""
    store.delete(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/genres")
async def list_genres():
    """Return all available genres with metadata."""
    return {
        "genres": [
            {
                "id": "fantasy",
                "label": "Fantasy",
                "description": "Dragons, magic, warriors — D&D style",
                "icon": "🏰",
                "recommended": True,
            },
            {
                "id": "horror",
                "label": "Horror",
                "description": "Dark, suspense, survival",
                "icon": "👻",
                "recommended": False,
            },
            {
                "id": "scifi",
                "label": "Sci-Fi",
                "description": "Space, AI, future worlds",
                "icon": "🚀",
                "recommended": False,
            },
            {
                "id": "romance",
                "label": "Romance",
                "description": "Emotion-based storytelling",
                "icon": "❤️",
                "recommended": False,
            },
        ]
    }


# ── Streaming core ────────────────────────────────────────────────

async def _stream_story(
    session: SessionState,
    player_message: str,
    session_id: str,
) -> AsyncIterator[str]:
    """
    Streams story text as SSE events, then emits a final 'choices' event.

    SSE format:
        data: {"type": "story_chunk", "text": "..."}
        data: {"type": "choices", "choices": [...], "turn": N, "chapter_title": "..."}
        data: {"type": "done"}
    """
    system = build_system_prompt(session.genre)
    history = session.history + [{"role": "user", "content": player_message}]

    full_response = ""
    story_text = ""

    try:
        with client.messages.stream(
            model=settings.model,
            max_tokens=1024,
            system=system,
            messages=history,
        ) as stream:
            for text_chunk in stream.text_stream:
                full_response += text_chunk

                # Emit story text until we hit the JSON fence
                if "```json" not in full_response:
                    story_text += text_chunk
                    chunk = StoryChunk(type="story_chunk", text=text_chunk)
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    await asyncio.sleep(0)  # yield control to event loop

        # Parse out structured choices from the full response
        parsed = _parse_story_response(full_response)

        # If story_text ended up empty (model put everything in JSON) fall back
        if not story_text.strip():
            story_text = parsed["story"]
            chunk = StoryChunk(type="story_chunk", text=story_text)
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Update session
        new_turn = session.turn + 1
        session.history = history + [{"role": "assistant", "content": full_response}]
        session.turn = new_turn
        session.last_choices = parsed["choices"]
        store.save(session_id, session)

        # Emit choices event
        choices_event = {
            "type": "choices",
            "choices": [c.model_dump() for c in parsed["choices"]],
            "turn": new_turn,
            "chapter_title": parsed.get("chapter_title", f"Chapter {new_turn}"),
        }
        yield f"data: {json.dumps(choices_event)}\n\n"

    except anthropic.APIError as e:
        error_event = {"type": "error", "message": str(e), "code": e.status_code}
        yield f"data: {json.dumps(error_event)}\n\n"

    finally:
        yield "data: {\"type\": \"done\"}\n\n"


# ── Helpers ───────────────────────────────────────────────────────

def _parse_story_response(raw: str) -> dict:
    """
    Model is prompted to return:
        <story prose>
        ```json
        { "chapter_title": "...", "choices": [{"key":"A","title":"...","subtitle":"..."},...] }
        ```
    This parser splits on the fence and handles graceful fallback.
    """
    story = raw
    choices = []
    chapter_title = "The Adventure Continues"

    try:
        if "```json" in raw:
            parts = raw.split("```json", 1)
            story = parts[0].strip()
            json_block = parts[1].split("```")[0].strip()
            data = json.loads(json_block)
            chapter_title = data.get("chapter_title", chapter_title)
            raw_choices = data.get("choices", [])
            choices = [
                Choice(
                    key=c.get("key", "A"),
                    title=c.get("title", "Continue"),
                    subtitle=c.get("subtitle", ""),
                )
                for c in raw_choices[:3]  # cap at 3
            ]
        elif "```" in raw:
            # fallback: try plain ``` block
            parts = raw.split("```", 1)
            story = parts[0].strip()
            try:
                json_block = parts[1].split("```")[0].strip()
                data = json.loads(json_block)
                chapter_title = data.get("chapter_title", chapter_title)
                choices = [
                    Choice(key=c["key"], title=c["title"], subtitle=c.get("subtitle", ""))
                    for c in data.get("choices", [])[:3]
                ]
            except Exception:
                pass
    except Exception:
        pass  # Return story as-is with empty choices; client handles gracefully

    # Final fallback choices if model didn't produce any
    if not choices:
        choices = [
            Choice(key="A", title="Press forward", subtitle="Into the unknown"),
            Choice(key="B", title="Investigate", subtitle="Look for clues"),
            Choice(key="C", title="Wait and observe", subtitle="Patience has its rewards"),
        ]

    return {"story": story, "choices": choices, "chapter_title": chapter_title}


def _build_player_message(req: ChoiceRequest, session: SessionState) -> str:
    """Construct the player's turn message from a choice or free text."""
    if req.choice_key and session.last_choices:
        # Find the chosen option for richer context
        chosen = next(
            (c for c in session.last_choices if c.key == req.choice_key), None
        )
        if chosen:
            return build_choice_prompt(req.choice_key, chosen.title, req.free_text)

    # Free text only
    if req.free_text:
        return req.free_text

    return f"I choose option {req.choice_key or 'A'}."
