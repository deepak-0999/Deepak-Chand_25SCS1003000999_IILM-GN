"""
Pydantic models — all request/response shapes for the Lore API.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Genre(str, Enum):
    fantasy = "fantasy"
    horror = "horror"
    scifi = "scifi"
    romance = "romance"


class Choice(BaseModel):
    key: str = Field(..., description="A, B, or C")
    title: str = Field(..., description="Short action label")
    subtitle: str = Field(default="", description="One-line consequence hint")


# ── Requests ──────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    genre: Genre
    player_name: str = Field(default="Adventurer", max_length=40)

    model_config = {"json_schema_extra": {
        "example": {"genre": "fantasy", "player_name": "Aria"}
    }}


class ChoiceRequest(BaseModel):
    choice_key: Optional[str] = Field(
        default=None,
        description="A, B, or C — required unless free_text is set",
    )
    free_text: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Player's own action (overrides choice_key if both given)",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {"choice_key": "A"},
            {"free_text": "I search the body for clues"},
            {"choice_key": "B", "free_text": "But I also call out to the stranger"},
        ]
    }}


# ── Responses ─────────────────────────────────────────────────────

class StartSessionResponse(BaseModel):
    session_id: str
    genre: Genre
    turn: int
    chapter_title: str
    story: str
    choices: list[Choice]


class ChoiceResponse(BaseModel):
    """Used internally; actual response is SSE stream."""
    turn: int
    chapter_title: str
    story: str
    choices: list[Choice]


# ── SSE chunk types ───────────────────────────────────────────────

class StoryChunk(BaseModel):
    type: str = "story_chunk"
    text: str


class ChoicesEvent(BaseModel):
    type: str = "choices"
    choices: list[Choice]
    turn: int
    chapter_title: str


class ErrorEvent(BaseModel):
    type: str = "error"
    message: str
    code: Optional[int] = None


# ── Session ───────────────────────────────────────────────────────

class SessionState(BaseModel):
    session_id: str
    genre: Genre
    player_name: str
    turn: int
    history: list[dict]          # raw message dicts for the API
    last_choices: list[Choice]   # most recent choices offered
