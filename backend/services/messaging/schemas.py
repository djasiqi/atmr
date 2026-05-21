"""Schémas Pydantic pour les payloads Socket.IO chat."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TeamChatInboundPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str | None = None
    receiver_id: int | None = None
    thread_id: str | None = None
    booking_id: int | None = None
    conversation_id: int | None = None
    client_message_id: str | None = Field(default=None, alias="client_message_id")
    message_type: str = "text"
    priority: str = "normal"
    image_url: str | None = None
    image: str | None = None
    pdf_url: str | None = None
    pdf: str | None = None
    pdf_filename: str | None = None
    pdf_size: int | None = None
    audio_url: str | None = None
    _localId: str | None = None

    @field_validator("receiver_id", "booking_id", "conversation_id", mode="before")
    @classmethod
    def _coerce_optional_int(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError("invalid integer")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value.strip())
        raise ValueError("invalid integer")

    @field_validator("receiver_id")
    @classmethod
    def _receiver_positive(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("receiver_id must be positive")
        return value

    def resolved_image_url(self) -> str | None:
        return self.image_url or self.image

    def resolved_pdf_url(self) -> str | None:
        return self.pdf_url or self.pdf

    def resolved_client_message_id(self) -> str | None:
        return self.client_message_id or self._localId


class CompanyChatInboundPayload(TeamChatInboundPayload):
    """Payload entreprise — même forme que team chat pour l'instant."""


class TypingPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sender_name: str | None = None
    surface: str | None = None
    conversation_id: int | str | None = None

    @field_validator("conversation_id", mode="before")
    @classmethod
    def _coerce_conversation_id(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value.strip())
        raise ValueError("invalid conversation_id")


class ReadReceiptPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message_id: int | None = None
    conversation_id: int | None = None
    thread_id: str | None = None

    @field_validator("message_id", "conversation_id", mode="before")
    @classmethod
    def _coerce_int(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value.strip())
        raise ValueError("invalid integer")
