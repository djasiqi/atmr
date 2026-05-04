from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime, timedelta

from models import ContactRequest

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_message(message: str) -> str:
    collapsed = _WHITESPACE_RE.sub(" ", (message or "").strip())
    return collapsed.lower()


def compute_dedupe_hash(email: str, category: str, message_normalized: str) -> str:
    material = f"{(email or '').strip().lower()}|{category}|{message_normalized}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def current_window_bucket(
    now: datetime | None = None, bucket_minutes: int = 5
) -> datetime:
    now = now or datetime.now(UTC)
    epoch = int(now.timestamp())
    bucket_seconds = bucket_minutes * 60
    rounded = math.floor(epoch / bucket_seconds) * bucket_seconds
    return datetime.fromtimestamp(rounded, tz=UTC)


def find_recent_duplicate(
    dedupe_hash: str, window_minutes: int = 5
) -> ContactRequest | None:
    threshold = datetime.now(UTC) - timedelta(minutes=window_minutes)
    return (
        ContactRequest.query.filter(ContactRequest.dedupe_hash == dedupe_hash)
        .filter(ContactRequest.created_at >= threshold)
        .filter(ContactRequest.status != "spam")
        .order_by(ContactRequest.created_at.asc(), ContactRequest.id.asc())
        .first()
    )
