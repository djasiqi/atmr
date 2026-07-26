"""Politique de renvoi email d'activation (sans import routes)."""

from __future__ import annotations

import os
from datetime import UTC, datetime


ACTIVATION_RESEND_COOLDOWN_SECONDS = int(
    os.getenv("ACTIVATION_RESEND_COOLDOWN_SECONDS", "60")
)
ACTIVATION_RESEND_DAILY_LIMIT = int(os.getenv("ACTIVATION_RESEND_DAILY_LIMIT", "10"))


def is_same_utc_day(a: datetime, b: datetime) -> bool:
    return a.astimezone(UTC).date() == b.astimezone(UTC).date()


def enforce_resend_policy(
    *, last_sent_at: datetime | None, resend_count: int
) -> tuple[bool, str | None, int]:
    """Retourne (allowed, error_code, retry_after_seconds)."""
    now = datetime.now(UTC)
    if last_sent_at:
        elapsed = int((now - last_sent_at).total_seconds())
        if elapsed < ACTIVATION_RESEND_COOLDOWN_SECONDS:
            return False, "cooldown", ACTIVATION_RESEND_COOLDOWN_SECONDS - elapsed

    daily_count = resend_count
    if last_sent_at and not is_same_utc_day(last_sent_at, now):
        daily_count = 0
    if daily_count >= ACTIVATION_RESEND_DAILY_LIMIT:
        return False, "daily_limit", 0

    return True, None, 0
