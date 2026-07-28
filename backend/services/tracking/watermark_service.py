"""Watermark persisted contigu — autorité PostgreSQL (pas Redis)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

WATERMARK_PAGE_SIZE = int(os.getenv("TRACKING_WATERMARK_PAGE_SIZE", "50"))
WATERMARK_HMAC_SECRET = os.getenv(
    "TRACKING_WATERMARK_HMAC_SECRET",
    os.getenv("SECRET_KEY", "dev-watermark-secret"),
)


def _sign_cursor(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(
        WATERMARK_HMAC_SECRET.encode("utf-8"), raw, hashlib.sha256
    ).hexdigest()[:16]
    return base64.urlsafe_b64encode(raw + b"|" + sig.encode("utf-8")).decode("ascii")


def _verify_cursor(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    try:
        decoded = base64.urlsafe_b64decode(token.encode("ascii"))
        raw, sig = decoded.rsplit(b"|", 1)
        expected = hmac.new(
            WATERMARK_HMAC_SECRET.encode("utf-8"), raw, hashlib.sha256
        ).hexdigest()[:16]
        if not hmac.compare_digest(sig.decode("ascii"), expected):
            return None
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def get_persisted_watermark(
    session: Session,
    *,
    driver_id: int,
    company_id: int,
    tracking_session_id: str,
    cursor: str | None = None,
) -> dict[str, Any]:
    sid = (tracking_session_id or "").strip()
    sess = (
        session.execute(
            text(
                """
            SELECT session_generation, status
            FROM tracking_sessions
            WHERE driver_id = :driver_id
              AND company_id = :company_id
              AND tracking_session_id = :sid
            """
            ),
            {"driver_id": driver_id, "company_id": company_id, "sid": sid},
        )
        .mappings()
        .first()
    )
    if sess is None:
        raise PermissionError("tracking_session_forbidden")

    state = (
        session.execute(
            text(
                """
            SELECT contiguous_persisted_through, max_seen_sequence, session_generation
            FROM tracking_session_state
            WHERE driver_id = :driver_id AND tracking_session_id = :sid
            FOR UPDATE
            """
            ),
            {"driver_id": driver_id, "sid": sid},
        )
        .mappings()
        .first()
    )

    contiguous = int(state["contiguous_persisted_through"]) if state else 0
    generation = int(
        state["session_generation"] if state else sess["session_generation"]
    )

    cursor_data = _verify_cursor(cursor) or {"after_sequence": contiguous}
    after_seq = int(cursor_data.get("after_sequence") or contiguous)

    ooo_rows = (
        session.execute(
            text(
                """
            SELECT sequence_id, location_event_id
            FROM driver_location_events
            WHERE driver_id = :driver_id
              AND tracking_session_id = :sid
              AND sequence_id > :contiguous
              AND sequence_id > :after_seq
            ORDER BY sequence_id ASC
            LIMIT :lim
            """
            ),
            {
                "driver_id": driver_id,
                "sid": sid,
                "contiguous": contiguous,
                "after_seq": after_seq,
                "lim": WATERMARK_PAGE_SIZE + 1,
            },
        )
        .mappings()
        .all()
    )

    has_more = len(ooo_rows) > WATERMARK_PAGE_SIZE
    page = list(ooo_rows[:WATERMARK_PAGE_SIZE])
    out_of_order = [
        {
            "sequence_id": int(r["sequence_id"]),
            "location_event_id": str(r["location_event_id"]),
        }
        for r in page
    ]

    gaps = (
        session.execute(
            text(
                """
            SELECT sequence_from, sequence_to
            FROM tracking_sequence_gaps
            WHERE driver_id = :driver_id
              AND tracking_session_id = :sid
              AND resolved_at IS NULL
            ORDER BY sequence_from ASC
            LIMIT 100
            """
            ),
            {"driver_id": driver_id, "sid": sid},
        )
        .mappings()
        .all()
    )
    missing_ranges = [[int(g["sequence_from"]), int(g["sequence_to"])] for g in gaps]

    next_cursor = None
    if has_more and page:
        next_cursor = _sign_cursor(
            {
                "tracking_session_id": sid,
                "after_sequence": int(page[-1]["sequence_id"]),
            }
        )

    return {
        "ack_status": "persisted",
        "tracking_session_id": sid,
        "session_generation": generation,
        "contiguous_persisted_through": contiguous,
        "out_of_order_persisted": out_of_order,
        "missing_ranges": missing_ranges,
        "next_cursor": next_cursor,
    }
