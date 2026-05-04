"""ChangeRequest — persistance DB (gouvernance plateforme V1)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from ext import db
from models.platform_change_request import PlatformChangeRequest
from services.platform_governance_constants import (
    CHANGE_REQUEST_COMPLETED,
    CHANGE_REQUEST_EXECUTING,
    CHANGE_REQUEST_FAILED,
)


def _row_to_dict(rec: PlatformChangeRequest) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": rec.id,
        "change_type": rec.change_type,
        "status": rec.status,
        "tenant_id": rec.tenant_id,
        "justification": rec.justification or "",
        "correlation_id": rec.correlation_id,
        "incident_id": rec.incident_id,
        "created_at": rec.created_at.isoformat() if rec.created_at else None,
        "updated_at": rec.updated_at.isoformat() if rec.updated_at else None,
        "metadata": rec.metadata_json or {},
    }
    if rec.result_json is not None:
        out["result"] = rec.result_json
    return out


def create_change_request(
    *,
    change_type: str,
    tenant_id: int | None,
    justification: str,
    correlation_id: str | None,
    incident_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    requested_by_user_id: int | None = None,
) -> dict[str, Any]:
    if correlation_id and tenant_id is not None:
        existing = db.session.scalar(
            select(PlatformChangeRequest).where(
                PlatformChangeRequest.correlation_id == correlation_id,
                PlatformChangeRequest.change_type == change_type,
                PlatformChangeRequest.tenant_id == tenant_id,
            )
        )
        if existing is not None:
            return _row_to_dict(existing)

    cid = str(uuid.uuid4())
    now = datetime.now(UTC)
    rec = PlatformChangeRequest(
        id=cid,
        change_type=change_type,
        status=CHANGE_REQUEST_EXECUTING,
        tenant_id=tenant_id,
        justification=justification,
        correlation_id=correlation_id,
        incident_id=incident_id,
        requested_by_user_id=requested_by_user_id,
        created_at=now,
        updated_at=now,
        metadata_json=metadata or {},
    )
    db.session.add(rec)
    db.session.flush()
    return _row_to_dict(rec)


def complete_change_request(
    change_request_id: str,
    *,
    status: str,
    result: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    rec = db.session.get(PlatformChangeRequest, change_request_id)
    if not rec:
        return None
    rec.status = status
    rec.updated_at = datetime.now(UTC)
    if result is not None:
        rec.result_json = result
    db.session.add(rec)
    db.session.flush()
    return _row_to_dict(rec)


def get_change_request(change_request_id: str) -> dict[str, Any] | None:
    rec = db.session.get(PlatformChangeRequest, change_request_id)
    if not rec:
        return None
    return _row_to_dict(rec)


def list_change_requests(*, limit: int = 50) -> list[dict[str, Any]]:
    limit = min(max(limit, 1), 200)
    rows = db.session.scalars(
        select(PlatformChangeRequest)
        .order_by(PlatformChangeRequest.created_at.desc())
        .limit(limit)
    ).all()
    return [_row_to_dict(r) for r in rows]


__all__ = [
    "CHANGE_REQUEST_COMPLETED",
    "CHANGE_REQUEST_EXECUTING",
    "CHANGE_REQUEST_FAILED",
    "complete_change_request",
    "create_change_request",
    "get_change_request",
    "list_change_requests",
]
