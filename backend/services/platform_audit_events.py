"""Lecture audit plateforme — filtres sur `audit_logs` (navigabilité grossière §17)."""

from __future__ import annotations

from typing import Any

from security.audit_log import AuditLog


def list_audit_events(
    *,
    page: int = 1,
    per_page: int = 50,
    action_type: str | None = None,
    action_category: str | None = None,
    company_id: int | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    per_page = min(max(per_page, 1), 200)
    page = max(page, 1)

    q = AuditLog.query
    if action_type:
        q = q.filter(AuditLog.action_type == action_type)
    if action_category:
        q = q.filter(AuditLog.action_category == action_category)
    if company_id is not None:
        q = q.filter(AuditLog.company_id == company_id)
    if correlation_id:
        q = q.filter(AuditLog.correlation_id == correlation_id)

    total = q.count()
    rows = (
        q.order_by(AuditLog.id.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    items = []
    for a in rows:
        items.append(
            {
                "id": a.id,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "action_type": a.action_type,
                "action_category": a.action_category,
                "user_id": a.user_id,
                "user_type": a.user_type,
                "result_status": a.result_status,
                "company_id": a.company_id,
                "correlation_id": a.correlation_id,
                "resource_type": a.resource_type,
                "resource_id": a.resource_id,
            }
        )

    pages = (total + per_page - 1) // per_page if total else 0
    return {
        "items": items,
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": pages,
    }


def replay_timeline_by_correlation_id(correlation_id: str) -> dict[str, Any]:
    """Reconstruction ordonnée des événements audit pour un correlation_id (replay V1)."""
    rows = (
        AuditLog.query.filter(AuditLog.correlation_id == correlation_id)
        .order_by(AuditLog.id.asc())
        .all()
    )
    events = []
    for a in rows:
        events.append(
            {
                "id": a.id,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "action_type": a.action_type,
                "result_status": a.result_status,
                "company_id": a.company_id,
                "resource_type": a.resource_type,
                "resource_id": a.resource_id,
            }
        )
    return {"correlation_id": correlation_id, "events": events, "count": len(events)}
