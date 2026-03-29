"""Recherche plateforme V1 — InvestigationContext minimal (§19E : 3 types d’IDs)."""

from __future__ import annotations

import re
import uuid
from typing import Any

from ext import db
from models.booking import Booking
from models.company import Company
from models.user import User


def _uuid_like(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except Exception:
        return False


def search_investigation(query: str) -> dict[str, Any]:
    """Construit un InvestigationContext : liens vers ressources connues."""
    raw = (query or "").strip()
    q = raw.lower()
    matched: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    if raw.isdigit():
        n = int(raw)
        c = db.session.get(Company, n)
        if c:
            matched.append(
                {
                    "entity_type": "tenant",
                    "id": c.id,
                    "label": c.name,
                    "href_hint": f"/api/v1/platform/tenants/{c.id}",
                }
            )
            links.append({"rel": "tenant", "tenant_id": c.id})
        b = db.session.get(Booking, n)
        if b:
            matched.append(
                {
                    "entity_type": "booking",
                    "id": b.id,
                    "label": f"booking#{b.id}",
                    "company_id": b.company_id,
                }
            )
            links.append({"rel": "booking", "booking_id": b.id, "tenant_id": b.company_id})

    if _uuid_like(raw):
        u = User.query.filter_by(public_id=raw).first()
        if u:
            matched.append(
                {
                    "entity_type": "user",
                    "public_id": u.public_id,
                    "id": u.id,
                    "role": u.role.value if u.role else None,
                }
            )
            links.append({"rel": "user", "user_id": u.id})

    # Heuristique : préfixe booking:, tenant:, user:
    m = re.match(r"^\s*tenant\s*:\s*(\d+)\s*$", raw, re.I)
    if m:
        return search_investigation(m.group(1))
    m = re.match(r"^\s*booking\s*:\s*(\d+)\s*$", raw, re.I)
    if m:
        return search_investigation(m.group(1))

    return {
        "query": raw,
        "query_normalized": q,
        "matched_entities": matched,
        "links": links,
        "note": "Search universel LATER ; V1 = IDs numériques (tenant, booking) et UUID user.public_id.",
    }
