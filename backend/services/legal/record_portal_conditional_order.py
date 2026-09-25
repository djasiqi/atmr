"""Enregistrement CLIENT_CONDITIONAL_ORDER (7B.5)."""

from __future__ import annotations

from typing import Any

from ext import db
from models.portal_client_conditional_order import PortalClientConditionalOrder
from services.billing.portal_booking_debtor import resolve_portal_booking_debtor_user_id
from services.legal.portal_channel_cancellation_caps import (
    get_current_channel_cancellation_policy,
)
from services.legal.portal_double_validation import FLOW_CONDITIONAL_ORDER_V1
from services.legal.portal_terms_catalog import prepared_portal_terms_v21


def build_eligible_carriers_snapshot(ceiling_result: Any) -> list[dict[str, Any]]:
    """Identités juridiques sans prix — depuis le calcul plafond."""
    from models import Company

    quotes = getattr(ceiling_result, "quotes", None) or []
    company_ids: list[int] = []
    for q in quotes:
        cid = getattr(q, "company_id", None)
        if cid is None and isinstance(q, dict):
            cid = q.get("company_id")
        if cid is None:
            continue
        try:
            company_ids.append(int(cid))
        except (TypeError, ValueError):
            continue
    # Déduplique en gardant l'ordre.
    seen: set[int] = set()
    ordered: list[int] = []
    for cid in company_ids:
        if cid in seen:
            continue
        seen.add(cid)
        ordered.append(cid)

    snap: list[dict[str, Any]] = []
    for cid in ordered:
        company = Company.query.get(cid)
        if company is None:
            continue
        legal = (
            str(getattr(company, "legal_name", None) or "").strip()
            or str(getattr(company, "name", None) or "").strip()
            or f"Entreprise #{cid}"
        )
        row: dict[str, Any] = {
            "company_id": int(cid),
            "legal_name": legal,
        }
        uid = getattr(company, "uid_ide", None)
        if uid:
            row["uid_ide"] = str(uid)
        addr = (
            getattr(company, "domicile_address_line1", None)
            or getattr(company, "address", None)
        )
        if addr:
            row["address"] = str(addr)
        snap.append(row)
    return snap


def record_portal_conditional_order(
    *,
    booking: Any,
    user_id: int | None,
    maximum_accepted_amount: float,
    estimated_amount: float | None,
    eligible_carriers_snapshot: list[dict[str, Any]],
    trip_snapshot: dict[str, Any] | None = None,
) -> PortalClientConditionalOrder:
    """Insert append-only — une commande par booking."""
    existing = PortalClientConditionalOrder.query.filter_by(
        booking_id=int(booking.id)
    ).one_or_none()
    if existing is not None:
        return existing

    channel = get_current_channel_cancellation_policy()
    if channel is None:
        raise ValueError("portal_channel_cancellation_policy_required")

    tos, transport = prepared_portal_terms_v21()
    debtor = None
    try:
        debtor = resolve_portal_booking_debtor_user_id(booking)
    except Exception:
        debtor = user_id

    row = PortalClientConditionalOrder(
        booking_id=int(booking.id),
        flow_version=FLOW_CONDITIONAL_ORDER_V1,
        client_ceiling=float(maximum_accepted_amount),
        estimated_amount_snapshot=(
            float(estimated_amount) if estimated_amount is not None else None
        ),
        currency="CHF",
        eligible_carriers_snapshot=list(eligible_carriers_snapshot or []),
        terms_of_service_version=str(tos.terms_version),
        terms_of_service_hash=str(tos.terms_hash),
        transport_terms_version=str(transport.terms_version),
        transport_terms_hash=str(transport.terms_hash),
        channel_policy_version=str(channel.version),
        channel_policy_hash=str(channel.content_hash),
        channel_policy_snapshot=str(channel.body_text),
        trip_snapshot=trip_snapshot,
        debtor_user_id=int(debtor) if debtor else None,
    )
    db.session.add(row)
    db.session.flush()
    return row
