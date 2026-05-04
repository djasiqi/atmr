"""Agrégats GET /admin/billing/pilotage/* — pilotage billing plateforme (entreprise porteuse)."""

from __future__ import annotations

import csv
import io
import logging
import math
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from sqlalchemy.orm import joinedload

from ext import db
from models import Booking, Client, Company
from services.admin_booking_billing_kernel import (
    CLASSIFICATION_VERSION,
    QUALIFICATION_VERSION,
    booking_is_executed,
    build_pilotage_payload_for_booking,
    classify_booking_source,
    reliability_bucket_and_percent,
)
from services.admin_platform_bookings import (
    _batch_list_transfer_flags,
    build_admin_bookings_query,
    parse_admin_booking_request_args,
)

logger = logging.getLogger(__name__)

MAX_PER_PAGE = 100
EXPORT_MAX_ROWS = 5000


def _json_safe_filter_value(v: Any) -> Any:
    """Valeurs utilisables par jsonify (pas datetime brut, Decimal, Enum, NaN)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, Decimal):
        try:
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (TypeError, ValueError, OverflowError):
            return str(v)
    if isinstance(v, Enum):
        ev = getattr(v, "value", v)
        return ev if isinstance(ev, (str, int, float, bool)) or ev is None else str(ev)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if hasattr(v, "isoformat") and callable(v.isoformat):
        try:
            return v.isoformat()
        except (TypeError, ValueError):
            return str(v)
    if isinstance(v, str):
        return v
    return str(v)


def _serialize_filters_applied(params: dict[str, Any]) -> dict[str, Any]:
    return {k: _json_safe_filter_value(v) for k, v in params.items() if v is not None}


def _dt_iso_maybe(v: Any) -> str | None:
    """Champ date/heure en JSON — tolère str héritée ou types SQL atypiques."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if hasattr(v, "isoformat") and callable(v.isoformat):
        try:
            return v.isoformat()
        except (TypeError, ValueError):
            return str(v)
    return str(v)


def _pilotage_json_sanitize(obj: Any) -> Any:
    """Récursif : garantit une structure sérialisable par jsonify (admin pilotage)."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Decimal):
        try:
            f = float(obj)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        ev = getattr(obj, "value", None)
        if isinstance(ev, (str, int, float, bool)) or ev is None:
            return ev
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _pilotage_json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pilotage_json_sanitize(x) for x in obj]
    if isinstance(obj, set):
        return [_pilotage_json_sanitize(x) for x in obj]
    return str(obj)


def _parse_date_start(s: str | None):
    if not s or not str(s).strip():
        return None
    return datetime.fromisoformat(f"{str(s).strip()}T00:00:00+00:00")


def _parse_date_end(s: str | None):
    if not s or not str(s).strip():
        return None
    return datetime.fromisoformat(f"{str(s).strip()}T23:59:59.999999+00:00")


def parse_pilotage_request_args(args) -> dict[str, Any]:
    """Query args pilotage (filtres alignés admin bookings + carrier_only implicite)."""
    base = parse_admin_booking_request_args(args)
    base.pop("page", None)
    base.pop("per_page", None)
    base.pop("sort", None)
    base.pop("order", None)
    return base


def parse_pilotage_list_args(args) -> dict[str, Any]:
    base = parse_pilotage_request_args(args)
    base["page"] = max(1, args.get("page", default=1, type=int) or 1)
    base["per_page"] = args.get("per_page", default=25, type=int) or 25
    base["sort"] = (args.get("sort") or "total_bookings").strip()
    base["order"] = (args.get("order") or "desc").strip()
    return base


def parse_pilotage_detail_args(args) -> dict[str, Any]:
    base = parse_pilotage_request_args(args)
    base["page"] = max(1, args.get("page", default=1, type=int) or 1)
    base["per_page"] = args.get("per_page", default=50, type=int) or 50
    return base


def _default_period_if_needed(
    kwargs: dict[str, Any],
) -> tuple[datetime | None, datetime | None]:
    """Si aucune borne temporelle : période glissante 30 jours sur created_at."""
    cf, ct = kwargs.get("created_from"), kwargs.get("created_to")
    sf, st = kwargs.get("scheduled_from"), kwargs.get("scheduled_to")
    if any(x is not None for x in (cf, ct, sf, st)):
        return None, None
    now = datetime.now(UTC)
    start = now - timedelta(days=30)
    return start, now


def _build_pilotage_query(**filter_kwargs: Any):
    d_from, d_to = _default_period_if_needed(filter_kwargs)
    if d_from is not None:
        filter_kwargs = {**filter_kwargs, "created_from": d_from, "created_to": d_to}

    return build_admin_bookings_query(
        q=filter_kwargs.get("q"),
        created_from=filter_kwargs.get("created_from"),
        created_to=filter_kwargs.get("created_to"),
        scheduled_from=filter_kwargs.get("scheduled_from"),
        scheduled_to=filter_kwargs.get("scheduled_to"),
        statuses=filter_kwargs.get("statuses"),
        institution_id=filter_kwargs.get("institution_id"),
        company_id=filter_kwargs.get("company_id"),
        institution_q=filter_kwargs.get("institution_q"),
        company_q=filter_kwargs.get("company_q"),
        cancelled_only=bool(filter_kwargs.get("cancelled_only")),
        exclude_cancelled=bool(filter_kwargs.get("exclude_cancelled")),
        with_transfer=filter_kwargs.get("with_transfer"),
        unassigned=filter_kwargs.get("unassigned"),
        incomplete_data=filter_kwargs.get("incomplete_data"),
        needs_investigation=filter_kwargs.get("needs_investigation"),
        company_scope="carrier_only",
    )


def _parse_status_list_for_pilotage(raw: str | None):
    from services.admin_platform_bookings import _parse_status_list

    return _parse_status_list(raw)


def _prepare_filter_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    fk = {k: v for k, v in params.items() if v is not None and k != "status"}
    fk["statuses"] = _parse_status_list_for_pilotage(params.get("status"))
    return fk


def _scan_bookings(
    base_query,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], list[Booking]]:
    """Un passage : KPIs globaux + agrégats par company_id porteur."""
    q = base_query.options(
        joinedload(Booking.client).joinedload(Client.user),
        joinedload(Booking.client).joinedload(Client.linked_institution),
        joinedload(Booking.company),
        joinedload(Booking.executing_company),
    ).order_by(None)

    rows: list[Booking] = q.all()
    flags = _batch_list_transfer_flags(rows)

    kpis_global = {
        "total_bookings": 0,
        "active_companies": set(),
        "institution": 0,
        "manual_direct": 0,
        "unknown_source": 0,
        "total_observed_amount": 0.0,
        "eligible": 0,
        "ambiguous": 0,
        "needs_review": 0,
        "excluded": 0,
        "executed": 0,
        "transferred": 0,
        "with_source_classified": 0,
        "with_observed_amount": 0,
    }
    fam_global: dict[str, int] = defaultdict(int)

    by_company: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "company_id": 0,
            "total_bookings": 0,
            "institution": 0,
            "manual_direct": 0,
            "company_manual": 0,
            "client_direct": 0,
            "admin_created": 0,
            "unknown_source": 0,
            "total_observed_amount": 0.0,
            "eligible": 0,
            "ambiguous": 0,
            "needs_review": 0,
            "excluded": 0,
            "executed": 0,
            "transferred": 0,
        }
    )

    for b in rows:
        kpis_global["total_bookings"] += 1
        cid = b.company_id
        if cid is not None:
            kpis_global["active_companies"].add(cid)

        ht, hp = flags[b.id]
        pl = build_pilotage_payload_for_booking(
            b, has_transfer=ht, has_pending_transfer=hp
        )
        src = pl["source_code"]
        st = pl["qualification"]["state"]
        amt = pl["observed_transport_amount"]

        if src == "institution_request":
            kpis_global["institution"] += 1
        elif src in ("company_manual", "client_direct", "admin_created"):
            kpis_global["manual_direct"] += 1
        elif src == "unknown_source":
            kpis_global["unknown_source"] += 1

        if src != "unknown_source":
            kpis_global["with_source_classified"] += 1
        if amt is not None:
            kpis_global["with_observed_amount"] += 1
            kpis_global["total_observed_amount"] += amt

        if st == "eligible":
            kpis_global["eligible"] += 1
        elif st == "ambiguous":
            kpis_global["ambiguous"] += 1
        elif st == "needs_review":
            kpis_global["needs_review"] += 1
        elif st == "excluded":
            kpis_global["excluded"] += 1

        if booking_is_executed(b):
            kpis_global["executed"] += 1
        if ht:
            kpis_global["transferred"] += 1

        for fam in pl["qualification"]["families"]:
            fam_global[fam] += 1

        if cid is None:
            continue

        bc = by_company[cid]
        bc["company_id"] = cid
        bc["total_bookings"] += 1
        if src == "institution_request":
            bc["institution"] += 1
        elif src in ("company_manual", "client_direct", "admin_created"):
            bc["manual_direct"] += 1
        elif src == "unknown_source":
            bc["unknown_source"] += 1
        if amt is not None:
            bc["total_observed_amount"] += amt
        if st == "eligible":
            bc["eligible"] += 1
        elif st == "ambiguous":
            bc["ambiguous"] += 1
        elif st == "needs_review":
            bc["needs_review"] += 1
        elif st == "excluded":
            bc["excluded"] += 1
        if booking_is_executed(b):
            bc["executed"] += 1
        if ht:
            bc["transferred"] += 1

    ac = len(kpis_global["active_companies"])
    del kpis_global["active_companies"]
    kpis_global["active_companies"] = ac

    return kpis_global, dict(by_company), dict(fam_global)


def _period_from_filters(params: dict[str, Any]) -> dict[str, Any]:
    cf, ct = params.get("created_from"), params.get("created_to")
    sf, st = params.get("scheduled_from"), params.get("scheduled_to")
    if cf is None and ct is None and sf is None and st is None:
        now = datetime.now(UTC)
        return {
            "from": (now - timedelta(days=30)).date().isoformat(),
            "to": now.date().isoformat(),
            "field": "created_at_default_30d",
        }
    return {
        "created_from": cf.isoformat() if cf else None,
        "created_to": ct.isoformat() if ct else None,
        "scheduled_from": sf.isoformat() if sf else None,
        "scheduled_to": st.isoformat() if st else None,
    }


def build_pilotage_summary(**params: Any) -> dict[str, Any]:
    fk = _prepare_filter_kwargs(params)
    base = _build_pilotage_query(**fk)
    kpis_global, _, fam_global = _scan_bookings(base)

    scope_summary = {
        "bookings_in_scope": kpis_global["total_bookings"],
        "with_source_classified": kpis_global["with_source_classified"],
        "with_observed_amount": kpis_global["with_observed_amount"],
        "by_qualification": {
            "eligible": kpis_global["eligible"],
            "ambiguous": kpis_global["ambiguous"],
            "needs_review": kpis_global["needs_review"],
            "excluded": kpis_global["excluded"],
        },
    }

    return _pilotage_json_sanitize(
        {
            "filters_applied": _serialize_filters_applied(params),
            "period": _period_from_filters(fk),
            "classification_version": CLASSIFICATION_VERSION,
            "qualification_version": QUALIFICATION_VERSION,
            "kpis": {
                "active_companies": kpis_global["active_companies"],
                "total_bookings": kpis_global["total_bookings"],
                "activity_institution": kpis_global["institution"],
                "activity_manual_direct": kpis_global["manual_direct"],
                "total_observed_amount": round(kpis_global["total_observed_amount"], 2),
                "reservations_eligible": kpis_global["eligible"],
                "reservations_needs_review": kpis_global["needs_review"],
                "reservations_ambiguous_secondary": kpis_global["ambiguous"],
                "executed": kpis_global["executed"],
                "transferred": kpis_global["transferred"],
            },
            "scope_summary": scope_summary,
            "anomaly_families": fam_global,
        }
    )


def _enrich_company_row(
    cid: int,
    agg: dict[str, Any],
) -> dict[str, Any]:
    co = db.session.get(Company, cid)
    name = co.name if co else None
    suspended = bool(getattr(co, "platform_suspended", False)) if co else False
    billing_email = getattr(co, "billing_email", None) if co else None
    tot = agg["total_bookings"]
    el = agg["eligible"]
    nr = agg["needs_review"]
    am = agg["ambiguous"]
    ex = agg["excluded"]
    bucket, pct = reliability_bucket_and_percent(
        eligible=el,
        needs_review=nr,
        ambiguous=am,
        excluded=ex,
        total=tot,
    )
    return {
        "company_id": cid,
        "company_name": name,
        "account_suspended": suspended,
        "billing_email": billing_email,
        "total_bookings": tot,
        "institution": agg["institution"],
        "manual_direct": agg["manual_direct"],
        "unknown_source": agg.get("unknown_source", 0),
        "manual_direct_breakdown": {
            "company_manual": agg.get("company_manual", 0),
            "client_direct": agg.get("client_direct", 0),
            "admin_created": agg.get("admin_created", 0),
        },
        "executed": agg["executed"],
        "transferred": agg["transferred"],
        "total_observed_amount": round(agg["total_observed_amount"], 2),
        "eligible": el,
        "ambiguous": am,
        "needs_review": nr,
        "excluded": ex,
        "reliability": {"bucket": bucket, "percent": pct},
    }


def list_pilotage_companies(
    *,
    page: int = 1,
    per_page: int = 25,
    sort: str = "total_bookings",
    order: str = "desc",
    **params: Any,
) -> dict[str, Any]:
    params = {
        k: v
        for k, v in params.items()
        if k not in ("page", "per_page", "sort", "order")
    }
    fk = _prepare_filter_kwargs(params)
    base = _build_pilotage_query(**fk)
    kpis_global, by_company, fam_global = _scan_bookings(base)

    rows = [_enrich_company_row(cid, agg) for cid, agg in by_company.items()]
    reverse = (order or "desc").lower() != "asc"
    sk = (sort or "total_bookings").lower()
    rows.sort(key=lambda r: r.get(sk, 0) or "", reverse=reverse)

    per_page = min(max(1, per_page), MAX_PER_PAGE)
    page = max(1, page)
    total_items = len(rows)
    start = (page - 1) * per_page
    chunk = rows[start : start + per_page]

    scope_summary = {
        "bookings_in_scope": kpis_global["total_bookings"],
        "with_source_classified": kpis_global["with_source_classified"],
        "with_observed_amount": kpis_global["with_observed_amount"],
        "by_qualification": {
            "eligible": kpis_global["eligible"],
            "ambiguous": kpis_global["ambiguous"],
            "needs_review": kpis_global["needs_review"],
            "excluded": kpis_global["excluded"],
        },
    }
    return _pilotage_json_sanitize(
        {
            "filters_applied": _serialize_filters_applied(params),
            "period": _period_from_filters(fk),
            "classification_version": CLASSIFICATION_VERSION,
            "qualification_version": QUALIFICATION_VERSION,
            "scope_summary": scope_summary,
            "anomaly_families": fam_global,
            "items": chunk,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_items": total_items,
                "total_pages": (total_items + per_page - 1) // per_page
                if total_items
                else 0,
            },
        }
    )


def _safe_created_ts(booking: Booking) -> float:
    """Tri par date ; naive en UTC + évite OSError sur .timestamp() (Windows)."""
    dt = booking.created_at
    if dt is None:
        return 0.0
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except (OSError, OverflowError, ValueError):
        return 0.0


def _sort_key_detail(booking: Booking, pl: dict[str, Any]) -> tuple[int, float]:
    st = pl["qualification"]["state"]
    prio = {"needs_review": 0, "ambiguous": 1, "eligible": 2, "excluded": 3}.get(st, 4)
    ts = _safe_created_ts(booking)
    return (prio, -ts)


def get_pilotage_company_detail(
    company_id: int,
    *,
    page: int = 1,
    per_page: int = 50,
    **params: Any,
) -> dict[str, Any] | None:
    params = {k: v for k, v in params.items() if k not in ("page", "per_page")}
    co = db.session.get(Company, company_id)
    if co is None:
        return None

    fk = _prepare_filter_kwargs({**params, "company_id": company_id})
    base = _build_pilotage_query(**fk)
    _, by_company, fam_global = _scan_bookings(base)
    agg = by_company.get(company_id)
    if not agg:
        agg = {
            "company_id": company_id,
            "total_bookings": 0,
            "institution": 0,
            "manual_direct": 0,
            "company_manual": 0,
            "client_direct": 0,
            "admin_created": 0,
            "unknown_source": 0,
            "total_observed_amount": 0.0,
            "eligible": 0,
            "ambiguous": 0,
            "needs_review": 0,
            "excluded": 0,
            "executed": 0,
            "transferred": 0,
        }

    q = base.filter(Booking.company_id == company_id).options(
        joinedload(Booking.client).joinedload(Client.user),
        joinedload(Booking.executing_company),
    )
    bookings = q.all()
    flags = _batch_list_transfer_flags(bookings)
    items_raw: list[tuple[Booking, dict[str, Any]]] = []
    for b in bookings:
        ht, hp = flags[b.id]
        pl = build_pilotage_payload_for_booking(
            b, has_transfer=ht, has_pending_transfer=hp
        )
        items_raw.append((b, pl))
    items_raw.sort(key=lambda x: _sort_key_detail(x[0], x[1]))

    per_page = min(max(1, per_page), MAX_PER_PAGE)
    page = max(1, page)
    total_items = len(items_raw)
    start = (page - 1) * per_page
    slice_raw = items_raw[start : start + per_page]

    booking_rows = []
    for b, pl in slice_raw:
        booking_rows.append(
            {
                "booking_id": b.id,
                "created_at": _dt_iso_maybe(b.created_at),
                "scheduled_at": _dt_iso_maybe(b.scheduled_time),
                "status": b.status.value
                if hasattr(b.status, "value")
                else str(b.status),
                "executing_company_id": b.executing_company_id,
                "executing_company_name": b.executing_company.name
                if b.executing_company
                else None,
                "pilotage": pl,
            }
        )

    br, pct = reliability_bucket_and_percent(
        eligible=agg["eligible"],
        needs_review=agg["needs_review"],
        ambiguous=agg["ambiguous"],
        excluded=agg["excluded"],
        total=agg["total_bookings"],
    )

    src_breakdown = defaultdict(int)
    for b in bookings:
        src_breakdown[classify_booking_source(b)] += 1

    payload = {
        "filters_applied": _serialize_filters_applied(
            {**params, "company_id": company_id}
        ),
        "period": _period_from_filters(fk),
        "classification_version": CLASSIFICATION_VERSION,
        "qualification_version": QUALIFICATION_VERSION,
        "company": {
            "id": co.id,
            "name": co.name,
            "billing_email": getattr(co, "billing_email", None),
            "platform_suspended": bool(getattr(co, "platform_suspended", False)),
        },
        "summary": _enrich_company_row(company_id, agg),
        "source_breakdown": dict(src_breakdown),
        "qualification_breakdown": {
            "eligible": agg["eligible"],
            "ambiguous": agg["ambiguous"],
            "needs_review": agg["needs_review"],
            "excluded": agg["excluded"],
        },
        "anomaly_families": fam_global,
        "reliability": {"bucket": br, "percent": pct},
        "bookings": booking_rows,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_items": total_items,
            "total_pages": (total_items + per_page - 1) // per_page
            if total_items
            else 0,
        },
    }
    return _pilotage_json_sanitize(payload)


def export_pilotage_csv(**params: Any) -> tuple[bytes, str]:
    fk = _prepare_filter_kwargs(params)
    base = _build_pilotage_query(**fk)
    q = base.options(
        joinedload(Booking.client).joinedload(Client.user),
        joinedload(Booking.company),
    ).order_by(Booking.id.desc())
    rows = q.limit(EXPORT_MAX_ROWS).all()
    flags = _batch_list_transfer_flags(rows)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "booking_id",
            "company_id",
            "source_code",
            "qualification_state",
            "observed_transport_amount",
            "needs_review",
            "ambiguous",
        ]
    )
    for b in rows:
        ht, hp = flags[b.id]
        pl = build_pilotage_payload_for_booking(
            b, has_transfer=ht, has_pending_transfer=hp
        )
        st = pl["qualification"]["state"]
        w.writerow(
            [
                b.id,
                b.company_id,
                pl["source_code"],
                st,
                pl["observed_transport_amount"]
                if pl["observed_transport_amount"] is not None
                else "",
                1 if st == "needs_review" else 0,
                1 if st == "ambiguous" else 0,
            ]
        )
    return buf.getvalue().encode(
        "utf-8-sig"
    ), f"pilotage_export_{datetime.now(UTC).date()}.csv"
