"""Liste et détail réservations — supervision plateforme admin."""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from marshmallow import ValidationError
from sqlalchemy import and_, exists, func, not_, or_, select
from sqlalchemy.orm import joinedload

from ext import db
from models import Booking, BookingStatus, Client, Company, Institution
from models.booking_transfer import BookingTransfer
from models.enums import TransferStatus
from schemas.validation_utils import ISO8601_DATE_REGEX
from security.audit_log import AuditLog
from services.admin_booking_billing_kernel import build_pilotage_payload_for_booking
from services.admin_booking_labels import booking_status_label_fr

logger = logging.getLogger(__name__)

MAX_PER_PAGE = 100
EXPORT_MAX_ROWS = 5000
ISO_DATE_PATTERN = re.compile(ISO8601_DATE_REGEX)


def _parse_bool_arg(raw: str | None) -> bool | None:
    if raw is None or raw == "":
        return None
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


def _parse_date_start(s: str | None, field_name: str):
    if not s or not str(s).strip():
        return None
    value = str(s).strip()
    if not ISO_DATE_PATTERN.fullmatch(value):
        raise ValidationError(
            {field_name: ["Date invalide (format attendu YYYY-MM-DD)"]}
        )
    return datetime.fromisoformat(f"{value}T00:00:00+00:00")


def _parse_date_end(s: str | None, field_name: str):
    if not s or not str(s).strip():
        return None
    value = str(s).strip()
    if not ISO_DATE_PATTERN.fullmatch(value):
        raise ValidationError(
            {field_name: ["Date invalide (format attendu YYYY-MM-DD)"]}
        )
    return datetime.fromisoformat(f"{value}T23:59:59.999999+00:00")


def parse_admin_booking_request_args(args) -> dict[str, Any]:
    """Transforme request.args en kwargs pour `list_admin_platform_bookings` / export."""
    institution_id = args.get("institution_id", type=int)
    company_id = args.get("company_id", type=int)
    created_from = _parse_date_start(args.get("created_from"), "created_from")
    created_to = _parse_date_end(args.get("created_to"), "created_to")
    scheduled_from = _parse_date_start(args.get("scheduled_from"), "scheduled_from")
    scheduled_to = _parse_date_end(args.get("scheduled_to"), "scheduled_to")
    if created_from and created_to and created_from > created_to:
        raise ValidationError(
            {"created_to": ["created_to doit être postérieure ou égale à created_from"]}
        )
    if scheduled_from and scheduled_to and scheduled_from > scheduled_to:
        raise ValidationError(
            {
                "scheduled_to": [
                    "scheduled_to doit être postérieure ou égale à scheduled_from"
                ]
            }
        )
    return {
        "page": max(1, args.get("page", default=1, type=int) or 1),
        "per_page": args.get("per_page", default=25, type=int) or 25,
        "sort": (args.get("sort") or "scheduled_time").strip(),
        "order": (args.get("order") or "desc").strip(),
        "q": (args.get("q") or "").strip() or None,
        "status": (args.get("status") or "").strip() or None,
        "created_from": created_from,
        "created_to": created_to,
        "scheduled_from": scheduled_from,
        "scheduled_to": scheduled_to,
        "institution_id": institution_id,
        "company_id": company_id,
        "institution_q": (args.get("institution_q") or "").strip() or None,
        "company_q": (args.get("company_q") or "").strip() or None,
        "cancelled_only": _parse_bool_arg(args.get("cancelled_only")),
        "exclude_cancelled": _parse_bool_arg(args.get("exclude_cancelled")),
        "with_transfer": _parse_bool_arg(args.get("with_transfer")),
        "unassigned": _parse_bool_arg(args.get("unassigned")),
        "incomplete_data": _parse_bool_arg(args.get("incomplete_data")),
        "needs_investigation": _parse_bool_arg(args.get("needs_investigation")),
    }


def _parse_status_list(raw: str | None) -> list[BookingStatus] | None:
    if not raw or not str(raw).strip():
        return None
    out: list[BookingStatus] = []
    for raw_part in str(raw).upper().split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            out.append(BookingStatus[part])
        except KeyError:
            try:
                out.append(BookingStatus(part))
            except ValueError:
                logger.warning("Statut booking ignoré: %s", part)
    return out or None


def _unassigned_condition():
    return and_(
        Booking.company_id.is_(None),
        Booking.executing_company_id.is_(None),
    )


def _incomplete_data_condition():
    return or_(
        Booking.scheduled_time.is_(None),
        func.coalesce(func.trim(Booking.customer_name), "") == "",
        func.coalesce(func.trim(Booking.pickup_location), "") == "",
        func.coalesce(func.trim(Booking.dropoff_location), "") == "",
    )


def _transfer_accepted_or_completed_exists():
    return exists(
        select(BookingTransfer.id).where(
            and_(
                BookingTransfer.booking_id == Booking.id,
                BookingTransfer.status.in_(
                    [TransferStatus.ACCEPTED, TransferStatus.COMPLETED]
                ),
            )
        )
    )


def _transfer_pending_blocked_exists():
    return exists(
        select(BookingTransfer.id).where(
            and_(
                BookingTransfer.booking_id == Booking.id,
                BookingTransfer.status == TransferStatus.PENDING,
            )
        )
    )


def _needs_investigation_condition(now: datetime):
    stale_threshold = now - timedelta(hours=24)
    return or_(
        _incomplete_data_condition(),
        and_(
            Booking.status == BookingStatus.PENDING,
            Booking.scheduled_time.isnot(None),
            Booking.scheduled_time < stale_threshold,
        ),
        _transfer_pending_blocked_exists(),
    )


def build_admin_bookings_query(
    *,
    q: str | None = None,
    created_from: datetime | None = None,
    created_to: datetime | None = None,
    scheduled_from: datetime | None = None,
    scheduled_to: datetime | None = None,
    statuses: list[BookingStatus] | None = None,
    institution_id: int | None = None,
    company_id: int | None = None,
    institution_q: str | None = None,
    company_q: str | None = None,
    cancelled_only: bool = False,
    exclude_cancelled: bool = False,
    with_transfer: bool | None = None,
    unassigned: bool | None = None,
    incomplete_data: bool | None = None,
    needs_investigation: bool | None = None,
    company_scope: str | None = None,
) -> Any:
    """Construit la requête SQLAlchemy filtrée (sans tri ni pagination).

    company_scope:
      - None ou \"default\" : filtre entreprise = porteur OU exécutant (comportement historique).
      - \"carrier_only\" : uniquement Booking.company_id (pilotage billing plateforme).
    """
    query = Booking.query

    if q and str(q).strip():
        term = str(q).strip()
        if term.isdigit():
            query = query.filter(Booking.id == int(term))
        else:
            like = f"%{term}%"
            query = query.filter(
                or_(
                    Booking.customer_name.ilike(like),
                    Booking.pickup_location.ilike(like),
                    Booking.dropoff_location.ilike(like),
                )
            )

    if created_from is not None:
        query = query.filter(Booking.created_at >= created_from)
    if created_to is not None:
        query = query.filter(Booking.created_at <= created_to)
    if scheduled_from is not None:
        query = query.filter(Booking.scheduled_time >= scheduled_from)
    if scheduled_to is not None:
        query = query.filter(Booking.scheduled_time <= scheduled_to)

    if statuses:
        query = query.filter(Booking.status.in_(statuses))

    if institution_id is not None:
        query = query.filter(
            Booking.client.has(Client.linked_institution_id == institution_id)
        )

    if company_id is not None:
        if company_scope == "carrier_only":
            query = query.filter(Booking.company_id == company_id)
        else:
            query = query.filter(
                or_(
                    Booking.company_id == company_id,
                    Booking.executing_company_id == company_id,
                )
            )

    if institution_q and str(institution_q).strip():
        iq = f"%{str(institution_q).strip()}%"
        query = query.filter(
            Booking.client.has(
                Client.linked_institution.has(Institution.name.ilike(iq))
            )
        )

    if company_q and str(company_q).strip():
        cq = f"%{str(company_q).strip()}%"
        query = query.filter(
            or_(
                Booking.company.has(Company.name.ilike(cq)),
                Booking.executing_company.has(Company.name.ilike(cq)),
            )
        )

    if cancelled_only:
        query = query.filter(Booking.status == BookingStatus.CANCELED)
    elif exclude_cancelled:
        query = query.filter(Booking.status != BookingStatus.CANCELED)

    if with_transfer is True:
        query = query.filter(_transfer_accepted_or_completed_exists())
    elif with_transfer is False:
        query = query.filter(~_transfer_accepted_or_completed_exists())

    if unassigned is True:
        query = query.filter(_unassigned_condition())
    elif unassigned is False:
        query = query.filter(~_unassigned_condition())

    if incomplete_data is True:
        query = query.filter(_incomplete_data_condition())
    elif incomplete_data is False:
        query = query.filter(~_incomplete_data_condition())

    if needs_investigation is True:
        query = query.filter(_needs_investigation_condition(datetime.now(UTC)))
    elif needs_investigation is False:
        query = query.filter(not_(_needs_investigation_condition(datetime.now(UTC))))

    return query


def _batch_list_transfer_flags(bookings: list[Booking]) -> dict[int, tuple[bool, bool]]:
    """Pour chaque booking : (has_transfer affichage, transfert PENDING).

    Évite N+1 sur `_is_transferred` et `_compute_needs_investigation_booking`.
    Logique alignée sur `Booking._is_transferred` (ACCEPTED/COMPLETED + owner != company).
    """
    if not bookings:
        return {}
    ids = [b.id for b in bookings]
    rows = BookingTransfer.query.filter(BookingTransfer.booking_id.in_(ids)).all()
    by_booking: dict[int, list[BookingTransfer]] = {}
    for t in rows:
        by_booking.setdefault(t.booking_id, []).append(t)
    out: dict[int, tuple[bool, bool]] = {}
    for b in bookings:
        transfers = by_booking.get(b.id, [])
        has_pending = any(t.status == TransferStatus.PENDING for t in transfers)
        has_transfer = False
        for t in transfers:
            if (
                t.status
                in (
                    TransferStatus.ACCEPTED,
                    TransferStatus.COMPLETED,
                )
                and t.owner_company_id != b.company_id
            ):
                has_transfer = True
                break
        out[b.id] = (has_transfer, has_pending)
    return out


def _order_clause(sort: str, order: str):
    col = {
        "scheduled_time": Booking.scheduled_time,
        "created_at": Booking.created_at,
        "id": Booking.id,
    }.get((sort or "scheduled_time").lower(), Booking.scheduled_time)
    asc = (order or "desc").lower() == "asc"
    return col.asc() if asc else col.desc()


def _serialize_created_by(booking: Booking) -> dict[str, Any]:
    try:
        tl = booking._get_institution_timeline()
        if tl and tl.get("created_by_name"):
            return {
                "source": "institution_request",
                "label": tl.get("created_by_name"),
                "institution_name": tl.get("institution_name"),
            }
    except Exception:
        pass
    cli = booking.client
    if cli and cli.user:
        u = cli.user
        name = f"{u.first_name or ''} {u.last_name or ''}".strip() or u.username
        return {"source": "client", "label": name}
    return {"source": "unknown", "label": None}


def _serialize_cancelled_by(booking: Booking) -> dict[str, Any] | None:
    if booking.status != BookingStatus.CANCELED:
        return None
    cancelled_at = getattr(booking, "cancelled_at", None)
    return {
        "role": booking.cancelled_by_role,
        "cancelled_at": cancelled_at.isoformat() if cancelled_at is not None else None,
        "reason_code": booking.cancellation_reason_code,
    }


def admin_booking_list_item(
    booking: Booking,
    *,
    has_transfer: bool | None = None,
) -> dict[str, Any]:
    status_val = booking.status
    key = status_val.value if hasattr(status_val, "value") else str(status_val).upper()
    inst_name = None
    if booking.client:
        li = getattr(booking.client, "linked_institution", None)
        if li is not None:
            inst_name = getattr(li, "name", None)
        if not inst_name:
            inst_name = getattr(booking.client, "institution_name", None)

    current_company = booking.executing_company or booking.company
    current_company_name = (
        booking.executing_company.name
        if booking.executing_company
        else (booking.company.name if booking.company else None)
    )
    created_at = getattr(booking, "created_at", None)
    scheduled_time = getattr(booking, "scheduled_time", None)
    amount = getattr(booking, "amount", None)

    from services.companies.booking_display import build_booking_display_blocks

    viewer_id = current_company.id if current_company else None
    display_blocks = build_booking_display_blocks(
        booking, viewer_company_id=viewer_id
    )
    scheduling = display_blocks.get("scheduling") or {}
    identity = display_blocks.get("identity") or {}

    return {
        "id": booking.id,
        "display_model": display_blocks.get("display_model"),
        "display_model_version": display_blocks.get("display_model_version"),
        "identity": identity,
        "scheduling": scheduling,
        "trip_flags": display_blocks.get("trip_flags"),
        "search_index": display_blocks.get("search_index"),
        "created_at": created_at.isoformat() if created_at is not None else None,
        "scheduled_at": (
            scheduled_time.isoformat() if scheduled_time is not None else None
        ),
        "client_name": identity.get("primary_label") or booking.customer_full_name,
        "institution_name": inst_name,
        "current_company_name": current_company_name,
        "current_company_id": current_company.id if current_company else None,
        "status": key.lower(),
        "status_label": booking_status_label_fr(status_val),
        "has_transfer": (
            bool(booking._is_transferred()) if has_transfer is None else has_transfer
        ),
        "amount_chf": float(amount) if amount is not None else None,
        "pickup_label": (booking.pickup_location or "")[:120],
        "dropoff_label": (booking.dropoff_location or "")[:120],
        "created_by": _serialize_created_by(booking),
        "cancelled_by": _serialize_cancelled_by(booking),
        "incomplete_data": _evaluate_incomplete(booking),
    }


def _evaluate_incomplete(booking: Booking) -> bool:
    if booking.scheduled_time is None:
        return True
    if not (booking.customer_name or "").strip():
        return True
    if not (booking.pickup_location or "").strip():
        return True
    return not (booking.dropoff_location or "").strip()


def _compute_needs_investigation_booking(
    booking: Booking,
    *,
    has_pending_transfer: bool | None = None,
) -> bool:
    now = datetime.now(UTC)
    if _evaluate_incomplete(booking):
        return True
    if booking.status == BookingStatus.PENDING and booking.scheduled_time:
        st = booking.scheduled_time
        if st.tzinfo is None:
            st = st.replace(tzinfo=UTC)
        if st < now - timedelta(hours=24):
            return True
    if has_pending_transfer is True:
        return True
    if has_pending_transfer is False:
        return False
    try:
        pending_tr = (
            BookingTransfer.query.filter_by(booking_id=booking.id)
            .filter_by(status=TransferStatus.PENDING)
            .first()
        )
        if pending_tr:
            return True
    except Exception:
        pass
    return False


def admin_booking_list_item_fixed(
    booking: Booking,
    *,
    has_transfer: bool | None = None,
    has_pending_transfer: bool | None = None,
) -> dict[str, Any]:
    """Item liste avec `needs_investigation` calculé en Python."""
    base = admin_booking_list_item(booking, has_transfer=has_transfer)
    base["needs_investigation"] = _compute_needs_investigation_booking(
        booking, has_pending_transfer=has_pending_transfer
    )
    ht = bool(booking._is_transferred()) if has_transfer is None else has_transfer
    hp = has_pending_transfer
    if hp is None:
        hp = _batch_list_transfer_flags([booking]).get(booking.id, (False, False))[1]
    base["pilotage"] = build_pilotage_payload_for_booking(
        booking, has_transfer=ht, has_pending_transfer=bool(hp)
    )
    return base


def compute_summary(query) -> dict[str, int]:
    """Compteurs sur le même jeu filtré que `query` (avant pagination)."""
    total = query.order_by(None).count()
    q_unassigned = query.filter(_unassigned_condition())
    q_canceled = query.filter(Booking.status == BookingStatus.CANCELED)
    q_transferred = query.filter(_transfer_accepted_or_completed_exists())
    q_incomplete = query.filter(_incomplete_data_condition())
    q_inv = query.filter(_needs_investigation_condition(datetime.now(UTC)))

    return {
        "total": total,
        "unassigned": q_unassigned.order_by(None).count(),
        "canceled": q_canceled.order_by(None).count(),
        "transferred": q_transferred.order_by(None).count(),
        "incomplete_data": q_incomplete.order_by(None).count(),
        "needs_investigation": q_inv.order_by(None).count(),
    }


def list_admin_platform_bookings(
    *,
    page: int = 1,
    per_page: int = 25,
    sort: str = "scheduled_time",
    order: str = "desc",
    **filter_kwargs: Any,
) -> dict[str, Any]:
    """Liste paginée + summary pour GET /admin/bookings."""
    per_page = min(max(1, per_page), MAX_PER_PAGE)
    page = max(1, page)

    statuses = _parse_status_list(filter_kwargs.get("status"))
    created_from = filter_kwargs.get("created_from")
    created_to = filter_kwargs.get("created_to")
    scheduled_from = filter_kwargs.get("scheduled_from")
    scheduled_to = filter_kwargs.get("scheduled_to")

    base = build_admin_bookings_query(
        q=filter_kwargs.get("q"),
        created_from=created_from,
        created_to=created_to,
        scheduled_from=scheduled_from,
        scheduled_to=scheduled_to,
        statuses=statuses,
        institution_id=filter_kwargs.get("institution_id"),
        company_id=filter_kwargs.get("company_id"),
        cancelled_only=bool(filter_kwargs.get("cancelled_only")),
        exclude_cancelled=bool(filter_kwargs.get("exclude_cancelled")),
        with_transfer=filter_kwargs.get("with_transfer"),
        unassigned=filter_kwargs.get("unassigned"),
        incomplete_data=filter_kwargs.get("incomplete_data"),
        needs_investigation=filter_kwargs.get("needs_investigation"),
    )

    summary = compute_summary(base)
    ordered = base.options(
        joinedload(Booking.client).joinedload(Client.user),
        joinedload(Booking.client).joinedload(Client.linked_institution),
        joinedload(Booking.company),
        joinedload(Booking.executing_company),
    ).order_by(_order_clause(sort, order))
    pagination = ordered.paginate(page=page, per_page=per_page, error_out=False)
    transfer_flags = _batch_list_transfer_flags(pagination.items)
    items = [
        admin_booking_list_item_fixed(
            b,
            has_transfer=transfer_flags[b.id][0],
            has_pending_transfer=transfer_flags[b.id][1],
        )
        for b in pagination.items
    ]

    total_pages = pagination.pages or 0
    return {
        "filters": {k: v for k, v in filter_kwargs.items() if v is not None},
        "summary": summary,
        "items": items,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "total_items": pagination.total or 0,
        },
    }


def _previous_company_from_transfers(booking_id: int) -> dict[str, Any] | None:
    tr = (
        BookingTransfer.query.filter_by(booking_id=booking_id)
        .filter(
            BookingTransfer.status.in_(
                [TransferStatus.ACCEPTED, TransferStatus.COMPLETED]
            )
        )
        .order_by(BookingTransfer.id.desc())
        .first()
    )
    if not tr:
        return None
    from models import Company

    owner = db.session.get(Company, tr.owner_company_id)
    if not owner:
        return None
    return {"id": owner.id, "name": owner.name}


def build_admin_booking_detail(
    booking: Booking, *, admin_public_id: str
) -> dict[str, Any]:
    """Payload GET /admin/bookings/:id."""
    full = booking.serialize
    status_val = booking.status
    key = status_val.value if hasattr(status_val, "value") else str(status_val).upper()

    current = booking.executing_company or booking.company
    previous = _previous_company_from_transfers(booking.id)

    timeline: list[dict[str, Any]] = []

    created_at = getattr(booking, "created_at", None)
    if created_at is not None:
        timeline.append(
            {
                "type": "booking_created",
                "at": created_at.isoformat(),
                "label": "Réservation créée",
            }
        )
    tl_inst = None
    try:
        tl_inst = booking._get_institution_timeline()
    except Exception:
        tl_inst = None
    if tl_inst:
        if tl_inst.get("sent_at"):
            timeline.append(
                {
                    "type": "request_sent",
                    "at": tl_inst["sent_at"],
                    "label": "Demande envoyée (institution)",
                }
            )
        if tl_inst.get("accepted_at"):
            timeline.append(
                {
                    "type": "request_accepted",
                    "at": tl_inst["accepted_at"],
                    "label": "Acceptée par entreprise",
                    "detail": tl_inst.get("accepted_by_company_name"),
                }
            )

    cancelled_at = getattr(booking, "cancelled_at", None)
    if cancelled_at is not None:
        timeline.append(
            {
                "type": "cancelled",
                "at": cancelled_at.isoformat(),
                "label": "Annulation",
                "detail": booking.cancelled_by_role,
            }
        )

    audit_rows = (
        AuditLog.query.filter_by(booking_id=booking.id)
        .order_by(AuditLog.created_at.asc())
        .limit(200)
        .all()
    )
    for row in audit_rows:
        try:
            details = json.loads(row.action_details or "{}")
        except json.JSONDecodeError:
            details = {}
        timeline.append(
            {
                "type": f"audit:{row.action_type}",
                "at": row.created_at.isoformat() if row.created_at else None,
                "label": row.action_type,
                "category": row.action_category,
                "user_type": row.user_type,
                "details": details,
            }
        )

    timeline.sort(key=lambda x: (x.get("at") or "",))

    base_path = f"/dashboard/admin/{admin_public_id}"
    links = {
        "admin_home": base_path,
        "platform_ops": f"{base_path}/platform-ops/overview",
    }
    if current:
        links["company"] = f"{base_path}/users"
    inst_id = None
    if booking.client and booking.client.linked_institution_id:
        inst_id = booking.client.linked_institution_id
        links["institution"] = f"{base_path}/platform-ops/overview"

    return {
        "id": booking.id,
        "client_name": booking.customer_full_name,
        "institution_name": full.get("client", {}).get("institution_name"),
        "status": key.lower(),
        "status_label": booking_status_label_fr(status_val),
        "created_by": _serialize_created_by(booking),
        "current_company": (
            {"id": current.id, "name": current.name} if current else None
        ),
        "previous_company": previous,
        "cancelled_by": _serialize_cancelled_by(booking),
        "booking": full,
        "timeline": timeline,
        "links": links,
        "linked_institution_id": inst_id,
    }


def export_admin_bookings_csv(**filter_kwargs: Any) -> tuple[bytes, str]:
    """CSV pour export (max EXPORT_MAX_ROWS)."""
    base = build_admin_bookings_query(
        q=filter_kwargs.get("q"),
        created_from=filter_kwargs.get("created_from"),
        created_to=filter_kwargs.get("created_to"),
        scheduled_from=filter_kwargs.get("scheduled_from"),
        scheduled_to=filter_kwargs.get("scheduled_to"),
        statuses=_parse_status_list(filter_kwargs.get("status")),
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
    )
    rows = (
        base.options(
            joinedload(Booking.client).joinedload(Client.user),
            joinedload(Booking.client).joinedload(Client.linked_institution),
            joinedload(Booking.company),
            joinedload(Booking.executing_company),
        )
        .order_by(Booking.id.desc())
        .limit(EXPORT_MAX_ROWS)
        .all()
    )
    export_flags = _batch_list_transfer_flags(rows)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "id",
            "status",
            "status_label",
            "created_at",
            "scheduled_at",
            "client_name",
            "institution_name",
            "current_company_name",
            "amount_chf",
            "pilotage_source_code",
            "pilotage_qualification_state",
            "observed_transport_amount",
        ]
    )
    for b in rows:
        item = admin_booking_list_item_fixed(
            b,
            has_transfer=export_flags[b.id][0],
            has_pending_transfer=export_flags[b.id][1],
        )
        pl = item.get("pilotage") or {}
        qual = pl.get("qualification") or {}
        ota = pl.get("observed_transport_amount")
        w.writerow(
            [
                item["id"],
                item["status"],
                item["status_label"],
                item["created_at"],
                item["scheduled_at"],
                item["client_name"],
                item["institution_name"] or "",
                item["current_company_name"] or "",
                item["amount_chf"] if item["amount_chf"] is not None else "",
                pl.get("source_code") or "",
                qual.get("state") or "",
                ota if ota is not None else "",
            ]
        )
    return buf.getvalue().encode(
        "utf-8-sig"
    ), f"bookings_export_{datetime.now(UTC).date()}.csv"
