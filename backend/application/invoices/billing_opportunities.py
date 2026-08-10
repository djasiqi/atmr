"""Agrégats read-only : opportunités de facturation (sujet + payeur + période)."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from flask import has_app_context
from sqlalchemy import and_, or_
from sqlalchemy.orm import joinedload

from application.invoices.billable_amount import calculate_billable_booking_amount
from application.invoices.institution_patient_resolution import (
    resolve_missing_institution_patient_ids,
)
from application.invoices.invoice_booking_units import resolve_invoice_booking_units
from application.invoices.period_invoice_preview import build_period_invoice_preview
from application.invoices.subject_identity import resolve_subject_identity
from ext import db
from models import BillingParty, Booking, Client, Company, InstitutionPatient
from models.enums import BookingStatus
from repositories.client_repository import ClientRepository
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)

logger = logging.getLogger(__name__)

CALENDAR_MONTHS = 12

RecipientStatus = Literal["ready", "missing_billing_address", "incomplete"]
IdentityStatus = Literal["resolved", "needs_review"]


@dataclass(frozen=True, slots=True)
class PatientOpportunity:
    """Opportunité patient (client classique ou InstitutionPatient)."""

    opportunity_key: str
    subject_key: str
    subject_type: str
    subject_id: int | None
    carrier_client_id: int
    billing_party_id: int | None
    display_name: str
    payer_display_name: str | None
    identity_status: IdentityStatus
    recipient_status: RecipientStatus
    can_generate: bool
    segments_count: int
    units_count: int
    unbilled_total_amount: float
    currency: str
    # Compat registre legacy
    client_id: int
    transports_count: int
    estimated_total: float


@dataclass(frozen=True, slots=True)
class ClinicOpportunity:
    clinic_company_id: int
    name: str
    transports_count: int
    estimated_total: float
    currency: str


@dataclass(frozen=True, slots=True)
class BillingOpportunitiesResult:
    period_year: int
    period_month: int
    patient_items: list[PatientOpportunity]
    clinic_items: list[ClinicOpportunity]
    total_draft_would_create: int
    ignored_missing_billing_party_count: int = 0


def _inc_ignored_missing_billing_party_metric(count: int = 1) -> None:
    """Métrique Prometheus optionnelle (no-op si client absent)."""
    if count <= 0:
        return
    try:
        from prometheus_client import Counter  # type: ignore

        counter = getattr(_inc_ignored_missing_billing_party_metric, "_counter", None)
        if counter is None:
            counter = Counter(
                "billing_opportunities_ignored_missing_billing_party_total",
                "Sujets patient ignorés faute de billing_party_id (registre V2)",
            )
            _inc_ignored_missing_billing_party_metric._counter = counter  # type: ignore[attr-defined]
        counter.inc(count)
    except Exception:
        logger.debug(
            "billing_opportunities metric ignored_missing_billing_party indisponible",
            exc_info=True,
        )


def _period_bounds(period_year: int, period_month: int) -> tuple[datetime, datetime]:
    start_date = datetime(period_year, period_month, 1)
    if period_month == CALENDAR_MONTHS:
        end_date = datetime(period_year + 1, 1, 1)
    else:
        end_date = datetime(period_year, period_month + 1, 1)
    return start_date, end_date


def _client_display_name(client: Client) -> str:
    user = getattr(client, "user", None)
    if user:
        fn = (getattr(user, "first_name", None) or "").strip()
        ln = (getattr(user, "last_name", None) or "").strip()
        if fn or ln:
            return f"{fn} {ln}".strip()
        un = (getattr(user, "username", None) or "").strip()
        if un:
            return un
    if client.institution_name:
        return str(client.institution_name)
    return f"Client #{client.id}"


def _institution_patient_display(
    patient: InstitutionPatient | None, fallback: str
) -> str:
    if patient is None:
        return fallback
    ln = (getattr(patient, "last_name", None) or "").strip()
    fn = (getattr(patient, "first_name", None) or "").strip()
    if ln or fn:
        return f"{ln} {fn}".strip() if ln else fn
    return fallback


def _address_has_street_postal_city(address: str | None) -> bool:
    text = (address or "").strip()
    if not text:
        return False
    # Au moins une ligne + un NPA (4 chiffres CH) et un mot ville
    if not re.search(r"\b\d{4}\b", text):
        return False
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n") if ln.strip()]
    if len(lines) < 1:
        return False
    # Exiger plus que le seul NPA
    return len(text) >= 8


def _recipient_status_for_party(
    bp: BillingParty | None,
    *,
    patient: InstitutionPatient | None = None,
) -> RecipientStatus:
    if bp is None:
        # Fallback InstitutionPatient fields
        if patient is not None:
            name_ok = bool(
                (patient.first_name or "").strip() or (patient.last_name or "").strip()
            )
            addr = (patient.address or "").strip()
            postal = (patient.postal_code or "").strip()
            city = (patient.city or "").strip()
            if name_ok and addr and postal and city:
                return "ready"
            return "missing_billing_address"
        return "incomplete"

    name_ok = bool((bp.display_name or "").strip())
    addr = (bp.billing_address or "").strip()
    bp_type = getattr(bp.type, "value", None) or str(getattr(bp, "type", "") or "")
    if bp_type.lower() == "patient":
        if not name_ok:
            return "incomplete"
        if patient is not None:
            postal = (patient.postal_code or "").strip()
            city = (patient.city or "").strip()
            street = (patient.address or "").strip() or addr
            if street and postal and city:
                return "ready"
        if _address_has_street_postal_city(addr):
            return "ready"
        return "missing_billing_address"

    if name_ok and addr:
        return "ready"
    return "missing_billing_address"


@dataclass(frozen=True, slots=True)
class ParsedBillingOpportunityKey:
    opportunity_key: str
    subject_type: str
    subject_id: int | None
    billing_party_id: int | None
    subject_key: str


def parse_billing_opportunity_key(key: str) -> ParsedBillingOpportunityKey:
    """Parse ``institution_patient:458|billing_party:901`` (ou ``client:72|...``)."""
    raw = (key or "").strip()
    parts = raw.split("|")
    subject_part = parts[0] if parts else ""
    party_id: int | None = None
    if len(parts) > 1 and parts[1].startswith("billing_party:"):
        try:
            party_id = int(parts[1].split(":", 1)[1])
        except (TypeError, ValueError):
            party_id = None

    subject_type = "unknown"
    subject_id: int | None = None
    if subject_part.startswith("institution_patient:"):
        subject_type = "institution_patient"
        try:
            subject_id = int(subject_part.split(":", 1)[1])
        except (TypeError, ValueError):
            subject_id = None
    elif subject_part.startswith("client:"):
        subject_type = "client"
        try:
            subject_id = int(subject_part.split(":", 1)[1])
        except (TypeError, ValueError):
            subject_id = None
    elif subject_part.startswith("legacy-institution-booking:"):
        subject_type = "legacy_institution_booking"
        try:
            subject_id = int(subject_part.split(":", 1)[1])
        except (TypeError, ValueError):
            subject_id = None

    if subject_type not in (
        "institution_patient",
        "client",
        "legacy_institution_booking",
    ):
        raise ValueError(
            "billing_opportunity_key invalide "
            "(attendu institution_patient:{id}|billing_party:{id} "
            "ou client:{id}|billing_party:{id})"
        )
    if party_id is None or subject_id is None:
        raise ValueError(
            "billing_opportunity_key invalide : billing_party_id et subject_id requis"
        )

    return ParsedBillingOpportunityKey(
        opportunity_key=raw,
        subject_type=subject_type,
        subject_id=subject_id,
        billing_party_id=party_id,
        subject_key=subject_part,
    )


def build_opportunity_key(subject_key: str, billing_party_id: int) -> str:
    return f"{subject_key}|billing_party:{int(billing_party_id)}"


def pick_canonical_billing_party_id(bookings: list[Booking]) -> int | None:
    """Payeur retenu pour un sujet facturable.

    Un même patient peut porter plusieurs ``BillingParty`` historiques (créés avant
    la déduplication par ``external_ref``). On retient le plus fréquent, puis le plus
    ancien, pour que le sujet ne produise qu'une seule opportunité.

    Guérit aussi les courses ``billed_to_type=patient`` encore liées à un BP
    établissement (clinique / EMS / hôpital).
    """
    from services.billing.billing_party_linker import (
        ensure_patient_destination_billing_party,
        is_establishment_billing_party,
    )

    # Le choix déterministe reste utilisable dans les helpers purs. Les
    # vérifications et corrections ORM ne sont pertinentes qu'en contexte Flask.
    can_access_database = has_app_context()

    for booking in bookings:
        btype = str(getattr(booking, "billed_to_type", None) or "").lower().strip()
        if btype != "patient" or not can_access_database:
            continue
        bp_id = getattr(booking, "billing_party_id", None)
        bp = db.session.get(BillingParty, int(bp_id)) if bp_id is not None else None
        if bp is None or is_establishment_billing_party(bp):
            ensure_patient_destination_billing_party(booking)

    counts: dict[int, int] = defaultdict(int)
    for booking in bookings:
        bp_id = getattr(booking, "billing_party_id", None)
        if bp_id is None:
            continue
        bp = (
            db.session.get(BillingParty, int(bp_id))
            if can_access_database
            else None
        )
        # Ne pas élire un BP établissement pour une opportunité patient.
        if bp is not None and is_establishment_billing_party(bp):
            continue
        counts[int(bp_id)] += 1
    if not counts:
        return None
    return min(counts, key=lambda bp_id: (-counts[bp_id], bp_id))


def resolve_recipient_status(
    *,
    billing_party: BillingParty | None,
    institution_patient: InstitutionPatient | None = None,
    display_name: str | None = None,
) -> RecipientStatus:
    """Alias public pour generate_invoice (adresse destinataire)."""
    _ = display_name
    return _recipient_status_for_party(billing_party, patient=institution_patient)


def build_billing_subject_snapshot(
    subject: Any,
    *,
    display_name: str,
    institution_patient: InstitutionPatient | None = None,
    client: Client | None = None,
) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "type": getattr(subject, "subject_type", None) or "unknown",
        "id": getattr(subject, "subject_id", None),
        "key": getattr(subject, "key", None),
        "display_name": display_name,
    }
    if institution_patient is not None:
        snap["birth_date"] = (
            institution_patient.dob.isoformat()
            if getattr(institution_patient, "dob", None)
            else None
        )
        snap["first_name"] = institution_patient.first_name
        snap["last_name"] = institution_patient.last_name
    elif client is not None:
        snap["client_id"] = client.id
    return snap


def build_recipient_snapshot(
    billing_party: BillingParty,
    *,
    recipient_status: str,
    institution_patient: InstitutionPatient | None = None,
) -> dict[str, Any]:
    addr = (billing_party.billing_address or "").strip()
    if not addr and institution_patient is not None:
        parts = []
        if institution_patient.address:
            parts.append(institution_patient.address.strip())
        pc = (institution_patient.postal_code or "").strip()
        city = (institution_patient.city or "").strip()
        if pc or city:
            parts.append(f"{pc} {city}".strip())
        addr = "\n".join(parts)
    return {
        "billing_party_id": billing_party.id,
        "display_name": billing_party.display_name,
        "billing_address": addr,
        "contact_email": billing_party.contact_email,
        "contact_phone": getattr(billing_party, "contact_phone", None),
        "type": getattr(billing_party.type, "value", None) or str(billing_party.type),
        "recipient_status": recipient_status,
    }


def load_eligible_bookings_for_opportunity(
    *,
    company_id: int,
    parsed: ParsedBillingOpportunityKey,
    period_year: int,
    period_month: int,
    excluded_booking_ids: set[int] | None = None,
) -> list[Booking]:
    """Charge tous les bookings patient éligibles pour l'opportunité (autorité serveur)."""
    from application.invoices.invoice_booking_units import (
        collect_explicit_peer_ids_to_load,
    )

    start_date, end_date = _period_bounds(period_year, period_month)
    excluded = excluded_booking_ids or set()
    canceled_ok = and_canceled_billable()

    if parsed.subject_id is None:
        return []

    period_bookings = (
        Booking.query.options(
            joinedload(Booking.client),
            joinedload(Booking.institution_patient),
        )
        .filter(
            Booking.company_id == company_id,
            Booking.billed_to_type == "patient",
            Booking.invoice_line_id.is_(None),
            Booking.scheduled_time >= start_date,
            Booking.scheduled_time < end_date,
            or_status_billable(canceled_ok),
        )
        .order_by(Booking.scheduled_time.asc())
        .all()
    )
    # Même rattrapage que la liste des opportunités, sinon le sujet ne matcherait pas.
    resolve_missing_institution_patient_ids(period_bookings)

    bookings = [
        b
        for b in period_bookings
        if resolve_subject_identity(b).key == parsed.subject_key
    ]
    if not bookings:
        return []

    # Expand pairs hors période
    peer_ids = collect_explicit_peer_ids_to_load(bookings)
    present = {int(b.id) for b in bookings}
    parent_ids = {
        int(b.parent_booking_id)
        for b in bookings
        if getattr(b, "parent_booking_id", None) is not None
    }
    to_load = (peer_ids | parent_ids) - present
    if present:
        children = Booking.query.filter(
            Booking.company_id == company_id,
            Booking.parent_booking_id.in_(present),
            Booking.billed_to_type == "patient",
            Booking.invoice_line_id.is_(None),
        ).all()
        for c in children:
            to_load.add(int(c.id))
    if to_load:
        extras = Booking.query.filter(Booking.id.in_(to_load)).all()
        resolve_missing_institution_patient_ids(extras)
        by_id = {int(b.id): b for b in bookings}
        for e in extras:
            if e.invoice_line_id is None:
                by_id[int(e.id)] = e
        bookings = list(by_id.values())

    # Filtrer sujet + exclusions
    out: list[Booking] = []
    for b in bookings:
        if int(b.id) in excluded:
            continue
        if b.billed_to_type != "patient":
            continue
        subj = resolve_subject_identity(b)
        if subj.key != parsed.subject_key and not (
            parsed.subject_type == "institution_patient"
            and getattr(b, "institution_patient_id", None) == parsed.subject_id
        ):
            continue
        out.append(b)
    return sorted(
        out,
        key=lambda b: (
            b.scheduled_time or datetime.min,
            int(b.id),
        ),
    )


def list_billing_opportunities(
    *,
    company_id: int,
    period_year: int,
    period_month: int,
) -> BillingOpportunitiesResult:
    """Liste les opportunités patient (sujet+payeur) et cliniques S2."""
    if not 1 <= period_month <= CALENDAR_MONTHS:
        raise ValueError("period_month invalide (1-12)")

    crepo = ClientRepository()
    start_date, end_date = _period_bounds(period_year, period_month)
    billing_settings = CompanyBillingSettingsRepository().find_or_create(company_id)

    canceled_ok = and_canceled_billable()
    patient_bookings = (
        Booking.query.options(
            joinedload(Booking.client).joinedload(Client.user),
            joinedload(Booking.institution_patient),
        )
        .filter(
            Booking.company_id == company_id,
            Booking.billed_to_type == "patient",
            Booking.invoice_line_id.is_(None),
            Booking.scheduled_time >= start_date,
            Booking.scheduled_time < end_date,
            or_status_billable(canceled_ok),
        )
        .all()
    )

    # Rattrapage : sans institution_patient_id, chaque transport formerait sa
    # propre opportunité (clé legacy-institution-booking).
    resolve_missing_institution_patient_ids(patient_bookings)

    # Grouper par sujet facturable : un patient = une facture pour la période.
    groups: dict[str, list[Booking]] = defaultdict(list)
    for b in patient_bookings:
        groups[resolve_subject_identity(b).key].append(b)

    patient_items: list[PatientOpportunity] = []
    ip_cache: dict[int, InstitutionPatient] = {}
    bp_cache: dict[int, BillingParty] = {}
    # Clients déjà eager-loadés via Booking.client (vérif multi-tenant en mémoire).
    carrier_ok: dict[int, bool] = {}
    ignored_missing_billing_party_count = 0

    for subject_key, bookings in groups.items():
        if not bookings:
            continue
        bp_id = pick_canonical_billing_party_id(bookings)
        sample = bookings[0]
        subj = resolve_subject_identity(sample)
        carrier = int(sample.client_id) if sample.client_id is not None else 0
        if carrier:
            if carrier not in carrier_ok:
                sample_client = getattr(sample, "client", None)
                if (
                    sample_client is not None
                    and int(getattr(sample_client, "company_id", 0) or 0) == company_id
                ):
                    carrier_ok[carrier] = True
                else:
                    # Fallback rare si relation absente / hors session
                    carrier_ok[carrier] = bool(
                        crepo.find_model_by_id_and_company(carrier, company_id)
                    )
            if not carrier_ok[carrier]:
                continue

        patient: InstitutionPatient | None = None
        if subj.subject_type == "institution_patient" and subj.subject_id is not None:
            if subj.subject_id not in ip_cache:
                ip_cache[subj.subject_id] = getattr(
                    sample, "institution_patient", None
                ) or db.session.get(InstitutionPatient, subj.subject_id)
            patient = ip_cache.get(subj.subject_id)

        bp: BillingParty | None = None
        if bp_id is not None:
            if bp_id not in bp_cache:
                bp_cache[bp_id] = db.session.get(BillingParty, bp_id)
            bp = bp_cache.get(bp_id)

        if subj.subject_type == "institution_patient":
            display = _institution_patient_display(
                patient, getattr(sample, "customer_name", None) or subject_key
            )
        elif subj.subject_type == "client":
            client = getattr(sample, "client", None)
            display = _client_display_name(client) if client else f"Client #{carrier}"
        else:
            display = getattr(sample, "customer_name", None) or subject_key or "Booking"

        bp_type_raw = (
            (getattr(bp.type, "value", None) or str(bp.type)).lower().strip()
            if bp is not None
            else ""
        )
        # BP PATIENT : pas de « facturé à {patient} » redondant (ni clinique obsolète).
        if bp_type_raw == "patient":
            payer_name = None
        else:
            payer_name = (bp.display_name if bp else None) or None
        recipient_status = _recipient_status_for_party(bp, patient=patient)
        identity_status: IdentityStatus = subj.status  # type: ignore[assignment]

        def _amt(bk: Booking) -> Decimal:
            return calculate_billable_booking_amount(
                bk, billing_settings=billing_settings
            ).amount_ht

        # Expand pairs explicites hors période (C3) pour units_count fiable
        from application.invoices.invoice_booking_units import (
            collect_explicit_peer_ids_to_load,
        )

        scope_by_id = {int(b.id): b for b in bookings}
        peer_ids = collect_explicit_peer_ids_to_load(bookings)
        present = set(scope_by_id)
        parent_ids = {
            int(b.parent_booking_id)
            for b in bookings
            if getattr(b, "parent_booking_id", None) is not None
        }
        to_load = (peer_ids | parent_ids) - present
        if present:
            for child in Booking.query.filter(
                Booking.company_id == company_id,
                Booking.parent_booking_id.in_(present),
                Booking.billed_to_type == "patient",
                Booking.invoice_line_id.is_(None),
            ).all():
                to_load.add(int(child.id))
        if to_load:
            extras = Booking.query.filter(Booking.id.in_(to_load)).all()
            resolve_missing_institution_patient_ids(extras)
            for extra in extras:
                if extra.invoice_line_id is None:
                    if resolve_subject_identity(extra).key != subject_key:
                        continue
                    scope_by_id[int(extra.id)] = extra
        scope_bookings = list(scope_by_id.values())

        units = resolve_invoice_booking_units(
            selected_ids={int(b.id) for b in bookings},
            scope_bookings=scope_bookings,
            subject_key_fn=lambda bk: resolve_subject_identity(bk).key,
            amount_ht_fn=_amt,
            expand_explicit_peers=True,
        )
        # Ne garder que les unités du sujet (filet)
        units = [u for u in units if u.subject_key == subject_key]
        if not units:
            continue
        total = float(sum((u.amount_ht for u in units), Decimal("0")))
        segments = sum(len(u.booking_ids) for u in units)
        if bp_id is None:
            ignored_missing_billing_party_count += 1
            logger.warning(
                "billing_opportunities: sujet %s sans billing_party_id ignoré",
                subject_key,
            )
            continue
        opportunity_key = build_opportunity_key(subject_key, bp_id)
        can_generate = (
            identity_status == "resolved"
            and recipient_status == "ready"
            and segments > 0
        )

        patient_items.append(
            PatientOpportunity(
                opportunity_key=opportunity_key,
                subject_key=subject_key,
                subject_type=subj.subject_type,
                subject_id=subj.subject_id,
                carrier_client_id=carrier,
                billing_party_id=bp_id,
                display_name=display,
                payer_display_name=payer_name if payer_name != display else None,
                identity_status=identity_status,
                recipient_status=recipient_status,
                can_generate=can_generate,
                segments_count=segments,
                units_count=len(units),
                unbilled_total_amount=total,
                currency="CHF",
                client_id=carrier,
                transports_count=segments,
                estimated_total=total,
            )
        )

    # --- S2 cliniques ---
    clinic_ids = [
        r[0]
        for r in (
            db.session.query(Booking.billed_to_company_id)
            .filter(
                Booking.company_id == company_id,
                Booking.billed_to_type == "clinic",
                Booking.invoice_line_id.is_(None),
                Booking.scheduled_time >= start_date,
                Booking.scheduled_time < end_date,
            )
            .distinct()
            .all()
        )
    ]
    clinic_items: list[ClinicOpportunity] = []
    for ccid in sorted({x for x in clinic_ids if x is not None}):
        prev = build_period_invoice_preview(
            company_id=company_id,
            period_year=period_year,
            period_month=period_month,
            client_id=None,
            clinic_company_id=ccid,
            include_line_details=False,
        )
        if prev.transports_count < 1:
            continue
        c = db.session.get(Company, ccid)
        name = c.name if c else f"Clinique #{ccid}"
        clinic_items.append(
            ClinicOpportunity(
                clinic_company_id=ccid,
                name=str(name or ""),
                transports_count=prev.transports_count,
                estimated_total=prev.estimated_total,
                currency=prev.currency,
            )
        )

    would_patient = sum(
        1 for p in patient_items if p.can_generate and p.segments_count > 0
    )
    would_clinic = sum(1 for c in clinic_items if c.transports_count > 0)

    if ignored_missing_billing_party_count:
        _inc_ignored_missing_billing_party_metric(ignored_missing_billing_party_count)

    # Persister les guérisons BP patient←établissement (sinon rollback fin de requête).
    if db.session.new or db.session.dirty:
        db.session.commit()

    return BillingOpportunitiesResult(
        period_year=period_year,
        period_month=period_month,
        patient_items=sorted(patient_items, key=lambda x: x.display_name.lower()),
        clinic_items=sorted(clinic_items, key=lambda x: (x.name or "").lower()),
        total_draft_would_create=would_patient + would_clinic,
        ignored_missing_billing_party_count=ignored_missing_billing_party_count,
    )


def and_canceled_billable():
    return and_(
        Booking.status == BookingStatus.CANCELED.value,
        Booking.cancellation_fee_amount.isnot(None),
        or_(
            Booking.is_cancellation_billable.is_(True),
            Booking.billing_override_reason.isnot(None),
        ),
    )


def or_status_billable(canceled_condition):
    return or_(
        Booking.status.in_(
            [BookingStatus.COMPLETED.value, BookingStatus.RETURN_COMPLETED.value]
        ),
        canceled_condition,
    )


def opportunities_to_dict(res: BillingOpportunitiesResult) -> dict[str, Any]:
    return {
        "period_year": res.period_year,
        "period_month": res.period_month,
        "patient_payers": [
            {
                "opportunity_key": p.opportunity_key,
                "subject_key": p.subject_key,
                "subject_type": p.subject_type,
                "subject_id": p.subject_id,
                "carrier_client_id": p.carrier_client_id,
                "client_id": p.client_id,
                "billing_party_id": p.billing_party_id,
                "display_name": p.display_name,
                "payer_display_name": p.payer_display_name,
                "identity_status": p.identity_status,
                "recipient_status": p.recipient_status,
                "can_generate": p.can_generate,
                "segments_count": p.segments_count,
                "units_count": p.units_count,
                "transports_count": p.transports_count,
                "unbilled_total_amount": p.unbilled_total_amount,
                "estimated_total": p.estimated_total,
                "currency": p.currency,
            }
            for p in res.patient_items
        ],
        "clinic_payers": [
            {
                "clinic_company_id": c.clinic_company_id,
                "name": c.name,
                "transports_count": c.transports_count,
                "estimated_total": c.estimated_total,
                "currency": c.currency,
            }
            for c in res.clinic_items
        ],
        "total_draft_would_create": res.total_draft_would_create,
        "ignored_missing_billing_party_count": res.ignored_missing_billing_party_count,
    }
