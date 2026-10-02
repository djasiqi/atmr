"""Sélecteur « Nouvelle facture » : lecture agrégée, sans brouillon ni clinique.

Le menu patient n'a besoin que du nom, du nombre de courses et d'un montant
estimé (montant indiqué sur la réservation). La prévisualisation et
« Préparer la facture » relisent la base ; ce cache n'est pas une source de vérité.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from contextlib import suppress
from decimal import Decimal
from typing import Any

from application.invoices.active_invoice_claim import (
    filter_bookings_without_active_invoice_claim,
)
from application.invoices.billing_opportunities import (
    _period_bounds,
    and_canceled_billable,
    build_opportunity_key,
    or_status_billable,
)
from application.invoices.booking_status import booking_status_is_canceled
from application.invoices.institution_invoice_eligibility import (
    attach_invoice_request_ids,
    filter_institution_invoice_eligible,
)
from application.invoices.institution_patient_resolution import (
    resolve_missing_institution_patient_ids,
)
from application.invoices.subject_identity import resolve_subject_identity

CANDIDATES_CACHE_TTL_SECONDS = 60


def patient_candidates_cache_key(
    company_id: int, period_year: int, period_month: int
) -> str:
    return (
        f"billing:candidates:patient:company_{int(company_id)}:"
        f"{int(period_year):04d}-{int(period_month):02d}"
    )


def invalidate_patient_invoice_candidates(
    company_id: int, period_year: int, period_month: int
) -> None:
    """Oublie le sélecteur après création d'une facture patient."""
    try:
        from ext import redis_client

        if redis_client is None:
            return
        redis_client.delete(
            patient_candidates_cache_key(company_id, period_year, period_month)
        )
    except Exception:
        return


def _indicated_amount(booking: Any) -> Decimal:
    """Montant affiché dans le menu : frais d'annulation, sinon ``booking.amount``.

    Le tarif contractuel et la grille sont recalculés à la prévisualisation.
    """
    if booking_status_is_canceled(booking):
        fee = getattr(booking, "cancellation_fee_amount", None)
        if fee is None:
            return Decimal("0.00")
        return Decimal(str(fee)).quantize(Decimal("0.01"))
    raw = getattr(booking, "amount", None) or 0
    return Decimal(str(raw)).quantize(Decimal("0.01"))


def group_patient_invoice_candidates(
    bookings: list[Any],
    *,
    party_id_of: Any = None,
) -> list[dict[str, Any]]:
    """Regroupe des courses déjà filtrées (facturables, non revendiquées).

    ``party_id_of`` permet de regrouper sur le payeur effectif sans écrire
    ``booking.billing_party_id``. Sans ce rappel, la valeur stockée est lue.
    """

    def _party_id(row: Any) -> int | None:
        if party_id_of is not None:
            resolved = party_id_of(row)
            if resolved is not None:
                return int(resolved)
        raw = getattr(row, "billing_party_id", None)
        return int(raw) if raw is not None else None

    grouped: dict[str, list[Any]] = defaultdict(list)
    for booking in bookings:
        subject = resolve_subject_identity(booking)
        if subject.status != "resolved" or subject.subject_id is None:
            continue
        grouped[subject.key].append(booking)

    patients: list[dict[str, Any]] = []
    for subject_key, rows in grouped.items():
        subject = resolve_subject_identity(rows[0])
        parties = [party_id for row in rows if (party_id := _party_id(row)) is not None]
        if not parties:
            continue
        billing_party_id = sorted(
            Counter(parties).items(), key=lambda item: (-item[1], item[0])
        )[0][0]
        total = sum((_indicated_amount(row) for row in rows), Decimal("0.00"))
        names = [
            str(getattr(row, "customer_name", "") or "").strip()
            for row in rows
            if str(getattr(row, "customer_name", "") or "").strip()
        ]
        fallback = Counter(names).most_common(1)[0][0] if names else "Patient"
        carrier = subject.carrier_client_id
        patients.append(
            {
                "id": build_opportunity_key(subject_key, billing_party_id),
                "name": fallback,
                "billable_count": len(rows),
                "amount": float(total.quantize(Decimal("0.01"))),
                "client_id": int(carrier) if carrier else None,
                "institution_patient_id": (
                    subject.subject_id
                    if subject.subject_type == "institution_patient"
                    else None
                ),
                "billing_party_id": billing_party_id,
                "_subject_type": subject.subject_type,
                "_subject_id": subject.subject_id,
            }
        )
    return patients


def _apply_display_names(patients: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from ext import db
    from models.institution_patient import InstitutionPatient
    from models.user import User

    ip_ids = {
        int(row["_subject_id"])
        for row in patients
        if row.get("_subject_type") == "institution_patient" and row.get("_subject_id")
    }
    client_ids = {int(row["client_id"]) for row in patients if row.get("client_id")}
    ip_names: dict[int, str] = {}
    if ip_ids:
        for pid, first_name, last_name in (
            db.session.query(
                InstitutionPatient.id,
                InstitutionPatient.first_name,
                InstitutionPatient.last_name,
            )
            .filter(InstitutionPatient.id.in_(ip_ids))
            .all()
        ):
            label = f"{first_name or ''} {last_name or ''}".strip()
            if label:
                ip_names[int(pid)] = label
    client_names: dict[int, str] = {}
    if client_ids:
        from models.client import Client

        for cid, first_name, last_name in (
            db.session.query(Client.id, User.first_name, User.last_name)
            .join(User, User.id == Client.user_id)
            .filter(Client.id.in_(client_ids))
            .all()
        ):
            label = f"{first_name or ''} {last_name or ''}".strip()
            if label:
                client_names[int(cid)] = label

    public: list[dict[str, Any]] = []
    for row in patients:
        name = row["name"]
        if row.get("_subject_type") == "institution_patient":
            name = ip_names.get(int(row["_subject_id"]), name)
        elif row.get("client_id"):
            name = client_names.get(int(row["client_id"]), name)
        public.append(
            {
                "id": row["id"],
                "name": name,
                "billable_count": row["billable_count"],
                "amount": row["amount"],
                "client_id": row["client_id"],
                "institution_patient_id": row["institution_patient_id"],
                "billing_party_id": row["billing_party_id"],
            }
        )
    public.sort(key=lambda item: str(item["name"]).lower())
    return public


def list_patient_invoice_candidates(
    *,
    company_id: int,
    period_year: int,
    period_month: int,
) -> dict[str, Any]:
    """Courses patient facturables du mois, agrégées pour le menu.

    Aucune écriture. Les cliniques, les lignes et le brouillon ne sont pas calculés.
    """
    from sqlalchemy.orm import selectinload

    from ext import redis_client
    from models.booking import Booking

    if not 1 <= int(period_month) <= 12:
        raise ValueError("period_month invalide (1-12)")

    cache_key = patient_candidates_cache_key(company_id, period_year, period_month)
    if redis_client is not None:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                payload = json.loads(cached)
                if isinstance(payload, dict) and isinstance(
                    payload.get("patients"), list
                ):
                    return payload
        except Exception:
            pass

    start_date, end_date = _period_bounds(int(period_year), int(period_month))
    bookings = (
        Booking.query.options(
            selectinload(Booking.client),
            selectinload(Booking.source_request),
        )
        .filter(
            Booking.company_id == int(company_id),
            Booking.billed_to_type == "patient",
            Booking.invoice_line_id.is_(None),
            Booking.scheduled_time >= start_date,
            Booking.scheduled_time < end_date,
            or_status_billable(and_canceled_billable()),
        )
        .all()
    )
    bookings = filter_bookings_without_active_invoice_claim(bookings)
    resolve_missing_institution_patient_ids(bookings, persist=False)
    attach_invoice_request_ids(bookings)
    eligible = filter_institution_invoice_eligible(bookings)
    def _effective_party_id(booking: Any) -> int | None:
        # Lecture seule : ne pas assigner booking.billing_party_id (sale la session).
        from services.billing.effective_patient_payer import (
            resolve_effective_patient_billing_party,
        )

        party = resolve_effective_patient_billing_party(
            booking=booking,
            company_id=int(company_id),
        )
        if party is not None:
            return int(party.id)
        raw = getattr(booking, "billing_party_id", None)
        return int(raw) if raw is not None else None

    patients = _apply_display_names(
        group_patient_invoice_candidates(eligible, party_id_of=_effective_party_id)
    )
    payload = {
        "period": f"{int(period_year):04d}-{int(period_month):02d}",
        "patients": patients,
    }
    if redis_client is not None:
        with suppress(Exception):
            redis_client.setex(
                cache_key,
                CANDIDATES_CACHE_TTL_SECONDS,
                json.dumps(payload, ensure_ascii=False),
            )
    return payload
