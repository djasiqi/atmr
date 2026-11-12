"""Registre des mandats de démonstration. Une relation, plusieurs courses.

Aucune facture réelle. Aucune signature juridique. Une entreprise ne lit
pas le mandat d'une autre.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from application.settlement.convention_demo import (
    DISCLAIMER,
    DOCUMENT_VERSION,
    DURATION_LABEL,
    SERVICE_SCOPE,
    TARIFF_VERSION,
    ConventionClosed,
    ConventionRefusal,
    _client_for_company,
    _pdf,
    _require_open,
    _trace,
    archive_reference,
    company_name_for,
    present,
    read_archived_pdf,
    store_mandate_pdf,
    tactile_reference,
)
from domain.end_of_trip_settlement.mandate_applicability import (
    assess_mandate,
    decide_coverage,
    real_invoice_blocked,
)
from domain.end_of_trip_settlement.recipe_simulation import (
    recipe_simulation_read_allowed,
)
from ext import db
from models.booking import Booking
from models.company import Company
from models.convention_demo import PatientCompanyConventionDemo

_BILLING = (
    "Facturation simulée de cette entreprise. "
    "Ce document n'autorise aucune facture réelle."
)


def _open() -> None:
    if not recipe_simulation_read_allowed():
        raise ConventionClosed
    _require_open()


def _payer(booking: Booking) -> str:
    raw = getattr(booking.billed_to_type, "value", booking.billed_to_type)
    return str(raw or "patient").strip().lower()


def _already_paid(booking: Booking) -> bool:
    from models.enums import PaymentStatus
    from models.payment import Payment

    amounts = (
        db.session.query(Payment.amount)
        .filter(
            Payment.booking_id == int(booking.id),
            Payment.status == PaymentStatus.COMPLETED,
        )
        .all()
    )
    if not amounts:
        return False
    paid = sum((Decimal(str(row.amount or 0)) for row in amounts), Decimal("0.00"))
    raw = booking.price_amount if booking.price_amount is not None else booking.amount
    if raw is None:
        return False
    return paid >= Decimal(str(raw))


def _tariff_label(booking: Booking) -> str | None:
    version_id = getattr(booking, "pricing_profile_version_id", None)
    if version_id is None:
        return None
    from models.service_area_pricing import PricingProfileVersion

    version = db.session.get(PricingProfileVersion, int(version_id))
    if version is None:
        return str(version_id)
    return str(version.version)


def _meta(row: PatientCompanyConventionDemo) -> dict[str, Any]:
    trace = row.specimen_trace if isinstance(row.specimen_trace, dict) else {}
    mandate = trace.get("mandate")
    if not isinstance(mandate, dict):
        return {}
    return mandate


def _covered_services(row: PatientCompanyConventionDemo) -> tuple[str, ...]:
    raw = _meta(row).get("covered_services")
    if isinstance(raw, list) and raw:
        return tuple(str(item) for item in raw if str(item).strip())
    return ()


def _booking_service(booking: Booking) -> str | None:
    detail = str(getattr(booking, "assistance_detail", "") or "").strip()
    return detail or None


def _covered(row: PatientCompanyConventionDemo) -> tuple[str, ...]:
    raw = _meta(row).get("covered_tariff_versions")
    if isinstance(raw, list) and raw:
        return tuple(str(item) for item in raw)
    if row.tariff_version:
        return (str(row.tariff_version),)
    return ()


def _parse_date(value: object) -> date | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def remember(
    row: PatientCompanyConventionDemo,
    action: str,
    detail: str,
    *,
    at: str | None = None,
) -> None:
    """Ajoute une ligne d'historique sans effacer un tracé de démonstration."""
    trace = dict(row.specimen_trace) if isinstance(row.specimen_trace, dict) else {}
    mandate = dict(trace.get("mandate") or {}) if isinstance(trace.get("mandate"), dict) else {}
    history = list(mandate.get("history") or [])
    history.append(
        {
            "action": action,
            "detail": detail,
            "at": at or datetime.now(UTC).isoformat(),
        }
    )
    mandate["history"] = history
    trace["mandate"] = mandate
    row.specimen_trace = trace


def _decision(row: PatientCompanyConventionDemo | None, booking: Booking):
    meta = _meta(row) if row is not None else {}
    return assess_mandate(
        mandate_company_id=None if row is None else int(row.company_id),
        booking_company_id=int(booking.company_id),
        status="demo_absent" if row is None else str(row.status),
        covered_tariff_versions=_covered(row) if row is not None else (),
        booking_tariff_version=_tariff_label(booking),
        valid_from=_parse_date(meta.get("valid_from")),
        valid_until=_parse_date(meta.get("valid_until")),
        on_date=datetime.now(UTC).date(),
        billed_to_type=_payer(booking),
        already_paid=_already_paid(booking),
    )


def _reservations(row: PatientCompanyConventionDemo) -> list[dict[str, Any]]:
    found = (
        db.session.query(Booking)
        .filter(
            Booking.client_id == int(row.client_id),
            Booking.company_id == int(row.company_id),
        )
        .order_by(Booking.id)
        .limit(20)
        .all()
    )
    return [
        {
            "id": int(item.id),
            "label": item.customer_name,
            "amount": None if item.price_amount is None else str(item.price_amount),
        }
        for item in found
    ]


def _public_signature(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    return {
        "signer": raw.get("signer"),
        "representative_name": raw.get("representative_name"),
        "representative_quality": raw.get("representative_quality"),
        "representative_relation": raw.get("representative_relation"),
        "document_version": raw.get("document_version"),
        "tariff_version_presented": raw.get("tariff_version_presented"),
        "admin_review_required": bool(raw.get("admin_review_required")),
        "billing_prepared": bool(raw.get("billing_prepared")),
        "invoice_issued": False,
        "real_signature_collected": False,
        "legally_effective": False,
        "version_number": raw.get("version_number"),
        "signed_at": raw.get("signed_at"),
        "tactile_reference": raw.get("tactile_reference"),
        "archive_reference": raw.get("archive_reference"),
        "documentary_status": raw.get("documentary_status"),
        "representation_review_required": bool(raw.get("admin_review_required")),
    }


def _public_transmissions(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [
        {
            "channel": item.get("channel"),
            "status": item.get("status"),
            "sent": False,
            "destination": item.get("destination"),
            "public_link": None,
        }
        for item in raw
        if isinstance(item, dict)
    ]


def _view(row: PatientCompanyConventionDemo, company_name: str, booking: Booking | None) -> dict[str, Any]:
    meta = _meta(row)
    decision = None if booking is None else _decision(row, booking)
    amendments = [item for item in list(meta.get("amendments") or []) if isinstance(item, dict)]
    unsigned = [item for item in amendments if not item.get("signed")]
    amendment_label = None
    if unsigned or (decision is not None and decision.amendment_required):
        amendment_label = "Avenant requis — version tarifaire non couverte."
    signature = meta.get("signature") if isinstance(meta.get("signature"), dict) else {}
    return {
        "id": int(row.id),
        "client_id": int(row.client_id),
        "company_id": int(row.company_id),
        "company_name": company_name,
        "debtor_label": row.debtor_label,
        "representative_name": row.representative_name,
        "tariff_version": row.tariff_version,
        "pricing_profile_label": row.tariff_version,
        "covered_tariff_versions": list(_covered(row)),
        "covered_services": list(_covered_services(row)),
        "service_scope": row.service_scope,
        "billing_conditions": str(meta.get("billing_conditions") or _BILLING),
        "prepared_at": None if row.created_at is None else row.created_at.isoformat(),
        "signed_at": signature.get("signed_at") if isinstance(signature, dict) else None,
        "valid_from": meta.get("valid_from"),
        "valid_until": meta.get("valid_until"),
        "duration_label": row.duration_label,
        "status": row.status,
        "status_label": _status_label(row.status),
        "applicability": None if decision is None else decision.kind,
        "amendment_required": bool(unsigned) or (
            False if decision is None else decision.amendment_required
        ),
        "amendment_label": amendment_label,
        "amendments": amendments,
        "reservations": _reservations(row),
        "document_version": row.document_version,
        "version_number": int(row.version_number),
        "active": bool(row.active),
        "legally_effective": False,
        "real_signature_collected": False,
        "billing_authorized": False,
        "invoice_issued": False,
        "history": list(meta.get("history") or []),
        "demonstration_signature": _public_signature(meta.get("signature")),
        "signature_recorded": row.status == "demo_specimen_signed",
        "admin_review_required": bool(
            isinstance(meta.get("signature"), dict)
            and meta["signature"].get("admin_review_required")
        ),
        "transmissions": _public_transmissions(meta.get("transmissions")),
        "disclaimer": DISCLAIMER,
        "covers_other_companies": False,
    }


def _status_label(status: str) -> str:
    return {
        "demo_draft": "Mandat à signer",
        "demo_specimen_signed": "Mandat signé",
        "demo_valid_simulated": "Mandat signé",
    }.get(status, "Mandat à signer")


def _company_name(company_id: int) -> str:
    company = db.session.get(Company, int(company_id))
    return str(getattr(company, "name", None) or "Entreprise fictive")


def _version_content(row: PatientCompanyConventionDemo) -> tuple[str, str, str, str, str]:
    return (
        str(row.debtor_label),
        str(row.tariff_version),
        str(row.service_scope),
        str(row.duration_label),
        str(row.document_version),
    )


def _lock_pair(client_id: int, company_id: int) -> None:
    """Sérialise les activations d'un même couple patient–entreprise."""
    from sqlalchemy import text

    db.session.execute(
        text("SELECT pg_advisory_xact_lock(:client_id, :company_id)"),
        {"client_id": int(client_id), "company_id": int(company_id)},
    )


def open_version(company_id: int, client_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Ouvre une version et désactive la précédente, sans la réécrire.

    Aucune facture. ``legally_effective`` reste faux.
    """
    from sqlalchemy.exc import IntegrityError

    _open()
    _client_for_company(int(company_id), int(client_id))
    key = str(payload.get("idempotency_key") or "").strip()[:80]
    content = (
        str(payload.get("debtor_label") or "").strip()[:160],
        str(payload.get("tariff_version") or "").strip()[:64],
        str(payload.get("service_scope") or "").strip(),
        str(payload.get("duration_label") or "").strip()[:160],
        str(payload.get("document_version") or "").strip()[:32],
    )
    if len(key) < 8 or any(not part for part in content):
        raise ConventionRefusal(
            "version_incomplete",
            "La nouvelle version doit indiquer une clé, un tarif et un périmètre.",
        )
    _lock_pair(int(client_id), int(company_id))
    rows = (
        db.session.query(PatientCompanyConventionDemo)
        .filter_by(company_id=int(company_id), client_id=int(client_id))
        .order_by(PatientCompanyConventionDemo.version_number)
        .with_for_update()
        .all()
    )
    for row in rows:
        if row.idempotency_key != key:
            continue
        if _version_content(row) == content:
            db.session.commit()
            view = _view(row, _company_name(int(company_id)), None)
            view["replay"] = True
            return view
        db.session.rollback()
        raise ConventionRefusal(
            "conflit_idempotence",
            "Cette clé a déjà ouvert une version au contenu différent.",
        )
    active = [row for row in rows if row.active]
    if len(active) > 1:
        raise ConventionRefusal(
            "activation_refusee",
            "Deux versions actives existent déjà pour ce patient.",
        )
    previous = active[0] if active else None
    number = 1 if not rows else max(int(row.version_number) for row in rows) + 1
    if previous is not None:
        previous.active = False
    created = PatientCompanyConventionDemo(
        client_id=int(client_id),
        company_id=int(company_id),
        debtor_label=content[0],
        tariff_version=content[1],
        service_scope=content[2],
        duration_label=content[3],
        document_version=content[4],
        version_number=number,
        active=True,
        idempotency_key=key,
        status="demo_draft",
        legally_effective=False,
        specimen_trace=None,
    )
    db.session.add(created)
    try:
        db.session.flush()
    except IntegrityError as exc:
        db.session.rollback()
        raise ConventionRefusal(
            "activation_refusee",
            "Une seule version peut rester active.",
        ) from exc
    remember(
        created,
        "version_opened",
        f"Version {number} ouverte. Les versions précédentes sont conservées.",
    )
    created.legally_effective = False
    db.session.commit()
    view = _view(created, _company_name(int(company_id)), None)
    view["replay"] = False
    return view


def refuse_historical_change(
    company_id: int, version_id: int, payload: dict[str, Any]
) -> None:
    """Refuse de réécrire une version archivée ou déjà signée."""
    _open()
    _lock_pair_from_version(int(company_id), int(version_id))
    row = (
        db.session.query(PatientCompanyConventionDemo)
        .filter_by(id=int(version_id), company_id=int(company_id))
        .with_for_update()
        .one_or_none()
    )
    if row is None:
        raise ConventionRefusal("version_absente", "Cette version est introuvable.")
    proposed = str(payload.get("tariff_version") or "")
    if (not row.active) or row.status != "demo_draft" or proposed != str(row.tariff_version):
        raise ConventionRefusal(
            "version_figee",
            "Une version signée ou archivée n'est pas réécrite.",
        )


def _lock_pair_from_version(company_id: int, version_id: int) -> None:
    row = (
        db.session.query(PatientCompanyConventionDemo)
        .filter_by(id=int(version_id), company_id=int(company_id))
        .one_or_none()
    )
    if row is not None:
        _lock_pair(int(row.client_id), int(company_id))


def list_for_client(company_id: int, client_id: int) -> dict[str, Any]:
    _open()
    _client_for_company(company_id, client_id)
    rows = (
        db.session.query(PatientCompanyConventionDemo)
        .filter_by(company_id=company_id, client_id=client_id)
        .order_by(PatientCompanyConventionDemo.id)
        .all()
    )
    name = _company_name(company_id)
    changed = False
    for row in rows:
        if row.active and _record_existing(row):
            changed = True
    if changed:
        db.session.commit()
    return {
        "disclaimer": DISCLAIMER,
        "billing_authorized": False,
        "mandates": [_view(row, name, None) for row in rows],
    }


def read_applicability(company_id: int, booking_id: int) -> dict[str, Any]:
    """Lit si la version active couvre le transport. N'écrit aucune ligne."""
    _open()
    booking = _booking_for_company(company_id, booking_id)
    return _applicability(int(company_id), booking)


def _applicability(company_id: int, booking: Booking) -> dict[str, Any]:
    """Assemble la réponse. Ne modifie ni le mandat, ni la réservation, ni une facture."""
    client_id = None if booking.client_id is None else int(booking.client_id)
    row = None if client_id is None else _row(int(company_id), client_id)
    booking_tariff = _tariff_label(booking)
    booking_service = _booking_service(booking)
    tariffs = () if row is None else _covered(row)
    services = () if row is None else _covered_services(row)
    if row is None and client_id is not None:
        other_patient = (
            db.session.query(PatientCompanyConventionDemo.id)
            .filter(
                PatientCompanyConventionDemo.company_id == int(company_id),
                PatientCompanyConventionDemo.active.is_(True),
                PatientCompanyConventionDemo.client_id != int(client_id),
            )
            .count()
        )
        if other_patient:
            decision = decide_coverage(
                mandate_client_id=int(client_id) + 1,
                booking_client_id=client_id,
                mandate_company_id=int(company_id),
                booking_company_id=int(booking.company_id),
                active=True,
                covered_tariffs=(),
                booking_tariff=None,
                covered_services=(),
                booking_service=None,
            )
        else:
            decision = decide_coverage(
                mandate_client_id=None,
                booking_client_id=client_id,
                mandate_company_id=None,
                booking_company_id=int(booking.company_id),
                active=False,
                covered_tariffs=(),
                booking_tariff=booking_tariff,
                covered_services=(),
                booking_service=booking_service,
            )
    else:
        decision = decide_coverage(
            mandate_client_id=None if row is None else int(row.client_id),
            booking_client_id=client_id,
            mandate_company_id=None if row is None else int(row.company_id),
            booking_company_id=int(booking.company_id),
            active=row is not None and bool(row.active),
            covered_tariffs=tariffs,
            booking_tariff=booking_tariff,
            covered_services=services,
            booking_service=booking_service,
        )
    review = bool(
        row is not None
        and isinstance(_meta(row).get("signature"), dict)
        and _meta(row)["signature"].get("admin_review_required")
    )
    amount = None if booking.price_amount is None else str(booking.price_amount)
    return {
        "booking_id": int(booking.id),
        "amount": amount,
        "mandate_id": None if row is None else int(row.id),
        "version_number": None if row is None else int(row.version_number),
        "active": bool(row.active) if row is not None else False,
        "company_id": int(booking.company_id),
        "client_id": client_id,
        "covered_services": list(services),
        "covered_tariff_versions": list(tariffs),
        "tariff_applicable": None if row is None else str(row.tariff_version),
        "booking_tariff": booking_tariff,
        "booking_service": booking_service,
        "code": decision.code,
        "label": decision.label,
        "avenant_requis": decision.avenant_requis,
        "avenant_motif": decision.motif,
        "admin_review_required": review,
        "legally_effective": False,
        "billing_authorized": False,
        "invoice_issued": False,
    }


def assess_for_booking(company_id: int, booking_id: int) -> dict[str, Any]:
    _open()
    booking = _booking_for_company(company_id, booking_id)
    row = _row(company_id, booking.client_id)
    decision = _decision(row, booking)
    payload: dict[str, Any] = {
        "booking_id": int(booking.id),
        "amount": None if booking.price_amount is None else str(booking.price_amount),
        "client_id": None if booking.client_id is None else int(booking.client_id),
        "kind": decision.kind,
        "label": decision.label,
        "covered_tariff_versions": [] if row is None else list(_covered(row)),
        "required_tariff_version": _tariff_label(booking),
        "billing_refusal": real_invoice_blocked(
            kind=decision.kind,
            legally_effective=False,
            real_signature_collected=False,
        ),
        "amendment_required": decision.amendment_required,
        "billing_authorized": False,
        "legally_effective": False,
        "real_signature_collected": False,
        "invoice_issued": False,
        "disclaimer": DISCLAIMER,
    }
    if row is not None and decision.kind != "other_company":
        payload["mandate"] = _view(row, _company_name(company_id), booking)
    else:
        payload["mandate"] = None
    from application.settlement.leg_settlement import leg_public

    payload.update(leg_public(booking))
    coverage = _applicability(int(company_id), booking)
    payload["code"] = coverage["code"]
    payload["avenant_requis"] = coverage["avenant_requis"]
    payload["avenant_motif"] = coverage["avenant_motif"]
    payload["version_number"] = coverage["version_number"]
    payload["active"] = coverage["active"]
    payload["covered_services"] = coverage["covered_services"]
    payload["admin_review_required"] = coverage["admin_review_required"]
    payload["billing_authorized"] = False
    payload["legally_effective"] = False
    if coverage["avenant_requis"]:
        payload["amendment_required"] = True
    return payload


def _record_existing(row: PatientCompanyConventionDemo) -> bool:
    """Une ligne factuelle pour une convention déjà là. Ne couvre aucune version nouvelle."""
    history = list(_meta(row).get("history") or [])
    if history:
        return False
    moment = None if row.created_at is None else row.created_at.isoformat()
    remember(
        row,
        "recorded",
        (
            f"Convention {int(row.id)} déjà présente. "
            f"Statut : {_status_label(row.status)}. "
            f"Version tarifaire citée : {row.tariff_version}. "
            "Aucune signature juridique."
        ),
        at=moment,
    )
    return True


def _note_booking(row: PatientCompanyConventionDemo, booking: Booking) -> str:
    """Rattache une course ou prépare un avenant. Ne élargit pas la couverture."""
    trace = dict(row.specimen_trace) if isinstance(row.specimen_trace, dict) else {}
    mandate = dict(trace.get("mandate") or {}) if isinstance(trace.get("mandate"), dict) else {}
    seen = [int(item) for item in list(mandate.get("seen_booking_ids") or [])]
    booking_id = int(booking.id)
    already = booking_id in seen
    if not already:
        seen.append(booking_id)
    mandate["seen_booking_ids"] = seen
    label = _tariff_label(booking)
    covered = _covered(row)
    trace["mandate"] = mandate
    row.specimen_trace = trace
    if label and label not in covered:
        amendments = [
            item for item in list(mandate.get("amendments") or []) if isinstance(item, dict)
        ]
        known = any(int(item.get("booking_id") or 0) == booking_id for item in amendments)
        if known:
            return "replay"
        amendments.append(
            {
                "booking_id": booking_id,
                "origin_convention_id": int(row.id),
                "covered_tariff_versions": list(covered),
                "required_tariff_version": label,
                "status": "prepared",
                "signed": False,
                "legally_effective": False,
            }
        )
        mandate["amendments"] = amendments
        trace["mandate"] = mandate
        row.specimen_trace = dict(trace)
        remember(
            row,
            "amendment_required",
            (
                f"Avenant requis pour la réservation {booking_id}. "
                f"Version {label} non couverte. "
                f"Couverture actuelle : {', '.join(covered) or 'aucune'}."
            ),
        )
        return "amendment"
    if already:
        return "replay"
    remember(row, "linked", f"Réservation {booking_id} rattachée à la convention existante.")
    return "reused"


def _insert_draft(booking: Booking) -> tuple[PatientCompanyConventionDemo | None, bool]:
    from sqlalchemy.exc import IntegrityError

    covered = [TARIFF_VERSION]
    tariff = _tariff_label(booking)
    if tariff and tariff not in covered:
        covered.append(tariff)
    label = (booking.customer_name or "").strip() or f"Patient {booking.client_id}"
    row = PatientCompanyConventionDemo(
        client_id=int(booking.client_id),
        company_id=int(booking.company_id),
        debtor_label=label,
        tariff_version=TARIFF_VERSION,
        service_scope=SERVICE_SCOPE,
        duration_label=DURATION_LABEL,
        status="demo_draft",
        document_version=DOCUMENT_VERSION,
        version_number=1,
        active=True,
        idempotency_key=f"ouverture-{int(booking.client_id)}-{int(booking.company_id)}",
        legally_effective=False,
        specimen_trace={
            "mandate": {
                "billing_conditions": _BILLING,
                "covered_tariff_versions": covered,
                "seen_booking_ids": [int(booking.id)],
                "amendments": [],
                "history": [
                    {
                        "action": "prepared",
                        "detail": f"Préparé pour la réservation {int(booking.id)}.",
                        "at": datetime.now(UTC).isoformat(),
                    }
                ],
            }
        },
    )
    nested = db.session.begin_nested()
    try:
        db.session.add(row)
        db.session.flush()
        nested.commit()
    except IntegrityError:
        nested.rollback()
        return _row(int(booking.company_id), int(booking.client_id)), False
    return row, True


def ensure_prepared_on_booking_created(booking_id: int) -> dict[str, Any] | None:
    """Prépare un brouillon administratif. N'écrit rien hors recette locale.

    Ne modifie ni le montant ni le débiteur de la réservation. N'émet aucune facture.
    """
    if not recipe_simulation_read_allowed():
        return None
    try:
        _require_open()
    except ConventionClosed:
        return None
    booking = db.session.get(Booking, int(booking_id))
    if booking is None or booking.client_id is None or booking.company_id is None:
        return None
    if _payer(booking) != "patient" or _already_paid(booking):
        return None
    try:
        _client_for_company(int(booking.company_id), int(booking.client_id))
    except ConventionRefusal:
        return None
    amount_before = booking.price_amount
    debtor_before = booking.billed_to_type
    row = (
        db.session.query(PatientCompanyConventionDemo)
        .filter_by(
            company_id=int(booking.company_id),
            client_id=int(booking.client_id),
            active=True,
        )
        .with_for_update()
        .one_or_none()
    )
    created = False
    if row is None:
        row, created = _insert_draft(booking)
        outcome = "prepared" if created else "reused"
        if row is not None and not created:
            outcome = _note_booking(row, booking)
    else:
        _record_existing(row)
        outcome = _note_booking(row, booking)
    if row is None:
        return None
    row.legally_effective = False
    db.session.flush()
    if booking.price_amount != amount_before or booking.billed_to_type != debtor_before:
        db.session.rollback()
        return None
    db.session.commit()
    return {
        "created": created,
        "convention_id": int(row.id),
        "outcome": outcome,
        "legally_effective": False,
        "real_signature_collected": False,
        "billing_authorized": False,
    }


def request_real_invoice_from_mandate(company_id: int, booking_id: int) -> None:
    """Barrière backend. Refuse avant toute écriture de facture."""
    _open()
    booking = _booking_for_company(company_id, booking_id)
    decision = _decision(_row(company_id, booking.client_id), booking)
    reason = real_invoice_blocked(
        kind=decision.kind,
        legally_effective=False,
        real_signature_collected=False,
    )
    raise ConventionRefusal(
        reason,
        "Aucune facture réelle n'est autorisée par ce mandat.",
    )


def prepare_for_booking(
    company_id: int, booking_id: int, *, debtor_label: str
) -> dict[str, Any]:
    """Prépare un brouillon s'il n'en existe pas. Ne couvre pas une institution."""
    _open()
    booking = _booking_for_company(company_id, booking_id)
    if _payer(booking) != "patient" or _already_paid(booking):
        raise ConventionRefusal(
            "mandat_non_concerne",
            "Cette réservation ne déclenche pas de mandat patient.",
        )
    if booking.client_id is None:
        raise ConventionRefusal(
            "patient_non_rattache",
            "Cette réservation n'est pas reliée à une fiche patient.",
        )
    _client_for_company(company_id, int(booking.client_id))
    _ = debtor_label
    result = ensure_prepared_on_booking_created(int(booking.id))
    if result is None:
        raise ConventionRefusal(
            "mandat_non_concerne",
            "Cette réservation ne déclenche pas de mandat patient.",
        )
    body = assess_for_booking(company_id, booking_id)
    body["created"] = bool(result["created"])
    body["booking_id"] = int(booking.id)
    return body


def _booking_for_company(company_id: int, booking_id: int) -> Booking:
    booking = db.session.get(Booking, int(booking_id))
    if booking is None or int(booking.company_id) != int(company_id):
        raise ConventionRefusal("reservation_absente", "Réservation introuvable.")
    return booking


def _row(company_id: int, client_id: int | None) -> PatientCompanyConventionDemo | None:
    if client_id is None:
        return None
    return (
        db.session.query(PatientCompanyConventionDemo)
        .filter_by(company_id=int(company_id), client_id=int(client_id), active=True)
        .one_or_none()
    )


def signature_offer(kind: str, status: object) -> str:
    """Décide l'écran chauffeur. N'écrit rien."""
    if kind == "applicable_demonstration" or status == "demo_valid_simulated":
        return "already"
    if kind in {"expired", "amendment_required", "other_company", "not_concerned"}:
        return "blocked"
    if status == "demo_specimen_signed" or kind == "specimen":
        return "document"
    return "wizard"


def driver_read(booking: Booking) -> dict[str, Any]:
    """Lecture chauffeur. Aucune écriture de mandat, de facture ou de paiement."""
    _open()
    amount = None if booking.price_amount is None else str(booking.price_amount)
    company_name = _company_name(int(booking.company_id))
    if booking.client_id is None:
        return {
            "status": "demo_unlinked",
            "status_label": "Patient non rattaché à une fiche",
            "company_name": company_name,
            "debtor_label": str(booking.customer_name or "Patient fictif"),
            "tariff_version": None,
            "amount": amount,
            "supplements": [],
            "signature_offer": "blocked",
            "applicability": "absent",
            "legally_effective": False,
            "real_signature_collected": False,
            "billing_authorized": False,
            "invoice_issued": False,
            "disclaimer": DISCLAIMER,
        }
    body = present(
        int(booking.company_id),
        int(booking.client_id),
        company_name=company_name,
        debtor_label=str(booking.customer_name or "Patient fictif"),
    )
    assessed = assess_for_booking(int(booking.company_id), int(booking.id))
    mandate = assessed.get("mandate") if isinstance(assessed.get("mandate"), dict) else {}
    email, phone = _suggested_contacts(booking)
    body.update(
        {
            "applicability": assessed["kind"],
            "label": assessed["label"],
            "amount": amount,
            "amendment_label": mandate.get("amendment_label"),
            "billing_conditions": mandate.get("billing_conditions") or _BILLING,
            "service_scope": mandate.get("service_scope") or body.get("service_scope"),
            "duration_label": mandate.get("duration_label") or body.get("duration_label"),
            "document_version": mandate.get("document_version") or body.get("document_version"),
            "convention_id": mandate.get("id"),
            "covered_tariff_versions": assessed.get("covered_tariff_versions") or [],
            "supplements": [],
            "signature_offer": signature_offer(str(assessed["kind"]), body.get("status")),
            "demonstration_signature": mandate.get("demonstration_signature"),
            "admin_review_required": bool(mandate.get("admin_review_required")),
            "transmissions": mandate.get("transmissions") or [],
            "history": mandate.get("history") or [],
            "suggested_email": email,
            "suggested_phone": phone,
            "billing_prepared": bool(
                isinstance(mandate.get("demonstration_signature"), dict)
                and mandate["demonstration_signature"].get("billing_prepared")
            ),
            "invoice_issued": False,
            "billing_authorized": False,
            "legally_effective": False,
            "real_signature_collected": False,
            "public_link": None,
        }
    )
    return body


def _refuse_other_version(locked: PatientCompanyConventionDemo, body: dict[str, Any]) -> None:
    """Une signature vise la version présentée. Une version inactive est refusée."""
    raw = body.get("version_number")
    if raw is None or str(raw).strip() == "":
        return
    try:
        requested = int(raw)
    except (TypeError, ValueError) as exc:
        db.session.rollback()
        raise ConventionRefusal(
            "version_changee",
            "La version présentée a changé pendant la signature.",
        ) from exc
    if requested == int(locked.version_number):
        return
    other = (
        db.session.query(PatientCompanyConventionDemo)
        .filter_by(
            company_id=int(locked.company_id),
            client_id=int(locked.client_id),
            version_number=requested,
        )
        .one_or_none()
    )
    inactive = other is not None and not bool(other.active)
    db.session.rollback()
    if inactive:
        raise ConventionRefusal(
            "version_inactive",
            "Cette version n'est plus active. Sa signature n'est pas recopiée.",
        )
    raise ConventionRefusal(
        "version_changee",
        "La version active a changé pendant la signature.",
    )


def _restore_archive(locked: PatientCompanyConventionDemo) -> None:
    """Rejoue le même fichier si l'archive a disparu. Ne crée pas de version."""
    if read_archived_pdf(locked) is not None:
        return
    trace = locked.specimen_trace if isinstance(locked.specimen_trace, dict) else {}
    mandate = trace.get("mandate") if isinstance(trace.get("mandate"), dict) else {}
    signature = mandate.get("signature") if isinstance(mandate.get("signature"), dict) else {}
    recorded = signature.get("version_number")
    if recorded is not None and int(recorded) != int(locked.version_number):
        raise ConventionRefusal(
            "version_changee",
            "L'archive ne correspond pas à cette version.",
        )
    reference = str(signature.get("archive_reference") or archive_reference(locked))
    try:
        payload = _pdf(locked, company_name_for(int(locked.company_id)))
        store_mandate_pdf(reference, payload)
    except OSError as exc:
        raise ConventionRefusal(
            "archive_indisponible",
            "Le document n'a pas pu être relu.",
        ) from exc
    if signature:
        signature["archive_reference"] = reference
        signature["documentary_status"] = "archived"
        from sqlalchemy.orm.attributes import flag_modified

        flag_modified(locked, "specimen_trace")
        db.session.commit()


def sign_demonstration(booking: Booking, body: dict[str, Any]) -> dict[str, Any]:
    """Enregistre une signature de démonstration, une seule fois."""
    _open()
    _guard_patient(booking)
    row = _row(int(booking.company_id), booking.client_id)
    if row is not None:
        _refuse_other_version(row, body)
    decision = _decision(row, booking)
    if decision.kind == "applicable_demonstration":
        raise ConventionRefusal(
            "mandat_deja_applicable",
            "Un mandat de démonstration couvre déjà cette réservation.",
        )
    if decision.kind in {"expired", "amendment_required", "other_company", "not_concerned"}:
        raise ConventionRefusal(decision.kind, decision.label)
    if row is None or row.status not in {"demo_draft", "demo_specimen_signed"}:
        raise ConventionRefusal(
            "specimen_sans_brouillon",
            "Préparez d'abord le document de démonstration.",
        )
    _refuse_other_version(row, body)
    locked = _lock(int(row.id))
    _refuse_other_version(locked, body)
    if locked.status == "demo_specimen_signed":
        _restore_archive(locked)
        _ensure_followup(booking)
        view = driver_read(booking)
        view["replay"] = True
        return view
    if body.get("acknowledged") is not True:
        raise ConventionRefusal(
            "conditions_non_lues",
            "Les conditions doivent être consultées avant la signature.",
        )
    signer = str(body.get("signer") or "")
    if signer not in {"patient", "representative"}:
        raise ConventionRefusal("signataire_inconnu", "Le signataire indiqué n'est pas reconnu.")
    name = quality = relation = None
    if signer == "representative":
        name = str(body.get("representative_name") or "").strip()[:160]
        quality = str(body.get("representative_quality") or "").strip()[:160]
        relation = str(body.get("representative_relation") or "").strip()[:160]
        if min(len(name), len(quality), len(relation)) < 2:
            raise ConventionRefusal(
                "representant_incomplet",
                "Le représentant doit indiquer son nom, sa qualité et son pouvoir déclaré.",
            )
    points = _trace(body.get("trace"))
    presented = str(body.get("tariff_version_presented") or "").strip()
    if presented and presented != str(locked.tariff_version):
        raise ConventionRefusal(
            "version_presentee_differente",
            "La version présentée ne correspond pas à la convention préparée.",
        )
    if not presented:
        presented = str(locked.tariff_version)
    trace = dict(locked.specimen_trace) if isinstance(locked.specimen_trace, dict) else {}
    mandate = dict(trace.get("mandate") or {}) if isinstance(trace.get("mandate"), dict) else {}
    reference = archive_reference(locked)
    signed_at = datetime.now(UTC).isoformat()
    mandate["signature"] = {
        "mandate_id": int(locked.id),
        "version_number": int(locked.version_number),
        "client_id": int(locked.client_id),
        "company_id": int(locked.company_id),
        "signer": signer,
        "representative_name": name,
        "representative_quality": quality,
        "representative_relation": relation,
        "document_version": locked.document_version,
        "tariff_version_presented": presented,
        "covered_services": list(mandate.get("covered_services") or []),
        "covered_tariff_versions": list(mandate.get("covered_tariff_versions") or [presented]),
        "acknowledged": True,
        "admin_review_required": signer == "representative",
        "representation_review_required": signer == "representative",
        "billing_prepared": True,
        "billing_authorized": False,
        "invoice_issued": False,
        "real_signature_collected": False,
        "legally_effective": False,
        "signed_at": signed_at,
        "tactile_reference": tactile_reference(points),
        "archive_reference": reference,
        "documentary_status": "pending",
        "idempotency_key": str(body.get("idempotency_key") or "")[:80] or None,
    }
    trace["points"] = points
    trace["kind"] = "demonstration"
    trace["mandate"] = mandate
    locked.specimen_trace = trace
    locked.representative_name = name
    locked.legally_effective = False
    try:
        payload = _pdf(locked, company_name_for(int(locked.company_id)))
        store_mandate_pdf(reference, payload)
    except OSError as exc:
        db.session.rollback()
        raise ConventionRefusal(
            "archive_indisponible",
            "Le document n'a pas été archivé. La version reste non signée.",
        ) from exc
    stored = read_archived_pdf(locked)
    if stored != payload:
        db.session.rollback()
        raise ConventionRefusal(
            "archive_indisponible",
            "Le document archivé n'est pas relisible. La version reste non signée.",
        )
    mandate["signature"]["documentary_status"] = "archived"
    locked.specimen_trace = dict(trace)
    locked.status = "demo_specimen_signed"
    remember(
        locked,
        "demonstration_signed",
        "Mandat signé pour cette version. Document archivé, sans valeur juridique.",
    )
    _unchanged_booking(booking)
    db.session.commit()
    _ensure_followup(booking)
    view = driver_read(booking)
    view["replay"] = False
    return view


def _ensure_followup(booking: Booking) -> None:
    from application.settlement.patient_billing_prep import ensure_patient_followup
    from domain.end_of_trip_settlement.production_prep_barrier import PrepRefusal

    try:
        ensure_patient_followup(
            int(booking.id),
            company_id=int(booking.company_id),
            actor_id=None,
        )
    except PrepRefusal as exc:
        raise ConventionRefusal(
            exc.code,
            "Le dossier de suivi n'a pas été préparé. Aucune facture n'est émise.",
        ) from exc


def refuse_signature(booking: Booking, body: dict[str, Any]) -> dict[str, Any]:
    """Enregistre un refus ou une impossibilité. N'invente aucun tracé."""
    _open()
    _guard_patient(booking)
    row = _row(int(booking.company_id), booking.client_id)
    if row is None or row.status != "demo_draft":
        raise ConventionRefusal(
            "deja_signe" if row is not None and row.status == "demo_specimen_signed" else "specimen_sans_brouillon",
            "Aucune signature n'est fabriquée.",
        )
    reason = str(body.get("reason") or "").strip()
    category = str(body.get("category") or "")
    if category not in {"refus", "incapable"} or not 3 <= len(reason) <= 500:
        raise ConventionRefusal("motif_absent", "Le motif doit être indiqué.")
    locked = _lock(int(row.id))
    if locked.status != "demo_draft":
        raise ConventionRefusal("deja_signe", "Une démonstration est déjà enregistrée. Elle n'est pas effacée.")
    trace = dict(locked.specimen_trace) if isinstance(locked.specimen_trace, dict) else {}
    mandate = dict(trace.get("mandate") or {}) if isinstance(trace.get("mandate"), dict) else {}
    previous = mandate.get("refusal")
    if isinstance(previous, dict) and int(previous.get("booking_id") or 0) == int(booking.id):
        view = driver_read(booking)
        view["replay"] = True
        return view
    mandate["refusal"] = {
        "booking_id": int(booking.id),
        "category": category,
        "reason": reason,
    }
    trace["mandate"] = mandate
    locked.specimen_trace = trace
    locked.legally_effective = False
    remember(
        locked,
        "signature_refused",
        f"Signature non recueillie ({category}). Motif transmis à l'entreprise. Aucun mandat activé.",
    )
    _notify(
        int(booking.company_id),
        f"mandate_refusal:{int(booking.id)}",
        "mandate_signature_refused",
        "Signature de mandat non recueillie",
        (
            f"Course {int(booking.id)}. {category}. Motif : {reason}. "
            "Aucune signature fabriquée, aucune facture."
        ),
        {"booking_id": int(booking.id), "category": category, "simulation": True},
    )
    _unchanged_booking(booking)
    db.session.commit()
    view = driver_read(booking)
    view["replay"] = False
    return view


def signal_blocked(booking: Booking) -> dict[str, Any]:
    """Signale une couverture inutilisable. N'enregistre aucune signature."""
    _open()
    _guard_patient(booking)
    row = _row(int(booking.company_id), booking.client_id)
    decision = _decision(row, booking)
    if decision.kind not in {"expired", "amendment_required", "other_company"}:
        raise ConventionRefusal(
            "signal_inutile",
            "Cette réservation n'appelle pas de signalement de couverture.",
        )
    if row is None:
        raise ConventionRefusal("specimen_sans_brouillon", "Aucune convention à signaler.")
    locked = _lock(int(row.id))
    marker = f"coverage_signaled:{int(booking.id)}:{decision.kind}"
    history = list(_meta(locked).get("history") or [])
    if any(isinstance(line, dict) and line.get("action") == marker for line in history):
        view = driver_read(booking)
        view["replay"] = True
        return view
    remember(locked, marker, f"{decision.label} Aucune signature enregistrée.")
    _notify(
        int(booking.company_id),
        f"mandate_signal:{int(booking.id)}:{decision.kind}",
        "mandate_coverage_blocked",
        "Couverture de mandat à vérifier",
        f"Course {int(booking.id)}. {decision.label} Aucune signature, aucune facture.",
        {"booking_id": int(booking.id), "kind": decision.kind, "simulation": True},
    )
    locked.legally_effective = False
    _unchanged_booking(booking)
    db.session.commit()
    view = driver_read(booking)
    view["replay"] = False
    return view


def dispatch_demonstration(booking: Booking, body: dict[str, Any]) -> dict[str, Any]:
    """Transmet le document déjà signé. Ne crée ni signature ni mandat."""
    _open()
    _guard_patient(booking)
    row = _row(int(booking.company_id), booking.client_id)
    if row is None or row.status != "demo_specimen_signed":
        raise ConventionRefusal(
            "signature_absente",
            "Aucune signature de démonstration n'est prête à être transmise.",
        )
    channel = str(body.get("channel") or "")
    if channel not in {"email", "sms"}:
        raise ConventionRefusal("canal_inconnu", "Le canal de transmission n'est pas reconnu.")
    destination = str(body.get("destination") or "").strip()
    key = str(body.get("idempotency_key") or "").strip()[:80]
    if len(key) < 8:
        raise ConventionRefusal("cle_absente", "La transmission exige une clé d'idempotence.")
    locked = _lock(int(row.id))
    if locked.status != "demo_specimen_signed":
        raise ConventionRefusal("signature_absente", "La démonstration n'est plus transmissible.")
    trace = dict(locked.specimen_trace) if isinstance(locked.specimen_trace, dict) else {}
    mandate = dict(trace.get("mandate") or {}) if isinstance(trace.get("mandate"), dict) else {}
    transmissions = [item for item in list(mandate.get("transmissions") or []) if isinstance(item, dict)]
    for item in transmissions:
        if item.get("key") == key:
            view = driver_read(booking)
            view["replay"] = True
            view["transmission"] = _public_transmissions([item])[0]
            return view
    status, masked = _dispatch_status(
        channel,
        destination,
        simulate_failure=bool(body.get("simulate_failure")),
    )
    entry = {
        "key": key,
        "channel": channel,
        "status": status,
        "sent": False,
        "destination": masked,
        "public_link": None,
        "message": "Référence authentifiée. Aucun lien public.",
    }
    transmissions.append(entry)
    mandate["transmissions"] = transmissions
    trace["mandate"] = mandate
    locked.specimen_trace = trace
    locked.legally_effective = False
    remember(
        locked,
        "transmission",
        f"Transmission fictive {channel} : {status}. Aucune nouvelle signature.",
    )
    _unchanged_booking(booking)
    db.session.commit()
    view = driver_read(booking)
    view["replay"] = False
    view["transmission"] = _public_transmissions([entry])[0]
    return view


def _guard_patient(booking: Booking) -> None:
    if _payer(booking) != "patient" or _already_paid(booking):
        raise ConventionRefusal(
            "reservation_non_concernee",
            "Cette réservation ne relève pas d'un mandat patient.",
        )
    if booking.client_id is None:
        raise ConventionRefusal(
            "patient_non_rattache",
            "Cette course n'est pas rattachée à une fiche patient.",
        )


def _lock(row_id: int) -> PatientCompanyConventionDemo:
    return (
        db.session.query(PatientCompanyConventionDemo)
        .filter_by(id=int(row_id))
        .with_for_update()
        .one()
    )


def _unchanged_booking(booking: Booking) -> None:
    amount = booking.price_amount
    debtor = booking.billed_to_type
    db.session.flush()
    if booking.price_amount != amount or booking.billed_to_type != debtor:
        db.session.rollback()
        raise ConventionRefusal(
            "montant_modifie",
            "Le montant de la réservation n'a pas été conservé.",
        )


def _notify(
    company_id: int,
    dedupe: str,
    event_type: str,
    title: str,
    message: str,
    metadata: dict[str, Any],
) -> None:
    from sqlalchemy import select

    from models.company_notification import CompanyNotification

    existing = db.session.scalar(
        select(CompanyNotification).where(
            CompanyNotification.company_id == int(company_id),
            CompanyNotification.dedupe_key == dedupe,
        )
    )
    if existing is not None:
        return
    db.session.add(
        CompanyNotification(
            company_id=int(company_id),
            event_type=event_type,
            title=title,
            message=message,
            metadata_json=metadata,
            is_read=False,
            dedupe_key=dedupe,
        )
    )


def _fictive_email(value: str) -> bool:
    host = value.strip().lower().split("@")[-1]
    return host in {"example.com", "example.ch"} or host.endswith(".invalid")


def _fictive_phone(value: str) -> bool:
    digits = "".join(character for character in value if character.isdigit())
    return digits.startswith("4100") or digits.startswith("000")


def _mask(value: str) -> str | None:
    if "@" in value:
        local, host = value.split("@", 1)
        return f"{local[:1]}***@{host}"
    digits = "".join(character for character in value if character.isdigit())
    return f"***{digits[-2:]}" if digits else None


def _dispatch_status(channel: str, destination: str, *, simulate_failure: bool) -> tuple[str, str | None]:
    if not destination:
        return "absent", None
    fictive = _fictive_email(destination) if channel == "email" else _fictive_phone(destination)
    if not fictive:
        return "blocked", _mask(destination)
    if simulate_failure:
        return "failed", _mask(destination)
    return "captured", _mask(destination)


def _suggested_contacts(booking: Booking) -> tuple[str | None, str | None]:
    from models.client import Client
    from models.user import User

    email = None
    phone = None
    if booking.client_id is not None:
        client = db.session.get(Client, int(booking.client_id))
        user = None if client is None or client.user_id is None else db.session.get(User, int(client.user_id))
        if user is not None and user.email and _fictive_email(str(user.email)):
            email = str(user.email).strip()
        raw_phone = None if user is None else getattr(user, "phone", None)
        if raw_phone and _fictive_phone(str(raw_phone)):
            phone = str(raw_phone).strip()
    passenger = str(getattr(booking, "passenger_phone", "") or "").strip()
    if phone is None and passenger and _fictive_phone(passenger):
        phone = passenger
    return email, phone

