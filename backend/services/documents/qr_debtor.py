"""Débiteur QR-facture (« Payable par ») — résolution live et snapshot immuable.

Le débiteur QR n'est **pas** le bloc « Facturé à » : cas curatelle / OPAD, la QR
demande le patient + son domicile structuré (name / street / pcode / city / country),
alors que le bloc imprime patient / c/o tiers / adresse du tiers.

Source de vérité : ``invoice.meta["qr_debtor_snapshot"]``, capturé avec le bloc
« Facturé à » (brouillon rafraîchi, gel à la sortie de DRAFT). Les renderers QR
relisent ce snapshot dès que la facture est figée. Aucun backfill silencieux.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from services.documents.invoice_recipient import invoice_billed_to_is_frozen

logger = logging.getLogger("qrbill_service")

QR_DEBTOR_SNAPSHOT_KEY = "qr_debtor_snapshot"
QR_DEBTOR_SNAPSHOT_VERSION = 1

QrDebtorRule = Literal[
    "s2_clinic",
    "legacy_institution",
    "institution_patient",
    "patient_direct",
]
QrDebtorSource = Literal["live", "snapshot", "legacy_live"]

_MIN_ADDRESS_PARTS = 2
_MIN_ADDRESS_PARTS_POSTAL = 3
_MIN_ADDRESS_PARTS_CITY = 4


def _normalize_meta(meta: Any) -> dict[str, Any]:
    if isinstance(meta, dict):
        return dict(meta)
    return {}


def _strategy_value(invoice: Any) -> str:
    raw = getattr(invoice, "billing_strategy", None)
    value = getattr(raw, "value", raw)
    return str(value or "").strip().lower()


@dataclass(frozen=True, slots=True)
class QrDebtor:
    """Identité structurée exigée par la norme QR-facture (pas les lignes « Facturé à »)."""

    name: str
    street: str
    pcode: str
    city: str
    country: str = "CH"
    rule: QrDebtorRule = "patient_direct"
    source: QrDebtorSource = "live"

    def to_qrbill_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "street": self.street,
            "pcode": self.pcode,
            "city": self.city,
            "country": self.country,
        }

    def to_snapshot(self, *, reason: str, frozen: bool) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        return {
            "version": QR_DEBTOR_SNAPSHOT_VERSION,
            "name": self.name,
            "street": self.street,
            "pcode": self.pcode,
            "city": self.city,
            "country": self.country,
            "rule": self.rule,
            "captured_at": now,
            "captured_reason": reason,
            "frozen_at": now if frozen else None,
        }


def parse_address_for_qrbill(address: str | None) -> tuple[str, str, str]:
    """Parse une adresse pour QR-bill en séparant rue, code postal et ville."""
    if not address:
        return ("", "1200", "Genève")
    parts = [p.strip() for p in str(address).replace("\n", ",").split(",") if p.strip()]
    if len(parts) >= _MIN_ADDRESS_PARTS_CITY:
        street = f"{parts[0]}, {parts[1]}" if len(parts) > 1 else parts[0]
        return (street, parts[2], parts[3])
    if len(parts) >= _MIN_ADDRESS_PARTS_POSTAL:
        street = parts[0]
        pcode_city = parts[1].strip().split()
        if len(pcode_city) >= _MIN_ADDRESS_PARTS:
            return (street, pcode_city[0], " ".join(pcode_city[1:]))
        return (
            street,
            parts[1],
            parts[2] if len(parts) >= _MIN_ADDRESS_PARTS_POSTAL else "Genève",
        )
    if len(parts) >= _MIN_ADDRESS_PARTS:
        last_part = parts[-1].strip().split()
        if len(last_part) >= _MIN_ADDRESS_PARTS:
            return (parts[0], last_part[0], " ".join(last_part[1:]))
    return (address or "", "1200", "Genève")


def read_qr_debtor_snapshot(invoice: Any) -> dict[str, Any] | None:
    snap = _normalize_meta(getattr(invoice, "meta", None)).get(QR_DEBTOR_SNAPSHOT_KEY)
    if isinstance(snap, dict) and (snap.get("name") or "").strip():
        return snap
    return None


def qr_debtor_from_snapshot(snap: dict[str, Any]) -> QrDebtor:
    rule = str(snap.get("rule") or "patient_direct")
    return QrDebtor(
        name=str(snap.get("name") or "Client"),
        street=str(snap.get("street") or "Adresse non renseignée"),
        pcode=str(snap.get("pcode") or "1200"),
        city=str(snap.get("city") or "Genève"),
        country=str(snap.get("country") or "CH"),
        rule=rule,  # type: ignore[arg-type]
        source="snapshot",
    )


def write_qr_debtor_snapshot(invoice: Any, snap: dict[str, Any]) -> None:
    meta = _normalize_meta(getattr(invoice, "meta", None))
    meta[QR_DEBTOR_SNAPSHOT_KEY] = snap
    invoice.meta = meta


def resolve_invoice_qr_debtor(invoice: Any, *, use_snapshot: bool = True) -> QrDebtor:
    """Point unique : snapshot si facture figée, sinon master data (règles QR inchangées)."""
    if use_snapshot and invoice_billed_to_is_frozen(invoice):
        snap = read_qr_debtor_snapshot(invoice)
        if snap is not None:
            return qr_debtor_from_snapshot(snap)
        logger.warning(
            "[QR-Bill] Facture figée sans qr_debtor_snapshot (invoice_id=%s, status=%s) : "
            "rendu depuis les master data courantes (legacy_live).",
            getattr(invoice, "id", None),
            getattr(getattr(invoice, "status", None), "value", None),
        )
        live = resolve_qr_debtor_live(invoice)
        return QrDebtor(
            name=live.name,
            street=live.street,
            pcode=live.pcode,
            city=live.city,
            country=live.country,
            rule=live.rule,
            source="legacy_live",
        )
    return resolve_qr_debtor_live(invoice)


def resolve_qr_debtor_live(invoice: Any) -> QrDebtor:
    """Règles historiques de ``QRBillService._get_debtor_info`` (master data)."""
    client = getattr(invoice, "client", None)
    strategy = _strategy_value(invoice)

    if strategy == "s2_clinic_monthly" and getattr(
        invoice, "billed_to_company_id", None
    ):
        return _live_s2_clinic(invoice)

    bill_to_client_id = getattr(invoice, "bill_to_client_id", None)
    client_id = getattr(invoice, "client_id", None)
    if bill_to_client_id and bill_to_client_id != client_id:
        return _live_legacy_institution(invoice, bill_to_client_id)

    if (
        client
        and bool(getattr(client, "is_institution", False))
        and strategy == "s1_patient"
    ):
        return _live_institution_patient(invoice, client)

    return _live_patient_direct(client)


def refresh_qr_debtor_snapshot_if_draft(
    invoice: Any, *, reason: str = "pdf_generation"
) -> dict[str, Any] | None:
    if invoice_billed_to_is_frozen(invoice):
        return read_qr_debtor_snapshot(invoice)
    snap = resolve_qr_debtor_live(invoice).to_snapshot(reason=reason, frozen=False)
    write_qr_debtor_snapshot(invoice, snap)
    return snap


def freeze_qr_debtor_snapshot(
    invoice: Any, *, reason: str = "status_transition"
) -> dict[str, Any]:
    existing = read_qr_debtor_snapshot(invoice)
    if existing is not None:
        if not existing.get("frozen_at"):
            frozen = dict(existing)
            frozen["frozen_at"] = datetime.now(UTC).isoformat()
            frozen["frozen_reason"] = reason
            write_qr_debtor_snapshot(invoice, frozen)
            return frozen
        return existing
    snap = resolve_qr_debtor_live(invoice).to_snapshot(reason=reason, frozen=True)
    snap["frozen_reason"] = reason
    write_qr_debtor_snapshot(invoice, snap)
    logger.info(
        "[QR-Bill] Snapshot débiteur figé (invoice_id=%s, rule=%s, reason=%s).",
        getattr(invoice, "id", None),
        snap.get("rule"),
        reason,
    )
    return snap


def _live_s2_clinic(invoice: Any) -> QrDebtor:
    name = "Clinique"
    street = "Adresse non renseignée"
    pcode = "1200"
    city = "Genève"
    bp = getattr(invoice, "billing_party", None)
    if bp is not None:
        name = (getattr(bp, "display_name", None) or "Clinique").strip()
        addr = (getattr(bp, "billing_address", None) or "").strip()
        if addr:
            street, pcode, city = parse_address_for_qrbill(addr)
    else:
        clinic = getattr(invoice, "billed_to_company", None)
        if clinic is not None:
            name = (getattr(clinic, "name", None) or "Clinique").strip()
            line1 = (getattr(clinic, "domicile_address_line1", None) or "").strip()
            line2 = (getattr(clinic, "domicile_address_line2", None) or "").strip()
            street = f"{line1} {line2}".strip() or "Adresse non renseignée"
            pcode = getattr(clinic, "domicile_zip", None) or "1200"
            city = getattr(clinic, "domicile_city", None) or "Genève"
    return QrDebtor(name=name, street=street, pcode=pcode, city=city, rule="s2_clinic")


def _live_legacy_institution(_invoice: Any, bill_to_client_id: int) -> QrDebtor:
    from models import Client as ClientModel

    institution = ClientModel.query.get(bill_to_client_id)
    if institution and getattr(institution, "is_institution", False):
        name = institution.institution_name or "Institution"
        street = (
            institution.billing_address
            or institution.contact_address
            or "Adresse non renseignée"
        )
    else:
        name = "Institution"
        street = "Adresse non renseignée"
    return QrDebtor(
        name=name,
        street=street,
        pcode="1200",
        city="Genève",
        rule="legacy_institution",
    )


def _live_institution_patient(invoice: Any, client: Any) -> QrDebtor:
    from models import Booking

    name = "Patient"
    street = "Adresse non renseignée"
    pcode = "1200"
    city = "Genève"
    for line in getattr(invoice, "lines", None) or []:
        reservation_id = getattr(line, "reservation_id", None)
        if not reservation_id:
            continue
        booking = Booking.query.get(reservation_id)
        if booking and getattr(booking, "customer_name", None):
            name = booking.customer_name
            break

    linked_inst_id = getattr(client, "linked_institution_id", None)
    if linked_inst_id:
        try:
            from models.institution_patient import InstitutionPatient
            from models.transport_request import TransportRequest

            request = (
                TransportRequest.query.filter_by(institution_id=linked_inst_id)
                .order_by(TransportRequest.id.desc())
                .first()
            )
            if request and request.patient_id:
                patient = InstitutionPatient.query.get(request.patient_id)
                if patient is not None:
                    if name == "Patient":
                        name = (
                            f"{patient.first_name or ''} {patient.last_name or ''}".strip()
                            or "Patient"
                        )
                    addr_str = ", ".join(
                        p
                        for p in (
                            patient.address or "",
                            patient.postal_code or "",
                            patient.city or "",
                        )
                        if p
                    )
                    if addr_str:
                        street, pcode, city = parse_address_for_qrbill(addr_str)
        except Exception as exc:
            logger.warning("[QR-Bill] Patient lookup error: %s", exc)

    return QrDebtor(
        name=name,
        street=street,
        pcode=pcode,
        city=city,
        rule="institution_patient",
    )


def _live_patient_direct(client: Any) -> QrDebtor:
    user = getattr(client, "user", None) if client is not None else None
    name = (
        (
            f"{getattr(user, 'first_name', '') or ''} {getattr(user, 'last_name', '') or ''}"
        ).strip()
        or getattr(user, "username", None)
        or "Client"
    )
    street = "Adresse non renseignée"
    pcode = "1200"
    city = "Genève"
    if client is not None and getattr(client, "domicile_address", None):
        street = client.domicile_address
        if getattr(client, "domicile_zip", None):
            pcode = client.domicile_zip
        if getattr(client, "domicile_city", None):
            city = client.domicile_city
    elif user is not None and getattr(user, "address", None):
        parts = [p.strip() for p in str(user.address).split(",")]
        if len(parts) >= _MIN_ADDRESS_PARTS:
            street = f"{parts[0]}, {parts[1]}"
        if len(parts) >= _MIN_ADDRESS_PARTS_POSTAL:
            pcode = parts[2]
        if len(parts) >= _MIN_ADDRESS_PARTS_CITY:
            city = parts[3]
    return QrDebtor(
        name=name, street=street, pcode=pcode, city=city, rule="patient_direct"
    )
