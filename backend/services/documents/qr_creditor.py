"""Créancier QR-facture (compte + identité) — snapshot immuable.

Le débiteur QR est déjà figé (`qr_debtor_snapshot`). Le compte et l'identité
créancier venaient encore des master data (`CompanyBillingProfile` / `Company` /
`CompanyBillingSettings`) : une régénération SENT pouvait changer l'IBAN.

Source de vérité : ``invoice.meta["qr_creditor_snapshot"]``, capturé avec le
bloc « Facturé à » (brouillon rafraîchi, gel à la sortie de DRAFT).

Ne duplique pas : montant (`Invoice.total_amount`), devise (CHF), référence
(`Invoice.qr_reference`), numéro / période (additional_information), langue (fr).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from services.documents.invoice_recipient import invoice_billed_to_is_frozen

logger = logging.getLogger("qrbill_service")

QR_CREDITOR_SNAPSHOT_KEY = "qr_creditor_snapshot"
QR_CREDITOR_SNAPSHOT_VERSION = 1

QrCreditorSource = Literal["live", "snapshot", "legacy_live"]


def _normalize_meta(meta: Any) -> dict[str, Any]:
    if isinstance(meta, dict):
        return dict(meta)
    return {}


def _clean(value: Any) -> str:
    return str(value or "").strip()


@dataclass(frozen=True, slots=True)
class QrCreditor:
    """Compte et identité « Payable à » réellement encodés dans la QR."""

    account: str
    name: str
    street: str
    pcode: str
    city: str
    country: str = "CH"
    address_type: str = "S"
    source: QrCreditorSource = "live"

    def to_qrbill_address(self) -> dict[str, str]:
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
            "version": QR_CREDITOR_SNAPSHOT_VERSION,
            "account": self.account,
            "name": self.name,
            "street": self.street,
            "pcode": self.pcode,
            "city": self.city,
            "country": self.country,
            "address_type": self.address_type,
            "captured_at": now,
            "captured_reason": reason,
            "frozen_at": now if frozen else None,
        }


def read_qr_creditor_snapshot(invoice: Any) -> dict[str, Any] | None:
    snap = _normalize_meta(getattr(invoice, "meta", None)).get(QR_CREDITOR_SNAPSHOT_KEY)
    if isinstance(snap, dict) and (
        _clean(snap.get("account")) or _clean(snap.get("name"))
    ):
        return snap
    return None


def qr_creditor_from_snapshot(snap: dict[str, Any]) -> QrCreditor:
    return QrCreditor(
        account=_clean(snap.get("account")),
        name=_clean(snap.get("name")) or "[Entreprise non configurée]",
        street=_clean(snap.get("street")) or "[Adresse non configurée]",
        pcode=_clean(snap.get("pcode")) or "0000",
        city=_clean(snap.get("city")) or "[Ville non configurée]",
        country=_clean(snap.get("country")) or "CH",
        address_type=_clean(snap.get("address_type")) or "S",
        source="snapshot",
    )


def write_qr_creditor_snapshot(invoice: Any, snap: dict[str, Any]) -> None:
    meta = _normalize_meta(getattr(invoice, "meta", None))
    meta[QR_CREDITOR_SNAPSHOT_KEY] = snap
    invoice.meta = meta


def resolve_invoice_qr_creditor(
    invoice: Any, *, use_snapshot: bool = True
) -> QrCreditor:
    """Snapshot si facture figée, sinon master data (même cascade que QRBillService)."""
    if use_snapshot and invoice_billed_to_is_frozen(invoice):
        snap = read_qr_creditor_snapshot(invoice)
        if snap is not None:
            return qr_creditor_from_snapshot(snap)
        logger.warning(
            "[QR-Bill] Facture figée sans qr_creditor_snapshot (invoice_id=%s, status=%s) : "
            "rendu depuis les master data courantes (legacy_live).",
            getattr(invoice, "id", None),
            getattr(getattr(invoice, "status", None), "value", None),
        )
        live = resolve_qr_creditor_live(invoice)
        return QrCreditor(
            account=live.account,
            name=live.name,
            street=live.street,
            pcode=live.pcode,
            city=live.city,
            country=live.country,
            address_type=live.address_type,
            source="legacy_live",
        )
    return resolve_qr_creditor_live(invoice)


def resolve_qr_creditor_live(invoice: Any) -> QrCreditor:
    """Cascade historique : profil → company → CompanyBillingSettings.iban."""
    from models import CompanyBillingSettings
    from services.billing import BillingProfileService

    company = getattr(invoice, "company", None)
    company_id = getattr(invoice, "company_id", None) or getattr(company, "id", None)
    profile = (
        BillingProfileService.get_by_company_id(int(company_id)) if company_id else None
    )

    if profile:
        if _clean(getattr(profile, "building_number", None)):
            street = (
                f"{_clean(profile.street_name)} {_clean(profile.building_number)}"
            ).strip()
        else:
            street = _clean(getattr(profile, "street_name", None))
        account = _clean(getattr(profile, "qr_iban", None)) or _clean(
            getattr(profile, "iban", None)
        )
        creditor = QrCreditor(
            account=account,
            name=_clean(getattr(profile, "legal_name", None))
            or "[Entreprise non configurée]",
            street=street or "[Adresse non configurée]",
            pcode=_clean(getattr(profile, "postal_code", None)) or "0000",
            city=_clean(getattr(profile, "city", None)) or "[Ville non configurée]",
            country=_clean(getattr(profile, "country_code", None)) or "CH",
            address_type="S",
            source="live",
        )
    else:
        street = (
            _clean(getattr(company, "domicile_address_line1", None))
            or _clean(getattr(company, "address", None))
            or "[Adresse non configurée]"
        )
        creditor = QrCreditor(
            account=_clean(getattr(company, "iban", None)),
            name=_clean(getattr(company, "name", None))
            or "[Entreprise non configurée]",
            street=street,
            pcode=_clean(getattr(company, "domicile_zip", None)) or "0000",
            city=_clean(getattr(company, "domicile_city", None))
            or "[Ville non configurée]",
            country=_clean(getattr(company, "domicile_country", None)) or "CH",
            address_type="K",
            source="live",
        )

    if not creditor.account and company_id:
        settings = CompanyBillingSettings.query.filter_by(
            company_id=int(company_id)
        ).first()
        fallback = _clean(getattr(settings, "iban", None)) if settings else ""
        if fallback:
            return QrCreditor(
                account=fallback,
                name=creditor.name,
                street=creditor.street,
                pcode=creditor.pcode,
                city=creditor.city,
                country=creditor.country,
                address_type=creditor.address_type,
                source="live",
            )
    return creditor


def refresh_qr_creditor_snapshot_if_draft(
    invoice: Any, *, reason: str = "pdf_generation"
) -> dict[str, Any] | None:
    if invoice_billed_to_is_frozen(invoice):
        return read_qr_creditor_snapshot(invoice)
    snap = resolve_qr_creditor_live(invoice).to_snapshot(reason=reason, frozen=False)
    write_qr_creditor_snapshot(invoice, snap)
    return snap


def freeze_qr_creditor_snapshot(
    invoice: Any, *, reason: str = "status_transition"
) -> dict[str, Any]:
    existing = read_qr_creditor_snapshot(invoice)
    if existing is not None:
        if not existing.get("frozen_at"):
            frozen = dict(existing)
            frozen["frozen_at"] = datetime.now(UTC).isoformat()
            frozen["frozen_reason"] = reason
            write_qr_creditor_snapshot(invoice, frozen)
            return frozen
        return existing
    snap = resolve_qr_creditor_live(invoice).to_snapshot(reason=reason, frozen=True)
    snap["frozen_reason"] = reason
    write_qr_creditor_snapshot(invoice, snap)
    logger.info(
        "[QR-Bill] Snapshot créancier figé (invoice_id=%s, account=%s, reason=%s).",
        getattr(invoice, "id", None),
        (snap.get("account") or "")[:8],
        reason,
    )
    return snap
