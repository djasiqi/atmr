"""Payload QR-facture suisse générique (plateforme : arrondi 0,05 CHF)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


@dataclass(frozen=True)
class QrParty:
    name: str
    street: str
    building_number: str | None
    postal_code: str
    city: str
    country_code: str


@dataclass(frozen=True)
class SwissQrBillPayload:
    creditor: QrParty
    debtor: QrParty
    iban: str
    reference_type: str
    reference: str | None
    amount: Decimal
    currency: str = "CHF"
    additional_information: str | None = None


def platform_qr_amount(total_ttc: Decimal) -> Decimal:
    """Montant QR plateforme = total TTC figé (sans arrondi aux 5 centimes)."""
    return Decimal(total_ttc).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def is_swiss_qr_iban(iban: str | None) -> bool:
    """QR-IBAN suisse : IID (positions 5–9) dans 30000–31999."""
    compact = (iban or "").replace(" ", "").upper()
    if len(compact) < 9 or not compact.startswith("CH"):
        return False
    try:
        iid = int(compact[4:9])
    except ValueError:
        return False
    return 30000 <= iid <= 31999


def resolve_platform_reference_mode(
    iban: str | None, requested_mode: str | None
) -> str:
    """Résout le type de référence QR-facture.

    - QRR : uniquement avec un QR-IBAN ; sinon bascule SCOR (IBAN classique).
    - SCOR : référence créancier ISO 11649 (RF…).
    - NON : aucune référence.
    """
    mode = (requested_mode or "QRR").upper()
    if mode == "NON":
        return "NON"
    if mode == "SCOR":
        return "SCOR"
    if mode == "QRR":
        return "QRR" if is_swiss_qr_iban(iban) else "SCOR"
    # Défaut prudent
    return "QRR" if is_swiss_qr_iban(iban) else "SCOR"


def render_swiss_qr_bill(payload: SwissQrBillPayload) -> dict[str, Any]:
    """Construit les données prêtes pour la lib qrbill (montant via money_round_chf)."""
    from qrbill import QRBill

    amount = platform_qr_amount(payload.amount)
    creditor_addr = {
        "name": payload.creditor.name,
        "street": payload.creditor.street,
        "house_num": payload.creditor.building_number or "",
        "pcode": payload.creditor.postal_code,
        "city": payload.creditor.city,
        "country": payload.creditor.country_code or "CH",
    }
    debtor_addr = {
        "name": payload.debtor.name,
        "street": payload.debtor.street,
        "house_num": payload.debtor.building_number or "",
        "pcode": payload.debtor.postal_code,
        "city": payload.debtor.city,
        "country": payload.debtor.country_code or "CH",
    }
    ref_type = resolve_platform_reference_mode(
        payload.iban, payload.reference_type or "NON"
    )
    kwargs: dict[str, Any] = {
        "account": payload.iban.replace(" ", ""),
        "creditor": creditor_addr,
        "amount": f"{amount:.2f}",
        "currency": payload.currency or "CHF",
        "debtor": debtor_addr,
        "language": "fr",
    }
    # QRR uniquement avec QR-IBAN + référence 27 chiffres
    if (ref_type == "QRR" and payload.reference) or (
        ref_type == "SCOR" and payload.reference
    ):
        kwargs["reference_number"] = payload.reference
    if payload.additional_information:
        kwargs["additional_information"] = payload.additional_information[:140]

    bill = QRBill(**kwargs)
    return {
        "amount": str(amount),
        "currency": payload.currency,
        "iban": payload.iban,
        "reference": payload.reference if ref_type in ("QRR", "SCOR") else None,
        "reference_type": ref_type,
        "qr_bill": bill,
    }
