"""Synchronisation IBAN / QR-IBAN entre Company, CompanyBillingSettings et CompanyBillingProfile.

Évite les factures avec un IBAN au pied de page (settings) différent du QR-facture (profil).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from models import Company

logger = logging.getLogger(__name__)


def _normalize_iban(value: str | None) -> str | None:
    if not value:
        return None
    s = "".join(str(value).split()).upper()
    return s if s else None


def sync_banking_identifiers(
    company: Company,
    *,
    source: Literal["billing_settings", "company"] = "billing_settings",
) -> None:
    """Propage IBAN / QR-IBAN pour aligner PDF (pied de page), QR-facture et profil entreprise.

    - ``billing_settings`` : les valeurs déjà écrites sur ``CompanyBillingSettings`` font foi
      (c’est la source utilisée pour le pied de page PDF). Recopie vers ``Company`` et
      ``CompanyBillingProfile``.
    - ``company`` : ``Company.iban`` fait foi (ex. ancien flux profil). Recopie vers les
      paramètres de facturation et le profil ; crée une ligne ``CompanyBillingSettings`` si
      absente.

    Ne supprime pas d’IBAN existant sur ``Company`` si les settings sont vides (évite pertes).
    """
    from ext import db
    from models import CompanyBillingProfile, CompanyBillingSettings

    cid = company.id
    billing = CompanyBillingSettings.query.filter_by(company_id=cid).first()
    profile = CompanyBillingProfile.query.filter_by(company_id=cid).first()

    if source == "billing_settings":
        if not billing:
            return
        main = _normalize_iban(billing.iban)
        qr = _normalize_iban(billing.qr_iban) if billing.qr_iban else main
        if main:
            company.iban = main
        if profile and main:
            profile.iban = main
            profile.qr_iban = qr if qr else main
        return

    # source == "company"
    main = _normalize_iban(company.iban)
    if not main:
        return

    if not billing:
        billing = CompanyBillingSettings()
        billing.company_id = cid
        billing.payment_terms_days = 10
        billing.overdue_fee = 15
        billing.reminder1_fee = 0
        billing.reminder2_fee = 40
        billing.reminder3_fee = 0
        billing.reminder_schedule_days = {"1": 10, "2": 5, "3": 5}
        billing.auto_reminders_enabled = True
        billing.invoice_number_format = "{PREFIX}-{YYYY}-{MM}-{SEQ4}"
        billing.invoice_prefix = "EM"
        billing.pdf_template_variant = "default"
        db.session.add(billing)

    billing.iban = main
    if not billing.qr_iban or not str(billing.qr_iban).strip():
        billing.qr_iban = main

    qr_resolved = _normalize_iban(billing.qr_iban) if billing.qr_iban else main
    if profile:
        profile.iban = main
        profile.qr_iban = qr_resolved if qr_resolved else main

    logger.info(
        "[BankingSync] company_id=%s source=company — aligné settings + profil sur IBAN",
        cid,
    )
