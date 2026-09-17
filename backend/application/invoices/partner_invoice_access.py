"""Accès tenant-safe aux factures du catalogue partenaire."""

from __future__ import annotations

from sqlalchemy.orm import joinedload, selectinload

from models.partner_invoice import PartnerInvoice
from models.partnership import Partnership


def company_can_access_partner_invoice(
    company_id: int, partner_invoice: PartnerInvoice
) -> bool:
    """True si l'entreprise est owner, partenaire ou exécutante."""
    partnership = partner_invoice.partnership
    if partnership is None:
        partnership = Partnership.query.get(partner_invoice.partnership_id)
    if partnership is None:
        return partner_invoice.executing_company_id == company_id
    allowed = {
        partnership.owner_company_id,
        partnership.partner_company_id,
        partner_invoice.executing_company_id,
    }
    return company_id in allowed


def load_accessible_partner_invoice(
    company_id: int, partner_invoice_id: int
) -> PartnerInvoice | None:
    """Charge une PartnerInvoice avec graphe utile, ou None si hors périmètre."""
    partner_invoice = (
        PartnerInvoice.query.options(
            selectinload(PartnerInvoice.lines),
            selectinload(PartnerInvoice.recorded_payments),
            selectinload(PartnerInvoice.transfers),
            joinedload(PartnerInvoice.partnership).joinedload(
                Partnership.owner_company
            ),
            joinedload(PartnerInvoice.partnership).joinedload(
                Partnership.partner_company
            ),
        )
        .filter_by(id=partner_invoice_id)
        .first()
    )
    if partner_invoice is None:
        return None
    if not company_can_access_partner_invoice(company_id, partner_invoice):
        return None
    return partner_invoice
