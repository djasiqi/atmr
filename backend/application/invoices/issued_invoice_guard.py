"""Garde-fou centralisé d'immutabilité des factures émises (LIRIE — étape 22).

Règle unique, appliquée dans les services backend faisant autorité (jamais
seulement côté frontend) :

* une facture **brouillon** (``draft``) conserve le flux existant (édition des
  lignes, frais papier, recalcul des totaux, génération / régénération du PDF) ;
* une facture **émise** — tout statut autre que ``draft`` : ``sent``,
  ``partially_paid``, ``paid``, ``overdue``, ``cancelled`` et toute valeur
  inconnue ou héritée — est **figée** : son contenu financier (lignes, totaux,
  TVA, références QR, snapshots débiteur / créancier) et son PDF original ne
  sont jamais recalculés, modifiés, supprimés ni remplacés automatiquement.

Le garde-fou est volontairement *fail-closed* : un statut absent, inconnu ou
non reconnu est traité comme « émis ». Aucun paramètre d'appel (ni a fortiori
une variable frontend) ne permet de le contourner ; une correction d'une
facture émise relève d'une procédure administrative explicite (hors de ce lot).

Les constantes et helpers couvrent le catalogue standard (``Invoice``) et le
catalogue partenaire (``PartnerInvoice``, statuts chaîne).
"""

from __future__ import annotations

import logging
from typing import Any

from models.enums import InvoiceStatus
from models.partner_invoice import PartnerInvoiceStatus

logger = logging.getLogger(__name__)

# Statuts « émis » connus du modèle réel (``models.enums.InvoiceStatus``).
# ``CANCELLED`` est inclus : une facture annulée reste un document comptable
# figé dont le PDF et les montants ne doivent pas être réécrits.
ISSUED_INVOICE_STATUSES: frozenset[InvoiceStatus] = frozenset(
    {
        InvoiceStatus.SENT,
        InvoiceStatus.PARTIALLY_PAID,
        InvoiceStatus.PAID,
        InvoiceStatus.OVERDUE,
        InvoiceStatus.CANCELLED,
    }
)

# Statuts partenaire émis (``PartnerInvoiceStatus`` est une classe de constantes
# chaîne, pas une ``Enum``).
ISSUED_PARTNER_INVOICE_STATUSES: frozenset[str] = frozenset(
    {
        PartnerInvoiceStatus.SENT,
        PartnerInvoiceStatus.PARTIALLY_PAID,
        PartnerInvoiceStatus.PAID,
        PartnerInvoiceStatus.OVERDUE,
        PartnerInvoiceStatus.CANCELLED,
    }
)

ERROR_CODE_ISSUED_INVOICE_IMMUTABLE = "ISSUED_INVOICE_IMMUTABLE"
ERROR_CODE_ISSUED_INVOICE_PDF_MISSING = "ISSUED_INVOICE_PDF_MISSING"

ISSUED_INVOICE_ADMIN_MESSAGE = (
    "Facture émise : son contenu financier et son PDF original sont figés. "
    "Aucune régénération ni modification automatique n'est possible. "
    "Toute correction doit être signalée à l'administration afin de préserver "
    "le document original et les montants comptabilisés."
)

ISSUED_INVOICE_PDF_MISSING_ADMIN_MESSAGE = (
    "Facture émise dont le PDF original est introuvable. Le système ne fabrique "
    "pas de nouveau document à la place de l'original : ce dossier doit être "
    "signalé à l'administration pour traitement."
)


class IssuedInvoiceImmutableError(Exception):
    """Refus d'une opération destructive ou recalculante sur une facture émise.

    ``status_code`` vaut 409 (conflit avec l'état figé de la ressource).
    ``to_error_payload()`` fournit la réponse HTTP normalisée.
    """

    status_code = 409

    def __init__(
        self,
        *,
        invoice_id: int | None,
        status: str | None,
        operation: str,
        code: str = ERROR_CODE_ISSUED_INVOICE_IMMUTABLE,
        message: str | None = None,
        catalog: str = "invoice",
    ) -> None:
        self.invoice_id = invoice_id
        self.status = status
        self.operation = operation
        self.code = code
        self.catalog = catalog
        self.message = message or (
            ISSUED_INVOICE_PDF_MISSING_ADMIN_MESSAGE
            if code == ERROR_CODE_ISSUED_INVOICE_PDF_MISSING
            else ISSUED_INVOICE_ADMIN_MESSAGE
        )
        super().__init__(self.message)

    def to_error_payload(self) -> dict[str, Any]:
        return issued_invoice_refusal_payload(
            invoice_id=self.invoice_id,
            status=self.status,
            operation=self.operation,
            code=self.code,
            message=self.message,
            catalog=self.catalog,
        )


def issued_invoice_refusal_payload(
    *,
    invoice_id: int | None,
    status: str | None,
    operation: str,
    code: str = ERROR_CODE_ISSUED_INVOICE_IMMUTABLE,
    message: str | None = None,
    catalog: str = "invoice",
) -> dict[str, Any]:
    """Corps de réponse normalisé d'un refus (409) — même forme partout."""
    msg = message or (
        ISSUED_INVOICE_PDF_MISSING_ADMIN_MESSAGE
        if code == ERROR_CODE_ISSUED_INVOICE_PDF_MISSING
        else ISSUED_INVOICE_ADMIN_MESSAGE
    )
    return {
        "success": False,
        "error": msg,
        "error_code": code,
        "details": {
            "invoice_id": invoice_id,
            "catalog": catalog,
            "status": status,
            "operation": operation,
            "administrative_procedure_required": True,
        },
    }


def _status_value(status: Any) -> str | None:
    """Valeur chaîne normalisée d'un statut (Enum, chaîne, None)."""
    if status is None:
        return None
    raw = getattr(status, "value", status)
    if raw is None:
        return None
    return str(raw).strip().lower()


def invoice_status_value(invoice: Any) -> str | None:
    """Valeur chaîne du statut d'une facture (standard ou partenaire)."""
    return _status_value(getattr(invoice, "status", None))


def invoice_is_issued(invoice: Any) -> bool:
    """True si la facture standard est émise (tout sauf ``draft``) — fail-closed.

    Une valeur absente, inconnue ou héritée (ex. ``UNPAID``) est considérée
    comme émise : on ne modifie jamais un document dont on ne sait pas
    prouver qu'il est encore un brouillon.
    """
    return invoice_status_value(invoice) != InvoiceStatus.DRAFT.value


def invoice_is_mutable(invoice: Any) -> bool:
    """True uniquement pour un brouillon (flux d'édition / génération existant)."""
    return not invoice_is_issued(invoice)


def partner_invoice_is_issued(partner_invoice: Any) -> bool:
    """True si la facture partenaire est émise (tout sauf ``draft``) — fail-closed."""
    return invoice_status_value(partner_invoice) != _status_value(
        PartnerInvoiceStatus.DRAFT
    )


def partner_invoice_is_mutable(partner_invoice: Any) -> bool:
    return not partner_invoice_is_issued(partner_invoice)


def assert_invoice_mutable(invoice: Any, operation: str) -> None:
    """Lève ``IssuedInvoiceImmutableError`` si la facture standard est émise.

    À appeler **avant** toute écriture en base, toute suppression de fichier
    et tout recalcul financier.
    """
    if invoice_is_issued(invoice):
        err = IssuedInvoiceImmutableError(
            invoice_id=getattr(invoice, "id", None),
            status=invoice_status_value(invoice),
            operation=operation,
        )
        logger.warning(
            "[ISSUED INVOICE GUARD] Opération refusée (invoice_id=%s, status=%s, "
            "operation=%s)",
            err.invoice_id,
            err.status,
            operation,
        )
        raise err


def assert_partner_invoice_mutable(partner_invoice: Any, operation: str) -> None:
    """Équivalent partenaire de ``assert_invoice_mutable``."""
    if partner_invoice_is_issued(partner_invoice):
        err = IssuedInvoiceImmutableError(
            invoice_id=getattr(partner_invoice, "id", None),
            status=invoice_status_value(partner_invoice),
            operation=operation,
            catalog="partner_invoice",
        )
        logger.warning(
            "[ISSUED INVOICE GUARD] Opération partenaire refusée "
            "(partner_invoice_id=%s, status=%s, operation=%s)",
            err.invoice_id,
            err.status,
            operation,
        )
        raise err


def issued_invoice_pdf_missing_error(
    invoice: Any, operation: str, *, catalog: str = "invoice"
) -> IssuedInvoiceImmutableError:
    """Erreur « PDF original absent » d'une facture émise (signalement admin)."""
    err = IssuedInvoiceImmutableError(
        invoice_id=getattr(invoice, "id", None),
        status=invoice_status_value(invoice),
        operation=operation,
        code=ERROR_CODE_ISSUED_INVOICE_PDF_MISSING,
        catalog=catalog,
    )
    logger.error(
        "[ISSUED INVOICE GUARD] PDF original absent sur facture émise "
        "(%s_id=%s, status=%s, operation=%s) : signalement administratif requis, "
        "aucun document fabriqué.",
        catalog,
        err.invoice_id,
        err.status,
        operation,
    )
    return err
