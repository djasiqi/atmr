"""Facture patient avec signature. Aucun document n'est émis ni conservé.

Le texte proposé n'est pas approuvé. Une signature financière n'est pas
une attestation de transport et ne signifie pas que la course est payée.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

DOCUMENT_STATUS = "pending_approval"
LEGAL_QUALIFICATION = "pending_approval"
DOCUMENT_VERSION = "2026-10-26.draft-1"
DOCUMENT_VERSION_REVISED = "2026-10-26.draft-2"
DOCUMENT_TEXT = (
    "Document soumis à approbation. Il n'est pas en vigueur. "
    "La prestation est le transport indiqué sur la réservation. "
    "Le débiteur est la personne déjà désignée, pas une personne choisie "
    "par le chauffeur. Le montant annoncé est en CHF. "
    "L'entreprise bénéficiaire émettra la facture. "
    "La facturation est différée : ce document n'est pas un paiement reçu. "
    "Les conditions de paiement seront celles que la direction aura "
    "approuvées. Une signature ultérieure porterait sur ce document "
    "seulement. Elle ne remplace pas l'attestation de réalisation "
    "du transport et n'est pas, en l'état, une reconnaissance de dette."
)
REVISED_TEXT = (
    "Brouillon non approuvé. "
    "Je confirme avoir pris connaissance des informations de facturation "
    "présentées pour le transport indiqué, notamment du montant annoncé, "
    "de l'entreprise émettrice et des modalités de règlement différé. "
    "Cette confirmation n'est pas un paiement reçu. "
    "Elle ne remplace pas l'attestation de réalisation du transport "
    "et n'est pas une reconnaissance de dette."
)

_EXCEPTIONS = frozenset(
    {
        "refused",
        "unable",
        "not_understood",
        "representative",
        "phone_unavailable",
        "offline",
        "document_unavailable",
        "document_withdrawn",
        "amount_changed",
        "payer_changed",
        "already_invoiced",
        "already_paid",
        "signature_prefilled",
    }
)


@dataclass(frozen=True, slots=True)
class SignatureDocument:
    """Document versionné soumis à la direction. Il n'est pas en vigueur."""

    version: str
    text: str
    approved: bool
    legal_qualification: str
    replaces_transport_attestation: bool
    means_payment_received: bool
    debt_acknowledgment: bool
    status: str = "draft"
    approved_at: str | None = None
    approved_by: str | None = None
    scope: str = "proposal"
    presented_reference: str | None = None
    prefilled: bool = False


@dataclass(frozen=True, slots=True)
class InvoicePreparation:
    """Dossier préparé pour une facturation différée. Aucune facture émise."""

    allowed: bool
    reason: str
    debtor_label: str | None
    amount_chf: Decimal | None
    currency: str
    company_id: int
    document: SignatureDocument | None
    invoice_means_paid: bool
    invoice_issued: bool
    signature_persisted: bool
    attestation_separate: bool
    amount_preserved: bool
    administrative: bool


def draft_signature_document(*, version: str, text: str) -> SignatureDocument:
    """Propose un texte. L'approbation ne peut pas être forcée ici."""
    if not version.strip() or not text.strip():
        raise ValueError("Document de signature incomplet.")
    return SignatureDocument(
        version=version.strip(),
        text=text.strip(),
        approved=False,
        legal_qualification=LEGAL_QUALIFICATION,
        replaces_transport_attestation=False,
        means_payment_received=False,
        debt_acknowledgment=False,
        status="draft",
        prefilled=False,
    )


def revise_signature_document(
    previous: SignatureDocument, *, version: str, text: str
) -> tuple[SignatureDocument, SignatureDocument]:
    """Nouvelle version. Le texte déjà présenté ne change pas."""
    revised = draft_signature_document(version=version, text=text)
    return previous, revised


def prepare_patient_invoice(
    *,
    patient_is_payer: bool,
    responsibility_locked: bool,
    institution_origin: bool,
    company_invoice_enabled: bool,
    policy_allows_invoice: bool,
    already_paid: bool,
    invoice_already_issued: bool,
    debtor_label: str | None,
    amount_chf: Decimal | None,
    company_id: int,
    document: SignatureDocument | None,
) -> InvoicePreparation:
    """Vérifie les conditions. Le chauffeur ne choisit pas le payeur."""
    document_ready = (
        document is not None
        and document.approved
        and document.status == "approved"
        and bool(document.version)
        and not document.prefilled
    )
    if document is not None and document.status == "withdrawn":
        document_ready = False
    reason = "invoice_prepared"
    allowed = True
    if not patient_is_payer:
        allowed, reason = False, "payer_not_patient"
    elif institution_origin and not responsibility_locked:
        allowed, reason = False, "payer_still_mutable"
    elif not company_invoice_enabled:
        allowed, reason = False, "company_invoice_disabled"
    elif not policy_allows_invoice:
        allowed, reason = False, "policy_without_invoice"
    elif already_paid:
        allowed, reason = False, "already_paid"
    elif invoice_already_issued:
        allowed, reason = False, "invoice_already_issued"
    elif document is not None and document.status == "withdrawn":
        allowed, reason = False, "document_withdrawn"
    elif not document_ready:
        allowed, reason = False, "signature_text_unapproved"
    return InvoicePreparation(
        allowed=allowed,
        reason=reason,
        debtor_label=debtor_label,
        amount_chf=amount_chf,
        currency="CHF",
        company_id=company_id,
        document=document,
        invoice_means_paid=False,
        invoice_issued=False,
        signature_persisted=False,
        attestation_separate=True,
        amount_preserved=amount_chf is not None,
        administrative=not allowed,
    )


def attempt_financial_signature(
    preparation: InvoicePreparation,
    *,
    signatory_role: str,
    situation: str | None,
    prefilled: bool = False,
) -> InvoicePreparation:
    """Aucune signature n'est conservée. Le chauffeur ne signe pas."""
    if prefilled:
        situation = "signature_prefilled"
    blocked = situation in _EXCEPTIONS or signatory_role != "patient"
    if signatory_role == "driver":
        reason = "driver_cannot_sign"
    elif situation == "representative" or signatory_role == "representative":
        reason = "representative_needs_administrative_review"
    elif situation in _EXCEPTIONS:
        reason = situation
    elif not preparation.allowed:
        reason = preparation.reason
    else:
        reason = "signature_not_persisted"
    stored = preparation.allowed and not blocked
    return InvoicePreparation(
        allowed=stored,
        reason=reason,
        debtor_label=preparation.debtor_label,
        amount_chf=preparation.amount_chf,
        currency="CHF",
        company_id=preparation.company_id,
        document=preparation.document,
        invoice_means_paid=False,
        invoice_issued=False,
        signature_persisted=False,
        attestation_separate=True,
        amount_preserved=preparation.amount_preserved,
        administrative=not stored,
    )


@dataclass(frozen=True, slots=True)
class BillingDocumentPresentation:
    """Écran de démonstration. La signature obligatoire n'est pas activée."""

    debtor_label: str
    issuer_name: str
    service_label: str
    amount_chf: Decimal
    currency: str
    version: str
    text: str
    full_conditions_available: bool
    awareness_only: bool
    contractual_acceptance_activated: bool
    mandatory_signature_enabled: bool
    means_payment: bool
    means_attestation: bool
    debt_acknowledgment: bool
    recovery_guarantee: bool
    banner: str


def present_billing_document(
    *,
    debtor_label: str,
    issuer_name: str,
    service_label: str,
    amount_chf: Decimal,
    document: SignatureDocument,
) -> BillingDocumentPresentation:
    """Montre le brouillon. Il ne prouve ni le paiement ni la créance."""
    return BillingDocumentPresentation(
        debtor_label=debtor_label,
        issuer_name=issuer_name,
        service_label=service_label,
        amount_chf=amount_chf,
        currency="CHF",
        version=document.version,
        text=document.text,
        full_conditions_available=True,
        awareness_only=True,
        contractual_acceptance_activated=False,
        mandatory_signature_enabled=False,
        means_payment=False,
        means_attestation=False,
        debt_acknowledgment=False,
        recovery_guarantee=False,
        banner="DRAFT — PENDING APPROVAL",
    )
