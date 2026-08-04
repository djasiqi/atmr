"""Statut opérationnel et actions SSOT des dossiers facturation plateforme."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from models.enums import (
    PlatformBillingPeriodStatus,
    PlatformIssuedInvoiceStatus,
    PlatformStatementStatus,
)
from models.platform_billing import (
    PlatformBillingPeriod,
    PlatformInvoice,
    PlatformIssuedInvoice,
)
from services.admin_authz import (
    CAP_BILLING_CANCEL,
    CAP_BILLING_CREDIT,
    CAP_BILLING_DUE_DATE,
    CAP_BILLING_ISSUE,
    CAP_BILLING_PAYMENT,
    CAP_BILLING_SEND,
    CAP_BILLING_VALIDATE,
)
from services.platform_billing.issued_status import (
    balance_due_for_registry,
    is_credit_note,
    ui_status,
)
from services.platform_billing.money import money_round_chf

# Statuts opérationnels (codes API)
STATUS_CREDITED = "CREDITED"
STATUS_CANCELLED = "CANCELLED"
STATUS_PAID = "PAID"
STATUS_OVERDUE = "OVERDUE"
STATUS_PARTIALLY_PAID = "PARTIALLY_PAID"
STATUS_A_ENCAISSER = "A_ENCAISSER"
STATUS_A_ENVOYER = "A_ENVOYER"
STATUS_PRETE_A_EMETTRE = "PRETE_A_EMETTRE"
STATUS_PRETE_A_CLOTURER = "PRETE_A_CLOTURER"
STATUS_A_CONTROLER = "A_CONTROLER"
STATUS_A_CALCULER = "A_CALCULER"

A_TRAITER_STATUSES = frozenset(
    {
        STATUS_A_CALCULER,
        STATUS_A_CONTROLER,
        STATUS_PRETE_A_CLOTURER,
        STATUS_PRETE_A_EMETTRE,
        STATUS_A_ENVOYER,
        STATUS_A_ENCAISSER,
        STATUS_PARTIALLY_PAID,
        STATUS_OVERDUE,
    }
)

# Actions
ACTION_VIEW = "VIEW"
ACTION_RECALCULATE_DOSSIER = "RECALCULATE_DOSSIER"
ACTION_REVIEW = "REVIEW"
ACTION_ISSUE = "ISSUE"
ACTION_MARK_SENT = "MARK_SENT"
ACTION_RECORD_PAYMENT = "RECORD_PAYMENT"
ACTION_DOWNLOAD_PDF = "DOWNLOAD_PDF"
ACTION_CHANGE_DUE_DATE = "CHANGE_DUE_DATE"
ACTION_CANCEL = "CANCEL"
ACTION_CREDIT = "CREDIT"
ACTION_VIEW_CREDIT_NOTE = "VIEW_CREDIT_NOTE"
ACTION_VIEW_PAYMENTS = "VIEW_PAYMENTS"
ACTION_SEND_REMINDER = "SEND_REMINDER"
ACTION_REVERSE_PAYMENT = "REVERSE_PAYMENT"
ACTION_EDIT_INVOICE = "EDIT_INVOICE"
ACTION_CORRECT_INVOICE = "CORRECT_INVOICE"

ACTION_CAPABILITY: dict[str, str | None] = {
    ACTION_VIEW: None,
    ACTION_RECALCULATE_DOSSIER: CAP_BILLING_VALIDATE,
    ACTION_REVIEW: CAP_BILLING_VALIDATE,
    ACTION_ISSUE: CAP_BILLING_ISSUE,
    ACTION_MARK_SENT: CAP_BILLING_SEND,
    ACTION_RECORD_PAYMENT: CAP_BILLING_PAYMENT,
    ACTION_DOWNLOAD_PDF: None,
    ACTION_CHANGE_DUE_DATE: CAP_BILLING_DUE_DATE,
    ACTION_CANCEL: CAP_BILLING_CANCEL,
    ACTION_CREDIT: CAP_BILLING_CREDIT,
    ACTION_VIEW_CREDIT_NOTE: None,
    ACTION_VIEW_PAYMENTS: None,
    ACTION_SEND_REMINDER: CAP_BILLING_SEND,
    ACTION_REVERSE_PAYMENT: CAP_BILLING_PAYMENT,
    # Projection : cap « principale » ; routes exigent CANCEL+ISSUE / CREDIT+ISSUE
    ACTION_EDIT_INVOICE: CAP_BILLING_ISSUE,
    ACTION_CORRECT_INVOICE: CAP_BILLING_CREDIT,
}

ACTION_GROUP: dict[str, str] = {
    ACTION_VIEW: "FACTURE",
    ACTION_DOWNLOAD_PDF: "FACTURE",
    ACTION_MARK_SENT: "FACTURE",
    ACTION_VIEW_CREDIT_NOTE: "FACTURE",
    ACTION_EDIT_INVOICE: "FACTURE",
    ACTION_CORRECT_INVOICE: "CORRECTION",
    ACTION_RECORD_PAYMENT: "PAIEMENT",
    ACTION_VIEW_PAYMENTS: "PAIEMENT",
    ACTION_REVERSE_PAYMENT: "PAIEMENT",
    ACTION_CHANGE_DUE_DATE: "CORRECTION",
    ACTION_CREDIT: "CORRECTION",
    ACTION_CANCEL: "EXCEPTION",
    ACTION_SEND_REMINDER: "SUIVI",
    ACTION_RECALCULATE_DOSSIER: "CORRECTION",
    ACTION_REVIEW: "FACTURE",
    ACTION_ISSUE: "FACTURE",
}


def dossier_key(period_id: int, company_id: int) -> str:
    return f"{int(period_id)}:{int(company_id)}"


def parse_dossier_key(key: str) -> tuple[int, int]:
    parts = str(key or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError("dossier_key invalide")
    return int(parts[0]), int(parts[1])


def _stmt_status(statement: PlatformInvoice | None) -> str | None:
    if statement is None:
        return None
    return (
        getattr(statement, "statement_status", None)
        or PlatformStatementStatus.DRAFT.value
    )


def _period_locked(period: PlatformBillingPeriod | None) -> bool:
    if period is None:
        return False
    return period.status == PlatformBillingPeriodStatus.LOCKED.value


def _active_primary_invoice(
    issued: PlatformIssuedInvoice | None,
) -> PlatformIssuedInvoice | None:
    """Facture primaire active liée au relevé (ignore DRAFT orphelin si besoin)."""
    if issued is None:
        return None
    if is_credit_note(issued):
        return None
    return issued


def zero_charge_flags(
    statement: PlatformInvoice | None,
) -> tuple[bool, str | None]:
    if statement is None:
        return False, None
    total = money_round_chf(Decimal(str(statement.total_amount or 0)))
    if total <= 0:
        return True, "Aucun montant facturable"
    return False, None


def compute_issuable(
    statement: PlatformInvoice | None,
    period: PlatformBillingPeriod | None,
    *,
    qr_ready: bool | None = None,
    qr_errors: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Aligné sur statement_qr_ready quand possible ; sinon heuristique locale."""
    if statement is None:
        return False, ["Aucun relevé"]
    if qr_ready is not None:
        return bool(qr_ready), list(qr_errors or [])
    errors: list[str] = []
    st = _stmt_status(statement)
    if st != PlatformStatementStatus.LOCKED.value:
        errors.append("Relevé non verrouillé")
    if not _period_locked(period):
        errors.append("Période non verrouillée")
    total = Decimal(str(statement.total_amount or 0))
    if total <= 0:
        errors.append("Montant total doit être > 0 pour QR")
    return len(errors) == 0, errors


def operational_status(
    *,
    statement: PlatformInvoice | None,
    period: PlatformBillingPeriod | None,
    primary_invoice: PlatformIssuedInvoice | None,
    now: datetime | None = None,
) -> str:
    """Priorité haute → basse (plan verrouillé)."""
    now = now or datetime.now(UTC)
    inv = _active_primary_invoice(primary_invoice)

    if inv is not None:
        st = inv.status or ""
        if st == PlatformIssuedInvoiceStatus.CREDITED.value:
            return STATUS_CREDITED
        if st == PlatformIssuedInvoiceStatus.CANCELLED.value:
            # Remplacement : nouvelle facture liée au relevé → pas CANCELLED sur le dossier
            return STATUS_CANCELLED
        u = ui_status(inv, now=now)
        if u == "PAID":
            return STATUS_PAID
        if u == "OVERDUE":
            return STATUS_OVERDUE
        if u == "PARTIALLY_PAID":
            return STATUS_PARTIALLY_PAID
        if inv.sent_at is None and st not in (
            PlatformIssuedInvoiceStatus.CANCELLED.value,
            PlatformIssuedInvoiceStatus.CREDITED.value,
        ):
            return STATUS_A_ENVOYER
        bal = balance_due_for_registry(inv)
        if bal > 0:
            return STATUS_A_ENCAISSER
        return STATUS_PAID

    # Pas de facture active
    stmt_st = _stmt_status(statement)
    if stmt_st is None or stmt_st == PlatformStatementStatus.DRAFT.value:
        return STATUS_A_CALCULER
    if stmt_st in (
        PlatformStatementStatus.CALCULATED.value,
        PlatformStatementStatus.NEEDS_REVIEW.value,
    ):
        return STATUS_A_CONTROLER
    if stmt_st == PlatformStatementStatus.VALIDATED.value and not _period_locked(
        period
    ):
        return STATUS_PRETE_A_CLOTURER
    if stmt_st == PlatformStatementStatus.LOCKED.value and _period_locked(period):
        return STATUS_PRETE_A_EMETTRE
    # VALIDATED + période déjà locked (transitoire) ou LOCKED sans période locked
    if stmt_st == PlatformStatementStatus.VALIDATED.value:
        return STATUS_PRETE_A_CLOTURER
    if stmt_st == PlatformStatementStatus.LOCKED.value:
        return STATUS_PRETE_A_EMETTRE
    return STATUS_A_CALCULER


def resolve_actions(
    *,
    status: str,
    statement: PlatformInvoice | None,
    primary_invoice: PlatformIssuedInvoice | None,
    credit_note_id: int | None,
    issuable: bool,
    issuer_errors: list[str],
    caps: set[str] | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Calcule primary_action, allowed_actions, blocked_actions."""
    now = now or datetime.now(UTC)
    caps = caps if caps is not None else set()
    allowed: list[str] = []
    blocked: dict[str, str] = {}
    inv = primary_invoice

    def _can(action: str) -> bool:
        need = ACTION_CAPABILITY.get(action)
        if need is None:
            return True
        # Si caps vide = caller non fourni → autoriser côté projection
        # (les routes mutantes restent gardées)
        if not caps:
            return True
        return need in caps

    def _add(action: str) -> None:
        if _can(action) and action not in allowed:
            allowed.append(action)

    def _block(action: str, reason: str) -> None:
        blocked[action] = reason

    _add(ACTION_VIEW)

    if status == STATUS_A_CALCULER:
        primary = ACTION_RECALCULATE_DOSSIER if _can(ACTION_RECALCULATE_DOSSIER) else ACTION_VIEW
        _add(ACTION_RECALCULATE_DOSSIER)
    elif status == STATUS_A_CONTROLER:
        primary = ACTION_REVIEW if _can(ACTION_REVIEW) else ACTION_VIEW
        _add(ACTION_REVIEW)
        _add(ACTION_RECALCULATE_DOSSIER)
    elif status == STATUS_PRETE_A_CLOTURER:
        primary = ACTION_VIEW
        _add(ACTION_REVIEW)
        _block(ACTION_ISSUE, "Clôturez d'abord la période")
    elif status == STATUS_PRETE_A_EMETTRE:
        if issuable and _can(ACTION_ISSUE):
            primary = ACTION_ISSUE
            _add(ACTION_ISSUE)
        else:
            primary = ACTION_VIEW
            reason = "; ".join(issuer_errors) if issuer_errors else "Émission non prête"
            _block(ACTION_ISSUE, reason)
    elif status == STATUS_A_ENVOYER:
        primary = ACTION_MARK_SENT if _can(ACTION_MARK_SENT) else ACTION_VIEW
        _add(ACTION_MARK_SENT)
        _add(ACTION_DOWNLOAD_PDF)
        _add(ACTION_CHANGE_DUE_DATE)
        paid = (
            money_round_chf(Decimal(str(inv.amount_paid or 0)))
            if inv
            else Decimal("0.00")
        )
        if inv and not inv.sent_at and paid <= 0:
            _add(ACTION_CANCEL)
            if _can(ACTION_EDIT_INVOICE) and _can(ACTION_CANCEL):
                _add(ACTION_EDIT_INVOICE)
        else:
            _block(ACTION_CANCEL, "Facture déjà envoyée ou payée")
            _block(
                ACTION_EDIT_INVOICE,
                "Correction impossible tant que des paiements sont enregistrés"
                if paid > 0
                else "Facture déjà envoyée",
            )
    elif status in (STATUS_A_ENCAISSER, STATUS_PARTIALLY_PAID):
        primary = (
            ACTION_RECORD_PAYMENT if _can(ACTION_RECORD_PAYMENT) else ACTION_VIEW
        )
        _add(ACTION_RECORD_PAYMENT)
        _add(ACTION_VIEW_PAYMENTS)
        _add(ACTION_DOWNLOAD_PDF)
        _add(ACTION_CHANGE_DUE_DATE)
        _add(ACTION_REVERSE_PAYMENT)
        paid = (
            money_round_chf(Decimal(str(inv.amount_paid or 0)))
            if inv
            else Decimal("0.00")
        )
        if inv and paid <= 0:
            _add(ACTION_CREDIT)
            if _can(ACTION_CORRECT_INVOICE) and _can(ACTION_ISSUE):
                _add(ACTION_CORRECT_INVOICE)
        else:
            _block(ACTION_CREDIT, "Un paiement est enregistré")
            _block(
                ACTION_CORRECT_INVOICE,
                "Correction impossible tant que des paiements sont enregistrés",
            )
        _block(ACTION_CANCEL, "Facture déjà envoyée")
    elif status == STATUS_OVERDUE:
        # Pas d'API rappel unitaire fiable en V1 → paiement en principal
        if _can(ACTION_RECORD_PAYMENT):
            primary = ACTION_RECORD_PAYMENT
        else:
            primary = ACTION_VIEW
        _add(ACTION_RECORD_PAYMENT)
        _add(ACTION_VIEW_PAYMENTS)
        _add(ACTION_DOWNLOAD_PDF)
        _add(ACTION_CHANGE_DUE_DATE)
        _add(ACTION_REVERSE_PAYMENT)
        paid = (
            money_round_chf(Decimal(str(inv.amount_paid or 0)))
            if inv
            else Decimal("0.00")
        )
        if inv and paid <= 0:
            _add(ACTION_CREDIT)
            if _can(ACTION_CORRECT_INVOICE) and _can(ACTION_ISSUE):
                _add(ACTION_CORRECT_INVOICE)
        else:
            _block(ACTION_CREDIT, "Un paiement est enregistré")
            _block(
                ACTION_CORRECT_INVOICE,
                "Correction impossible tant que des paiements sont enregistrés",
            )
        _block(ACTION_CANCEL, "Facture déjà envoyée")
    elif status == STATUS_PAID:
        primary = ACTION_VIEW
        _add(ACTION_VIEW_PAYMENTS)
        _add(ACTION_DOWNLOAD_PDF)
        _add(ACTION_REVERSE_PAYMENT)
        _block(
            ACTION_CREDIT,
            "Correction impossible tant que des paiements sont enregistrés",
        )
        _block(
            ACTION_CORRECT_INVOICE,
            "Correction impossible tant que des paiements sont enregistrés",
        )
    elif status == STATUS_CREDITED:
        primary = (
            ACTION_VIEW_CREDIT_NOTE if credit_note_id else ACTION_VIEW
        )
        if credit_note_id:
            _add(ACTION_VIEW_CREDIT_NOTE)
        _add(ACTION_DOWNLOAD_PDF)
    elif status == STATUS_CANCELLED:
        primary = ACTION_VIEW
    else:
        primary = ACTION_VIEW

    if inv is not None and ACTION_DOWNLOAD_PDF not in allowed:
        if status not in (STATUS_A_CALCULER, STATUS_A_CONTROLER, STATUS_PRETE_A_CLOTURER):
            _add(ACTION_DOWNLOAD_PDF)

    # Filtrer allowed par caps si fournis
    if caps:
        allowed = [a for a in allowed if _can(a)]
        if primary not in allowed and ACTION_VIEW in allowed:
            primary = ACTION_VIEW
        elif primary not in allowed:
            primary = allowed[0] if allowed else ACTION_VIEW

    return {
        "primary_action": primary,
        "allowed_actions": allowed,
        "blocked_actions": blocked,
    }
