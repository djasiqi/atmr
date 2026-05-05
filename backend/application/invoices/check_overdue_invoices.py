"""Use-case: marquer les factures échues comme en retard (OVERDUE)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy.exc import DBAPIError, OperationalError

from ext import db
from models import Invoice
from models.enums import InvoiceStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CheckOverdueInvoicesInput:
    """Entrée pour la vérification des factures en retard.

    Attributes:
        company_id: Si renseigné, limite aux factures de cette entreprise.
    """

    company_id: int | None = None


@dataclass(frozen=True, slots=True)
class CheckOverdueInvoicesOutput:
    """Résultat de la vérification des factures en retard."""

    success: bool
    updated_count: int = 0
    error: dict[str, str] | None = None
    status_code: int | None = None


class _InvoiceLineRepoPort(Protocol):
    def create(self, line_data: dict[str, Any]) -> Any: ...


class CheckOverdueInvoicesUseCase:
    """Passe en statut OVERDUE les factures émises/partiellement payées encore dues après l'échéance."""

    def __init__(
        self,
        invoice_line_repo: _InvoiceLineRepoPort | None = None,
    ) -> None:
        """invoice_line_repo est réservé à l'injection de tests / extensions (frais de retard)."""
        self._invoice_line_repo = invoice_line_repo

    def execute(
        self, input_data: CheckOverdueInvoicesInput
    ) -> CheckOverdueInvoicesOutput:
        now = datetime.now(UTC)
        try:
            query = Invoice.query.filter(
                Invoice.balance_due > 0,
                Invoice.due_date.isnot(None),
                Invoice.due_date < now,
                Invoice.status.in_(
                    (InvoiceStatus.SENT, InvoiceStatus.PARTIALLY_PAID),
                ),
            )
            if input_data.company_id is not None:
                query = query.filter(Invoice.company_id == input_data.company_id)

            overdue_invoices = query.all()
            updated = 0
            for inv in overdue_invoices:
                if not inv.is_overdue:
                    continue
                inv.status = InvoiceStatus.OVERDUE
                inv.updated_at = now
                updated += 1

            db.session.commit()
            return CheckOverdueInvoicesOutput(success=True, updated_count=updated)
        except (OperationalError, DBAPIError) as exc:
            db.session.rollback()
            logger.exception("Échec CheckOverdueInvoicesUseCase")
            return CheckOverdueInvoicesOutput(
                success=False,
                updated_count=0,
                error={"message": str(exc)},
                status_code=500,
            )
