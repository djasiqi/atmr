"""Cas d'usage pour le module Invoices."""

from .cancel_invoice import (
    CancelInvoiceInput,
    CancelInvoiceOutput,
    CancelInvoiceUseCase,
)
from .duplicate_invoice import DuplicateInvoiceUseCase
from .generate_consolidated_invoice import GenerateConsolidatedInvoiceUseCase
from .generate_invoice import (
    GenerateInvoiceInput,
    GenerateInvoiceOutput,
    GenerateInvoiceUseCase,
)
from .generate_invoice_pdf import GenerateInvoicePdfUseCase
from .generate_invoice_reminder import GenerateInvoiceReminderUseCase
from .get_invoice import GetInvoiceInput, GetInvoiceOutput, GetInvoiceUseCase
from .list_invoices import (
    ListInvoicesInput,
    ListInvoicesOutput,
    ListInvoicesUseCase,
)

__all__ = [
    "CancelInvoiceInput",
    "CancelInvoiceOutput",
    "CancelInvoiceUseCase",
    "DuplicateInvoiceUseCase",
    "GenerateConsolidatedInvoiceUseCase",
    "GenerateInvoiceInput",
    "GenerateInvoiceOutput",
    "GenerateInvoicePdfUseCase",
    "GenerateInvoiceReminderUseCase",
    "GenerateInvoiceUseCase",
    "GetInvoiceInput",
    "GetInvoiceOutput",
    "GetInvoiceUseCase",
    "ListInvoicesInput",
    "ListInvoicesOutput",
    "ListInvoicesUseCase",
]
