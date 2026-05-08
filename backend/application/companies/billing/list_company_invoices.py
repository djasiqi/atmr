from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class _InvoiceRepo(Protocol):
    def find_by_company_id_with_lines(self, company_id: int) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class ListCompanyInvoicesResult:
    payload: dict[str, Any]


class ListCompanyInvoicesUseCase:
    """Use-case Application: lister les factures d'une company."""

    def __init__(self, *, invoice_repo: _InvoiceRepo) -> None:
        super().__init__()
        self._invoice_repo = invoice_repo

    def execute(self, *, company_id: int) -> ListCompanyInvoicesResult:
        invoices = self._invoice_repo.find_by_company_id_with_lines(company_id)
        serialized: list[dict[str, Any]] = []
        for inv in invoices:
            to_dict = getattr(inv, "to_dict", None)
            if callable(to_dict):
                serialized.append(to_dict(list_view=True))
            elif hasattr(inv, "serialize"):
                ser = inv.serialize
                if isinstance(ser, dict):
                    serialized.append(ser)
                else:
                    serialized.append({"id": getattr(inv, "id", None)})
            else:
                serialized.append({"id": getattr(inv, "id", None)})
        return ListCompanyInvoicesResult(
            payload={"invoices": serialized, "total": len(invoices)}
        )
