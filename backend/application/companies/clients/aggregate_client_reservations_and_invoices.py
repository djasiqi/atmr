from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


def _status_value(x: Any) -> str:
    if x is None:
        return ""
    v = getattr(x, "value", None)
    if isinstance(v, str):
        return v
    return str(x)


class _ClientRepo(Protocol):
    def find_model_by_id_with_user(
        self, client_id: int, company_id: int
    ) -> Any | None: ...


class _BookingRepo(Protocol):
    def find_models_by_client_and_company(
        self, client_id: int, company_id: int, limit: int | None = None
    ) -> list[Any]: ...


class _InvoiceRepo(Protocol):
    def find_by_client_id_and_company(
        self, client_id: int, company_id: int
    ) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class AggregateClientReservationsAndInvoicesResult:
    ok: bool
    payload: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class AggregateClientReservationsAndInvoicesUseCase:
    """Use-case Application: détails client + réservations + factures
    + total_pending_amount."""

    def __init__(
        self,
        *,
        client_repo: _ClientRepo,
        booking_repo: _BookingRepo,
        invoice_repo: _InvoiceRepo,
    ) -> None:
        super().__init__()
        self._client_repo = client_repo
        self._booking_repo = booking_repo
        self._invoice_repo = invoice_repo

    def execute(
        self,
        *,
        company_id: int,
        client_id: int,
        limit: int | None = None,
        include_invoices: bool = True,
    ) -> AggregateClientReservationsAndInvoicesResult:
        client = self._client_repo.find_model_by_id_with_user(client_id, company_id)
        if not client:
            return AggregateClientReservationsAndInvoicesResult(
                ok=False,
                error={"error": "Client not found"},
                status_code=404,
            )

        user = getattr(client, "user", None)
        client_info = {
            "id": getattr(client, "id", None),
            "first_name": getattr(user, "first_name", "") if user else "",
            "last_name": getattr(user, "last_name", "") if user else "",
            "email": getattr(user, "email", "") if user else "",
            "phone": getattr(user, "phone", "") if user else "",
            "is_active": getattr(client, "is_active", None),
            "created_at": getattr(
                getattr(client, "created_at", None), "isoformat", lambda: None
            )(),
        }

        bookings = self._booking_repo.find_models_by_client_and_company(
            int(client.id),
            company_id,
            limit=limit,
        )
        invoices = (
            self._invoice_repo.find_by_client_id_and_company(client_id, company_id)
            if include_invoices
            else []
        )

        invoice_list: list[dict[str, Any]] = []
        for inv in invoices:
            ser = getattr(inv, "serialize", None)
            invoice_list.append(
                ser if isinstance(ser, dict) else {"id": getattr(inv, "id", None)}
            )

        total_pending_amount = 0.0
        enriched_bookings: list[dict[str, Any]] = []
        for booking in bookings:
            booking_ser = getattr(booking, "serialize", None)
            booking_data = (
                booking_ser
                if isinstance(booking_ser, dict)
                else {"id": getattr(booking, "id", None)}
            )

            invoice = getattr(booking, "invoice", None) if include_invoices else None
            amount = getattr(booking, "amount", 0) or 0
            try:
                amount_f = float(amount)
            except Exception:
                amount_f = 0.0

            if include_invoices and invoice:
                inv_ser = getattr(invoice, "serialize", None)
                booking_data["invoice"] = (
                    inv_ser
                    if isinstance(inv_ser, dict)
                    else {"id": getattr(invoice, "id", None)}
                )
                if _status_value(getattr(invoice, "status", None)).lower() != "paid":
                    total_pending_amount += amount_f
            else:
                total_pending_amount += amount_f

            enriched_bookings.append(booking_data)

        return AggregateClientReservationsAndInvoicesResult(
            ok=True,
            payload={
                "client": client_info,
                "reservations": enriched_bookings,
                "invoices": invoice_list,
                "total_pending_amount": round(total_pending_amount, 2),
            },
        )
