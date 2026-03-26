"""Contrats DTO liste/détail LIRIE (référence Annexe E / plan perf).

Les endpoints peuvent utiliser Marshmallow / sérialisation manuelle alignée sur ces champs.
"""

from __future__ import annotations

from typing import Any, TypedDict


class BookingListDTO(TypedDict):
    id: int
    status: str
    scheduled_time: str | None
    pickup_summary: str | None
    dropoff_summary: str | None
    customer_display_name: str | None
    driver_id: int | None
    is_urgent: bool
    assignment: dict[str, Any] | None
    parent_booking_id: int | None
    is_return: bool


class BookingDetailDTO(BookingListDTO, total=False):
    customer: dict[str, Any]
    notes: str
    timeline: list[Any]
    payments: list[Any]


class InvoiceListDTO(TypedDict):
    id: int
    invoice_kind: str
    invoice_number: str | None
    status: str
    issued_at: str | None
    due_date: str | None
    total_amount: float | str
    balance_due: float | str
    client_label: str
    currency: str


class InvoiceDetailDTO(TypedDict, total=False):
    id: int
    invoice_kind: str
    invoice_number: str | None
    status: str
    issued_at: str | None
    due_date: str | None
    total_amount: float | str
    balance_due: float | str
    client_label: str
    currency: str
    lines: list[Any]
    payments: list[Any]
    client: dict[str, Any]
    raw_metadata: dict[str, Any]


class DriverMapDTO(TypedDict):
    driver_id: int
    first_name: str
    last_name: str
    latitude: float | None
    longitude: float | None
    status: str
    mission_status: str | None
    updated_at: str | None
    current_booking_id: int | None
