"""Conventions de rooms alignées backend (underscore)."""

from __future__ import annotations


def driver_room(driver_id: int) -> str:
    return f"driver_{driver_id}"


def company_room(company_id: int) -> str:
    return f"company_{company_id}"
