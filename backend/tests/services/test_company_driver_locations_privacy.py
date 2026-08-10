"""Confidentialité : pas d'adresse / client_short dans les logs drivers/locations."""

from __future__ import annotations

from pathlib import Path


def test_busy_driver_log_format_omits_client_address():
    src = (
        Path(__file__).resolve().parents[2] / "services" / "company_driver_locations.py"
    ).read_text(encoding="utf-8")
    # Le log « Chauffeur en course » ne doit plus formater client=%s / adresse.
    assert (
        'Chauffeur en course: driver_id=%s booking_id=%s mission_status=%s"' in src
        or "Chauffeur en course: driver_id=%s booking_id=%s mission_status=%s" in src
    )
    busy_block = src.split("Chauffeur en course", 1)[1].split("locations.append", 1)[0]
    assert "client=%s" not in busy_block
    assert "client_short" not in busy_block
