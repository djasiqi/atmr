"""Étape 5B (PDF) : tests unitaires du libellé d'annulation dans la colonne Transport.

On teste le builder (transport_display) sans rendu binaire du PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from services.documents.pdf import (
    _detect_and_group_round_trips,
    _get_cancellation_transport_display,
)


@dataclass
class _MockBooking:
    """Booking minimal pour tester transport_display."""

    id: int = 1
    status: Any = "CANCELED"
    pickup_location: str = "Rue A, 1000 Lausanne"
    dropoff_location: str = "Rue B, 1000 Lausanne"
    scheduled_time: datetime | None = None
    cancellation_display_label: str | None = None


class TestGetCancellationTransportDisplay:
    """Tests pour _get_cancellation_transport_display (libellé prioritaire si annulé)."""

    def test_no_show_contains_client_ne_sest_pas_presente(self) -> None:
        """NO_SHOW ⇒ transport_display contient 'Client ne s'est pas présenté'."""
        booking = _MockBooking(
            cancellation_display_label="Client ne s'est pas présenté"
        )
        result = _get_cancellation_transport_display(booking)
        assert "Client ne s'est pas présenté" in result
        assert result == "Client ne s'est pas présenté"

    def test_last_minute_contains_annulation_derniere_minute(self) -> None:
        """LAST_MINUTE ⇒ contient 'Annulation dernière minute'."""
        booking = _MockBooking(cancellation_display_label="Annulation dernière minute")
        result = _get_cancellation_transport_display(booking)
        assert "Annulation dernière minute" in result
        assert result == "Annulation dernière minute"

    def test_company_issue_contains_probleme_entreprise(self) -> None:
        """COMPANY_ISSUE ⇒ contient 'Problème entreprise' (même si non facturé)."""
        booking = _MockBooking(cancellation_display_label="Problème entreprise")
        result = _get_cancellation_transport_display(booking)
        assert "Problème entreprise" in result
        assert result == "Problème entreprise"

    def test_legacy_none_returns_historique(self) -> None:
        """Legacy (cancellation_display_label None) ⇒ 'Annulation (historique)'."""
        booking = _MockBooking(cancellation_display_label=None)
        result = _get_cancellation_transport_display(booking)
        assert result == "Annulation (historique)"

    def test_booking_none_returns_historique(self) -> None:
        """booking None ⇒ fallback 'Annulation (historique)'."""
        result = _get_cancellation_transport_display(None)
        assert result == "Annulation (historique)"


class TestDetectAndGroupRoundTripsCancelled:
    """Annulés : transport_display = libellé, jamais 'pickup → dropoff'."""

    def test_single_cancelled_booking_has_label_not_pickup_dropoff(self) -> None:
        """Un seul booking annulé ⇒ transport_display = libellé, pas 'A → B'."""
        mock_line = type("Line", (), {"type": None, "meta": {}})()
        mock_line.type = "RIDE"
        mock_line.meta = {"patient_name": "Dupont Jean"}
        booking = _MockBooking(
            cancellation_display_label="Client ne s'est pas présenté",
            pickup_location="HUG, Genève",
            dropoff_location="Home, Lausanne",
        )
        lines_with_bookings = [
            {
                "line": mock_line,
                "booking": booking,
                "patient_id": 10,
                "patient_name": "Dupont Jean",
                "date": datetime(2025, 2, 1, 10, 0, tzinfo=UTC),
                "pickup": "HUG, Genève",
                "dropoff": "Home, Lausanne",
                "amount": Decimal("50.00"),
            }
        ]
        consolidated = _detect_and_group_round_trips(lines_with_bookings)
        assert len(consolidated) == 1
        item = consolidated[0]
        transport = item.get("transport_display", "")
        assert "Client ne s'est pas présenté" in transport
        # Contrainte : ne doit jamais afficher pickup → dropoff pour annulé
        assert "HUG" not in transport or "→" not in transport
        assert "Home" not in transport or "→" not in transport

    def test_cancelled_company_issue_display_label_in_output(self) -> None:
        """Annulé COMPANY_ISSUE ⇒ transport_display contient 'Problème entreprise'."""
        mock_line = type("Line", (), {"type": None, "meta": {}})()
        mock_line.type = "RIDE"
        mock_line.meta = {"patient_name": "Martin"}
        booking = _MockBooking(
            cancellation_display_label="Problème entreprise",
            pickup_location="A",
            dropoff_location="B",
        )
        lines_with_bookings = [
            {
                "line": mock_line,
                "booking": booking,
                "patient_id": 1,
                "patient_name": "Martin",
                "date": datetime(2025, 2, 1, 9, 0, tzinfo=UTC),
                "pickup": "A",
                "dropoff": "B",
                "amount": Decimal("0.00"),
            }
        ]
        consolidated = _detect_and_group_round_trips(lines_with_bookings)
        assert len(consolidated) == 1
        assert "Problème entreprise" in consolidated[0].get("transport_display", "")
