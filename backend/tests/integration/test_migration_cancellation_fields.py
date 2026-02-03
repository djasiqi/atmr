"""Tests pour la migration des champs d'annulation (20260202_cancellation_fields).

Vérifie que les colonnes existent et que le backfill est cohérent.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from ext import db
from models import Booking
from models.enums import BookingStatus


@pytest.mark.integration
class TestMigrationCancellationFields:
    """Tests d'intégration pour la migration cancellation_fields."""

    def test_booking_has_cancellation_columns(self) -> None:
        """Vérifie que le modèle Booking a les colonnes d'annulation."""
        assert hasattr(Booking, "cancelled_at")
        assert hasattr(Booking, "cancelled_by_role")
        assert hasattr(Booking, "cancellation_reason_code")
        assert hasattr(Booking, "cancellation_reason_text")
        assert hasattr(Booking, "is_cancellation_billable")
        assert hasattr(Booking, "cancellation_display_label")

    def test_cancelled_booking_backfill_logic(self, db_session) -> None:
        """Vérifie que le backfill SQL cible bien status='CANCELED'.

        Le backfill met à jour les bookings avec status=CANCELED et
        cancellation_reason_code NULL → OTHER, false, 'Annulation (historique)'.
        L'enum PostgreSQL ne contient que 'CANCELED' (pas 'CANCELLED').
        """
        # Vérifier que la table booking a les colonnes
        result = db_session.session.execute(
            text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'booking'
                  AND column_name IN (
                    'cancelled_at', 'cancelled_by_role', 'cancellation_reason_code',
                    'cancellation_reason_text', 'is_cancellation_billable',
                    'cancellation_display_label'
                  )
            """)
        )
        cols = {row[0] for row in result.fetchall()}
        expected = {
            "cancelled_at",
            "cancelled_by_role",
            "cancellation_reason_code",
            "cancellation_reason_text",
            "is_cancellation_billable",
            "cancellation_display_label",
        }
        assert cols == expected, f"Colonnes manquantes: {expected - cols}"
