"""Seam de test autorisé pour le CreateBookingUseCase canonique.

Les tests hors ``test_booking_create_use_case.py`` ne doivent pas importer
directement ``application.bookings.create_booking.CreateBookingUseCase``.
Ce module est le seul point d'accès test dédié (allowlist architecture).
"""

from __future__ import annotations

from application.bookings.create_booking import CreateBookingUseCase

__all__ = ["CreateBookingUseCase"]
