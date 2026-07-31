"""Façade de compatibilité.

La création client canonique appartient à
`application.bookings.create_booking`.
Aucune règle métier ne doit être ajoutée ici.
"""

from application.bookings.create_booking import CreateBookingUseCase

__all__ = ["CreateBookingUseCase"]
