"""Adaptateurs persistence pour le module Bookings."""

from infrastructure.persistence.bookings.booking_writer import (
    BookingWriterPort,
    SqlAlchemyBookingWriter,
)

# Exports publics (utilisés par le wiring / services).
__all__ = ["BookingWriterPort", "SqlAlchemyBookingWriter"]

# Marquer explicitement comme utilisés pour satisfaire les analyseurs statiques.
_EXPORTED = (BookingWriterPort, SqlAlchemyBookingWriter)
del _EXPORTED
# end
