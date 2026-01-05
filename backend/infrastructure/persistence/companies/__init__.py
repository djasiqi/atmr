"""Adaptateurs persistence pour le module Companies.

Espace réservé : les opérations SQLAlchemy spécifiques (multi-tables, transactions) migreront ici.
"""

from .client_writer import SqlAlchemyClientWriter
from .driver_writer import SqlAlchemyDriverWriter
from .vehicle_writer import SqlAlchemyVehicleWriter

__all__ = [
    "SqlAlchemyClientWriter",
    "SqlAlchemyDriverWriter",
    "SqlAlchemyVehicleWriter",
]
