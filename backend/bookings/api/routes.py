"""Routes API pour le Bounded Context Bookings.

Migration depuis routes/bookings.py vers la structure DDD.
"""

from __future__ import annotations

import logging

from flask_restx import Namespace

# Pour l'instant, on garde les routes dans routes/bookings.py
# et on les migrera progressivement ici

logger = logging.getLogger(__name__)

# Création du Namespace pour les réservations
bookings_ns = Namespace("bookings", description="Opérations relatives aux réservations")

# TODO: Migrer les routes depuis routes/bookings.py
# - GET /bookings/<id>
# - GET /bookings/
# - POST /bookings/clients/<public_id>/bookings
# - PUT /bookings/<id>
# - DELETE /bookings/<id>

__all__ = ["bookings_ns"]
