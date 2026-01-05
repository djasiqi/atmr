"""Schémas Marshmallow pour validation des routes OSRM.

Centralise la validation des paramètres de requête pour les endpoints OSRM.
"""

from marshmallow import (  # pyright: ignore[reportMissingImports]
    Schema,
    fields,
)

from schemas.validators import validate_latitude, validate_longitude


class OSRMRouteQuerySchema(Schema):
    """Schéma pour validation des paramètres de requête GET /osrm/route.

    Valide que tous les paramètres de coordonnées sont présents et valides.
    """

    pickup_lat = fields.Float(
        required=True,
        validate=validate_latitude,
        error_messages={
            "required": "pickup_lat est requis",
            "invalid": "pickup_lat doit être un nombre valide",
            "null": "pickup_lat ne peut pas être nul",
        },
    )
    pickup_lon = fields.Float(
        required=True,
        validate=validate_longitude,
        error_messages={
            "required": "pickup_lon est requis",
            "invalid": "pickup_lon doit être un nombre valide",
            "null": "pickup_lon ne peut pas être nul",
        },
    )
    dropoff_lat = fields.Float(
        required=True,
        validate=validate_latitude,
        error_messages={
            "required": "dropoff_lat est requis",
            "invalid": "dropoff_lat doit être un nombre valide",
            "null": "dropoff_lat ne peut pas être nul",
        },
    )
    dropoff_lon = fields.Float(
        required=True,
        validate=validate_longitude,
        error_messages={
            "required": "dropoff_lon est requis",
            "invalid": "dropoff_lon doit être un nombre valide",
            "null": "dropoff_lon ne peut pas être nul",
        },
    )

    class Meta:
        ordered = True
        unknown = "EXCLUDE"
