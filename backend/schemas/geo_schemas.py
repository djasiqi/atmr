"""Schémas Marshmallow pour validation des coordonnées géographiques.

✅ Refactoring: Centralise la validation des coordonnées pour éviter la duplication.
Utilise les validateurs existants de schemas/validators.py qui utilisent GeoValidator.
"""

from marshmallow import (  # pyright: ignore[reportMissingImports]
    Schema,
    fields,
)

from schemas.validators import validate_latitude, validate_longitude
from shared.geo_utils import GeoValidator


class CoordinatesSchema(Schema):
    """Schéma pour validation des coordonnées géographiques.

    Valide que lat est entre -90 et 90, et lon entre -180 et 180.
    Utilise les validateurs centralisés de schemas/validators.py.
    """

    lat = fields.Float(
        required=True,
        validate=validate_latitude,
        error_messages={
            "required": "La latitude est requise",
            "invalid": "La latitude doit être un nombre",
            "null": "La latitude ne peut pas être nulle",
        },
    )
    lon = fields.Float(
        required=True,
        validate=validate_longitude,
        error_messages={
            "required": "La longitude est requise",
            "invalid": "La longitude doit être un nombre",
            "null": "La longitude ne peut pas être nulle",
        },
    )


class OptionalCoordinatesSchema(Schema):
    """Schéma pour validation optionnelle des coordonnées géographiques.

    Permet que lat et lon soient None, mais si présents, ils doivent être valides.
    """

    lat = fields.Float(
        required=False,
        allow_none=True,
        validate=validate_latitude,
        error_messages={
            "invalid": "La latitude doit être un nombre",
        },
    )
    lon = fields.Float(
        required=False,
        allow_none=True,
        validate=validate_longitude,
        error_messages={
            "invalid": "La longitude doit être un nombre",
        },
    )


# Constantes pour utilisation dans d'autres schémas
LATITUDE_VALIDATOR = validate_latitude
LONGITUDE_VALIDATOR = validate_longitude
LATITUDE_RANGE = {"min": GeoValidator.LAT_MIN, "max": GeoValidator.LAT_MAX}
LONGITUDE_RANGE = {"min": GeoValidator.LON_MIN, "max": GeoValidator.LON_MAX}
