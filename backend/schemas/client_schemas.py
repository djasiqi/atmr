"""✅ Schemas Marshmallow pour validation des endpoints clients."""

from marshmallow import Schema, fields, validate

from schemas.validation_utils import ISO8601_DATE_REGEX


class ClientUpdateSchema(Schema):
    """Schema pour mise à jour profil client (PUT /api/clients/<id>)."""

    first_name = fields.Str(validate=validate.Length(min=1, max=100))
    last_name = fields.Str(validate=validate.Length(min=1, max=100))
    phone = fields.Str(
        validate=validate.Regexp(
            r"^\+?[0-9]{7,15}$",
            error="Numéro de téléphone invalide (format: +33123456789 ou 0123456789)",
        )
    )
    address = fields.Str(validate=validate.Length(min=1, max=500))
    birth_date = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATE_REGEX, error="birth_date doit être au format YYYY-MM-DD"
        ),
        allow_none=True,
    )
    gender = fields.Str(
        validate=validate.OneOf(
            ["HOMME", "FEMME", "AUTRE"], error="gender doit être: HOMME, FEMME ou AUTRE"
        )
    )
