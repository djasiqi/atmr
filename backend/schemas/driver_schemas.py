"""✅ Schemas Marshmallow pour validation des endpoints drivers."""

from marshmallow import Schema, fields, validate


class DriverProfileUpdateSchema(Schema):
    """Schema pour mise à jour profil chauffeur (PUT /api/driver/me/profile)."""

    # Champs utilisateur
    first_name = fields.Str(validate=validate.Length(min=1, max=100))
    last_name = fields.Str(validate=validate.Length(min=1, max=100))
    phone = fields.Str(validate=validate.Length(max=20))

    # Statut
    status = fields.Str(
        validate=validate.OneOf(
            ["disponible", "hors service"], error="status doit être: 'disponible' ou 'hors service'"
        )
    )

    # HR champs
    contract_type = fields.Str(validate=validate.Length(max=50))
    weekly_hours = fields.Int(validate=validate.Range(min=0, max=168, error="weekly_hours doit être entre 0 et 168"))
    hourly_rate_cents = fields.Int(validate=validate.Range(min=0, error="hourly_rate_cents doit être >= 0"))

    # Dates (format YYYY-MM-DD)
    # Note: fields.Date gère déjà le format ISO8601, pas besoin de Regexp
    employment_start_date = fields.Date(allow_none=True)
    employment_end_date = fields.Date(allow_none=True)
    license_valid_until = fields.Date(allow_none=True)
    medical_valid_until = fields.Date(allow_none=True)

    # Listes
    license_categories = fields.List(
        fields.Str(validate=validate.Length(max=10)),
        validate=validate.Length(max=10, error="Maximum 10 catégories de permis"),
    )
    trainings = fields.List(fields.Dict(), validate=validate.Length(max=50, error="Maximum 50 formations"))
