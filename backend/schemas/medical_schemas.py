"""✅ Schemas Marshmallow pour validation des query params des endpoints medical.

Note: Les endpoints medical sont principalement GET avec query params.
Ces schemas sont prêts pour validation si nécessaire à l'avenir.
"""

from marshmallow import Schema, fields, validate


class MedicalEstablishmentQuerySchema(Schema):
    """Schema pour validation query params GET /api/medical/establishments."""

    q = fields.Str(
        validate=validate.Length(max=200, error="Query trop longue (max 200 caractères)"),
        allow_none=True,
        load_default=None,
    )
    limit = fields.Int(validate=validate.Range(min=1, max=25, error="limit doit être entre 1 et 25"), load_default=8)


class MedicalServiceQuerySchema(Schema):
    """Schema pour validation query params GET /api/medical/services."""

    establishment_id = fields.Int(required=True, validate=validate.Range(min=1, error="establishment_id doit être > 0"))
    q = fields.Str(
        validate=validate.Length(max=200, error="Query trop longue (max 200 caractères)"),
        allow_none=True,
        load_default=None,
    )
