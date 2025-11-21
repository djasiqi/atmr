"""✅ Schemas Marshmallow pour validation des endpoints secret rotation."""

from marshmallow import fields, validate

from schemas.query_schemas import LimitOffsetQuerySchema


class SecretRotationMonitoringQuerySchema(LimitOffsetQuerySchema):
    """Schema pour validation query params GET /api/secret-rotation/monitoring."""

    secret_type = fields.Str(
        load_default=None,
        validate=validate.Length(max=50, error="secret_type doit faire max 50 caractères"),
        allow_none=True,
    )
    status = fields.Str(
        load_default=None,
        validate=validate.Length(max=50, error="status doit faire max 50 caractères"),
        allow_none=True,
    )
    environment = fields.Str(
        load_default=None,
        validate=validate.Length(max=50, error="environment doit faire max 50 caractères"),
        allow_none=True,
    )

    class Meta:  # type: ignore
        unknown = "INCLUDE"  # Permettre des champs supplémentaires pour compatibilité
