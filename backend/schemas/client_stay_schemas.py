"""✅ Schemas Marshmallow pour ClientStay (P2)."""

from marshmallow import (  # pyright: ignore[reportMissingImports]
    Schema,
    fields,
    validate,
)


class ClientStayCreateSchema(Schema):
    """Schema pour création d'un séjour client (POST /clients/{id}/stays)."""

    company_id = fields.Int(required=True, validate=validate.Range(min=1))
    start_date = fields.Str(required=True)  # ISO datetime
    end_date = fields.Str(allow_none=True)  # ISO datetime
    status = fields.Str(
        allow_none=True,
        validate=validate.OneOf(["active", "closed", "cancelled"]),
    )
    source = fields.Str(allow_none=True, validate=validate.Length(max=50))
    notes = fields.Str(allow_none=True)


class ClientStayUpdateSchema(Schema):
    """Schema pour modification d'un séjour (PATCH /client-stays/{id})."""

    company_id = fields.Int(allow_none=True, validate=validate.Range(min=1))
    start_date = fields.Str(allow_none=True)
    end_date = fields.Str(allow_none=True)
    status = fields.Str(
        allow_none=True,
        validate=validate.OneOf(["active", "closed", "cancelled"]),
    )
    source = fields.Str(allow_none=True, validate=validate.Length(max=50))
    notes = fields.Str(allow_none=True)


class ClientStayCloseSchema(Schema):
    """Schema pour clôture (POST /client-stays/{id}/close)."""

    end_date = fields.Str(allow_none=True)
