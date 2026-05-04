"""✅ Schemas Marshmallow pour TransportVoucher (P3)."""

from marshmallow import (  # pyright: ignore[reportMissingImports]
    Schema,
    fields,
    validate,
)


class TransportVoucherCreateSchema(Schema):
    """Schema pour création d'un bon de transport (POST /transport-vouchers)."""

    company_id = fields.Int(required=True, validate=validate.Range(min=1))
    client_id = fields.Int(required=True, validate=validate.Range(min=1))
    booking_id = fields.Int(allow_none=True, validate=validate.Range(min=1))
    billing_party_id = fields.Int(allow_none=True, validate=validate.Range(min=1))
    type = fields.Str(
        required=True,
        validate=validate.OneOf(["clinic", "insurance", "other"]),
    )
    status = fields.Str(
        allow_none=True,
        validate=validate.OneOf(
            ["draft", "submitted", "validated", "rejected", "expired"]
        ),
    )
    valid_from = fields.DateTime(allow_none=True)
    valid_to = fields.DateTime(allow_none=True)
    external_ref = fields.Str(allow_none=True, validate=validate.Length(max=255))
    notes = fields.Str(allow_none=True)


class TransportVoucherUpdateSchema(Schema):
    """Schema pour modification d'un bon (PATCH /transport-vouchers/{id})."""

    booking_id = fields.Int(allow_none=True, validate=validate.Range(min=1))
    billing_party_id = fields.Int(allow_none=True, validate=validate.Range(min=1))
    type = fields.Str(
        allow_none=True,
        validate=validate.OneOf(["clinic", "insurance", "other"]),
    )
    status = fields.Str(
        allow_none=True,
        validate=validate.OneOf(
            ["draft", "submitted", "validated", "rejected", "expired"]
        ),
    )
    valid_from = fields.DateTime(allow_none=True)
    valid_to = fields.DateTime(allow_none=True)
    external_ref = fields.Str(allow_none=True, validate=validate.Length(max=255))
    notes = fields.Str(allow_none=True)


class TransportVoucherValidateSchema(Schema):
    """Schema pour validation d'un bon (POST /transport-vouchers/{id}/validate)."""

    billing_party_id = fields.Int(allow_none=True, validate=validate.Range(min=1))
    notes = fields.Str(allow_none=True)


class TransportVoucherRejectSchema(Schema):
    """Schema pour rejet d'un bon (POST /transport-vouchers/{id}/reject)."""

    reason = fields.Str(required=True, validate=validate.Length(min=1, max=500))
    notes = fields.Str(allow_none=True)
