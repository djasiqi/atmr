"""✅ Schemas Marshmallow pour les liens Client ↔ BillingParty."""

from marshmallow import (  # pyright: ignore[reportMissingImports]
    Schema,
    fields,
    validate,
)


class ClientBillingPartyLinkCreateSchema(Schema):
    """Schema pour création d'un lien (POST /clients/{id}/billing-parties)."""

    billing_party_id = fields.Int(required=True, validate=validate.Range(min=1))
    role = fields.Str(allow_none=True, validate=validate.Length(max=50))
    is_default = fields.Bool(allow_none=True)
    contact_name = fields.Str(allow_none=True, validate=validate.Length(max=120))
    contact_email = fields.Str(allow_none=True, validate=validate.Length(max=255))
    contact_phone = fields.Str(allow_none=True, validate=validate.Length(max=50))


class ClientBillingPartyLinkUpdateSchema(Schema):
    """Schema pour modification d'un lien (PATCH /clients/billing-party-links/{id})."""

    role = fields.Str(allow_none=True, validate=validate.Length(max=50))
    is_default = fields.Bool(allow_none=True)
    contact_name = fields.Str(allow_none=True, validate=validate.Length(max=120))
    contact_email = fields.Str(allow_none=True, validate=validate.Length(max=255))
    contact_phone = fields.Str(allow_none=True, validate=validate.Length(max=50))
