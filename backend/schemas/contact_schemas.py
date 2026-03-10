from marshmallow import RAISE, Schema, ValidationError, fields, validates_schema
from marshmallow.validate import Length, OneOf

from schemas.validation_utils import EMAIL_VALIDATOR, PHONE_VALIDATOR

CONTACT_CATEGORIES = ("support", "institution", "transport", "demo", "billing", "family")
# Compatibilite legacy imports.
CONTACT_SUBJECTS = CONTACT_CATEGORIES


class BaseContactSchema(Schema):
    class Meta:
        unknown = RAISE

    category = fields.Str(required=True, validate=OneOf(CONTACT_CATEGORIES))
    name = fields.Str(required=True, validate=Length(min=2, max=120))
    email = fields.Email(required=True, validate=EMAIL_VALIDATOR)
    phone = fields.Str(required=False, allow_none=True, validate=PHONE_VALIDATOR)
    organization = fields.Str(required=False, allow_none=True, validate=Length(max=180))
    company = fields.Str(required=False, allow_none=True, validate=Length(max=180))
    message = fields.Str(required=True, validate=Length(min=5, max=4000))
    privacy_consent = fields.Bool(required=True)
    website = fields.Str(required=False, allow_none=True, validate=Length(max=256))
    client_request_id = fields.Str(required=False, allow_none=True, validate=Length(max=64))
    subject_detail = fields.Str(required=False, allow_none=True, validate=Length(max=64))
    reference = fields.Str(required=False, allow_none=True, validate=Length(max=120))
    urgency = fields.Str(
        required=False,
        allow_none=True,
        validate=OneOf(("normal", "priority")),
    )
    organization_type = fields.Str(required=False, allow_none=True, validate=Length(max=64))
    sites_count = fields.Str(required=False, allow_none=True, validate=Length(max=32))
    integration_required = fields.Str(
        required=False,
        allow_none=True,
        validate=OneOf(("yes", "no", "evaluate")),
    )
    integration_system = fields.Str(required=False, allow_none=True, validate=Length(max=120))
    fleet_size_range = fields.Str(required=False, allow_none=True, validate=Length(max=64))
    service_area = fields.Str(required=False, allow_none=True, validate=Length(max=160))
    timing = fields.Str(required=False, allow_none=True, validate=Length(max=64))
    preferred_slot = fields.Str(required=False, allow_none=True, validate=Length(max=64))
    volume_range = fields.Str(required=False, allow_none=True, validate=Length(max=64))
    situation = fields.Str(required=False, allow_none=True, validate=Length(max=220))

    @validates_schema
    def validate_privacy_consent(self, data, **_kwargs):
        if not data.get("privacy_consent"):
            raise ValidationError(
                {"privacy_consent": ["Le consentement est requis pour envoyer la demande."]}
            )


class SupportContactSchema(BaseContactSchema):
    pass


class InstitutionContactSchema(BaseContactSchema):
    @validates_schema
    def validate_required(self, data, **_kwargs):
        errors = {}
        if not (data.get("organization") or "").strip():
            errors["organization"] = ["Organisation requise."]
        if not data.get("organization_type"):
            errors["organization_type"] = ["Type d'organisation requis."]
        if not data.get("integration_required"):
            errors["integration_required"] = ["Indiquer si une integration est requise."]
        if errors:
            raise ValidationError(errors)


class TransportContactSchema(BaseContactSchema):
    @validates_schema
    def validate_required(self, data, **_kwargs):
        if not (data.get("organization") or "").strip():
            raise ValidationError({"organization": ["Organisation requise."]})


class DemoContactSchema(BaseContactSchema):
    @validates_schema
    def validate_required(self, data, **_kwargs):
        errors = {}
        if not (data.get("organization") or "").strip():
            errors["organization"] = ["Organisation requise."]
        if not data.get("organization_type"):
            errors["organization_type"] = ["Type d'organisation requis."]
        if not data.get("timing"):
            errors["timing"] = ["Timing requis."]
        if not data.get("preferred_slot"):
            errors["preferred_slot"] = ["Creneau souhaite requis."]
        if errors:
            raise ValidationError(errors)


class BillingContactSchema(BaseContactSchema):
    pass


class FamilyContactSchema(BaseContactSchema):
    @validates_schema
    def validate_required(self, data, **_kwargs):
        if (data.get("organization") or "").strip():
            raise ValidationError({"organization": ["Ne pas renseigner ce champ pour cette categorie."]})


def schema_for_category(category: str) -> Schema:
    mapping = {
        "support": SupportContactSchema,
        "institution": InstitutionContactSchema,
        "transport": TransportContactSchema,
        "demo": DemoContactSchema,
        "billing": BillingContactSchema,
        "family": FamilyContactSchema,
    }
    schema_cls = mapping.get(category)
    if not schema_cls:
        raise ValidationError({"category": ["Categorie invalide."]})
    return schema_cls()


class ContactRequestBaseOnlySchema(Schema):
    class Meta:
        unknown = RAISE

    category = fields.Str(required=True, validate=OneOf(CONTACT_CATEGORIES))
    name = fields.Str(required=True, validate=Length(min=2, max=120))
    email = fields.Email(required=True, validate=EMAIL_VALIDATOR)
    message = fields.Str(required=True, validate=Length(min=5, max=4000))
    phone = fields.Str(required=False, allow_none=True, validate=PHONE_VALIDATOR)
    privacy_consent = fields.Bool(required=True)
    website = fields.Str(required=False, allow_none=True, validate=Length(max=256))
    client_request_id = fields.Str(required=False, allow_none=True, validate=Length(max=64))


class ContactRequestSchema(BaseContactSchema):
    """Alias legacy pour compatibilite tests/imports existants."""

    pass
