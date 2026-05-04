from marshmallow import Schema, fields
from marshmallow.validate import Length, OneOf

from schemas.validation_utils import EMAIL_VALIDATOR, PHONE_VALIDATOR

ORGANIZATION_TYPES = (
    "transport_company",
    "institution",
    "ems",
    "clinic",
    "hospital",
    "curatorship",
    "other",
)

USE_CASES = (
    "planning_dispatch",
    "billing",
    "transport_tracking",
    "multi_company_coordination",
    "reporting",
    "si_integration",
    "other",
)

VOLUME_RANGES = ("1_5", "5_20", "20_100", "100_plus")
INTEGRATION_REQUIRED_VALUES = ("yes", "no", "evaluate")
TIMINGS = ("immediate", "one_three_months", "three_plus_months", "exploration")
PREFERRED_SLOTS = ("this_week", "next_week", "to_define")
PREFERRED_PERIODS = ("morning", "afternoon", "flexible")


class DemoRequestSchema(Schema):
    name = fields.Str(required=True, validate=Length(min=2, max=120))
    email = fields.Email(required=True, validate=EMAIL_VALIDATOR)
    phone = fields.Str(required=False, allow_none=True, validate=PHONE_VALIDATOR)
    organization = fields.Str(required=True, validate=Length(min=2, max=180))
    organization_type = fields.Str(required=True, validate=OneOf(ORGANIZATION_TYPES))
    use_case = fields.Str(required=True, validate=OneOf(USE_CASES))
    volume_range = fields.Str(
        required=False, allow_none=True, validate=OneOf(VOLUME_RANGES)
    )
    integration_required = fields.Str(
        required=True, validate=OneOf(INTEGRATION_REQUIRED_VALUES)
    )
    integration_system = fields.Str(
        required=False, allow_none=True, validate=Length(max=180)
    )
    timing = fields.Str(required=True, validate=OneOf(TIMINGS))
    preferred_slot = fields.Str(required=True, validate=OneOf(PREFERRED_SLOTS))
    preferred_period = fields.Str(required=True, validate=OneOf(PREFERRED_PERIODS))
    comment = fields.Str(required=False, allow_none=True, validate=Length(max=3000))

    privacy_consent = fields.Bool(required=True)
    honeypot = fields.Str(required=False, allow_none=True, validate=Length(max=256))
    form_started_at_ms = fields.Integer(required=False, allow_none=True)
    acknowledgement_already_sent = fields.Bool(required=False, allow_none=True)
    source = fields.Str(required=False, allow_none=True, validate=Length(max=64))
