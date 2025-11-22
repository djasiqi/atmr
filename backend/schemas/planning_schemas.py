"""✅ Schemas Marshmallow pour validation des query params des endpoints planning."""

from marshmallow import Schema, fields, validate


class PlanningShiftsQuerySchema(Schema):
    """Schema pour validation query params GET /api/planning/companies/me/planning/shifts."""

    driver_id = fields.Int(
        validate=validate.Range(min=1, error="driver_id doit être > 0"),
        allow_none=True,
        load_default=None,
    )
    # TODO: from/to pour filtre par date (à implémenter plus tard)
    # from_date = fields.Str(...)
    # to_date = fields.Str(...)


class PlanningUnavailabilityQuerySchema(Schema):
    """Schema pour validation query params GET /api/planning/companies/me/planning/unavailability."""

    driver_id = fields.Int(
        validate=validate.Range(min=1, error="driver_id doit être > 0"),
        allow_none=True,
        load_default=None,
    )


class PlanningWeeklyTemplateQuerySchema(Schema):
    """Schema pour validation query params GET /api/planning/companies/me/planning/weekly-template."""

    driver_id = fields.Int(
        validate=validate.Range(min=1, error="driver_id doit être > 0"),
        allow_none=True,
        load_default=None,
    )
