"""✅ Schemas Marshmallow pour validation des endpoints de réservations."""

from marshmallow import Schema, fields, validate

from schemas.validation_utils import ISO8601_DATE_REGEX, ISO8601_DATETIME_REGEX


class BookingCreateSchema(Schema):
    """Schema pour création de réservation (POST /api/bookings/clients/<id>/bookings)."""

    customer_name = fields.Str(required=True, validate=validate.Length(min=1, max=200))
    pickup_location = fields.Str(
        required=True, validate=validate.Length(min=1, max=500)
    )
    dropoff_location = fields.Str(
        required=True, validate=validate.Length(min=1, max=500)
    )
    scheduled_time = fields.Str(
        required=True,
        validate=validate.Regexp(
            ISO8601_DATETIME_REGEX, error="scheduled_time doit être au format ISO 8601"
        ),
    )
    amount = fields.Float(
        required=True, validate=validate.Range(min=0, error="amount doit être >= 0")
    )

    # Champs optionnels
    medical_facility = fields.Str(load_default="", validate=validate.Length(max=200))
    doctor_name = fields.Str(load_default="", validate=validate.Length(max=200))
    is_round_trip = fields.Bool(load_default=False)
    return_time = fields.Str(
        load_default=None,
        validate=validate.Regexp(
            ISO8601_DATETIME_REGEX, error="return_time doit être au format ISO 8601"
        ),
        allow_none=True,
    )


class BookingUpdateSchema(Schema):
    """Schema pour mise à jour de réservation (PUT /api/bookings/<id>)."""

    pickup_location = fields.Str(validate=validate.Length(min=1, max=500))
    dropoff_location = fields.Str(validate=validate.Length(min=1, max=500))
    scheduled_time = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATETIME_REGEX, error="scheduled_time doit être au format ISO 8601"
        )
    )
    amount = fields.Float(validate=validate.Range(min=0))
    status = fields.Str(
        validate=validate.OneOf(
            ["pending", "confirmed", "in_progress", "completed", "cancelled"]
        )
    )

    # Champs médicaux optionnels
    medical_facility = fields.Str(validate=validate.Length(max=200))
    doctor_name = fields.Str(validate=validate.Length(max=200))
    is_round_trip = fields.Bool()
    notes_medical = fields.Str(validate=validate.Length(max=1000))


class BookingListSchema(Schema):
    """Schema pour paramètres de liste de réservations (GET /api/bookings)."""

    page = fields.Int(
        load_default=1, validate=validate.Range(min=1, error="page doit être >= 1")
    )
    per_page = fields.Int(
        load_default=100,
        validate=validate.Range(
            min=1, max=500, error="per_page doit être entre 1 et 500"
        ),
    )
    status = fields.Str(
        load_default=None,
        validate=validate.OneOf(
            ["pending", "confirmed", "in_progress", "completed", "cancelled"]
        ),
        allow_none=True,
    )
    from_date = fields.Str(
        load_default=None,
        validate=validate.Regexp(
            ISO8601_DATE_REGEX, error="from_date doit être au format YYYY-MM-DD"
        ),
        allow_none=True,
    )
    to_date = fields.Str(
        load_default=None,
        validate=validate.Regexp(
            ISO8601_DATE_REGEX, error="to_date doit être au format YYYY-MM-DD"
        ),
        allow_none=True,
    )
