"""✅ Schemas Marshmallow pour validation des endpoints de réservations."""

from datetime import datetime

from marshmallow import (  # pyright: ignore[reportMissingImports]
    Schema,
    ValidationError,
    fields,
    validate,
    validates_schema,
)

from schemas.validation_utils import ISO8601_DATE_REGEX, ISO8601_DATETIME_REGEX


class BookingCreateSchema(Schema):
    """Schema pour création de réservation
    (POST /api/bookings/clients/<id>/bookings)."""

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
        required=True,
        validate=validate.Range(min=0.5, error="Le montant minimum accepté est 0.5"),
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

    @validates_schema
    def validate_return_time_after_scheduled_time(self, data, **kwargs):  # noqa: ARG002
        """Valide que return_time > scheduled_time si is_round_trip=True."""
        is_round_trip = data.get("is_round_trip", False)
        return_time_str = data.get("return_time")
        scheduled_time_str = data.get("scheduled_time")

        # Si is_round_trip=True, return_time doit être fourni
        if is_round_trip and not return_time_str:
            raise ValidationError("return_time est requis lorsque is_round_trip=True")

        # Si return_time est fourni, il doit être > scheduled_time
        if return_time_str and scheduled_time_str:
            try:
                scheduled_time = datetime.fromisoformat(
                    scheduled_time_str.replace("Z", "+00:00")
                )
                return_time = datetime.fromisoformat(
                    return_time_str.replace("Z", "+00:00")
                )

                if return_time <= scheduled_time:
                    raise ValidationError(
                        "return_time doit être postérieur à scheduled_time"
                    )
            except (ValueError, AttributeError):
                # Si les dates ne peuvent pas être parsées, la validation du format
                # regex aura déjà échoué, donc on ne fait rien ici
                pass


class BookingUpdateSchema(Schema):
    """Schema pour mise à jour de réservation (PUT /api/bookings/<id>)."""

    pickup_location = fields.Str(validate=validate.Length(min=1, max=500))
    dropoff_location = fields.Str(validate=validate.Length(min=1, max=500))
    scheduled_time = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATETIME_REGEX, error="scheduled_time doit être au format ISO 8601"
        )
    )
    amount = fields.Float(
        validate=validate.Range(min=0.5, error="Le montant minimum accepté est 0.5")
    )
    status = fields.Str(
        validate=validate.OneOf(
            ["pending", "confirmed", "in_progress", "completed", "cancelled"]
        )
    )

    # Champs médicaux optionnels
    medical_facility = fields.Str(validate=validate.Length(max=200))
    doctor_name = fields.Str(validate=validate.Length(max=200))
    is_round_trip = fields.Bool()
    return_time = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATETIME_REGEX, error="return_time doit être au format ISO 8601"
        ),
        allow_none=True,
    )
    notes_medical = fields.Str(validate=validate.Length(max=1000))

    @validates_schema
    def validate_return_time_after_scheduled_time(self, data, **kwargs):  # noqa: ARG002
        """Valide que return_time > scheduled_time si is_round_trip=True."""
        is_round_trip = data.get("is_round_trip", False)
        return_time_str = data.get("return_time")
        scheduled_time_str = data.get("scheduled_time")

        # Si is_round_trip=True, return_time doit être fourni
        if is_round_trip and not return_time_str:
            raise ValidationError("return_time est requis lorsque is_round_trip=True")

        # Si return_time est fourni, il doit être > scheduled_time
        # (scheduled_time peut être dans les données existantes du booking)
        if return_time_str and scheduled_time_str:
            try:
                scheduled_time = datetime.fromisoformat(
                    scheduled_time_str.replace("Z", "+00:00")
                )
                return_time = datetime.fromisoformat(
                    return_time_str.replace("Z", "+00:00")
                )

                if return_time <= scheduled_time:
                    raise ValidationError(
                        "return_time doit être postérieur à scheduled_time"
                    )
            except (ValueError, AttributeError):
                # Si les dates ne peuvent pas être parsées, la validation du format
                # regex aura déjà échoué, donc on ne fait rien ici
                pass


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
