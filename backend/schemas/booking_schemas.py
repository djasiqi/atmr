"""✅ Schemas Marshmallow pour validation des endpoints de réservations."""

from datetime import UTC, date, datetime, timedelta

from marshmallow import (
    Schema,
    ValidationError,
    fields,
    pre_load,
    validate,
    validates_schema,
)

from schemas.validation_utils import ISO8601_DATE_REGEX, ISO8601_DATETIME_REGEX
from shared.client_portal_notes import CLIENT_PORTAL_FREE_NOTE_MAX_LENGTH
from shared.constants import ErrorCodes
from shared.time_utils import api_scheduled_iso_to_naive_geneva

DELIVERY_DESCRIPTION_MAX_LENGTH = 500


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
    # Portail client : « Dès que possible » → asap=true et scheduled_time absent / null
    asap = fields.Bool(load_default=False)
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
    hospital_service = fields.Str(load_default="", validate=validate.Length(max=100))
    is_round_trip = fields.Bool(load_default=False)
    return_time = fields.Str(
        load_default=None,
        validate=validate.Regexp(
            ISO8601_DATETIME_REGEX, error="return_time doit être au format ISO 8601"
        ),
        allow_none=True,
    )
    return_date = fields.Str(
        load_default=None,
        allow_none=True,
        validate=validate.Regexp(
            ISO8601_DATE_REGEX,
            error="return_date doit être au format YYYY-MM-DD",
        ),
    )
    occurrences = fields.Int(
        load_default=1,
        validate=validate.Range(
            min=1, max=20, error="occurrences doit être entre 1 et 20"
        ),
    )
    client_note = fields.Str(
        load_default="",
        validate=validate.Length(max=CLIENT_PORTAL_FREE_NOTE_MAX_LENGTH),
    )
    is_recurring = fields.Bool(load_default=False)
    recurrence_type = fields.Str(
        load_default=None,
        allow_none=True,
        validate=validate.OneOf(["daily", "weekly", "custom"]),
    )
    recurrence_days = fields.List(
        fields.Int(validate=validate.Range(min=0, max=6)),
        load_default=None,
        allow_none=True,
    )
    recurrence_end_date = fields.Str(
        load_default=None,
        allow_none=True,
        validate=validate.Regexp(
            ISO8601_DATE_REGEX,
            error="recurrence_end_date doit être au format YYYY-MM-DD",
        ),
    )
    recurrence_series_length = fields.Int(
        load_default=None,
        allow_none=True,
        validate=validate.Range(min=1, max=52),
    )

    @pre_load
    def _default_scheduled_time_for_asap(self, data, **kwargs):  # noqa: ARG002
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        asap_val = raw.get("asap")
        asap = asap_val is True or (
            isinstance(asap_val, str)
            and asap_val.strip().lower() in ("true", "1", "yes")
        )
        st = raw.get("scheduled_time")
        if asap and (st is None or (isinstance(st, str) and not st.strip())):
            # Marge pour éviter « dans le passé » (latence réseau / parsing fuseau)
            soon = (datetime.now(UTC) + timedelta(minutes=5)).replace(microsecond=0)
            raw["scheduled_time"] = soon.isoformat().replace("+00:00", "Z")
        red = raw.get("recurrence_end_date")
        if isinstance(red, str) and not red.strip():
            raw["recurrence_end_date"] = None
        rd = raw.get("return_date")
        if isinstance(rd, str) and not rd.strip():
            raw["return_date"] = None
        return raw

    @validates_schema
    def validate_return_time_after_scheduled_time(self, data, **kwargs):  # noqa: ARG002
        """Si return_time est fourni, il doit être postérieur à scheduled_time (instant)."""
        return_time_str = data.get("return_time")
        scheduled_time_str = data.get("scheduled_time")

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

    @validates_schema
    def validate_round_trip_return_plan(self, data, **kwargs):  # noqa: ARG002
        """Aller-retour : au moins ``return_date`` ou ``return_time`` (date seule côté portail)."""
        if not data.get("is_round_trip"):
            return
        rt = data.get("return_time")
        rd_raw = data.get("return_date")
        if rd_raw is None:
            rd = None
        elif isinstance(rd_raw, str):
            rd = rd_raw.strip() or None
        else:
            rd = None
        if not rt and not rd:
            raise ValidationError(
                "Pour un aller-retour, indiquez return_date (YYYY-MM-DD) ou return_time (ISO 8601)."
            )
        if rd and not rt:
            st_str = data.get("scheduled_time")
            if not st_str:
                return
            outbound = api_scheduled_iso_to_naive_geneva(st_str)
            if outbound is None:
                return
            try:
                rday = date.fromisoformat(str(rd))
            except ValueError:
                return
            if rday < outbound.date():
                raise ValidationError(
                    {
                        "return_date": [
                            "return_date ne peut pas précéder la date du départ."
                        ]
                    }
                )

    @validates_schema
    def validate_recurrence(self, data, **kwargs):  # noqa: ARG002
        """Récurrence portail client : métadonnées pour le transporteur (une réservation créée)."""
        if not data.get("is_recurring"):
            return
        rlen = data.get("recurrence_series_length")
        if rlen is None or int(rlen) < 1:
            raise ValidationError(
                "recurrence_series_length est requis (1–52) lorsque is_recurring=True"
            )
        rtype = (data.get("recurrence_type") or "").strip()
        if not rtype:
            raise ValidationError(
                "recurrence_type est requis lorsque is_recurring=True"
            )
        if rtype == "custom":
            days = data.get("recurrence_days") or []
            if not days:
                raise ValidationError(
                    "recurrence_days est requis pour recurrence_type=custom (au moins un jour)"
                )


class BookingPreviewSchema(BookingCreateSchema):
    """Schema pour prévisualisation de réservation côté client mobile.

    La preview partage les mêmes validations métier qu'une création,
    mais n'exige pas un ``amount`` côté client car il est calculé côté backend.
    """

    amount = fields.Float(
        required=False,
        load_default=1.0,
        validate=validate.Range(min=0.5, error="Le montant minimum accepté est 0.5"),
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
    hospital_service = fields.Str(validate=validate.Length(max=100))
    is_round_trip = fields.Bool()
    return_time = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATETIME_REGEX, error="return_time doit être au format ISO 8601"
        ),
        allow_none=True,
    )
    notes_medical = fields.Str(validate=validate.Length(max=1000))
    pickup_access_notes = fields.Str(
        validate=validate.Length(max=1000), allow_none=True
    )
    dropoff_access_notes = fields.Str(
        validate=validate.Length(max=1000), allow_none=True
    )
    pickup_floor = fields.Str(validate=validate.Length(max=50), allow_none=True)
    pickup_door_code = fields.Str(validate=validate.Length(max=50), allow_none=True)
    dropoff_floor = fields.Str(validate=validate.Length(max=50), allow_none=True)
    dropoff_door_code = fields.Str(validate=validate.Length(max=50), allow_none=True)

    # ✅ Livraison matériel : permettre de corriger mission_type et delivery_description
    mission_type = fields.Str(
        validate=validate.OneOf(["patient_transport", "material_delivery"]),
        allow_none=True,
    )
    delivery_description = fields.Str(
        validate=validate.Length(max=DELIVERY_DESCRIPTION_MAX_LENGTH),
        allow_none=True,
    )

    @validates_schema
    def validate_delivery_description(self, data, **kwargs):  # noqa: ARG002
        """Si mission_type == material_delivery (dans le payload), delivery_description requis."""
        mission_type = (data.get("mission_type") or "").strip().lower()
        if mission_type != "material_delivery":
            return
        raw = (data.get("delivery_description") or "").strip()
        delivery_desc = " ".join(raw.split()) if raw else ""
        if not delivery_desc:
            raise ValidationError(
                {
                    "delivery_description": [
                        "Veuillez saisir une description pour la livraison."
                    ],
                    "_error_code": ErrorCodes.MATERIAL_DELIVERY_DESCRIPTION_REQUIRED,
                    "_details": {"field": "delivery_description"},
                }
            )
        data["delivery_description"] = delivery_desc

    @validates_schema
    def validate_return_time_after_scheduled_time(self, data, **kwargs):  # noqa: ARG002
        """Aller-retour : return_time optionnel à la mise à jour. Si fourni, > scheduled_time."""
        return_time_str = data.get("return_time")
        scheduled_time_str = data.get("scheduled_time")

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


class CompanyBookingBillingAdjustmentSchema(Schema):
    """PATCH entreprise : ajustement montant / facturation (hors PUT opérationnel)."""

    amount = fields.Float(allow_none=True, validate=validate.Range(min=0))
    billed_to_type = fields.Str(
        allow_none=True,
        validate=validate.OneOf(["patient", "clinic", "insurance"]),
    )
    billed_to_company_id = fields.Int(allow_none=True)
    override_reason = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=2000),
    )


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
