"""✅ Schemas Marshmallow pour validation des endpoints company."""

from marshmallow import Schema, fields, validate

from schemas.validation_utils import (
    EMAIL_VALIDATOR,
    ISO8601_DATE_REGEX,
    ISO8601_DATETIME_REGEX,
    PASSWORD_VALIDATOR,
    USERNAME_VALIDATOR,
)


class ManualBookingCreateSchema(Schema):
    """Schema pour création manuelle de réservation (POST /api/companies/me/reservations/manual)."""

    # Champs requis
    client_id = fields.Int(required=True, validate=validate.Range(min=1))
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

    # Champs optionnels
    customer_first_name = fields.Str(validate=validate.Length(max=100))
    customer_last_name = fields.Str(validate=validate.Length(max=100))
    customer_email = fields.Email()
    customer_phone = fields.Str(validate=validate.Length(max=20))

    # Round trip
    is_round_trip = fields.Bool(load_default=False)
    return_time = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATETIME_REGEX, error="return_time doit être au format ISO 8601"
        ),
        allow_none=True,
    )
    return_date = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATE_REGEX, error="return_date doit être au format YYYY-MM-DD"
        ),
        allow_none=True,
    )

    # Montant et facturation
    amount = fields.Float(validate=validate.Range(min=0))
    billed_to_type = fields.Str(
        validate=validate.OneOf(["patient", "clinic", "insurance"]), allow_none=True
    )
    billed_to_company_id = fields.Int(validate=validate.Range(min=1), allow_none=True)
    billed_to_contact = fields.Str(validate=validate.Length(max=200))

    # Champs médicaux
    medical_facility = fields.Str(validate=validate.Length(max=200))
    doctor_name = fields.Str(validate=validate.Length(max=200))
    hospital_service = fields.Str(validate=validate.Length(max=200))
    notes_medical = fields.Str(validate=validate.Length(max=1000))
    wheelchair_client_has = fields.Bool()
    wheelchair_need = fields.Bool()

    # Champs médicaux structurés
    establishment_id = fields.Int(validate=validate.Range(min=1))
    medical_service_id = fields.Int(validate=validate.Range(min=1))

    # Coordonnées GPS
    pickup_lat = fields.Float(validate=validate.Range(min=-90, max=90))
    pickup_lon = fields.Float(validate=validate.Range(min=-180, max=180))
    dropoff_lat = fields.Float(validate=validate.Range(min=-90, max=90))
    dropoff_lon = fields.Float(validate=validate.Range(min=-180, max=180))

    # Récurrence
    is_recurring = fields.Bool(load_default=False)
    recurrence_type = fields.Str(
        validate=validate.OneOf(["daily", "weekly", "custom"]), allow_none=True
    )
    recurrence_days = fields.List(
        fields.Int(validate=validate.Range(min=0, max=6)), allow_none=True
    )
    recurrence_end_date = fields.Str(
        validate=validate.Regexp(ISO8601_DATE_REGEX), allow_none=True
    )
    occurrences = fields.Int(validate=validate.Range(min=1), allow_none=True)

    class Meta:  # type: ignore
        unknown = "INCLUDE"  # Permettre des champs supplémentaires pour compatibilité


class ClientCreateSchema(Schema):
    """Schema pour création de client (POST /api/companies/me/clients)."""

    # Champs selon client_type
    client_type = fields.Str(
        required=True, validate=validate.OneOf(["SELF_SERVICE", "PRIVATE", "CORPORATE"])
    )
    email = fields.Email(validate=validate.Length(max=254))

    # Champs facturation (requis pour PRIVATE/CORPORATE)
    first_name = fields.Str(validate=validate.Length(min=1, max=100))
    last_name = fields.Str(validate=validate.Length(min=1, max=100))
    address = fields.Str(validate=validate.Length(min=1, max=500))

    # Champs optionnels
    phone = fields.Str(validate=validate.Length(max=20), allow_none=True)
    birth_date = fields.Date(allow_none=True)
    notes = fields.Str(validate=validate.Length(max=1000), allow_none=True)

    # Champs institution
    is_institution = fields.Bool(load_default=False)
    institution_name = fields.Str(validate=validate.Length(max=200), allow_none=True)

    # Champs facturation/contact
    billing_address = fields.Str(validate=validate.Length(max=500), allow_none=True)
    billing_lat = fields.Float(
        validate=validate.Range(min=-90, max=90), allow_none=True
    )
    billing_lon = fields.Float(
        validate=validate.Range(min=-180, max=180), allow_none=True
    )
    contact_email = fields.Email(validate=validate.Length(max=254), allow_none=True)
    contact_phone = fields.Str(validate=validate.Length(max=20), allow_none=True)

    # Adresse domicile
    domicile_address = fields.Str(validate=validate.Length(max=500), allow_none=True)
    domicile_zip = fields.Str(validate=validate.Length(max=20), allow_none=True)
    domicile_city = fields.Str(validate=validate.Length(max=100), allow_none=True)
    domicile_lat = fields.Float(
        validate=validate.Range(min=-90, max=90), allow_none=True
    )
    domicile_lon = fields.Float(
        validate=validate.Range(min=-180, max=180), allow_none=True
    )

    # Tarif préférentiel
    preferential_rate = fields.Decimal(places=2, allow_none=True)

    class Meta:  # type: ignore
        unknown = "INCLUDE"  # Permettre champs additionnels


class CompanyUpdateSchema(Schema):
    """Schema pour mise à jour entreprise (PUT /api/companies/me)."""

    name = fields.Str(validate=validate.Length(min=1, max=200))
    address = fields.Str(validate=validate.Length(max=500))
    latitude = fields.Float(validate=validate.Range(min=-90, max=90), allow_none=True)
    longitude = fields.Float(
        validate=validate.Range(min=-180, max=180), allow_none=True
    )
    contact_email = fields.Email(validate=validate.Length(max=254))
    contact_phone = fields.Str(validate=validate.Length(max=20))
    billing_email = fields.Email(validate=validate.Length(max=254))
    billing_notes = fields.Str(validate=validate.Length(max=1000))

    # IBAN: format CH + 19 chiffres/lettres (2 lettres + 2 chiffres + 1-30 caractères)
    iban = fields.Str(
        validate=validate.Regexp(
            r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$",
            error="IBAN invalide (format: 2 lettres pays + 2 chiffres + 1-30 caractères)",
        )
    )
    # UID IDE Suisse: CHE-123.456.789
    uid_ide = fields.Str(
        validate=validate.Regexp(
            r"^CHE-\d{3}\.\d{3}\.\d{3}$",
            error="UID IDE invalide (format: CHE-123.456.789)",
        )
    )

    # Adresse de domiciliation
    domicile_address_line1 = fields.Str(validate=validate.Length(max=200))
    domicile_address_line2 = fields.Str(validate=validate.Length(max=200))
    domicile_zip = fields.Str(validate=validate.Length(max=20))
    domicile_city = fields.Str(validate=validate.Length(max=100))
    domicile_country = fields.Str(
        validate=validate.Length(
            equal=2, error="Code pays doit faire 2 caractères (ISO-2)"
        )
    )

    logo_url = fields.Str(validate=validate.Length(max=500), allow_none=True)

    class Meta:  # type: ignore
        unknown = "INCLUDE"  # Permettre des champs supplémentaires


class DriverCreateSchema(Schema):
    """Schema pour création de chauffeur (POST /api/companies/me/drivers/create)."""

    username = fields.Str(required=True, validate=USERNAME_VALIDATOR)
    first_name = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    last_name = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    email = fields.Email(required=True, validate=EMAIL_VALIDATOR)
    password = fields.Str(required=True, validate=PASSWORD_VALIDATOR)
    vehicle_assigned = fields.Str(
        required=True, validate=validate.Length(min=1, max=100)
    )
    brand = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    license_plate = fields.Str(required=True, validate=validate.Length(min=1, max=20))


class DriverVacationCreateSchema(Schema):
    """Schema pour création de congés/vacances (POST /api/companies/me/drivers/<id>/vacations)."""

    start_date = fields.Str(
        required=True,
        validate=validate.Regexp(
            ISO8601_DATE_REGEX, error="start_date doit être au format YYYY-MM-DD"
        ),
    )
    end_date = fields.Str(
        required=True,
        validate=validate.Regexp(
            ISO8601_DATE_REGEX, error="end_date doit être au format YYYY-MM-DD"
        ),
    )
    vacation_type = fields.Str(
        validate=validate.OneOf(
            ["VACANCES", "CONGE", "MALADIE", "AUTRE"], error="vacation_type invalide"
        ),
        load_default="VACANCES",
    )


class VehicleUpdateSchema(Schema):
    """Schema pour mise à jour de véhicule (PUT /api/companies/me/vehicles/<id>)."""

    brand = fields.Str(validate=validate.Length(min=1, max=100))
    model = fields.Str(validate=validate.Length(max=100))
    license_plate = fields.Str(validate=validate.Length(min=1, max=20))
    color = fields.Str(validate=validate.Length(max=50))
    year = fields.Int(validate=validate.Range(min=1900, max=2100))
    seats = fields.Int(validate=validate.Range(min=1, max=50))
    is_wheelchair_accessible = fields.Bool()
    is_active = fields.Bool()
    notes = fields.Str(validate=validate.Length(max=1000), allow_none=True)

    class Meta:  # type: ignore
        unknown = "INCLUDE"  # Permettre des champs supplémentaires
