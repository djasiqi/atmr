"""Schemas Marshmallow pour validation des endpoints institutionnels.

Inclut les schemas pour:
- Institution (profil institution)
- InstitutionPatient (CRUD patients)
- TransportRequest (CRUD demandes de transport)
"""

import pytz
from marshmallow import (
    EXCLUDE,
    Schema,
    ValidationError,
    fields,
    pre_load,
    validate,
    validates_schema,
)

from models.enums import (
    BillingIntent,
    GenderEnum,
    InstitutionRole,
    LocationType,
    MissionType,
    RequestStatus,
)

# ========== Constants ==========

VALID_GENDERS = [g.value for g in GenderEnum] + [g.name for g in GenderEnum]
VALID_MISSION_TYPES = MissionType.choices()
VALID_BILLING_INTENTS = BillingIntent.choices()
VALID_REQUEST_STATUSES = RequestStatus.choices()
VALID_INSTITUTION_ROLES = [r.value for r in InstitutionRole]
VALID_LOCATION_TYPES = LocationType.choices()

# Regex patterns
PHONE_REGEX = r"^\+?[0-9]{7,15}$"
ISO8601_DATE_REGEX = r"^\d{4}-\d{2}-\d{2}$"
ISO8601_DATETIME_REGEX = (
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:\d{2})?$"
)


# ========== Institution Profile Schema ==========


VALID_INSTITUTION_TYPES = ["clinic", "ems", "imad", "hospital", "curatelle", "other"]


class InstitutionUpdateSchema(Schema):
    """Schema pour mise à jour du profil institution (admin only).

    Tous les champs sont optionnels (update partiel).
    """

    class Meta:
        unknown = EXCLUDE

    name = fields.Str(
        validate=validate.Length(min=1, max=200),
        metadata={"description": "Nom de l'institution"},
    )
    institution_type = fields.Str(
        validate=validate.OneOf(VALID_INSTITUTION_TYPES),
        metadata={
            "description": "Type d'institution (clinic, ems, imad, hospital, curatelle, other)"
        },
    )
    address = fields.Str(
        validate=validate.Length(max=255),
        allow_none=True,
        metadata={"description": "Adresse postale"},
    )
    contact_email = fields.Email(
        allow_none=True,
        metadata={"description": "Email de contact"},
    )
    contact_phone = fields.Str(
        validate=validate.Regexp(
            PHONE_REGEX,
            error="Numéro de téléphone invalide (format: +41221234567)",
        ),
        allow_none=True,
        metadata={"description": "Téléphone de contact"},
    )
    notes = fields.Str(
        validate=validate.Length(max=2000),
        allow_none=True,
        metadata={"description": "Notes internes"},
    )


# ========== Institution Settings Schema ==========

VALID_BILLING_INTENT_SETTINGS = ["patient", "institution", "third_party"]
# Liste restreinte pour le frontend (dropdown), mais le backend accepte
# toute timezone IANA valide via pytz.
COMMON_TIMEZONES = [
    "Europe/Zurich",
    "Europe/Paris",
    "Europe/Berlin",
    "Europe/London",
    "Europe/Rome",
    "Europe/Vienna",
    "Europe/Brussels",
    "UTC",
]


def _validate_timezone(value: str) -> None:
    """Valide qu'une timezone est reconnue par pytz (IANA)."""
    if value not in pytz.all_timezones:
        raise ValidationError(
            f"Timezone invalide: '{value}'. Utilisez une timezone IANA valide (ex: Europe/Zurich, UTC)."
        )


class InstitutionSettingsUpdateSchema(Schema):
    """Schema pour PUT /institutions/settings (admin only).

    Tous les champs sont optionnels (update partiel).
    Section 'institution' = colonnes billing sur Institution.
    Section 'settings' = colonnes sur InstitutionSettings.
    """

    class Meta:
        unknown = EXCLUDE

    # ── Institution billing fields ──
    billing_email = fields.Email(
        allow_none=True,
        metadata={"description": "Email de facturation"},
    )
    billing_address = fields.Str(
        validate=validate.Length(max=500),
        allow_none=True,
        metadata={"description": "Adresse de facturation"},
    )
    vat_number = fields.Str(
        validate=validate.Length(max=50),
        allow_none=True,
        metadata={"description": "Numéro TVA (ex: CHE-123.456.789)"},
    )

    # ── Timeouts ──
    timeout_same_day_minutes = fields.Int(
        validate=validate.Range(min=1, max=240),
        metadata={"description": "Timeout same-day (minutes, 1-240)"},
    )
    timeout_default_minutes = fields.Int(
        validate=validate.Range(min=1, max=10080),
        metadata={"description": "Timeout par défaut (minutes, 1-10080)"},
    )

    # ── Billing defaults ──
    default_billing_intent = fields.Str(
        validate=validate.OneOf(VALID_BILLING_INTENT_SETTINGS),
        metadata={"description": "Intent facturation par défaut"},
    )
    default_vat_rate = fields.Float(
        validate=validate.Range(min=0, max=100),
        allow_none=True,
        metadata={"description": "Taux TVA par défaut (0-100%)"},
    )
    default_payment_terms_days = fields.Int(
        validate=validate.Range(min=0, max=365),
        metadata={"description": "Délai paiement (jours, 0-365)"},
    )

    # ── Notifications ──
    notification_emails = fields.List(
        fields.Email(),
        metadata={"description": "Liste d'emails de notification"},
    )
    notify_request_sent = fields.Bool()
    notify_offer_accepted = fields.Bool()
    notify_request_expired = fields.Bool()

    # ── Divers ──
    timezone = fields.Str(
        validate=_validate_timezone,
        metadata={"description": "Timezone IANA (ex: Europe/Zurich, UTC)"},
    )

    # ── Transport UX ──
    default_pickup_mode = fields.Str(
        validate=validate.OneOf(["institution", "domicile"]),
        metadata={"description": "Mode par défaut du lieu de départ"},
    )
    entry_points = fields.List(
        fields.Str(validate=validate.Length(min=1, max=100)),
        validate=validate.Length(max=20),
        metadata={"description": "Points d'accueil suggérés (max 20)"},
    )
    default_contact_phone = fields.Str(
        validate=validate.Length(max=50),
        allow_none=True,
        metadata={"description": "Téléphone standard institution"},
    )


# ========== Patient Schemas ==========


class InstitutionPatientCreateSchema(Schema):
    """Schema pour création d'un patient institution."""

    class Meta:
        unknown = EXCLUDE

    external_reference = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
        metadata={"description": "Référence externe DPI (unique par institution)"},
    )
    first_name = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=100),
        metadata={"description": "Prénom du patient"},
    )
    last_name = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=100),
        metadata={"description": "Nom du patient"},
    )
    dob = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATE_REGEX,
            error="dob doit être au format YYYY-MM-DD",
        ),
        allow_none=True,
        metadata={"description": "Date de naissance (YYYY-MM-DD)"},
    )
    gender = fields.Str(
        validate=validate.OneOf(
            VALID_GENDERS,
            error=f"gender doit être: {', '.join(VALID_GENDERS)}",
        ),
        allow_none=True,
        metadata={"description": "Genre (HOMME, FEMME, AUTRE)"},
    )
    address = fields.Str(
        validate=validate.Length(max=255),
        allow_none=True,
    )
    city = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
    )
    postal_code = fields.Str(
        validate=validate.Length(max=20),
        allow_none=True,
    )
    phone = fields.Str(
        validate=validate.Regexp(
            PHONE_REGEX,
            error="Numéro de téléphone invalide (format: +41791234567)",
        ),
        allow_none=True,
    )
    door_code = fields.Str(
        validate=validate.Length(max=50),
        allow_none=True,
        metadata={"description": "Code porte / digicode"},
    )
    floor = fields.Str(
        validate=validate.Length(max=20),
        allow_none=True,
        metadata={"description": "Étage (ex: 3, RDC, 2B)"},
    )
    access_notes = fields.Str(
        validate=validate.Length(max=2000),
        allow_none=True,
        metadata={"description": "Notes d'accès (ascenseur, rampe, concierge...)"},
    )
    residence_name = fields.Str(
        validate=validate.Length(max=200),
        allow_none=True,
        metadata={"description": "Établissement de résidence (EMS, foyer)"},
    )
    avs_number = fields.Str(
        validate=validate.Length(max=16),
        allow_none=True,
        metadata={"description": "Numéro AVS (756.XXXX.XXXX.XX)"},
    )
    insurance_name = fields.Str(
        validate=validate.Length(max=200),
        allow_none=True,
        metadata={"description": "Nom de la caisse maladie"},
    )
    insurance_number = fields.Str(
        validate=validate.Length(max=50),
        allow_none=True,
        metadata={"description": "Numéro d'assuré"},
    )
    has_guardianship = fields.Bool(
        load_default=False,
        metadata={"description": "Patient sous curatelle"},
    )
    guardianship_type = fields.Str(
        validate=validate.OneOf(
            ["curatorship", "opad", "lawyer", "family", "other"],
            error="guardianship_type doit être: curatorship, opad, lawyer, family, other",
        ),
        allow_none=True,
        metadata={"description": "Type de curatelle"},
    )
    guardian_name = fields.Str(
        validate=validate.Length(max=200),
        allow_none=True,
        metadata={"description": "Nom du curateur / représentant légal"},
    )
    guardian_organization = fields.Str(
        validate=validate.Length(max=200),
        allow_none=True,
        metadata={"description": "Organisation du curateur (OPAD, étude, etc.)"},
    )
    guardian_phone = fields.Str(
        validate=validate.Regexp(
            PHONE_REGEX,
            error="Numéro de téléphone du curateur invalide",
        ),
        allow_none=True,
        metadata={"description": "Téléphone du curateur"},
    )
    guardian_email = fields.Str(
        validate=[
            validate.Length(max=200),
            validate.Email(error="Email curateur invalide"),
        ],
        allow_none=True,
        metadata={"description": "Email du curateur"},
    )
    guardian_address = fields.Str(
        validate=validate.Length(max=500),
        allow_none=True,
        metadata={"description": "Adresse complète du curateur (facturation)"},
    )
    notes = fields.Str(
        validate=validate.Length(max=2000),
        allow_none=True,
    )


class InstitutionPatientUpdateSchema(Schema):
    """Schema pour mise à jour d'un patient institution.

    Tous les champs sont optionnels (update partiel).
    """

    class Meta:
        unknown = EXCLUDE

    external_reference = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
    )
    first_name = fields.Str(
        validate=validate.Length(min=1, max=100),
    )
    last_name = fields.Str(
        validate=validate.Length(min=1, max=100),
    )
    dob = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATE_REGEX,
            error="dob doit être au format YYYY-MM-DD",
        ),
        allow_none=True,
    )
    gender = fields.Str(
        validate=validate.OneOf(
            VALID_GENDERS,
            error=f"gender doit être: {', '.join(VALID_GENDERS)}",
        ),
        allow_none=True,
    )
    address = fields.Str(
        validate=validate.Length(max=255),
        allow_none=True,
    )
    city = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
    )
    postal_code = fields.Str(
        validate=validate.Length(max=20),
        allow_none=True,
    )
    phone = fields.Str(
        validate=validate.Regexp(
            PHONE_REGEX,
            error="Numéro de téléphone invalide",
        ),
        allow_none=True,
    )
    door_code = fields.Str(
        validate=validate.Length(max=50),
        allow_none=True,
    )
    floor = fields.Str(
        validate=validate.Length(max=20),
        allow_none=True,
    )
    access_notes = fields.Str(
        validate=validate.Length(max=2000),
        allow_none=True,
    )
    residence_name = fields.Str(
        validate=validate.Length(max=200),
        allow_none=True,
    )
    avs_number = fields.Str(
        validate=validate.Length(max=16),
        allow_none=True,
    )
    insurance_name = fields.Str(
        validate=validate.Length(max=200),
        allow_none=True,
    )
    insurance_number = fields.Str(
        validate=validate.Length(max=50),
        allow_none=True,
    )
    has_guardianship = fields.Bool()
    guardianship_type = fields.Str(
        validate=validate.OneOf(
            ["curatorship", "opad", "lawyer", "family", "other"],
            error="guardianship_type doit être: curatorship, opad, lawyer, family, other",
        ),
        allow_none=True,
    )
    guardian_name = fields.Str(
        validate=validate.Length(max=200),
        allow_none=True,
    )
    guardian_organization = fields.Str(
        validate=validate.Length(max=200),
        allow_none=True,
    )
    guardian_phone = fields.Str(
        validate=validate.Regexp(
            PHONE_REGEX,
            error="Numéro de téléphone du curateur invalide",
        ),
        allow_none=True,
    )
    guardian_email = fields.Str(
        validate=[
            validate.Length(max=200),
            validate.Email(error="Email curateur invalide"),
        ],
        allow_none=True,
    )
    guardian_address = fields.Str(
        validate=validate.Length(max=500),
        allow_none=True,
    )
    notes = fields.Str(
        validate=validate.Length(max=2000),
        allow_none=True,
    )


class InstitutionPatientQuerySchema(Schema):
    """Schema pour recherche de patients."""

    class Meta:
        unknown = EXCLUDE

    query = fields.Str(
        validate=validate.Length(min=1, max=100),
        allow_none=True,
        metadata={"description": "Recherche par nom/prénom"},
    )
    external_reference = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
    )
    page = fields.Int(
        validate=validate.Range(min=1),
        load_default=1,
    )
    per_page = fields.Int(
        validate=validate.Range(min=1, max=100),
        load_default=20,
    )


# ========== Transport Request Schemas ==========


class MobilitySchema(Schema):
    """Schema pour les informations de mobilité."""

    class Meta:
        unknown = EXCLUDE

    wheelchair = fields.Bool(load_default=False)
    stretcher = fields.Bool(load_default=False)
    needs_assistance = fields.Bool(load_default=False)
    oxygen = fields.Bool(load_default=False)
    walking = fields.Bool(load_default=True)


class ContactOnSiteSchema(Schema):
    """Schema pour le contact sur site (structure enrichie, rétrocompatible).

    Champs rétrocompatibles (lus par le code existant):
    - name, phone: contact principal (requis)

    Champs enrichis (optionnels):
    - requester_name, requester_phone, requester_service: demandeur
    - onsite_name, onsite_phone: contact sur place si différent
    - onsite_is_different: flag
    """

    class Meta:
        unknown = EXCLUDE

    # Rétrocompatible: champs historiques (requis)
    name = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=100),
    )
    phone = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=50),
    )
    role = fields.Str(
        validate=validate.Length(max=50),
        allow_none=True,
    )

    # Champs enrichis (optionnels)
    requester_name = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
    )
    requester_phone = fields.Str(
        validate=validate.Length(max=50),
        allow_none=True,
    )
    requester_service = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
    )
    onsite_is_different = fields.Bool(load_default=False)
    onsite_name = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
    )
    onsite_phone = fields.Str(
        validate=validate.Length(max=50),
        allow_none=True,
    )


class BillingDetailsSchema(Schema):
    """Schema pour les détails de facturation."""

    class Meta:
        unknown = EXCLUDE

    payer_name = fields.Str(validate=validate.Length(max=200))
    payer_address = fields.Str(validate=validate.Length(max=500))
    insurance_number = fields.Str(validate=validate.Length(max=50))
    reference = fields.Str(validate=validate.Length(max=100))
    notes = fields.Str(validate=validate.Length(max=500))


class TransportRequestCreateSchema(Schema):
    """Schema pour création d'une demande de transport.

    external_reference est OBLIGATOIRE et doit être unique par institution.
    """

    class Meta:
        unknown = EXCLUDE

    # Référence externe (OBLIGATOIRE)
    external_reference = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=100),
        metadata={"description": "Référence externe DPI (unique par institution)"},
    )

    # Patient (optionnel pour livraisons)
    patient_id = fields.Int(
        allow_none=True,
        metadata={"description": "ID du patient (null pour livraisons)"},
    )
    patient_external_reference = fields.Str(
        allow_none=True,
        validate=validate.Length(max=100),
        metadata={
            "description": "Référence externe patient (alternative à patient_id)"
        },
    )

    # Type de mission
    mission_type = fields.Str(
        validate=validate.OneOf(
            VALID_MISSION_TYPES,
            error=f"mission_type doit être: {', '.join(VALID_MISSION_TYPES)}",
        ),
        load_default=MissionType.PATIENT_TRANSPORT.value,
    )
    delivery_description = fields.Str(
        validate=validate.Length(max=500),
        allow_none=True,
        metadata={
            "description": "Description livraison (requis si mission_type=material_delivery)"
        },
    )

    # Horaire (OBLIGATOIRE)
    scheduled_time = fields.Str(
        required=True,
        validate=validate.Regexp(
            ISO8601_DATETIME_REGEX,
            error="scheduled_time doit être au format ISO8601 (ex: 2026-02-04T14:30:00+01:00)",
        ),
        metadata={"description": "Date/heure prévue (ISO8601 avec timezone)"},
    )
    scheduled_time_type = fields.Str(
        load_default="departure",
        validate=validate.OneOf(["departure", "arrival"]),
        metadata={
            "description": "Type d'horaire: 'departure' = heure de départ, "
            "'arrival' = heure du rendez-vous (arrivée)"
        },
    )

    # Lieux (OBLIGATOIRE)
    pickup_location = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=255),
    )
    pickup_lat = fields.Float(
        validate=validate.Range(min=-90, max=90),
        allow_none=True,
    )
    pickup_lng = fields.Float(
        validate=validate.Range(min=-180, max=180),
        allow_none=True,
    )
    pickup_floor = fields.Str(validate=validate.Length(max=50), allow_none=True)
    pickup_door_code = fields.Str(validate=validate.Length(max=50), allow_none=True)

    dropoff_location = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=255),
    )
    dropoff_lat = fields.Float(
        validate=validate.Range(min=-90, max=90),
        allow_none=True,
    )
    dropoff_lng = fields.Float(
        validate=validate.Range(min=-180, max=180),
        allow_none=True,
    )
    dropoff_floor = fields.Str(validate=validate.Length(max=50), allow_none=True)
    dropoff_door_code = fields.Str(validate=validate.Length(max=50), allow_none=True)

    # Type de lieu (optionnel, nullable pour rétrocompatibilité)
    pickup_type = fields.Str(
        validate=validate.OneOf(VALID_LOCATION_TYPES),
        allow_none=True,
        metadata={"description": "Type de lieu départ: institution | domicile | other"},
    )
    dropoff_type = fields.Str(
        validate=validate.OneOf(VALID_LOCATION_TYPES),
        allow_none=True,
        metadata={
            "description": "Type de lieu arrivée: institution | domicile | other"
        },
    )
    pickup_entry_point = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
        metadata={"description": "Point d'accueil départ (ex: Réception, Urgences)"},
    )
    dropoff_entry_point = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
        metadata={"description": "Point d'accueil arrivée"},
    )

    # Options trajet
    is_round_trip = fields.Bool(load_default=False)
    return_time = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATETIME_REGEX,
            error="return_time doit être au format ISO8601",
        ),
        allow_none=True,
    )

    # Mobilité
    mobility = fields.Nested(MobilitySchema, allow_none=True)

    # Accès
    floor_elevator_info = fields.Str(
        validate=validate.Length(max=500),
        allow_none=True,
    )

    # Contact
    contact_on_site = fields.Nested(ContactOnSiteSchema, allow_none=True)

    # Notes
    notes = fields.Str(
        validate=validate.Length(max=2000),
        allow_none=True,
    )

    # Facturation
    billing_intent = fields.Str(
        validate=validate.OneOf(
            VALID_BILLING_INTENTS,
            error=f"billing_intent doit être: {', '.join(VALID_BILLING_INTENTS)}",
        ),
        load_default=BillingIntent.PATIENT.value,
    )
    billing_details = fields.Nested(BillingDetailsSchema, allow_none=True)

    @validates_schema
    def validate_delivery_description(self, data, **_kwargs):
        """Valide que delivery_description est présent si mission_type != patient_transport."""
        mission_type = data.get("mission_type", MissionType.PATIENT_TRANSPORT.value)
        delivery_description = data.get("delivery_description")

        if (
            mission_type != MissionType.PATIENT_TRANSPORT.value
            and not delivery_description
        ):
            raise ValidationError(
                "delivery_description est requis pour mission_type=material_delivery",
                field_name="delivery_description",
            )

    @validates_schema
    def validate_round_trip(self, data, **_kwargs):
        """Valide que return_time est fourni si is_round_trip=True."""
        if data.get("is_round_trip") and not data.get("return_time"):
            raise ValidationError(
                "L'heure de retour est requise pour un trajet aller/retour.",
                field_name="return_time",
            )


class TransportRequestUpdateSchema(Schema):
    """Schema pour mise à jour d'une demande de transport.

    Validation: seuls DRAFT et SENT peuvent être modifiés.
    """

    class Meta:
        unknown = EXCLUDE

    # Patient
    patient_id = fields.Int(allow_none=True)
    patient_external_reference = fields.Str(
        allow_none=True,
        validate=validate.Length(max=100),
    )

    # Type de mission
    mission_type = fields.Str(
        validate=validate.OneOf(VALID_MISSION_TYPES),
    )
    delivery_description = fields.Str(
        validate=validate.Length(max=500),
        allow_none=True,
    )

    # Horaire
    scheduled_time = fields.Str(
        validate=validate.Regexp(
            ISO8601_DATETIME_REGEX,
            error="scheduled_time doit être au format ISO8601",
        ),
    )
    scheduled_time_type = fields.Str(
        validate=validate.OneOf(["departure", "arrival"]),
    )

    # Lieux
    pickup_location = fields.Str(validate=validate.Length(min=1, max=255))
    pickup_lat = fields.Float(validate=validate.Range(min=-90, max=90), allow_none=True)
    pickup_lng = fields.Float(
        validate=validate.Range(min=-180, max=180), allow_none=True
    )
    pickup_floor = fields.Str(validate=validate.Length(max=50), allow_none=True)
    pickup_door_code = fields.Str(validate=validate.Length(max=50), allow_none=True)

    dropoff_location = fields.Str(validate=validate.Length(min=1, max=255))
    dropoff_lat = fields.Float(
        validate=validate.Range(min=-90, max=90), allow_none=True
    )
    dropoff_lng = fields.Float(
        validate=validate.Range(min=-180, max=180), allow_none=True
    )
    dropoff_floor = fields.Str(validate=validate.Length(max=50), allow_none=True)
    dropoff_door_code = fields.Str(validate=validate.Length(max=50), allow_none=True)

    # Type de lieu
    pickup_type = fields.Str(
        validate=validate.OneOf(VALID_LOCATION_TYPES),
        allow_none=True,
    )
    dropoff_type = fields.Str(
        validate=validate.OneOf(VALID_LOCATION_TYPES),
        allow_none=True,
    )
    pickup_entry_point = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
    )
    dropoff_entry_point = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
    )

    # Options
    is_round_trip = fields.Bool()
    return_time = fields.Str(
        validate=validate.Regexp(ISO8601_DATETIME_REGEX),
        allow_none=True,
    )

    # Mobilité
    mobility = fields.Nested(MobilitySchema, allow_none=True)

    # Accès
    floor_elevator_info = fields.Str(validate=validate.Length(max=500), allow_none=True)

    # Contact
    contact_on_site = fields.Nested(ContactOnSiteSchema, allow_none=True)

    # Notes
    notes = fields.Str(validate=validate.Length(max=2000), allow_none=True)

    # Facturation
    billing_intent = fields.Str(validate=validate.OneOf(VALID_BILLING_INTENTS))
    billing_details = fields.Nested(BillingDetailsSchema, allow_none=True)


class TransportRequestQuerySchema(Schema):
    """Schema pour recherche de demandes."""

    class Meta:
        unknown = EXCLUDE

    status = fields.Str(
        validate=validate.OneOf(VALID_REQUEST_STATUSES),
        allow_none=True,
        load_default=None,
    )
    external_reference = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
        load_default=None,
    )
    patient_id = fields.Int(allow_none=True, load_default=None)
    date_from = fields.Str(
        validate=validate.Regexp(ISO8601_DATE_REGEX),
        allow_none=True,
        load_default=None,
    )
    date_to = fields.Str(
        validate=validate.Regexp(ISO8601_DATE_REGEX),
        allow_none=True,
        load_default=None,
    )
    page = fields.Int(
        validate=validate.Range(min=1),
        load_default=1,
    )
    per_page = fields.Int(
        validate=validate.Range(min=1, max=100),
        load_default=20,
    )

    @pre_load
    def empty_strings_to_none(self, data, **_kwargs):
        """Convertit les chaînes vides des query params en None."""
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, str) and value.strip() == "":
                cleaned[key] = None
            else:
                cleaned[key] = value
        return cleaned


# ========== Institution Users Management ==========


class InstitutionUserInviteSchema(Schema):
    """Schema pour inviter/ajouter un utilisateur à l'institution."""

    class Meta:
        unknown = EXCLUDE

    email = fields.Email(
        required=True,
        error_messages={"required": "L'email est requis", "invalid": "Email invalide"},
    )
    institution_role = fields.Str(
        required=True,
        validate=validate.OneOf(
            VALID_INSTITUTION_ROLES,
            error="Rôle invalide. Valeurs acceptées: {choices}",
        ),
    )
    first_name = fields.Str(
        validate=validate.Length(max=100),
        load_default=None,
    )
    last_name = fields.Str(
        validate=validate.Length(max=100),
        load_default=None,
    )


class InstitutionUserUpdateRoleSchema(Schema):
    """Schema pour modifier le rôle d'un utilisateur dans l'institution."""

    class Meta:
        unknown = EXCLUDE

    institution_role = fields.Str(
        required=True,
        validate=validate.OneOf(
            VALID_INSTITUTION_ROLES,
            error="Rôle invalide. Valeurs acceptées: {choices}",
        ),
    )


# ========== User Profile Schema ==========


class InstitutionMyProfileUpdateSchema(Schema):
    """Schema pour mise à jour du profil personnel de l'utilisateur institution.

    Tous les champs sont optionnels (update partiel).
    Chaque utilisateur peut modifier ses propres informations.
    """

    class Meta:
        unknown = EXCLUDE

    first_name = fields.Str(
        validate=validate.Length(max=100),
        load_default=None,
        metadata={"description": "Prénom"},
    )
    last_name = fields.Str(
        validate=validate.Length(max=100),
        load_default=None,
        metadata={"description": "Nom de famille"},
    )
    phone = fields.Str(
        validate=[
            validate.Length(max=20),
            validate.Regexp(
                PHONE_REGEX,
                error="Format de téléphone invalide. Utilisez le format international (+41...)",
            ),
        ],
        load_default=None,
        allow_none=True,
        metadata={"description": "Numéro de téléphone"},
    )


# ========== Permission Request Schema ==========


class PermissionRequestCreateSchema(Schema):
    """Schema pour créer une demande de droits auprès de l'admin.

    L'utilisateur spécifie le rôle souhaité et un message justificatif.
    """

    class Meta:
        unknown = EXCLUDE

    requested_role = fields.Str(
        required=True,
        validate=validate.OneOf(
            VALID_INSTITUTION_ROLES,
            error="Rôle invalide. Valeurs acceptées: {choices}",
        ),
        metadata={"description": "Rôle demandé"},
    )
    message = fields.Str(
        required=True,
        validate=validate.Length(min=5, max=500),
        metadata={
            "description": "Message justificatif de la demande (5-500 caractères)"
        },
    )
