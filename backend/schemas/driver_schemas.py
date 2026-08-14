"""✅ Schemas Marshmallow pour validation des endpoints drivers."""

from marshmallow import (  # pyright: ignore[reportMissingImports]
    Schema,
    fields,
    validate,
)


class DeviceHealthStatusSchema(Schema):
    """Schema pour POST /api/v1/driver/me/device-status.

    Canal séparé du tracking GPS : le mobile remonte ici l'état du device
    (foreground service, permissions, OEM battery optimization, taux de
    succès des fixes…). Permet de distinguer "téléphone éteint" d'une
    contrainte OEM (Samsung One UI, Doze…) qui tue le BG GPS de l'app.
    """

    kind = fields.Str(
        required=True,
        validate=validate.OneOf(
            ["tracking_health"],
            error="kind doit valoir 'tracking_health'",
        ),
    )
    fgs_running = fields.Bool(required=True)
    fg_permission = fields.Str(
        required=True,
        validate=validate.OneOf(
            ["granted", "denied", "undetermined"],
            error=("fg_permission doit être: 'granted', 'denied' ou 'undetermined'"),
        ),
    )
    bg_permission = fields.Str(
        required=True,
        validate=validate.OneOf(
            ["granted", "denied", "undetermined"],
            error=("bg_permission doit être: 'granted', 'denied' ou 'undetermined'"),
        ),
    )
    gps_provider_enabled = fields.Bool(required=True)
    battery_optimized = fields.Bool(required=True)

    battery_level = fields.Float(
        required=False,
        allow_none=True,
        validate=validate.Range(
            min=0.0, max=1.0, error="battery_level doit être entre 0 et 1"
        ),
    )
    is_charging = fields.Bool(required=False, allow_none=True)
    last_fix_age_seconds = fields.Int(
        required=False,
        allow_none=True,
        validate=validate.Range(min=0, error="last_fix_age_seconds doit être >= 0"),
        metadata={
            "description": (
                "Âge GNSS (s) = now - Location.timestamp "
                "(alias de location_fix_age_seconds)."
            )
        },
    )
    location_fix_age_seconds = fields.Int(
        required=False,
        allow_none=True,
        validate=validate.Range(
            min=0, error="location_fix_age_seconds doit être >= 0"
        ),
    )
    task_invoke_age_seconds = fields.Int(
        required=False,
        allow_none=True,
        validate=validate.Range(min=0, error="task_invoke_age_seconds doit être >= 0"),
        metadata={"description": "Âge dernière invocation task natif (≠ GNSS)."},
    )
    native_last_fix_age_seconds = fields.Int(
        required=False,
        allow_none=True,
        validate=validate.Range(
            min=0, error="native_last_fix_age_seconds doit être >= 0"
        ),
        metadata={
            "description": (
                "Compat : alias de task_invoke_age_seconds "
                "(≠ fraîcheur GNSS)."
            )
        },
    )
    watch_callback_age_seconds = fields.Int(
        required=False,
        allow_none=True,
        validate=validate.Range(
            min=0, error="watch_callback_age_seconds doit être >= 0"
        ),
    )
    observability_class = fields.Str(
        required=False,
        allow_none=True,
        validate=validate.Length(max=32),
    )
    oldest_queue_item_age_seconds = fields.Int(
        required=False,
        allow_none=True,
        validate=validate.Range(
            min=0, error="oldest_queue_item_age_seconds doit être >= 0"
        ),
    )
    persistence_lag_seconds = fields.Int(
        required=False,
        allow_none=True,
        validate=validate.Range(min=0, error="persistence_lag_seconds doit être >= 0"),
    )
    fix_success_rate_last_5min = fields.Float(
        required=False,
        allow_none=True,
        validate=validate.Range(
            min=0.0,
            max=1.0,
            error="fix_success_rate_last_5min doit être entre 0 et 1",
        ),
    )
    constraint_reason = fields.Str(
        required=False,
        allow_none=True,
        validate=validate.Length(max=64, error="constraint_reason max 64 caractères"),
    )


class DriverProfileUpdateSchema(Schema):
    """Schema pour mise à jour profil chauffeur (PUT /api/driver/me/profile)."""

    # Champs utilisateur
    first_name = fields.Str(validate=validate.Length(min=1, max=100))
    last_name = fields.Str(validate=validate.Length(min=1, max=100))
    phone = fields.Str(validate=validate.Length(max=20))

    # Statut
    status = fields.Str(
        validate=validate.OneOf(
            ["disponible", "hors service"],
            error="status doit être: 'disponible' ou 'hors service'",
        )
    )

    # HR champs
    contract_type = fields.Str(validate=validate.Length(max=50))
    weekly_hours = fields.Int(
        validate=validate.Range(
            min=0, max=168, error="weekly_hours doit être entre 0 et 168"
        )
    )
    hourly_rate_cents = fields.Int(
        validate=validate.Range(min=0, error="hourly_rate_cents doit être >= 0")
    )

    # Dates (format YYYY-MM-DD)
    # Note: fields.Date gère déjà le format ISO8601, pas besoin de Regexp
    employment_start_date = fields.Date(allow_none=True)
    employment_end_date = fields.Date(allow_none=True)
    license_valid_until = fields.Date(allow_none=True)
    medical_valid_until = fields.Date(allow_none=True)

    # Véhicule
    vehicle_assigned = fields.Str(validate=validate.Length(max=100), allow_none=True)
    brand = fields.Str(validate=validate.Length(max=100), allow_none=True)
    license_plate = fields.Str(validate=validate.Length(max=50), allow_none=True)

    # Listes
    license_categories = fields.List(
        fields.Str(validate=validate.Length(max=10)),
        validate=validate.Length(max=10, error="Maximum 10 catégories de permis"),
    )
    trainings = fields.List(
        fields.Dict(), validate=validate.Length(max=50, error="Maximum 50 formations")
    )
