# backend/routes/dispatch/dispatch_schemas.py
"""Schémas Marshmallow et RESTX partagés pour les routes de dispatch."""

# ruff: noqa: I001  # Imports organisés manuellement pour meilleure lisibilité
from datetime import date, datetime

from flask_restx import fields  # pyright: ignore[reportMissingImports]
from marshmallow import (  # pyright: ignore[reportMissingImports]
    Schema,
    fields as ma_fields,
    validate,
)

from models.enums import AssignmentStatus
from routes.dispatch import dispatch_ns
from schemas.dispatch_overrides_schema import DispatchOverridesSchema

# ===== Schémas de validation Marshmallow =====


class DispatchRunSchema(Schema):
    """Schéma de validation pour les paramètres de lancement de dispatch."""

    for_date = ma_fields.Str(
        required=True, validate=validate.Regexp(r"^\d{4}-\d{2}-\d{2}$")
    )
    mode = ma_fields.Str(
        validate=validate.OneOf(["auto", "heuristic_only", "solver_only"])
    )
    regular_first = ma_fields.Bool()
    allow_emergency = ma_fields.Bool()
    overrides = ma_fields.Nested(DispatchOverridesSchema)
    # ✅ UNIFIÉ : Une seule variante pour 'async' avec valeur par défaut
    async_param = ma_fields.Bool(data_key="async", load_default=True)


# ===== Types RESTX personnalisés (nullable) =====


class NullableBoolean(fields.Raw):
    """Type RESTX pour bool|null."""

    def format(self, value):
        if value is None:
            return None
        return bool(value)


class NullableDict(fields.Raw):
    """Type RESTX pour dict|null."""

    def format(self, value):
        if value is None:
            return None
        return dict(value)


class NullableList(fields.Raw):
    """Type RESTX pour list|null."""

    def format(self, value):
        if value is None:
            return None
        return list(value)


class NullableString(fields.Raw):
    """Type RESTX pour string|null."""

    def format(self, value):
        if value is None:
            return None
        return str(value)


class NullableInteger(fields.Raw):
    """Type RESTX pour int|null."""

    def format(self, value):
        if value is None:
            return None
        return int(value)


class NullableFloat(fields.Raw):
    """Type RESTX pour float|null."""

    def format(self, value):
        if value is None:
            return None
        return float(value)


class NullableDate(fields.Raw):
    """Type RESTX pour date|null."""

    def format(self, value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return str(value)


class NullableDateTime(fields.Raw):
    """Type RESTX pour datetime|null."""

    def format(self, value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)


class NullableAny(fields.Raw):
    """Type RESTX pour any|null."""

    def format(self, value):
        if value is None:
            return None
        return value


class NullableEnum(fields.Raw):
    """Type RESTX pour enum|null."""

    def __init__(self, enum_class, **kwargs):
        super().__init__(**kwargs)
        self.enum_class = enum_class

    def format(self, value):
        if value is None:
            return None
        return str(value)


# ===== Modèles RESTX simples =====

preview_response = dispatch_ns.model(
    "DispatchPreviewResponse",
    {
        "bookings": fields.Integer,
        "drivers": fields.Integer,
        "horizon_minutes": fields.Integer,
        "ready": fields.Boolean,
        "reason": fields.String,
    },
)

run_model = dispatch_ns.model(
    "DispatchRunRequest",
    {
        "for_date": fields.String(required=True, description="Date YYYY-MM-DD"),
        "regular_first": fields.Boolean(
            default=True, description="Priorité aux chauffeurs réguliers"
        ),
        "allow_emergency": NullableBoolean(
            description="Autoriser les chauffeurs d'urgence"
        ),
        "async": fields.Boolean(default=True, description="Mode asynchrone"),
        "overrides": NullableDict(description="Surcharges de paramètres"),
        "mode": fields.String(
            description="Mode d'opération (auto|solver_only|heuristic_only)"
        ),
    },
)

trigger_model = dispatch_ns.model(
    "DispatchTriggerRequest",
    {
        "for_date": fields.String(required=True, description="Date YYYY-MM-DD"),
        "regular_first": fields.Boolean(
            default=True, description="Priorité aux chauffeurs réguliers"
        ),
        "allow_emergency": NullableBoolean(
            description="Autoriser les chauffeurs d'urgence"
        ),
    },
)

autorun_model = dispatch_ns.model(
    "DispatchAutorunRequest",
    {
        "enabled": fields.Boolean(
            required=True, description="Activer/désactiver l'autorun"
        ),
        "interval_sec": fields.Integer(
            required=False, description="Intervalle en secondes (optionnel)"
        ),
    },
)

# ===== Modèles pour les assignments =====

booking_model = dispatch_ns.model(
    "BookingBrief",
    {
        "id": fields.Integer,
        "reference": NullableString,
        "company_id": fields.Integer,
        "customer_name": NullableString,
        "pickup_address": NullableString,
        "dropoff_address": NullableString,
        "scheduled_time": NullableDateTime,
        "status": NullableString,
    },
)

driver_user_model = dispatch_ns.model(
    "DriverUserBrief",
    {
        "id": fields.Integer,
        "first_name": NullableString,
        "last_name": NullableString,
        "username": NullableString,
    },
)

driver_model = dispatch_ns.model(
    "DriverBrief",
    {
        "id": fields.Integer,
        "company_id": fields.Integer,
        "user": fields.Nested(driver_user_model, skip_none=True),
        "username": NullableString,  # Champ flat pour faciliter l'accès
        "first_name": NullableString,  # Nom du user
        "last_name": NullableString,  # Prénom du user
        "full_name": NullableString,  # Nom complet calculé
    },
)

assignment_model = dispatch_ns.model(
    "Assignment",
    {
        "id": fields.Integer,
        "booking_id": fields.Integer,
        "driver_id": fields.Integer,
        "dispatch_run_id": fields.Integer,
        "status": NullableString,
        "pickup_eta": NullableString,
        "dropoff_eta": NullableString,
        "created_at": NullableDateTime,
        "updated_at": NullableDateTime,
        "booking": fields.Nested(booking_model, skip_none=True),
        "driver": fields.Nested(driver_model, skip_none=True),
    },
)

assignment_patch_model = dispatch_ns.model(
    "AssignmentPatch",
    {
        "driver_id": fields.Integer,
        "status": fields.String(enum=[s.value for s in AssignmentStatus]),
    },
)

reassign_model = dispatch_ns.model(
    "ReassignRequest",
    {
        "new_driver_id": fields.Integer(required=True),
    },
)

# ===== Modèles pour les runs =====

dispatch_run_model = dispatch_ns.model(
    "DispatchRun",
    {
        "id": fields.Integer,
        "company_id": fields.Integer,
        "day": NullableDate,
        "created_at": NullableDateTime,
        "started_at": NullableDateTime,
        "completed_at": NullableDateTime,
        "status": NullableString,
        "meta": NullableDict,
    },
)

dispatch_run_detail_model = dispatch_ns.model(
    "DispatchRunDetail",
    {
        "id": fields.Integer,
        "company_id": fields.Integer,
        "day": NullableDate,
        "created_at": NullableDateTime,
        "started_at": NullableDateTime,
        "completed_at": NullableDateTime,
        "status": NullableString,
        "meta": NullableDict,
        "assignments": fields.List(fields.Nested(assignment_model)),
    },
)

# ===== Modèles pour les delays =====

delay_model = dispatch_ns.model(
    "Delay",
    {
        "id": fields.Integer,
        "booking_id": fields.Integer,
        "driver_id": fields.Integer,
        "assignment_id": fields.Integer,
        "pickup_time": NullableDateTime,
        "dropoff_time": NullableDateTime,
        "pickup_eta": NullableDateTime,
        "dropoff_eta": NullableDateTime,
        # Agrégé (max pickup/dropoff) — doit figurer dans le marshal sinon le mobile ne le voit pas.
        "delay_minutes": fields.Integer,
        "pickup_delay_minutes": fields.Integer,
        "dropoff_delay_minutes": fields.Integer,
        "booking": fields.Nested(booking_model, skip_none=True),
        "driver": fields.Nested(driver_model, skip_none=True),
    },
)
