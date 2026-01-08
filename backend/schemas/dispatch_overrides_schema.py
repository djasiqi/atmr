"""Schéma Marshmallow pour les overrides de dispatch.

Placée dans `schemas/` (et non `routes/`) pour éviter les cycles d'imports
avec Flask-RESTX.
"""

from marshmallow import Schema  # pyright: ignore[reportMissingImports]
from marshmallow import fields as ma_fields  # pyright: ignore[reportMissingImports]


class DispatchOverridesSchema(Schema):
    """Valide que seules les clés autorisées sont présentes dans overrides."""

    heuristic = ma_fields.Dict(required=False, allow_none=True)
    solver = ma_fields.Dict(required=False, allow_none=True)
    service_times = ma_fields.Dict(required=False, allow_none=True)
    pooling = ma_fields.Dict(required=False, allow_none=True)
    time = ma_fields.Dict(required=False, allow_none=True)
    realtime = ma_fields.Dict(required=False, allow_none=True)
    fairness = ma_fields.Dict(required=False, allow_none=True)
    emergency = ma_fields.Dict(required=False, allow_none=True)
    matrix = ma_fields.Dict(required=False, allow_none=True)
    logging = ma_fields.Dict(required=False, allow_none=True)
    features = ma_fields.Dict(required=False, allow_none=True)
    autorun = ma_fields.Dict(required=False, allow_none=True)
    rl = ma_fields.Dict(required=False, allow_none=True)
    clustering = ma_fields.Dict(required=False, allow_none=True)
    multi_objective = ma_fields.Dict(required=False, allow_none=True)
    safety = ma_fields.Dict(required=False, allow_none=True)
    # ⚡ Champs supplémentaires pour fonctionnalités avancées
    reset_existing = ma_fields.Bool(required=False, allow_none=True)
    preferred_driver_id = ma_fields.Int(required=False, allow_none=True)
    fast_mode = ma_fields.Bool(required=False, allow_none=True)
    driver_load_multipliers = ma_fields.Dict(required=False, allow_none=True)

    class Meta:
        unknown = "EXCLUDE"
