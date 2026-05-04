"""Indicatif portail client : formule, arrondi, validation admin (hors compute_price / preview)."""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any

from models import PlatformClientIndicativeFareConfig

# Aligné sur Number.EPSILON (JavaScript) — utilisé seulement dans l'équivalent de round.
_JS_NUMBER_EPSILON = 2.220446049250313e-16


def round_chf_to_five_rappen(value: Any) -> Decimal:
    """Même sémantique que `roundChfToFiveRappen` dans ClientDashboard.jsx (ES)."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return Decimal("0")
    if not math.isfinite(x):
        return Decimal("0")
    y = (x + _JS_NUMBER_EPSILON) * 20.0
    n = math.floor(y + 0.5) if y >= 0.0 else math.ceil(y - 0.5)
    r = n / 20.0
    return Decimal(str(r))


def derive_per_km_chf(config: PlatformClientIndicativeFareConfig) -> Decimal:
    ref_km = Decimal(str(config.ref_km))
    if ref_km <= 0:
        return Decimal("0")
    return (
        Decimal(str(config.min_fare_chf))
        - Decimal(str(config.base_chf))
        - Decimal(str(config.ref_min)) * Decimal(str(config.per_minute_chf))
    ) / ref_km


def compute_indicative_amount_chf(
    distance_m: int, duration_s: int, config: PlatformClientIndicativeFareConfig
) -> Decimal:
    """Devis indicatif (CHF) : brut = base + per_km*km + per_min*min, puis plancher min_fare, arrondi 5 c."""
    if distance_m <= 0:
        return Decimal("0")
    per_km = derive_per_km_chf(config)
    km = Decimal(str(distance_m)) / Decimal(1000)
    if duration_s and duration_s > 0:
        minutes = Decimal(str(duration_s)) / Decimal(60)
    else:
        minutes = Decimal(0)
    raw = (
        Decimal(str(config.base_chf))
        + km * per_km
        + minutes * Decimal(str(config.per_minute_chf))
    )
    clamped = max(raw, Decimal(str(config.min_fare_chf)))
    return round_chf_to_five_rappen(clamped)


class IndicativeFareValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _to_decimal(name: str, raw: Any) -> Decimal:
    if raw is None:
        raise IndicativeFareValidationError(
            "invalid_type", f"Valeur manquante : {name}"
        )
    try:
        d = Decimal(str(raw))
    except (InvalidOperation, TypeError) as e:
        raise IndicativeFareValidationError(
            "invalid_type", f"Valeur non numérique : {name}"
        ) from e
    return d


def assert_coherence(
    min_fare: Decimal,
    base: Decimal,
    per_minute: Decimal,
    ref_min: Decimal,
    ref_km: Decimal,
) -> None:
    if ref_km <= 0:
        raise IndicativeFareValidationError(
            "ref_km", "ref_km doit être strictement positif."
        )
    if min_fare <= 0:
        raise IndicativeFareValidationError(
            "min_fare_chf", "min_fare_chf doit être strictement positif."
        )
    if base < 0 or per_minute < 0 or ref_min < 0:
        raise IndicativeFareValidationError(
            "non_negative", "base_chf, per_minute_chf et ref_min doivent être >= 0."
        )
    slack = min_fare - base - ref_min * per_minute
    if slack < 0:
        raise IndicativeFareValidationError(
            "negative_per_km",
            "Incohérence: min_fare - base - ref_min*per_minute doit être >= 0 (per_km implicite >= 0).",
        )


def merge_admin_update(
    row: PlatformClientIndicativeFareConfig, body: dict[str, Any]
) -> None:
    """Applique les champs reconnus sur `row` (PUT admin : incrément de version côté route)."""
    min_fare = _to_decimal("min_fare_chf", body.get("min_fare_chf", row.min_fare_chf))
    base = _to_decimal("base_chf", body.get("base_chf", row.base_chf))
    per_minute = _to_decimal(
        "per_minute_chf", body.get("per_minute_chf", row.per_minute_chf)
    )
    ref_km = _to_decimal("ref_km", body.get("ref_km", row.ref_km))
    ref_min = _to_decimal("ref_min", body.get("ref_min", row.ref_min))
    if "is_enabled" in body and body["is_enabled"] is not None:
        row.is_enabled = bool(body["is_enabled"])
    if "calibration_note" in body:
        note = body.get("calibration_note")
        row.calibration_note = (str(note) if note is not None else None) or None
    assert_coherence(min_fare, base, per_minute, ref_min, ref_km)
    row.min_fare_chf = min_fare
    row.base_chf = base
    row.per_minute_chf = per_minute
    row.ref_km = ref_km
    row.ref_min = ref_min


def config_to_public_dict(
    row: PlatformClientIndicativeFareConfig,
) -> dict[str, Any]:
    return {
        "is_enabled": bool(row.is_enabled),
        "min_fare_chf": float(row.min_fare_chf),
        "base_chf": float(row.base_chf),
        "per_minute_chf": float(row.per_minute_chf),
        "ref_km": float(row.ref_km),
        "ref_min": float(row.ref_min),
        "derived_per_km_chf": float(derive_per_km_chf(row)),
        "config_version": int(row.config_version or 0),
        "calibration_note": row.calibration_note,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "updated_by_user_id": row.updated_by_user_id,
    }
