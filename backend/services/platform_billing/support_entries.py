"""Saisie et validation des heures de support plateforme."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from ext import db
from models.enums import PlatformSupportEntryCategory
from models.platform_billing import (
    CompanyPlatformBillingConfig,
    PlatformSupportEntry,
)
from services.platform_billing.contracts import effective_config_for_period
from services.platform_billing.decimal_json import decimal_to_str, parse_decimal
from services.platform_billing.money import money_round_chf

_VALID_CATEGORIES = {c.value for c in PlatformSupportEntryCategory}


def serialize_support_entry(e: PlatformSupportEntry) -> dict[str, Any]:
    hours = (Decimal(e.duration_minutes) / Decimal(60)).quantize(Decimal("0.01"))
    return {
        "id": e.id,
        "company_id": e.company_id,
        "occurred_at": e.occurred_at.isoformat() if e.occurred_at else None,
        "duration_minutes": e.duration_minutes,
        "duration_hours": decimal_to_str(hours, places=2),
        "category": e.category,
        "description": e.description,
        "hourly_rate_snapshot": decimal_to_str(e.hourly_rate_snapshot),
        "amount": decimal_to_str(e.amount),
        "validated_at": e.validated_at.isoformat() if e.validated_at else None,
        "validated_by_user_id": e.validated_by_user_id,
        "billing_period_id": e.billing_period_id,
    }


def _resolve_duration_minutes(data: dict[str, Any]) -> int:
    if data.get("duration_minutes") is not None:
        minutes = int(data["duration_minutes"])
    elif data.get("duration_hours") is not None:
        hours = parse_decimal(data["duration_hours"], field="duration_hours")
        if hours is None:
            raise ValueError("duration_hours invalide")
        minutes = int((hours * Decimal(60)).to_integral_value())
    else:
        raise ValueError("Indiquez duration_hours ou duration_minutes")
    if minutes <= 0:
        raise ValueError("La durée doit être strictement positive")
    if minutes > 60 * 24 * 31:
        raise ValueError("Durée trop élevée")
    return minutes


def create_support_entry(
    data: dict[str, Any],
    *,
    validated_by_user_id: int | None = None,
) -> PlatformSupportEntry:
    """Crée une entrée support ; calcule le montant et valide si demandé."""
    try:
        company_id = int(data["company_id"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("company_id obligatoire") from e

    occurred_raw = data.get("occurred_at")
    if occurred_raw:
        occurred_at = datetime.fromisoformat(
            str(occurred_raw).replace("Z", "+00:00")
        )
    else:
        occurred_at = datetime.now(UTC)

    duration_minutes = _resolve_duration_minutes(data)
    category = str(data.get("category") or PlatformSupportEntryCategory.SUPPORT.value)
    if category not in _VALID_CATEGORIES:
        raise ValueError(
            f"category invalide (attendu: {', '.join(sorted(_VALID_CATEGORIES))})"
        )

    cfg: CompanyPlatformBillingConfig | None = effective_config_for_period(
        company_id, occurred_at
    )
    hourly = parse_decimal(
        data.get("hourly_rate_snapshot"),
        field="hourly_rate_snapshot",
        min_value=Decimal("0"),
        allow_none=True,
    )
    if hourly is None:
        if cfg and cfg.support_hourly_rate_default is not None:
            hourly = Decimal(str(cfg.support_hourly_rate_default))
        else:
            raise ValueError(
                "Tarif horaire manquant — renseignez-le ou activez-le dans la config "
                "entreprise (support_hourly_rate_default)."
            )

    amount = parse_decimal(
        data.get("amount"), field="amount", min_value=Decimal("0"), allow_none=True
    )
    if amount is None:
        hours = Decimal(duration_minutes) / Decimal(60)
        amount = money_round_chf(hours * hourly)
    else:
        amount = money_round_chf(amount)

    period_id = data.get("billing_period_id")
    billing_period_id = int(period_id) if period_id is not None else None

    auto_validate = data.get("auto_validate", True)
    if isinstance(auto_validate, str):
        auto_validate = auto_validate.lower() in ("1", "true", "yes", "on")

    se = PlatformSupportEntry(
        company_id=company_id,
        occurred_at=occurred_at,
        duration_minutes=duration_minutes,
        category=category,
        description=(data.get("description") or None),
        hourly_rate_snapshot=money_round_chf(hourly),
        amount=amount,
        billing_period_id=billing_period_id,
    )
    if auto_validate:
        se.validated_at = datetime.now(UTC)
        se.validated_by_user_id = validated_by_user_id

    db.session.add(se)
    db.session.commit()
    db.session.refresh(se)
    return se


def update_support_entry(
    entry_id: int,
    data: dict[str, Any],
    *,
    validated_by_user_id: int | None = None,
) -> PlatformSupportEntry:
    """Rectifie une entrée support (durée, catégorie, description, tarif)."""
    se = db.session.get(PlatformSupportEntry, entry_id)
    if not se:
        raise LookupError("Entrée support introuvable")

    if "duration_hours" in data or "duration_minutes" in data:
        se.duration_minutes = _resolve_duration_minutes(data)

    if "category" in data and data["category"] is not None:
        category = str(data["category"])
        if category not in _VALID_CATEGORIES:
            raise ValueError(
                f"category invalide (attendu: {', '.join(sorted(_VALID_CATEGORIES))})"
            )
        se.category = category

    if "description" in data:
        desc = data.get("description")
        se.description = (str(desc).strip() or None) if desc is not None else None

    if se.category == PlatformSupportEntryCategory.OTHER.value and not se.description:
        raise ValueError("Précisez la description pour la catégorie « Autre ».")

    if "hourly_rate_snapshot" in data and data["hourly_rate_snapshot"] is not None:
        hourly = parse_decimal(
            data["hourly_rate_snapshot"],
            field="hourly_rate_snapshot",
            min_value=Decimal("0"),
        )
        if hourly is None:
            raise ValueError("hourly_rate_snapshot invalide")
        se.hourly_rate_snapshot = money_round_chf(hourly)

    if "amount" in data and data["amount"] is not None:
        amount = parse_decimal(
            data["amount"], field="amount", min_value=Decimal("0")
        )
        if amount is None:
            raise ValueError("amount invalide")
        se.amount = money_round_chf(amount)
    else:
        hours = Decimal(se.duration_minutes) / Decimal(60)
        se.amount = money_round_chf(hours * Decimal(str(se.hourly_rate_snapshot)))

    # Revalider après correction
    se.validated_at = datetime.now(UTC)
    se.validated_by_user_id = validated_by_user_id

    db.session.commit()
    db.session.refresh(se)
    return se


def delete_support_entry(entry_id: int) -> tuple[dict[str, Any], int | None]:
    """Supprime une entrée support. Retourne (snapshot sérialisé, period_id)."""
    se = db.session.get(PlatformSupportEntry, entry_id)
    if not se:
        raise LookupError("Entrée support introuvable")
    snapshot = serialize_support_entry(se)
    period_id = se.billing_period_id
    db.session.delete(se)
    db.session.commit()
    return snapshot, period_id
