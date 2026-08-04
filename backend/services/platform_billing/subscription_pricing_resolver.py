"""Résolution partagée des grilles d'abonnement volume (moteur + contrats)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from ext import db
from models.platform_billing import (
    CompanyPlatformBillingConfig,
    PlatformSubscriptionPricing,
    PlatformSubscriptionPricingGrid,
    PlatformSubscriptionPricingTier,
)
from services.platform_billing.decimal_json import decimal_to_str

SourceKind = Literal[
    "explicit_grid",
    "global_grid",
    "legacy_dispatch",
    "fixed",
    "free",
    "disabled",
]


@dataclass(frozen=True)
class PricingTierSnapshot:
    volume_min: int
    volume_max: int | None
    price_monthly: Decimal
    label: str | None


@dataclass(frozen=True)
class SubscriptionPricingResolution:
    source_kind: SourceKind
    pricing_mode: str
    requested_grid_id: int | None
    resolved_grid_id: int | None
    grid_key: str | None
    grid_label: str | None
    currency: str
    valid_from: datetime | None
    valid_until: datetime | None
    legacy_dispatch_mode: str | None
    tiers: tuple[PricingTierSnapshot, ...]
    validation_errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.validation_errors

    def to_snapshot_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "pricing_mode": self.pricing_mode,
            "requested_grid_id": self.requested_grid_id,
            "resolved_grid_id": self.resolved_grid_id,
            "grid_key": self.grid_key,
            "grid_label": self.grid_label,
            "currency": self.currency,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "legacy_dispatch_mode": self.legacy_dispatch_mode,
            "tiers": [
                {
                    "volume_min": t.volume_min,
                    "volume_max": t.volume_max,
                    "price_monthly": decimal_to_str(t.price_monthly),
                    "label": t.label,
                }
                for t in self.tiers
            ],
            "validation_errors": list(self.validation_errors),
            "validation_ok": self.is_valid,
        }


def _validate_tiers(tiers: list[PricingTierSnapshot]) -> list[str]:
    errors: list[str] = []
    if not tiers:
        errors.append("aucun_palier")
        return errors
    ordered = sorted(tiers, key=lambda t: t.volume_min)
    prev_max: int | None = None
    for idx, tier in enumerate(ordered):
        if tier.volume_max is not None and tier.volume_max < tier.volume_min:
            errors.append(f"palier_invalide_{idx}")
        if prev_max is not None:
            if tier.volume_min > prev_max + 1:
                errors.append(f"trou_avant_{tier.volume_min}")
            if tier.volume_min <= prev_max:
                errors.append(f"chevauchement_a_{tier.volume_min}")
        prev_max = tier.volume_max if tier.volume_max is not None else 10**9
    return errors


def _tiers_from_grid(grid_id: int) -> list[PricingTierSnapshot]:
    rows = (
        PlatformSubscriptionPricingTier.query.filter_by(grid_id=grid_id)
        .order_by(PlatformSubscriptionPricingTier.volume_min.asc())
        .all()
    )
    return [
        PricingTierSnapshot(
            volume_min=int(row.volume_min),
            volume_max=int(row.volume_max) if row.volume_max is not None else None,
            price_monthly=Decimal(str(row.price_monthly)),
            label=row.label,
        )
        for row in rows
    ]


def _tiers_from_legacy(dispatch_mode: str) -> list[PricingTierSnapshot]:
    rows = (
        PlatformSubscriptionPricing.query.filter_by(dispatch_mode=dispatch_mode)
        .order_by(PlatformSubscriptionPricing.volume_min.asc())
        .all()
    )
    return [
        PricingTierSnapshot(
            volume_min=int(row.volume_min),
            volume_max=int(row.volume_max) if row.volume_max is not None else None,
            price_monthly=Decimal(str(row.price_monthly)),
            label=row.label,
        )
        for row in rows
    ]


def active_default_grid(
    period_start: datetime,
) -> PlatformSubscriptionPricingGrid | None:
    """Grille globale active pour la période (même sémantique que l'ancien moteur)."""
    grids = (
        PlatformSubscriptionPricingGrid.query.filter_by(
            grid_key="default", is_active=True
        )
        .order_by(PlatformSubscriptionPricingGrid.id.desc())
        .all()
    )
    for g in grids:
        ef = g.valid_from
        et = g.valid_until
        if ef is not None:
            if ef.tzinfo is None:
                ef = ef.replace(tzinfo=UTC)
            if period_start < ef:
                continue
        if et is not None:
            if et.tzinfo is None:
                et = et.replace(tzinfo=UTC)
            if period_start >= et:
                continue
        return g
    return grids[0] if grids else None


def select_tier_from_grid(
    grid_id: int, volume: int
) -> PlatformSubscriptionPricingTier | None:
    rows = (
        PlatformSubscriptionPricingTier.query.filter_by(grid_id=grid_id)
        .order_by(PlatformSubscriptionPricingTier.volume_min.asc())
        .all()
    )
    for row in rows:
        if volume < row.volume_min:
            continue
        if row.volume_max is not None and volume > row.volume_max:
            continue
        return row
    return None


def resolve_subscription_pricing(
    cfg: CompanyPlatformBillingConfig,
    *,
    period_start: datetime,
    pricing_mode: str,
    dispatch_mode: str,
    own_portfolio_enabled: bool,
) -> SubscriptionPricingResolution:
    """Résout la source tarifaire pour un mode et une date donnés."""
    if not own_portfolio_enabled:
        return SubscriptionPricingResolution(
            source_kind="disabled",
            pricing_mode=pricing_mode,
            requested_grid_id=getattr(cfg, "pricing_grid_id", None),
            resolved_grid_id=None,
            grid_key=None,
            grid_label=None,
            currency="CHF",
            valid_from=None,
            valid_until=None,
            legacy_dispatch_mode=None,
            tiers=(),
            validation_errors=(),
        )
    if pricing_mode == "free":
        return SubscriptionPricingResolution(
            source_kind="free",
            pricing_mode=pricing_mode,
            requested_grid_id=getattr(cfg, "pricing_grid_id", None),
            resolved_grid_id=None,
            grid_key=None,
            grid_label="Gratuit",
            currency="CHF",
            valid_from=None,
            valid_until=None,
            legacy_dispatch_mode=None,
            tiers=(),
            validation_errors=(),
        )
    if pricing_mode == "fixed":
        return SubscriptionPricingResolution(
            source_kind="fixed",
            pricing_mode=pricing_mode,
            requested_grid_id=getattr(cfg, "pricing_grid_id", None),
            resolved_grid_id=None,
            grid_key=None,
            grid_label="Montant fixe",
            currency="CHF",
            valid_from=None,
            valid_until=None,
            legacy_dispatch_mode=None,
            tiers=(),
            validation_errors=(),
        )

    requested_grid_id = getattr(cfg, "pricing_grid_id", None)
    use_global = bool(getattr(cfg, "use_global_pricing_grid", True))
    grid: PlatformSubscriptionPricingGrid | None = None
    source_kind: SourceKind = "legacy_dispatch"

    if use_global or requested_grid_id:
        if requested_grid_id:
            grid = db.session.get(PlatformSubscriptionPricingGrid, requested_grid_id)
            if grid is not None:
                source_kind = "explicit_grid"
        if grid is None:
            grid = active_default_grid(period_start)
            if grid is not None:
                source_kind = "global_grid"

    if grid is not None:
        tiers = _tiers_from_grid(grid.id)
        errors = _validate_tiers(tiers)
        return SubscriptionPricingResolution(
            source_kind=source_kind,
            pricing_mode=pricing_mode,
            requested_grid_id=requested_grid_id,
            resolved_grid_id=grid.id,
            grid_key=grid.grid_key,
            grid_label=grid.label,
            currency=grid.currency or "CHF",
            valid_from=grid.valid_from,
            valid_until=grid.valid_until,
            legacy_dispatch_mode=None,
            tiers=tuple(tiers),
            validation_errors=tuple(errors),
        )

    tiers = _tiers_from_legacy(dispatch_mode)
    errors = _validate_tiers(tiers)
    return SubscriptionPricingResolution(
        source_kind="legacy_dispatch",
        pricing_mode=pricing_mode,
        requested_grid_id=requested_grid_id,
        resolved_grid_id=None,
        grid_key=None,
        grid_label=None,
        currency="CHF",
        valid_from=None,
        valid_until=None,
        legacy_dispatch_mode=dispatch_mode,
        tiers=tuple(tiers),
        validation_errors=tuple(errors),
    )


def ensure_contract_pricing_grid(
    *,
    billing_config_id: int,
    revision_number: int,
    reference: str,
    resolution: SubscriptionPricingResolution,
) -> PlatformSubscriptionPricingGrid:
    """Crée ou remplace la grille contractuelle immuable (brouillon)."""
    if not resolution.tiers:
        raise ValueError("Impossible de matérialiser une grille sans paliers")

    grid_key = f"contract-cfg-{billing_config_id}-r{revision_number}"
    existing = (
        PlatformSubscriptionPricingGrid.query.filter_by(grid_key=grid_key)
        .order_by(PlatformSubscriptionPricingGrid.id.desc())
        .first()
    )
    if existing is None:
        grid = PlatformSubscriptionPricingGrid(
            grid_key=grid_key,
            label=f"Grille contractuelle {reference}",
            currency=resolution.currency or "CHF",
            valid_from=None,
            valid_until=None,
            is_active=False,
        )
        db.session.add(grid)
        db.session.flush()
    else:
        grid = existing
        grid.label = f"Grille contractuelle {reference}"
        grid.currency = resolution.currency or "CHF"
        grid.is_active = False
        PlatformSubscriptionPricingTier.query.filter_by(grid_id=grid.id).delete()
        db.session.flush()

    for tier in resolution.tiers:
        db.session.add(
            PlatformSubscriptionPricingTier(
                grid_id=grid.id,
                volume_min=tier.volume_min,
                volume_max=tier.volume_max,
                price_monthly=tier.price_monthly,
                label=tier.label,
            )
        )
    db.session.flush()
    return grid
