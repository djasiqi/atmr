"""Contrats commerciaux versionnés (fenêtres semi-ouvertes [from, to))."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select

from ext import db
from models.enums import (
    CommissionCancellationPolicy,
    SubscriptionPricingMode,
)
from models.platform_billing import CompanyPlatformBillingConfig
from services.platform_billing.decimal_json import decimal_to_str, parse_decimal
from services.platform_billing.errors import BillingInvariantError

# Import tardif dans les fonctions pour éviter cycles avec partner_agreement

logger = logging.getLogger(__name__)
_ZURICH = ZoneInfo("Europe/Zurich")

_PRICING_MODES = {m.value for m in SubscriptionPricingMode}
_CANCEL_POLICIES = {p.value for p in CommissionCancellationPolicy}


def month_start_zurich_utc(year: int, month: int) -> datetime:
    """Premier jour du mois à 00:00 Europe/Zurich, en UTC."""
    local = datetime(year, month, 1, 0, 0, 0, tzinfo=_ZURICH)
    return local.astimezone(UTC)


def normalize_effective_to_month_start(dt: datetime | None) -> datetime | None:
    """Normalise une date d'effet au 1er du mois 00:00 Zurich → UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    local = dt.astimezone(_ZURICH)
    normalized = datetime(local.year, local.month, 1, 0, 0, 0, tzinfo=_ZURICH)
    return normalized.astimezone(UTC)


def _as_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def window_contains(effective_from: datetime | None, effective_to: datetime | None, at: datetime) -> bool:
    """Vrai si at ∈ [effective_from, effective_to) (NULL from = -∞, NULL to = +∞)."""
    at = _as_aware(at) or at
    ef = _as_aware(effective_from)
    et = _as_aware(effective_to)
    if ef is not None and at < ef:
        return False
    if et is not None and at >= et:
        return False
    return True


def windows_overlap(
    a_from: datetime | None,
    a_to: datetime | None,
    b_from: datetime | None,
    b_to: datetime | None,
) -> bool:
    """Chevauchement de deux intervalles semi-ouverts [from, to)."""
    # [a_from, a_to) overlaps [b_from, b_to) iff a_from < b_to and b_from < a_to
    # with None = unbounded
    a_from_u = _as_aware(a_from) or datetime.min.replace(tzinfo=UTC)
    b_from_u = _as_aware(b_from) or datetime.min.replace(tzinfo=UTC)
    a_to_u = _as_aware(a_to) or datetime.max.replace(tzinfo=UTC)
    b_to_u = _as_aware(b_to) or datetime.max.replace(tzinfo=UTC)
    return a_from_u < b_to_u and b_from_u < a_to_u


def effective_config_for_period(
    company_id: int,
    period_start_utc: datetime,
) -> CompanyPlatformBillingConfig | None:
    """Sélectionne exactement une config active+enabled pour le début de période."""
    period_start_utc = _as_aware(period_start_utc) or period_start_utc
    rows = (
        CompanyPlatformBillingConfig.query.filter(
            CompanyPlatformBillingConfig.company_id == company_id,
            CompanyPlatformBillingConfig.is_active.is_(True),
            CompanyPlatformBillingConfig.is_billing_enabled.is_(True),
        )
        .order_by(CompanyPlatformBillingConfig.id.desc())
        .all()
    )
    matches = [
        r
        for r in rows
        if window_contains(r.effective_from, r.effective_to, period_start_utc)
    ]
    if not matches:
        return None
    # Si plusieurs (données legacy), prendre l'id le plus récent
    return matches[0]


def lock_company_contracts_for_update(company_id: int) -> list[CompanyPlatformBillingConfig]:
    """Verrouille les versions actives d'une entreprise (anti-chevauchement concurrent)."""
    stmt = (
        select(CompanyPlatformBillingConfig)
        .where(
            CompanyPlatformBillingConfig.company_id == company_id,
            CompanyPlatformBillingConfig.is_active.is_(True),
        )
        .order_by(CompanyPlatformBillingConfig.id.asc())
        .with_for_update()
    )
    return list(db.session.execute(stmt).scalars().all())


def assert_no_overlap(
    company_id: int,
    effective_from: datetime | None,
    effective_to: datetime | None,
    *,
    exclude_id: int | None = None,
) -> None:
    locked = lock_company_contracts_for_update(company_id)
    for row in locked:
        if exclude_id is not None and row.id == exclude_id:
            continue
        if not row.is_active:
            continue
        if windows_overlap(
            effective_from, effective_to, row.effective_from, row.effective_to
        ):
            raise ValueError(
                "Chevauchement interdit avec le contrat "
                f"id={row.id} [{row.effective_from} → {row.effective_to})"
            )


def supersede_overlapping_contracts(
    company_id: int,
    new_from: datetime | None,
    new_to: datetime | None,
) -> list[int]:
    """Clôture / neutralise les contrats actifs qui chevaucheraient la nouvelle fenêtre.

    - Si l'ancien démarre avant ``new_from`` : ``effective_to = new_from`` (chaînage).
    - Sinon (même mois / même début) : fenêtre vide ``[from, from)`` + ``is_active=False``
      pour garder l'historique sans bloquer la nouvelle version.
    """
    close_at = _as_aware(new_from)
    if close_at is None:
        now_local = datetime.now(_ZURICH)
        close_at = month_start_zurich_utc(now_local.year, now_local.month)

    locked = lock_company_contracts_for_update(company_id)
    touched: list[int] = []
    for row in locked:
        if not row.is_active:
            continue
        if not windows_overlap(
            new_from, new_to, row.effective_from, row.effective_to
        ):
            continue
        old_from = _as_aware(row.effective_from)
        if old_from is None or old_from < close_at:
            row.effective_to = close_at
        else:
            # [A, A) ne chevauche rien ; on archive la version remplacée
            row.effective_to = old_from
            row.is_active = False
        touched.append(row.id)
        logger.info(
            "Contrat plateforme supersédé company_id=%s contract_id=%s → to=%s active=%s",
            company_id,
            row.id,
            row.effective_to,
            row.is_active,
        )
    if touched:
        db.session.flush()
    return touched


def calendar_year_month_zurich(dt: datetime | None) -> tuple[int | None, int | None]:
    """Année / mois calendaires Europe/Zurich pour une instant UTC."""
    if dt is None:
        return None, None
    aware = _as_aware(dt)
    if aware is None:
        return None, None
    local = aware.astimezone(_ZURICH)
    return local.year, local.month


def resolve_effective_instant_from_payload(
    data: dict[str, Any],
    *,
    year_key: str,
    month_key: str,
    iso_key: str,
) -> datetime | None:
    """Résout une date d'effet : year/month prioritaires, sinon ISO normalisé.

    Si year/month et ISO sont tous présents et contradictoires → 409.
    """
    year_raw = data.get(year_key)
    month_raw = data.get(month_key)
    has_ym = year_raw is not None and year_raw != ""
    has_m = month_raw is not None and month_raw != ""
    if has_ym ^ has_m:
        raise BillingInvariantError(
            "EFFECTIVE_YM_INCOMPLETE",
            f"{year_key} et {month_key} doivent être fournis ensemble.",
            details={year_key: year_raw, month_key: month_raw},
        )
    from_ym: datetime | None = None
    if has_ym and has_m:
        try:
            year = int(year_raw)
            month = int(month_raw)
        except (TypeError, ValueError) as exc:
            raise BillingInvariantError(
                "EFFECTIVE_YM_INVALID",
                f"{year_key}/{month_key} invalides.",
                details={year_key: year_raw, month_key: month_raw},
            ) from exc
        if month < 1 or month > 12:
            raise BillingInvariantError(
                "EFFECTIVE_MONTH_OUT_OF_RANGE",
                f"{month_key} doit être entre 1 et 12.",
                details={month_key: month},
            )
        from_ym = month_start_zurich_utc(year, month)

    iso_dt = normalize_effective_to_month_start(_parse_iso_dt(data.get(iso_key)))
    if from_ym is not None and iso_dt is not None and from_ym != iso_dt:
        raise BillingInvariantError(
            "EFFECTIVE_DATE_CONFLICT",
            f"{year_key}/{month_key} et {iso_key} sont contradictoires.",
            details={
                year_key: year_raw,
                month_key: month_raw,
                iso_key: data.get(iso_key),
                "from_year_month": from_ym.isoformat(),
                "from_iso": iso_dt.isoformat(),
            },
        )
    return from_ym if from_ym is not None else iso_dt


def serialize_contract(cfg: CompanyPlatformBillingConfig) -> dict[str, Any]:
    ef_year, ef_month = calendar_year_month_zurich(cfg.effective_from)
    et_year, et_month = calendar_year_month_zurich(cfg.effective_to)
    return {
        "id": cfg.id,
        "company_id": cfg.company_id,
        "is_billing_enabled": cfg.is_billing_enabled,
        "own_portfolio_billing_enabled": bool(
            getattr(cfg, "own_portfolio_billing_enabled", False)
        ),
        "lirie_commission_enabled": bool(
            getattr(cfg, "lirie_commission_enabled", False)
        ),
        "support_enabled": bool(getattr(cfg, "support_enabled", False)),
        "subscription_pricing_mode": getattr(
            cfg, "subscription_pricing_mode", SubscriptionPricingMode.VOLUME.value
        ),
        "custom_subscription_amount": decimal_to_str(
            getattr(cfg, "custom_subscription_amount", None)
        ),
        "use_global_pricing_grid": bool(
            getattr(cfg, "use_global_pricing_grid", True)
        ),
        "pricing_grid_id": getattr(cfg, "pricing_grid_id", None),
        "commission_cancellation_policy": getattr(
            cfg,
            "commission_cancellation_policy",
            CommissionCancellationPolicy.EXCLUDE.value,
        ),
        "free_license_max_months": getattr(cfg, "free_license_max_months", None),
        "statement_dispute_days": getattr(cfg, "statement_dispute_days", None),
        "payment_terms_days": getattr(cfg, "payment_terms_days", None),
        "amounts_are_tax_inclusive": bool(
            getattr(cfg, "amounts_are_tax_inclusive", False)
        ),
        "tax_rate_override": decimal_to_str(
            getattr(cfg, "tax_rate_override", None), places=4
        ),
        "legacy_dispatch_mode_override": cfg.dispatch_mode_override,
        "commission_rate": decimal_to_str(cfg.commission_rate, places=6),
        "support_hourly_rate_default": decimal_to_str(
            cfg.support_hourly_rate_default
        ),
        "effective_year": ef_year,
        "effective_month": ef_month,
        "effective_from": cfg.effective_from.isoformat()
        if cfg.effective_from
        else None,
        "effective_to_year": et_year,
        "effective_to_month": et_month,
        "effective_to": cfg.effective_to.isoformat() if cfg.effective_to else None,
        "effective_timezone": "Europe/Zurich",
        "is_active": cfg.is_active,
        "notes": cfg.notes,
        "commercially_frozen": _is_commercially_frozen(cfg.id),
        **_serialize_dunning(cfg),
        "dunning_automation_ready": _dunning_ready_payload(cfg),
    }


def _serialize_dunning(cfg: CompanyPlatformBillingConfig) -> dict[str, Any]:
    from services.platform_billing.dunning_policy import serialize_dunning_fields

    return serialize_dunning_fields(cfg)


def _dunning_ready_payload(cfg: CompanyPlatformBillingConfig) -> dict[str, Any]:
    from models.enums import PartnerAgreementStatus
    from models.platform_billing import PlatformPartnerAgreement
    from services.platform_billing.dunning_policy import (
        compute_dunning_automation_ready,
    )

    agr = (
        PlatformPartnerAgreement.query.filter_by(
            billing_config_id=cfg.id,
            status=PartnerAgreementStatus.SIGNED.value,
        )
        .order_by(PlatformPartnerAgreement.id.desc())
        .first()
    )
    return compute_dunning_automation_ready(cfg=cfg, agreement=agr)


def _is_commercially_frozen(billing_config_id: int) -> bool:
    from services.platform_billing.partner_agreement import (
        config_is_commercially_frozen,
    )

    return config_is_commercially_frozen(billing_config_id)


def _parse_iso_dt(raw: Any) -> datetime | None:
    if raw is None or raw == "":
        return None
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))


def create_contract_version(
    company_id: int, data: dict[str, Any]
) -> CompanyPlatformBillingConfig:
    """Crée une nouvelle version de contrat (jamais d'écrasement silencieux)."""
    effective_from = resolve_effective_instant_from_payload(
        data,
        year_key="effective_year",
        month_key="effective_month",
        iso_key="effective_from",
    )
    effective_to = resolve_effective_instant_from_payload(
        data,
        year_key="effective_to_year",
        month_key="effective_to_month",
        iso_key="effective_to",
    )
    if effective_from and effective_to and effective_to <= effective_from:
        raise ValueError("effective_to doit être strictement après effective_from")

    mode = str(
        data.get("subscription_pricing_mode") or SubscriptionPricingMode.VOLUME.value
    )
    if mode not in _PRICING_MODES:
        raise ValueError(
            f"subscription_pricing_mode doit être parmi {sorted(_PRICING_MODES)}"
        )
    policy = str(
        data.get("commission_cancellation_policy")
        or CommissionCancellationPolicy.EXCLUDE.value
    )
    if policy not in _CANCEL_POLICIES:
        raise ValueError(
            f"commission_cancellation_policy doit être parmi {sorted(_CANCEL_POLICIES)}"
        )

    commission_rate = parse_decimal(
        data.get("commission_rate"),
        field="commission_rate",
        min_value=Decimal("0"),
        max_value=Decimal("1"),
    )
    custom_sub = parse_decimal(
        data.get("custom_subscription_amount"),
        field="custom_subscription_amount",
        min_value=Decimal("0"),
    )
    support_rate = parse_decimal(
        data.get("support_hourly_rate_default"),
        field="support_hourly_rate_default",
        min_value=Decimal("0"),
    )
    tax_override = parse_decimal(
        data.get("tax_rate_override"),
        field="tax_rate_override",
        min_value=Decimal("0"),
        max_value=Decimal("100"),
    )

    payment_terms = data.get("payment_terms_days")
    if payment_terms is not None and payment_terms != "":
        payment_terms = int(payment_terms)
        if payment_terms < 0:
            raise ValueError("payment_terms_days doit être >= 0")
    else:
        payment_terms = None

    free_months = data.get("free_license_max_months")
    if free_months is not None and free_months != "":
        free_months = int(free_months)
        if free_months < 1 or free_months > 120:
            raise ValueError("free_license_max_months doit être entre 1 et 120")
    else:
        free_months = 60 if mode == SubscriptionPricingMode.FREE.value else None

    dispute_days = data.get("statement_dispute_days")
    if dispute_days is not None and dispute_days != "":
        dispute_days = int(dispute_days)
        if dispute_days < 1 or dispute_days > 60:
            raise ValueError("statement_dispute_days doit être entre 1 et 60")
    else:
        dispute_days = 10

    # Par défaut : clôturer / archiver les versions ouvertes qui bloqueraient l'enregistrement
    auto_close = data.get("auto_close_overlapping", True)
    if auto_close is not False:
        supersede_overlapping_contracts(company_id, effective_from, effective_to)

    assert_no_overlap(company_id, effective_from, effective_to)

    # Défauts sécurisés : inactif tant que non explicitement demandé
    is_billing = bool(data.get("is_billing_enabled", False))
    own = bool(data.get("own_portfolio_billing_enabled", False))
    comm = bool(data.get("lirie_commission_enabled", False))
    support = bool(data.get("support_enabled", False))
    if is_billing and not (own or comm or support):
        raise BillingInvariantError(
            "BILLING_PRODUCTS_REQUIRED",
            "Sélectionnez au moins un produit avant d’activer la facturation.",
            details={
                "is_billing_enabled": is_billing,
                "own_portfolio_billing_enabled": own,
                "lirie_commission_enabled": comm,
                "support_enabled": support,
            },
        )

    from services.platform_billing.dunning_policy import parse_dunning_fields

    dunning = parse_dunning_fields(data)

    cfg = CompanyPlatformBillingConfig(
        company_id=company_id,
        is_billing_enabled=is_billing,
        own_portfolio_billing_enabled=own,
        lirie_commission_enabled=comm,
        support_enabled=support,
        subscription_pricing_mode=mode,
        custom_subscription_amount=custom_sub,
        use_global_pricing_grid=bool(data.get("use_global_pricing_grid", True)),
        pricing_grid_id=data.get("pricing_grid_id"),
        commission_cancellation_policy=policy,
        free_license_max_months=free_months,
        statement_dispute_days=dispute_days,
        payment_terms_days=payment_terms,
        amounts_are_tax_inclusive=bool(data.get("amounts_are_tax_inclusive", False)),
        tax_rate_override=tax_override,
        commission_rate=commission_rate,
        support_hourly_rate_default=support_rate,
        effective_from=effective_from,
        effective_to=effective_to,
        is_active=bool(data.get("is_active", True)),
        notes=data.get("notes"),
        # legacy non écrit depuis la nouvelle API
        dispatch_mode_override=None,
        **dunning,
    )
    db.session.add(cfg)
    db.session.commit()
    db.session.refresh(cfg)
    return cfg


def close_contract(
    contract_id: int,
    effective_to: datetime | None = None,
    *,
    data: dict[str, Any] | None = None,
) -> CompanyPlatformBillingConfig:
    """Clôture un contrat (effective_to = début du mois suivant par défaut / fourni).

    Interdit si déjà clôturé, ou si un accord juridique est envoyé/signé :
    la clôture temporelle passe alors uniquement par create_contract_version / supersede.
    """
    cfg = db.session.get(CompanyPlatformBillingConfig, contract_id)
    if not cfg:
        raise BillingInvariantError(
            "CONTRACT_NOT_FOUND",
            "Contrat introuvable",
            status_code=404,
            details={"contract_id": contract_id},
        )
    if cfg.effective_to is not None:
        raise BillingInvariantError(
            "CONTRACT_ALREADY_CLOSED",
            "Cette version contractuelle est déjà clôturée.",
            details={
                "contract_id": contract_id,
                "effective_to": cfg.effective_to.isoformat(),
            },
        )
    from services.platform_billing.partner_agreement import (
        PartnerAgreementError,
        assert_config_mutable,
    )

    try:
        assert_config_mutable(contract_id)
    except PartnerAgreementError as exc:
        raise ValueError(exc.message) from exc
    rows = lock_company_contracts_for_update(cfg.company_id)
    cfg = next((r for r in rows if r.id == contract_id), cfg)
    if cfg.effective_to is not None:
        raise BillingInvariantError(
            "CONTRACT_ALREADY_CLOSED",
            "Cette version contractuelle est déjà clôturée.",
            details={
                "contract_id": contract_id,
                "effective_to": cfg.effective_to.isoformat(),
            },
        )

    to = effective_to
    if data:
        to = resolve_effective_instant_from_payload(
            data,
            year_key="effective_to_year",
            month_key="effective_to_month",
            iso_key="effective_to",
        )
    to = normalize_effective_to_month_start(to)
    if to is None:
        # Clôture immédiate au début du mois Zurich courant
        now_local = datetime.now(_ZURICH)
        to = month_start_zurich_utc(now_local.year, now_local.month)
    if cfg.effective_from and to <= _as_aware(cfg.effective_from):
        raise ValueError("effective_to de clôture invalide")
    # Vérifier que la fenêtre fermée ne chevauche pas une autre (hors soi)
    assert_no_overlap(
        cfg.company_id,
        cfg.effective_from,
        to,
        exclude_id=cfg.id,
    )
    cfg.effective_to = to
    db.session.commit()
    db.session.refresh(cfg)
    return cfg


def list_contracts(company_id: int) -> list[CompanyPlatformBillingConfig]:
    return (
        CompanyPlatformBillingConfig.query.filter_by(company_id=company_id)
        .order_by(
            CompanyPlatformBillingConfig.effective_from.desc().nullslast(),
            CompanyPlatformBillingConfig.id.desc(),
        )
        .all()
    )


def distinct_billable_company_ids() -> list[int]:
    """Entreprises ayant au moins une config active+enabled (pour recalcul)."""
    rows = (
        db.session.query(CompanyPlatformBillingConfig.company_id)
        .filter(
            CompanyPlatformBillingConfig.is_active.is_(True),
            CompanyPlatformBillingConfig.is_billing_enabled.is_(True),
        )
        .distinct()
        .all()
    )
    return [int(r[0]) for r in rows]
