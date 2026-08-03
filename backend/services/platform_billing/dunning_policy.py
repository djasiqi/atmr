"""Politique de dunning art. 6 bis — snapshot et validation."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from models.enums import PartnerAgreementStatus
from models.platform_billing import (
    CompanyPlatformBillingConfig,
    PlatformPartnerAgreement,
)

DUNNING_POLICY_VERSION = 1

_DUNNING_FIELD_DEFAULTS = {
    "automated_dunning_enabled": True,
    "reminder_delay_days_after_due": 0,
    "reminder_grace_days": 10,
    "full_suspend_days_after_due": 30,
    "full_suspend_overdue_invoice_count": 2,
    "termination_notice_days": 10,
    "partial_block_marketplace_offers": True,
    "partial_block_marketplace_acceptance": True,
    "partial_block_billable_support": True,
    "partial_block_billable_configuration": True,
}


def parse_dunning_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Valide et normalise les champs dunning depuis un payload API."""
    enabled = data.get("automated_dunning_enabled")
    if enabled is None:
        enabled = True
    enabled = bool(enabled)

    def _int(name: str, default: int, lo: int, hi: int) -> int:
        raw = data.get(name, default)
        if raw is None or raw == "":
            raw = default
        value = int(raw)
        if value < lo or value > hi:
            raise ValueError(f"{name} doit être entre {lo} et {hi}")
        return value

    delay = _int("reminder_delay_days_after_due", 0, 0, 30)
    grace = _int("reminder_grace_days", 10, 1, 30)
    full_days = _int("full_suspend_days_after_due", 30, 7, 90)
    full_count = _int("full_suspend_overdue_invoice_count", 2, 1, 12)
    term = _int("termination_notice_days", 10, 1, 30)

    if full_days <= delay + grace:
        raise ValueError(
            "full_suspend_days_after_due doit être strictement supérieur à "
            "reminder_delay_days_after_due + reminder_grace_days"
        )

    def _bool(name: str, default: bool = True) -> bool:
        raw = data.get(name)
        if raw is None:
            return default
        return bool(raw)

    return {
        "automated_dunning_enabled": enabled,
        "reminder_delay_days_after_due": delay,
        "reminder_grace_days": grace,
        "full_suspend_days_after_due": full_days,
        "full_suspend_overdue_invoice_count": full_count,
        "termination_notice_days": term,
        "partial_block_marketplace_offers": _bool(
            "partial_block_marketplace_offers"
        ),
        "partial_block_marketplace_acceptance": _bool(
            "partial_block_marketplace_acceptance"
        ),
        "partial_block_billable_support": _bool("partial_block_billable_support"),
        "partial_block_billable_configuration": _bool(
            "partial_block_billable_configuration"
        ),
    }


def build_dunning_policy_snapshot(
    cfg: CompanyPlatformBillingConfig | dict[str, Any],
) -> dict[str, Any]:
    """Construit le JSON figé à l'émission / ouverture de dossier."""

    def _get(key: str, default: Any) -> Any:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    snap: dict[str, Any] = {}
    for key, default in _DUNNING_FIELD_DEFAULTS.items():
        val = _get(key, default)
        if val is None:
            val = default
        snap[key] = bool(val) if isinstance(default, bool) else int(val)
    snap["policy_version"] = DUNNING_POLICY_VERSION
    return snap


def serialize_dunning_fields(cfg: CompanyPlatformBillingConfig) -> dict[str, Any]:
    return {
        "automated_dunning_enabled": bool(
            getattr(cfg, "automated_dunning_enabled", True)
        ),
        "reminder_delay_days_after_due": int(
            getattr(cfg, "reminder_delay_days_after_due", 0) or 0
        ),
        "reminder_grace_days": int(getattr(cfg, "reminder_grace_days", 10) or 10),
        "full_suspend_days_after_due": int(
            getattr(cfg, "full_suspend_days_after_due", 30) or 30
        ),
        "full_suspend_overdue_invoice_count": int(
            getattr(cfg, "full_suspend_overdue_invoice_count", 2) or 2
        ),
        "termination_notice_days": int(
            getattr(cfg, "termination_notice_days", 10) or 10
        ),
        "partial_block_marketplace_offers": bool(
            getattr(cfg, "partial_block_marketplace_offers", True)
        ),
        "partial_block_marketplace_acceptance": bool(
            getattr(cfg, "partial_block_marketplace_acceptance", True)
        ),
        "partial_block_billable_support": bool(
            getattr(cfg, "partial_block_billable_support", True)
        ),
        "partial_block_billable_configuration": bool(
            getattr(cfg, "partial_block_billable_configuration", True)
        ),
    }


def compute_dunning_automation_ready(
    *,
    cfg: CompanyPlatformBillingConfig | None,
    agreement: PlatformPartnerAgreement | None,
    today: date | None = None,
) -> dict[str, Any]:
    """Indicateur UI pour futures émissions (non rétroactif)."""
    today = today or datetime.now(UTC).date()
    reasons: list[str] = []
    if cfg is None:
        return {"ready": False, "reasons": ["no_billing_config"]}
    if not bool(getattr(cfg, "automated_dunning_enabled", False)):
        reasons.append("automated_dunning_disabled")
    if agreement is None:
        reasons.append("no_partner_agreement")
    else:
        if agreement.status != PartnerAgreementStatus.SIGNED.value:
            reasons.append("agreement_not_signed")
        if agreement.billing_config_id != cfg.id:
            reasons.append("agreement_config_mismatch")
        eff = agreement.agreement_effective_from
        if eff is not None and eff > today:
            reasons.append("agreement_not_effective")
    return {"ready": len(reasons) == 0, "reasons": reasons}


def is_dunning_authorized_at_issuance(
    *,
    cfg: CompanyPlatformBillingConfig | None,
    agreement: PlatformPartnerAgreement | None,
    today: date | None = None,
) -> bool:
    return bool(
        compute_dunning_automation_ready(
            cfg=cfg, agreement=agreement, today=today
        )["ready"]
    )
