"""CLI Flask — facturation plateforme LIRIE V1 (config pilote, recalcul, grilles)."""

# pyright: reportUnusedFunction=false
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import click
from flask import Flask
from flask.cli import with_appcontext

from ext import db
from models import Company
from models.enums import DispatchMode
from models.platform_billing import (
    CompanyPlatformBillingConfig,
    PlatformSubscriptionPricing,
)

_ISO_DATE_STRING_LEN = 10
_CALENDAR_MONTH_COUNT = 12


def _parse_optional_datetime(s: str | None) -> datetime | None:
    if s is None or not str(s).strip():
        return None
    s = str(s).strip()
    if (
        len(s) == _ISO_DATE_STRING_LEN
        and s[4] == "-"
        and s[7] == "-"
    ):
        return datetime.fromisoformat(f"{s}T00:00:00+00:00")
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _parse_optional_bool(s: str | None) -> bool | None:
    if s is None or s == "":
        return None
    v = str(s).strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    raise click.BadParameter("attendu: true ou false")


def _latest_company_config(company_id: int) -> CompanyPlatformBillingConfig | None:
    return (
        CompanyPlatformBillingConfig.query.filter_by(company_id=company_id)
        .order_by(CompanyPlatformBillingConfig.id.desc())
        .first()
    )


def _serialize_config(cfg: CompanyPlatformBillingConfig) -> dict[str, Any]:
    return {
        "id": cfg.id,
        "company_id": cfg.company_id,
        "is_billing_enabled": cfg.is_billing_enabled,
        "dispatch_mode_override": cfg.dispatch_mode_override,
        "commission_rate": str(cfg.commission_rate)
        if cfg.commission_rate is not None
        else None,
        "support_hourly_rate_default": str(cfg.support_hourly_rate_default)
        if cfg.support_hourly_rate_default is not None
        else None,
        "effective_from": cfg.effective_from.isoformat() if cfg.effective_from else None,
        "effective_to": cfg.effective_to.isoformat() if cfg.effective_to else None,
        "is_active": cfg.is_active,
        "notes": cfg.notes,
    }


def register_platform_billing_cli(app: Flask) -> None:
    @app.cli.group("platform-billing")
    def platform_billing() -> None:
        """Facturation plateforme LIRIE V1 (config entreprise, périodes, grilles)."""

    @platform_billing.command("set-company-config")
    @click.option("--company-id", type=int, required=True)
    @click.option(
        "--enabled",
        "billing_enabled",
        default=None,
        help="true/false : activer le billing plateforme pour cette entreprise",
    )
    @click.option("--commission-rate", type=str, default=None, help="Taux décimal ex. 0.10")
    @click.option(
        "--support-hourly-rate",
        type=str,
        default=None,
        help="Tarif horaire support défaut (CHF)",
    )
    @click.option(
        "--dispatch-mode-override",
        type=str,
        default=None,
        help="manual | semi_auto | fully_auto, ou chaîne vide pour effacer",
    )
    @click.option(
        "--effective-from",
        type=str,
        default=None,
        help="Date début validité (ISO ou YYYY-MM-DD)",
    )
    @click.option(
        "--effective-to",
        type=str,
        default=None,
        help="Date fin validité (ISO ou YYYY-MM-DD)",
    )
    @click.option(
        "--is-active",
        type=str,
        default=None,
        help="true/false : enregistrement actif",
    )
    @click.option("--notes", type=str, default=None)
    @with_appcontext
    def set_company_config(
        company_id: int,
        billing_enabled: str | None,
        commission_rate: str | None,
        support_hourly_rate: str | None,
        dispatch_mode_override: str | None,
        effective_from: str | None,
        effective_to: str | None,
        is_active: str | None,
        notes: str | None,
    ) -> None:
        co = db.session.get(Company, company_id)
        if not co:
            raise click.ClickException(f"Entreprise introuvable: {company_id}")

        cfg = _latest_company_config(company_id)
        if not cfg:
            cfg = CompanyPlatformBillingConfig()
            cfg.company_id = company_id
            db.session.add(cfg)

        be = _parse_optional_bool(billing_enabled)
        if be is not None:
            cfg.is_billing_enabled = be

        if commission_rate is not None:
            cfg.commission_rate = Decimal(str(commission_rate))

        if support_hourly_rate is not None:
            cfg.support_hourly_rate_default = Decimal(str(support_hourly_rate))

        if dispatch_mode_override is not None:
            dmo = dispatch_mode_override.strip()
            if dmo == "":
                cfg.dispatch_mode_override = None
            else:
                allowed = {e.value for e in DispatchMode}
                if dmo not in allowed:
                    raise click.BadParameter(
                        f"dispatch_mode_override doit être parmi {sorted(allowed)}"
                    )
                cfg.dispatch_mode_override = dmo

        if effective_from is not None:
            cfg.effective_from = _parse_optional_datetime(effective_from)
        if effective_to is not None:
            cfg.effective_to = _parse_optional_datetime(effective_to)

        ia = _parse_optional_bool(is_active)
        if ia is not None:
            cfg.is_active = ia

        if notes is not None:
            cfg.notes = notes

        db.session.commit()
        click.echo(
            f"OK — config enregistrée (id={cfg.id}, company_id={company_id})",
            err=False,
        )
        click.echo(_serialize_config(cfg))

    @platform_billing.command("show-company-config")
    @click.option("--company-id", type=int, required=True)
    @with_appcontext
    def show_company_config(company_id: int) -> None:
        co = db.session.get(Company, company_id)
        if not co:
            raise click.ClickException(f"Entreprise introuvable: {company_id}")
        cfg = _latest_company_config(company_id)
        if not cfg:
            click.echo("Aucune company_platform_billing_config pour cette entreprise.")
            return
        click.echo(_serialize_config(cfg))

    @platform_billing.command("list-pricing-tiers")
    @with_appcontext
    def list_pricing_tiers() -> None:
        rows = (
            PlatformSubscriptionPricing.query.order_by(
                PlatformSubscriptionPricing.dispatch_mode.asc(),
                PlatformSubscriptionPricing.volume_min.asc(),
            ).all()
        )
        if not rows:
            click.echo("Aucune ligne dans platform_subscription_pricing.")
            return
        for r in rows:
            vmax = r.volume_max if r.volume_max is not None else "∞"
            click.echo(
                f"id={r.id} mode={r.dispatch_mode} volume=[{r.volume_min}, {vmax}] price={r.price_monthly} CHF label={r.label!r}"
            )

    @platform_billing.command("ensure-period")
    @click.option("--year", type=int, required=True)
    @click.option("--month", type=int, required=True)
    @with_appcontext
    def ensure_period(year: int, month: int) -> None:
        if month < 1 or month > _CALENDAR_MONTH_COUNT:
            raise click.BadParameter("month doit être entre 1 et 12")
        from services.platform_billing.engine import get_or_create_period

        p = get_or_create_period(year, month)
        click.echo(
            {
                "id": p.id,
                "billing_year": p.billing_year,
                "billing_month": p.billing_month,
                "status": p.status,
            }
        )

    @platform_billing.command("recalculate-period")
    @click.option("--period-id", type=int, required=True)
    @with_appcontext
    def recalculate_period(period_id: int) -> None:
        from services.platform_billing.engine import recalculate_platform_period_drafts

        try:
            out = recalculate_platform_period_drafts(period_id)
        except ValueError as e:
            raise click.ClickException(str(e)) from e
        click.echo(out)
