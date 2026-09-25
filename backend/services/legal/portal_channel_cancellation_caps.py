"""Caps canal LIRIE — validation stricte (jamais min silencieux).

Politique canal = dimensions + maxima %.
Politique entreprise admissible ssi :
  - clés top-level ⊆ allowlist canal
  - dimensions utilisées ⊆ dimensions canal
  - chaque taux <= cap de la dimension
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ext import db
from models.lirie_channel_cancellation_policy import LirieChannelCancellationPolicy
from shared.time_utils import now_utc

# Clés JSON entreprise autorisées sur le canal (hors dimensions custom).
_COMPANY_ALLOWED_TOP_LEVEL = frozenset(
    {
        "enabled",
        "tiers",
        "reason_overrides",
        "apply_when_driver_assigned_only",
        "min_fee_chf",
        "max_fee_chf",
    }
)

ERROR_EXCEEDS_CAP = "portal_cancellation_exceeds_channel_cap"
ERROR_DIMENSION = "portal_cancellation_dimension_rejected"
ERROR_CHANNEL_REQUIRED = "portal_channel_cancellation_policy_required"


def _hash_body(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def get_current_channel_cancellation_policy() -> (
    LirieChannelCancellationPolicy | None
):
    return (
        LirieChannelCancellationPolicy.query.filter_by(is_current=True)
        .order_by(LirieChannelCancellationPolicy.id.desc())
        .first()
    )


def render_channel_policy_text(body_json: dict[str, Any]) -> str:
    """Texte client — cadre maximal d'annulation du canal."""
    lines = [
        "Cadre d'annulation du canal Client privé LIRIE (plafonds maximaux)",
        "",
        "Ces pourcentages sont des maximums. Chaque entreprise peut appliquer "
        "moins, y compris 0 %. Aucune entreprise ne peut dépasser ces plafonds "
        "ni facturer des frais d'annulation hors de ce cadre.",
        "",
    ]
    dims = body_json.get("dimensions") or []
    if isinstance(dims, list):
        for d in dims:
            if not isinstance(d, dict):
                continue
            label = str(d.get("label") or d.get("key") or "").strip()
            try:
                max_pct = int(d.get("max_percent"))
            except (TypeError, ValueError):
                continue
            if label:
                lines.append(f"- {label} : maximum {max_pct} %")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


@dataclass(frozen=True, slots=True)
class ChannelCapValidationResult:
    ok: bool
    error: str | None = None
    message: str | None = None


def _time_dimension_cap(
    hours_before: float, dimensions: list[dict[str, Any]]
) -> int | None:
    """Cap pour un palier entreprise « moins de H heures ».

    Convention éditeur ATMR : hours_before=H signifie frais si délai restant < H.
    Mapping vers fenêtres canal :
      H >= 24 → window applicable 4–24 h (et plus strictement le max de
                 toutes les fenêtres intersectant (0, H])
      sinon → max des fenêtres time intersectant (0, H]
    """
    windows: list[tuple[float, float, int]] = []
    for d in dimensions:
        if str(d.get("kind") or "") != "time":
            continue
        try:
            # Fenêtre [gte, lt) en heures avant départ ; lt=None → infini.
            gte = float(d.get("hours_before_gte") or 0)
            lt_raw = d.get("hours_before_lt")
            lt = float(lt_raw) if lt_raw is not None else float("inf")
            max_pct = int(d.get("max_percent"))
        except (TypeError, ValueError):
            continue
        windows.append((gte, lt, max_pct))
    if not windows:
        return None
    # Intersecte (0, hours_before)
    applicable = [
        pct
        for gte, lt, pct in windows
        if gte < hours_before and lt > 0  # overlap with (0, H)
    ]
    if not applicable:
        return None
    return min(applicable)


def validate_company_policy_against_channel(
    *,
    company_policy: dict[str, Any] | None,
    channel_body: dict[str, Any],
) -> ChannelCapValidationResult:
    """Valide dimensions ⊆ canal et rates <= caps. Jamais de clamp."""
    if not isinstance(channel_body, dict):
        return ChannelCapValidationResult(
            ok=False,
            error=ERROR_CHANNEL_REQUIRED,
            message="Politique canal invalide.",
        )

    dims_raw = channel_body.get("dimensions") or []
    if not isinstance(dims_raw, list) or not dims_raw:
        return ChannelCapValidationResult(
            ok=False,
            error=ERROR_CHANNEL_REQUIRED,
            message="La politique canal ne définit aucune dimension.",
        )
    dimensions = [d for d in dims_raw if isinstance(d, dict)]

    allowed_reasons = {
        str(d.get("code") or "").upper()
        for d in dimensions
        if str(d.get("kind") or "") == "reason" and d.get("code")
    }
    allowed_statuses = {
        str(d.get("status") or "").upper()
        for d in dimensions
        if str(d.get("kind") or "") == "status" and d.get("status")
    }
    allow_abs = bool(channel_body.get("allow_absolute_fee_bounds", True))
    allow_driver_gate = bool(channel_body.get("allow_driver_assigned_gate", True))

    if not company_policy or not company_policy.get("enabled"):
        return ChannelCapValidationResult(ok=True)

    if not isinstance(company_policy, dict):
        return ChannelCapValidationResult(
            ok=False,
            error=ERROR_DIMENSION,
            message="Configuration d'annulation entreprise invalide.",
        )

    unknown = set(company_policy.keys()) - _COMPANY_ALLOWED_TOP_LEVEL
    if unknown:
        return ChannelCapValidationResult(
            ok=False,
            error=ERROR_DIMENSION,
            message=(
                "Dimensions ou frais hors cadre canal : "
                + ", ".join(sorted(unknown))
            ),
        )

    if (
        company_policy.get("apply_when_driver_assigned_only") is not None
        and not allow_driver_gate
    ):
        return ChannelCapValidationResult(
            ok=False,
            error=ERROR_DIMENSION,
            message="Le canal n'autorise pas le filtre chauffeur assigné.",
        )

    if not allow_abs:
        for k in ("min_fee_chf", "max_fee_chf"):
            v = company_policy.get(k)
            if v is not None and str(v).strip() != "":
                try:
                    if float(v) > 0:
                        return ChannelCapValidationResult(
                            ok=False,
                            error=ERROR_DIMENSION,
                            message=(
                                "Frais absolus d'annulation non autorisés "
                                "par le canal LIRIE."
                            ),
                        )
                except (TypeError, ValueError):
                    return ChannelCapValidationResult(
                        ok=False,
                        error=ERROR_DIMENSION,
                        message=f"Valeur invalide pour {k}.",
                    )

    tiers = company_policy.get("tiers") or []
    if not isinstance(tiers, list):
        return ChannelCapValidationResult(
            ok=False,
            error=ERROR_DIMENSION,
            message="tiers invalides.",
        )

    for t in tiers:
        if not isinstance(t, dict):
            return ChannelCapValidationResult(
                ok=False,
                error=ERROR_DIMENSION,
                message="Palier d'annulation invalide.",
            )
        ttype = str(t.get("type") or "").lower()
        try:
            pct = int(t.get("percent") or 0)
        except (TypeError, ValueError):
            return ChannelCapValidationResult(
                ok=False,
                error=ERROR_DIMENSION,
                message="Pourcentage de palier invalide.",
            )
        if pct < 0:
            return ChannelCapValidationResult(
                ok=False,
                error=ERROR_DIMENSION,
                message="Pourcentage négatif interdit.",
            )
        if ttype == "time":
            try:
                hours = float(t.get("hours_before") or 0)
            except (TypeError, ValueError):
                return ChannelCapValidationResult(
                    ok=False,
                    error=ERROR_DIMENSION,
                    message="Seuil horaire invalide.",
                )
            cap = _time_dimension_cap(hours, dimensions)
            if cap is None:
                return ChannelCapValidationResult(
                    ok=False,
                    error=ERROR_DIMENSION,
                    message=(
                        f"Palier temps ({hours} h) hors dimensions du canal LIRIE."
                    ),
                )
            if pct > cap:
                return ChannelCapValidationResult(
                    ok=False,
                    error=ERROR_EXCEEDS_CAP,
                    message=(
                        f"Palier {hours} h : {pct} % dépasse le plafond canal "
                        f"de {cap} %."
                    ),
                )
        elif ttype == "status":
            status = str(t.get("status") or "").upper()
            if status not in allowed_statuses:
                return ChannelCapValidationResult(
                    ok=False,
                    error=ERROR_DIMENSION,
                    message=f"Statut {status!r} hors dimensions canal.",
                )
            status_cap = next(
                (
                    int(d["max_percent"])
                    for d in dimensions
                    if str(d.get("kind")) == "status"
                    and str(d.get("status") or "").upper() == status
                ),
                None,
            )
            if status_cap is None or pct > status_cap:
                return ChannelCapValidationResult(
                    ok=False,
                    error=ERROR_EXCEEDS_CAP,
                    message=(
                        f"Statut {status} : {pct} % dépasse le plafond canal."
                    ),
                )
        else:
            return ChannelCapValidationResult(
                ok=False,
                error=ERROR_DIMENSION,
                message=f"Type de palier non autorisé : {ttype!r}.",
            )

    overrides = company_policy.get("reason_overrides") or {}
    if overrides and not isinstance(overrides, dict):
        return ChannelCapValidationResult(
            ok=False,
            error=ERROR_DIMENSION,
            message="reason_overrides invalide.",
        )
    if isinstance(overrides, dict):
        for code, meta in overrides.items():
            code_u = str(code).upper()
            if code_u not in allowed_reasons:
                return ChannelCapValidationResult(
                    ok=False,
                    error=ERROR_DIMENSION,
                    message=f"Motif {code_u!r} hors dimensions canal.",
                )
            if not isinstance(meta, dict):
                return ChannelCapValidationResult(
                    ok=False,
                    error=ERROR_DIMENSION,
                    message=f"Override motif {code_u} invalide.",
                )
            extra = set(meta.keys()) - {"billable"}
            if extra:
                return ChannelCapValidationResult(
                    ok=False,
                    error=ERROR_DIMENSION,
                    message=(
                        f"Frais hors cadre pour motif {code_u} : "
                        + ", ".join(sorted(extra))
                    ),
                )

    return ChannelCapValidationResult(ok=True)


@dataclass(frozen=True, slots=True)
class PublishChannelPolicyResult:
    ok: bool
    policy: LirieChannelCancellationPolicy | None = None
    error: str | None = None


def publish_channel_cancellation_policy(
    *,
    body_json: dict[str, Any],
    version: str | None = None,
) -> PublishChannelPolicyResult:
    """Publie une nouvelle version courante des caps canal (tests / admin)."""
    if not isinstance(body_json, dict) or not body_json.get("dimensions"):
        return PublishChannelPolicyResult(
            ok=False, error="body_json.dimensions requis"
        )
    text = render_channel_policy_text(body_json)
    content_hash = _hash_body(
        json.dumps(body_json, sort_keys=True, ensure_ascii=False) + "\n" + text
    )
    count = LirieChannelCancellationPolicy.query.count()
    ver = (version or f"v{int(count) + 1}").strip()
    for row in LirieChannelCancellationPolicy.query.filter_by(is_current=True).all():
        row.is_current = False
    row = LirieChannelCancellationPolicy(
        version=ver,
        body_json=body_json,
        body_text=text,
        content_hash=content_hash,
        is_current=True,
        effective_at=now_utc(),
    )
    db.session.add(row)
    db.session.flush()
    return PublishChannelPolicyResult(ok=True, policy=row)


def synthetic_test_channel_caps() -> dict[str, Any]:
    """Fixture de test uniquement — pas de valeurs prod inventées au boot."""
    return {
        "dimensions": [
            {
                "kind": "time",
                "hours_before_gte": 24,
                "hours_before_lt": None,
                "max_percent": 0,
                "label": "Plus de 24 h avant départ",
            },
            {
                "kind": "time",
                "hours_before_gte": 4,
                "hours_before_lt": 24,
                "max_percent": 50,
                "label": "Entre 4 h et 24 h",
            },
            {
                "kind": "time",
                "hours_before_gte": 0,
                "hours_before_lt": 4,
                "max_percent": 100,
                "label": "Moins de 4 h",
            },
            {
                "kind": "status",
                "status": "EN_ROUTE",
                "max_percent": 100,
                "label": "Chauffeur en route",
            },
            {
                "kind": "reason",
                "code": "NO_SHOW",
                "max_percent": 100,
                "label": "No-show",
            },
            {
                "kind": "reason",
                "code": "COMPANY_ISSUE",
                "max_percent": 0,
                "label": "Annulation imputable au transporteur",
            },
            {
                "kind": "reason",
                "code": "CLIENT_REQUEST",
                "max_percent": 100,
                "label": "Demande client",
            },
            {
                "kind": "reason",
                "code": "LAST_MINUTE",
                "max_percent": 100,
                "label": "Dernière minute",
            },
            {
                "kind": "reason",
                "code": "MAJOR_DELAY",
                "max_percent": 100,
                "label": "Retard important",
            },
            {
                "kind": "reason",
                "code": "VEHICLE_ISSUE",
                "max_percent": 0,
                "label": "Problème véhicule",
            },
            {
                "kind": "reason",
                "code": "OTHER",
                "max_percent": 100,
                "label": "Autre",
            },
        ],
        "allow_absolute_fee_bounds": True,
        "allow_driver_assigned_gate": True,
    }
