"""Politiques d'annulation PORTAL versionnées par entreprise.

Source de vérité opérationnelle : ``CompanyBillingSettings.cancellation_policy``
(même JSON que l'éditeur « Frais d'annulation »).

Publication PORTAL : snapshot texte client + hash immuable dans
``company_portal_cancellation_policy`` (append-only, ``is_current``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ext import db
from models.company_portal_cancellation_policy import CompanyPortalCancellationPolicy
from shared.time_utils import now_utc

# Aligné sur frontend CancellationPolicyEditor + cancellation_rules.
_REASON_LABELS: dict[str, str] = {
    "LAST_MINUTE": "Annulation dernière minute",
    "NO_SHOW": "Client ne s'est pas présenté",
    "CLIENT_REQUEST": "Client a demandé l'annulation",
    "COMPANY_ISSUE": "Problème entreprise",
    "MAJOR_DELAY": "Retard important",
    "VEHICLE_ISSUE": "Problème véhicule",
    "OTHER": "Autre raison",
}


def _hash_body(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PublishPolicyResult:
    ok: bool
    policy: CompanyPortalCancellationPolicy | None = None
    error: str | None = None


def get_current_cancellation_policy(
    company_id: int,
) -> CompanyPortalCancellationPolicy | None:
    return (
        CompanyPortalCancellationPolicy.query.filter_by(
            company_id=int(company_id), is_current=True
        )
        .order_by(CompanyPortalCancellationPolicy.id.desc())
        .first()
    )


def _next_version_label(company_id: int) -> str:
    count = CompanyPortalCancellationPolicy.query.filter_by(
        company_id=int(company_id)
    ).count()
    return f"v{int(count) + 1}"


def load_company_cancellation_config(company_id: int) -> dict[str, Any] | None:
    """JSON éditeur frais d'annulation (peut être None ou ``enabled: false``)."""
    from models.invoice import CompanyBillingSettings

    billing = CompanyBillingSettings.query.filter_by(company_id=int(company_id)).first()
    if billing is None:
        return None
    raw = getattr(billing, "cancellation_policy", None)
    return raw if isinstance(raw, dict) else None


def render_portal_cancellation_policy_text(
    *,
    company_name: str,
    policy: dict[str, Any] | None,
) -> str:
    """Texte client dérivé de la config — ordre aligné sur ``compute_cancellation_fee``.

    Précédence moteur (résumé) :
    1. politique désactivée / absente → pas de frais paramétrés ;
    2. overrides motif (non facturable) avant paliers ;
    3. gate « chauffeur assigné » ;
    4. palier statut EN_ROUTE, sinon premier palier temps (heures < seuil).
    """
    name = (company_name or "l'entreprise de transport").strip() or (
        "l'entreprise de transport"
    )
    lines: list[str] = [
        f"Conditions d'annulation de {name}",
        "",
    ]

    if not policy or not policy.get("enabled"):
        lines.extend(
            [
                "Selon la configuration actuelle de l'entreprise, aucun frais "
                "d'annulation paramétrable n'est applicable via cette politique.",
                "",
                "Des règles particulières (par exemple no-show) peuvent toutefois "
                "s'appliquer selon le motif d'annulation.",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    if policy.get("apply_when_driver_assigned_only", True):
        lines.append(
            "Les frais d'annulation ne s'appliquent que lorsqu'un chauffeur "
            "a été assigné à la course."
        )
        lines.append("")

    time_tiers = sorted(
        [t for t in (policy.get("tiers") or []) if t.get("type") == "time"],
        key=lambda t: float(t.get("hours_before") or 0),
    )
    status_tiers = [
        t for t in (policy.get("tiers") or []) if t.get("type") == "status"
    ]

    if time_tiers:
        lines.append("Paliers selon le délai avant le départ :")
        for t in time_tiers:
            try:
                hours = float(t.get("hours_before") or 0)
                pct = int(t.get("percent") or 0)
            except (TypeError, ValueError):
                continue
            label = (t.get("label") or "").strip() or f"Moins de {int(hours)} heures"
            lines.append(
                f"- {label} avant le départ : {pct} % du prix du transport."
            )
        lines.append("")

    en_route = next(
        (
            t
            for t in status_tiers
            if str(t.get("status") or "").upper() == "EN_ROUTE"
        ),
        None,
    )
    if en_route is not None:
        try:
            pct = int(en_route.get("percent") or 0)
        except (TypeError, ValueError):
            pct = 0
        lines.append(
            f"Lorsque le chauffeur est en route : {pct} % du prix du transport."
        )
        lines.append("")

    min_fee = policy.get("min_fee_chf")
    max_fee = policy.get("max_fee_chf")
    bounds: list[str] = []
    try:
        if min_fee is not None and float(min_fee) > 0:
            bounds.append(f"minimum {float(min_fee):.2f} CHF")
    except (TypeError, ValueError):
        pass
    try:
        if max_fee is not None and str(max_fee).strip() != "" and float(max_fee) > 0:
            bounds.append(f"maximum {float(max_fee):.2f} CHF")
    except (TypeError, ValueError):
        pass
    if bounds:
        lines.append("Plafonds de frais : " + ", ".join(bounds) + ".")
        lines.append("")

    overrides = policy.get("reason_overrides") or {}
    if isinstance(overrides, dict) and overrides:
        lines.append("Exceptions selon le motif d'annulation (prioritaires) :")
        for code, meta in overrides.items():
            if not isinstance(meta, dict):
                continue
            label = _REASON_LABELS.get(str(code).upper(), str(code))
            billable = bool(meta.get("billable", True))
            lines.append(
                f"- {label} : "
                + ("facturable selon les paliers." if billable else "non facturable.")
            )
        lines.append("")

    lines.append(
        "Ces conditions correspondent à la configuration d'annulation de "
        "l'entreprise au moment de la publication. Une modification ultérieure "
        "de la configuration n'altère pas la version déjà acceptée."
    )
    return "\n".join(lines).strip() + "\n"


def get_portal_policy_publication_status(
    *,
    company_id: int,
    company_name: str,
) -> dict[str, Any]:
    """État publication pour l'UI paramètres facturation."""
    config = load_company_cancellation_config(company_id)
    draft_text = render_portal_cancellation_policy_text(
        company_name=company_name, policy=config
    )
    draft_hash = _hash_body(draft_text)
    current = get_current_cancellation_policy(company_id)
    published = None
    if current is not None:
        published = {
            "id": int(current.id),
            "version": current.version,
            "body_text": current.body_text,
            "content_hash": current.content_hash,
            "effective_at": current.effective_at.isoformat()
            if current.effective_at
            else None,
        }
    # Compare au corps « métier » sans pied de ref config (publication ajoute un pied).
    published_core = (published or {}).get("body_text") or ""
    if "\n\n[Réf. config " in published_core:
        published_core = published_core.split("\n\n[Réf. config ", 1)[0].rstrip() + "\n"
    has_unpublished = current is None or _hash_body(published_core) != draft_hash
    return {
        "config_enabled": bool(config and config.get("enabled")),
        "config_present": config is not None,
        "draft_preview": draft_text,
        "draft_content_hash": draft_hash,
        "policy": published,
        "has_unpublished_changes": has_unpublished,
        "next_version": _next_version_label(company_id),
    }


def publish_cancellation_policy(
    *,
    company_id: int,
    version: str,
    body_text: str,
) -> PublishPolicyResult:
    """Publie une nouvelle version courante (append-only des anciennes)."""
    body = (body_text or "").strip()
    ver = (version or "").strip()
    if not body or not ver:
        return PublishPolicyResult(ok=False, error="version_and_body_required")

    if not body.endswith("\n"):
        body = body + "\n"
    content_hash = _hash_body(body)

    CompanyPortalCancellationPolicy.query.filter_by(
        company_id=int(company_id), is_current=True
    ).update({"is_current": False}, synchronize_session=False)

    row = CompanyPortalCancellationPolicy(
        company_id=int(company_id),
        version=ver,
        body_text=body,
        content_hash=content_hash,
        is_current=True,
        effective_at=now_utc(),
    )
    db.session.add(row)
    db.session.flush()
    return PublishPolicyResult(ok=True, policy=row)


def publish_portal_policy_from_billing_settings(
    *,
    company_id: int,
    company_name: str,
) -> PublishPolicyResult:
    """Publie le texte client dérivé de la config frais d'annulation actuelle.

    Même si les frais sont désactivés : une version explicite « aucun frais
    paramétrable » est créée (policy missing ≠ zero fees).

    Si une politique canal LIRIE est publiée : validation stricte
    company_rate <= cap + dimensions ⊆ canal (jamais min silencieux).
    """
    config = load_company_cancellation_config(company_id)

    from services.legal.portal_channel_cancellation_caps import (
        get_current_channel_cancellation_policy,
        validate_company_policy_against_channel,
    )

    channel = get_current_channel_cancellation_policy()
    if channel is not None and isinstance(channel.body_json, dict):
        cap_check = validate_company_policy_against_channel(
            company_policy=config if isinstance(config, dict) else None,
            channel_body=channel.body_json,
        )
        if not cap_check.ok:
            return PublishPolicyResult(
                ok=False,
                error=str(cap_check.error or "portal_cancellation_cap_rejected"),
            )

    body = render_portal_cancellation_policy_text(
        company_name=company_name, policy=config
    )
    version = _next_version_label(company_id)
    try:
        cfg_raw = json.dumps(config or {"enabled": False}, sort_keys=True, default=str)
        cfg_hash = hashlib.sha256(cfg_raw.encode("utf-8")).hexdigest()[:16]
        body = body.rstrip() + f"\n\n[Réf. config {cfg_hash}]\n"
    except Exception:
        pass
    return publish_cancellation_policy(
        company_id=int(company_id),
        version=version,
        body_text=body,
    )
