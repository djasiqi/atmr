"""Règles d'annulation standardisées : motifs, facturation, libellés.

Backend = source of truth. Utilisé par company web/mobile, driver, invoice, PDF.

Phase 1 : pas de validation stricte (LAST_MINUTE sélectionnable manuellement).
Phase 2 : seuil configurable (ex. 2h) + validation/override.

Référence – Choix proposés au chauffeur (mobile CancelJustificationModal) :
┌──────────────────┬─────────────────────────────────────┬────────────┐
│ Code (backend)   │ Libellé                             │ Facturable │
├──────────────────┼─────────────────────────────────────┼────────────┤
│ LAST_MINUTE      │ Annulation dernière minute          │ Oui        │
│ NO_SHOW          │ Client ne s'est pas présenté        │ Oui        │
│ CLIENT_REQUEST   │ Client a demandé l'annulation       │ Oui        │
├──────────────────┼─────────────────────────────────────┼────────────┤
│ COMPANY_ISSUE    │ Problème entreprise                 │ Non        │
│ MAJOR_DELAY      │ Retard important (mobile: DELAY)    │ Non        │
│ VEHICLE_ISSUE    │ Problème véhicule                   │ Non        │
│ OTHER            │ Autre raison                        │ Non        │
└──────────────────┴─────────────────────────────────────┴────────────┘
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Mapping

logger = logging.getLogger("cancellation")

# Libellés affichés dans facture/PDF (alignés avec le modal chauffeur)
CANCELLATION_REASON_LABELS: dict[str, str] = {
    "LAST_MINUTE": "Annulation dernière minute",
    "NO_SHOW": "Client ne s'est pas présenté",
    "CLIENT_REQUEST": "Client a demandé l'annulation",
    "COMPANY_ISSUE": "Problème entreprise",
    "MAJOR_DELAY": "Retard important",
    "VEHICLE_ISSUE": "Problème véhicule",
    "OTHER": "Autre raison",
}

# Motifs facturables (annulation facturée à la clinique/client)
# = choix chauffeur où isClientFault / "Facturation prévue" dans l'app
BILLABLE_REASONS: frozenset[str] = frozenset(
    {"LAST_MINUTE", "NO_SHOW", "CLIENT_REQUEST"}
)

# Mapping transition : anciens codes mobile / UI → codes facturation
LEGACY_REASON_MAP: dict[str, str] = {
    "OPERATOR_CANCELLED": "COMPANY_ISSUE",
    "CANCEL": "OTHER",
    "CLIENT_NO_SHOW": "NO_SHOW",  # mobile CancelJustificationModal
    "DELAY": "MAJOR_DELAY",  # mobile option id
}


def _normalize_reason_code(code: str | None) -> str:
    """Normalise le code (uppercase, mapping legacy)."""
    if not code or not str(code).strip():
        return "OTHER"
    raw = str(code).strip().upper()
    return LEGACY_REASON_MAP.get(raw, raw)


def is_cancellation_billable(reason_code: str | None) -> bool:
    """Indique si l'annulation est facturable selon le motif."""
    code = _normalize_reason_code(reason_code)
    return code in BILLABLE_REASONS


def get_cancellation_display_label(
    reason_code: str | None,
    reason_text: str | None = None,
) -> str:
    """Retourne le libellé exact à afficher dans facture/PDF."""
    # None/vide = annulation historique (avant motif obligatoire)
    if not reason_code or not str(reason_code).strip():
        return "Annulation (historique)"
    code = _normalize_reason_code(reason_code)
    if code == "OTHER" and reason_text and str(reason_text).strip():
        # Tronquer à 80 caractères pour cohérence DB
        text = str(reason_text).strip()[:80]
        return f"Annulation – {text}"
    return CANCELLATION_REASON_LABELS.get(code, "Annulation (historique)")


def compute_cancellation_fields(
    *,
    reason_code: str | None,
    reason_text: str | None,
    cancelled_by_role: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Calcule tous les champs à persister sur le booking à l'annulation.

    Returns:
        Dict avec: cancelled_at, cancelled_by_role, cancellation_reason_code,
        cancellation_reason_text, is_cancellation_billable, cancellation_display_label
    """
    now_utc = now or datetime.now(UTC)
    code = _normalize_reason_code(reason_code)
    # reason_code None/vide → "Annulation (historique)" (legacy)
    label = (
        get_cancellation_display_label(None, None)
        if not reason_code or not str(reason_code).strip()
        else get_cancellation_display_label(code, reason_text)
    )
    billable = is_cancellation_billable(code)

    reason_text_val = None
    if reason_text and str(reason_text).strip():
        reason_text_val = str(reason_text).strip()[:500]  # Limite DB

    return {
        "cancelled_at": now_utc,
        "cancelled_by_role": cancelled_by_role,
        "cancellation_reason_code": code,
        "cancellation_reason_text": reason_text_val,
        "is_cancellation_billable": billable,
        "cancellation_display_label": label,
    }


def get_all_reason_codes() -> list[str]:
    """Liste des codes valides (pour validation/sélecteur UI)."""
    return list(CANCELLATION_REASON_LABELS.keys())


def log_cancellation_persisted(
    booking: Any, cancel_fields: Mapping[str, Any]
) -> None:
    """Log structuré après persistance d'une annulation (observabilité / litiges).

    À appeler une fois par annulation, uniquement quand on vient d'écrire les 6 champs
    (évite double log sur retries idempotents).
    Pas de PII : pas de reason_text (peut contenir infos sensibles si OTHER).
    company_id / tenant_id pour filtrer par tenant en multi-tenant.
    pickup_time toujours en UTC (timezone-aware).
    """
    booking_id = getattr(booking, "id", None)
    company_id = getattr(booking, "company_id", None)
    scheduled_time = getattr(booking, "scheduled_time", None)
    if not scheduled_time:
        pickup_time_str: str | None = None
    elif hasattr(scheduled_time, "astimezone"):
        try:
            if getattr(scheduled_time, "tzinfo", None) is None:
                # Naive datetime : on force UTC pour cohérence du log (observabilité, pas calcul métier)
                pickup_time_str = scheduled_time.replace(tzinfo=UTC).isoformat()
            else:
                pickup_time_str = scheduled_time.astimezone(UTC).isoformat()
        except Exception:
            pickup_time_str = (
                scheduled_time.isoformat()
                if hasattr(scheduled_time, "isoformat")
                else str(scheduled_time)
            )
    else:
        pickup_time_str = str(scheduled_time)
    logger.info(
        "[cancellation] booking_id=%s company_id=%s cancelled_by_role=%s reason_code=%s "
        "is_billable=%s label=%s pickup_time=%s",
        booking_id,
        company_id,
        cancel_fields.get("cancelled_by_role"),
        cancel_fields.get("cancellation_reason_code"),
        cancel_fields.get("is_cancellation_billable"),
        cancel_fields.get("cancellation_display_label"),
        pickup_time_str,
    )
