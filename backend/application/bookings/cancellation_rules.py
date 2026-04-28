"""Règles d'annulation standardisées : motifs, facturation, libellés.

Backend = source of truth. Utilisé par company web/mobile, driver, invoice, PDF.

Phase 1 : pas de validation stricte (LAST_MINUTE sélectionnable manuellement).
Phase 2 : seuil configurable (ex. 2h) + validation/override.

ÉTAPE 5 - Règles statut:
┌────────────────────┬─────────────────────────────────────┬────────────┐
│ Statut Booking     │ Règle                               │ Frais      │
├────────────────────┼─────────────────────────────────────┼────────────┤
│ PENDING/ACCEPTED   │ Annulation libre                    │ Selon motif│
│ ASSIGNED           │ Annulation libre                    │ Selon motif│
│ EN_ROUTE           │ Chauffeur en route → frais dus      │ Oui        │
│ IN_PROGRESS        │ Course en cours → frais majorés     │ Oui (100%) │
│ COMPLETED          │ Non annulable                       │ N/A        │
└────────────────────┴─────────────────────────────────────┴────────────┘

Reference - Choix proposes au chauffeur (mobile CancelJustificationModal) :
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
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
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
    "OUTBOUND_CANCELLED": "Retour annulé (aller annulé)",
    "PAYMENT_TIMEOUT": "Paiement en ligne non finalisé dans le délai (15 min)",
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


@dataclass(frozen=True, slots=True)
class CancellationFeeResult:
    """Result of cancellation fee computation."""

    is_billable: bool
    percent: int | None
    tier_id: str | None
    fee_amount: Decimal
    fee_label: str


_ZERO = Decimal("0")


def compute_cancellation_fee(  # noqa: PLR0911
    booking: Any,
    *,
    status_at_cancel: str,
    cancelled_at: datetime,
    reason_code: str | None,
    cancel_source: str | None = None,
    policy: dict[str, Any] | None = None,
) -> CancellationFeeResult:
    """Compute cancellation fee based on company policy.

    Args:
        booking: Booking ORM instance (needs .amount, .driver_id, .scheduled_time)
        status_at_cancel: booking.status captured BEFORE setting CANCELED
        cancelled_at: timestamp of cancellation
        reason_code: raw reason code (will be normalized)
        cancel_source: "cascade_from_outbound" for R5, else None
        policy: CompanyBillingSettings.cancellation_policy JSON dict, or None

    Returns:
        CancellationFeeResult with is_billable = (fee_amount > 0)
    """
    code = _normalize_reason_code(reason_code)

    if cancel_source == "cascade_from_outbound":
        return CancellationFeeResult(
            is_billable=False, percent=None, tier_id=None,
            fee_amount=_ZERO, fee_label="Cascade A/R",
        )

    if not policy or not policy.get("enabled"):
        return CancellationFeeResult(
            is_billable=(code in BILLABLE_REASONS),
            percent=None, tier_id=None,
            fee_amount=_ZERO, fee_label="Legacy",
        )

    overrides = policy.get("reason_overrides") or {}
    if code in overrides:
        if not overrides[code].get("billable", True):
            return CancellationFeeResult(
                is_billable=False, percent=None, tier_id=None,
                fee_amount=_ZERO, fee_label="Override non facturable",
            )
    elif code not in BILLABLE_REASONS:
        return CancellationFeeResult(
            is_billable=False, percent=None, tier_id=None,
            fee_amount=_ZERO, fee_label="Motif non facturable",
        )

    if policy.get("apply_when_driver_assigned_only") and not getattr(booking, "driver_id", None):
        return CancellationFeeResult(
            is_billable=False, percent=None, tier_id=None,
            fee_amount=_ZERO, fee_label="Pas de chauffeur assigné",
        )

    amount = getattr(booking, "amount", None)
    if amount is None or float(amount) <= 0:
        return CancellationFeeResult(
            is_billable=False, percent=None, tier_id=None,
            fee_amount=_ZERO, fee_label="Tarif non défini",
        )

    tiers = policy.get("tiers") or []
    selected_tier: dict[str, Any] | None = None
    status_upper = (status_at_cancel or "").upper()

    if status_upper == "EN_ROUTE":
        selected_tier = next(
            (t for t in tiers if t.get("type") == "status" and t.get("status") == "EN_ROUTE"),
            None,
        )
    else:
        scheduled = getattr(booking, "scheduled_time", None)
        if scheduled:
            delta = (scheduled - cancelled_at).total_seconds() / 3600
            hours_before = max(0.0, delta)
        else:
            hours_before = 0.0

        time_tiers = [t for t in tiers if t.get("type") == "time"]
        for t in time_tiers:
            if hours_before < t["hours_before"]:
                selected_tier = t
                break

    if selected_tier is None:
        return CancellationFeeResult(
            is_billable=False, percent=None, tier_id=None,
            fee_amount=_ZERO, fee_label="Aucun palier applicable",
        )

    base = Decimal(str(amount))
    pct = selected_tier["percent"]
    fee = base * pct / 100

    min_fee = Decimal(str(policy.get("min_fee_chf") or 0))
    max_fee_raw = policy.get("max_fee_chf")
    fee = max(fee, min_fee)
    if max_fee_raw is not None:
        fee = min(fee, Decimal(str(max_fee_raw)))

    fee = fee.quantize(Decimal("0.01"))

    return CancellationFeeResult(
        is_billable=(fee > 0),
        percent=pct,
        tier_id=selected_tier.get("id"),
        fee_amount=fee,
        fee_label=selected_tier.get("label") or f"< {selected_tier.get('hours_before', '?')}h",
    )


def compute_cancellation_fields(
    *,
    reason_code: str | None,
    reason_text: str | None,
    cancelled_by_role: str,
    now: datetime | None = None,
    booking: Any | None = None,
    policy: dict[str, Any] | None = None,
    cancel_source: str | None = None,
    status_at_cancel: str | None = None,
) -> dict[str, Any]:
    """Calcule tous les champs à persister sur le booking à l'annulation.

    Returns:
        Dict avec: cancelled_at, cancelled_by_role, cancellation_reason_code,
        cancellation_reason_text, is_cancellation_billable, cancellation_display_label,
        and optionally cancellation_fee_amount, cancellation_fee_percent, cancellation_fee_tier_id
    """
    now_utc = now or datetime.now(UTC)
    code = _normalize_reason_code(reason_code)
    label = (
        get_cancellation_display_label(None, None)
        if not reason_code or not str(reason_code).strip()
        else get_cancellation_display_label(code, reason_text)
    )
    billable = is_cancellation_billable(code)

    reason_text_val = None
    if reason_text and str(reason_text).strip():
        reason_text_val = str(reason_text).strip()[:500]

    fields: dict[str, Any] = {
        "cancelled_at": now_utc,
        "cancelled_by_role": cancelled_by_role,
        "cancellation_reason_code": code,
        "cancellation_reason_text": reason_text_val,
        "is_cancellation_billable": billable,
        "cancellation_display_label": label,
    }

    if booking:
        fee_result = compute_cancellation_fee(
            booking,
            status_at_cancel=status_at_cancel or getattr(booking, "status", "") or "",
            cancelled_at=now_utc,
            reason_code=reason_code,
            cancel_source=cancel_source,
            policy=policy,
        )
        fields["is_cancellation_billable"] = fee_result.is_billable
        fields["cancellation_fee_amount"] = fee_result.fee_amount
        fields["cancellation_fee_percent"] = fee_result.percent
        fields["cancellation_fee_tier_id"] = fee_result.tier_id

    return fields


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
        "[cancellation] booking_id=%s company_id=%s cancelled_by_role=%s reason_code=%s is_billable=%s label=%s pickup_time=%s",
        booking_id,
        company_id,
        cancel_fields.get("cancelled_by_role"),
        cancel_fields.get("cancellation_reason_code"),
        cancel_fields.get("is_cancellation_billable"),
        cancel_fields.get("cancellation_display_label"),
        pickup_time_str,
    )


# ========== ÉTAPE 5: Règles annulation basées sur le statut booking ==========

# Statuts où l'annulation est automatiquement facturable (chauffeur déjà mobilisé)
STATUS_ALWAYS_BILLABLE: frozenset[str] = frozenset({"EN_ROUTE", "IN_PROGRESS"})

# Statuts où l'annulation est impossible (course terminée)
STATUS_NOT_CANCELLABLE: frozenset[str] = frozenset({
    "COMPLETED",
    "RETURN_COMPLETED",
    "CANCELED",  # Déjà annulé
})

# Statuts annulables librement (selon motif)
STATUS_FREE_CANCELLATION: frozenset[str] = frozenset({
    "PENDING",
    "ACCEPTED",
    "ASSIGNED",
})


def is_booking_cancellable(status: str | None) -> bool:
    """Vérifie si un booking peut être annulé selon son statut.

    ÉTAPE 5:
    - PENDING, ACCEPTED, ASSIGNED, EN_ROUTE, IN_PROGRESS → annulable
    - COMPLETED, RETURN_COMPLETED, CANCELED → non annulable

    Args:
        status: Statut actuel du booking (BookingStatus value)

    Returns:
        True si annulable, False sinon
    """
    if not status:
        return True  # Cas edge, on permet
    status_upper = status.upper()
    return status_upper not in STATUS_NOT_CANCELLABLE


def is_status_billable_cancellation(status: str | None) -> bool:
    """Vérifie si l'annulation est automatiquement facturable selon le statut.

    ÉTAPE 5:
    - EN_ROUTE → chauffeur en route, frais dus (déplacement engagé)
    - IN_PROGRESS → course en cours, frais dus (100%)

    Args:
        status: Statut actuel du booking

    Returns:
        True si l'annulation doit forcément être facturée
    """
    if not status:
        return False
    return status.upper() in STATUS_ALWAYS_BILLABLE


def get_cancellation_billing_info(
    status: str | None,
    reason_code: str | None,
) -> dict[str, Any]:
    """Calcule les informations de facturation pour une annulation.

    ÉTAPE 5 - Logique:
    1. Si statut EN_ROUTE ou IN_PROGRESS → facturable (override motif)
    2. Sinon → facturable selon motif (LAST_MINUTE, NO_SHOW, etc.)

    Args:
        status: Statut actuel du booking
        reason_code: Code motif d'annulation

    Returns:
        Dict avec is_billable, billing_reason, surcharge_percent
    """
    status_upper = (status or "").upper()

    # EN_ROUTE: frais dus (déplacement engagé)
    if status_upper == "EN_ROUTE":
        return {
            "is_billable": True,
            "billing_reason": "status_en_route",
            "billing_description": "Annulation après départ chauffeur - frais de déplacement",
            "surcharge_percent": 0,  # Pas de majoration, juste frais standard
        }

    # IN_PROGRESS: course en cours, frais à 100%
    if status_upper == "IN_PROGRESS":
        return {
            "is_billable": True,
            "billing_reason": "status_in_progress",
            "billing_description": "Annulation course en cours - facturation intégrale",
            "surcharge_percent": 100,  # Majoration 100% (course complète)
        }

    # Autres statuts: selon le motif
    billable = is_cancellation_billable(reason_code)
    return {
        "is_billable": billable,
        "billing_reason": "reason_code" if billable else "none",
        "billing_description": None,
        "surcharge_percent": 0,
    }


def compute_cancellation_fields_with_status(
    *,
    booking_status: str | None,
    reason_code: str | None,
    reason_text: str | None,
    cancelled_by_role: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Calcule tous les champs d'annulation en tenant compte du statut.

    ÉTAPE 5: Extension de compute_cancellation_fields pour forcer
    is_cancellation_billable si le statut l'impose.

    Returns:
        Dict avec tous les champs + billing_info
    """
    # Calculer les champs de base
    base_fields = compute_cancellation_fields(
        reason_code=reason_code,
        reason_text=reason_text,
        cancelled_by_role=cancelled_by_role,
        now=now,
    )

    # Obtenir les infos de facturation basées sur le statut
    billing_info = get_cancellation_billing_info(booking_status, reason_code)

    # Override is_cancellation_billable si le statut l'impose
    if billing_info["is_billable"] and not base_fields["is_cancellation_billable"]:
        base_fields["is_cancellation_billable"] = True
        # Adapter le label si besoin
        if billing_info["billing_description"]:
            base_fields["cancellation_display_label"] = billing_info["billing_description"]

    # Ajouter les infos de billing
    base_fields["billing_info"] = billing_info

    return base_fields
