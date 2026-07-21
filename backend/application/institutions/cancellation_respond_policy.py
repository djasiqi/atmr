"""CancellationRespondPolicy — applicabilité + éligibilité commerciale (OUTBOUND_ONLY).

Référence : docs/domain/transport-decision-workflow.md §13
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from models import Booking
from models.booking_change_request import BookingChangeRequest, TransportActionType
from models.enums import BookingStatus

logger = logging.getLogger(__name__)

_ZERO = Decimal("0.00")
_CENT = Decimal("0.01")
DEFAULT_FREE_THRESHOLD_HOURS = 24
CURRENCY = "CHF"

TERMINAL_STATUSES = frozenset(
    {
        BookingStatus.COMPLETED.value,
        BookingStatus.RETURN_COMPLETED.value,
        BookingStatus.CANCELED.value,
        "CANCELLED",
        "cancelled",
        "canceled",
        "completed",
        "return_completed",
    }
)


class TripLeg(StrEnum):
    OUTBOUND = "OUTBOUND"
    RETURN = "RETURN"


class CancellationSituation(StrEnum):
    NON_BILLABLE_RETURN = "NON_BILLABLE_RETURN"
    FREE_WINDOW = "FREE_WINDOW"
    FEE_WINDOW = "FEE_WINDOW"
    EN_ROUTE = "EN_ROUTE"


class BillingOutcome(StrEnum):
    ZERO = "ZERO"
    POLICY_FEE = "POLICY_FEE"
    APPROACH_FEE = "APPROACH_FEE"
    FULL_FARE = "FULL_FARE"
    CUSTOM = "CUSTOM"


class BillingScope(StrEnum):
    OUTBOUND_ONLY = "OUTBOUND_ONLY"
    NONE = "NONE"


class PrimaryCta(StrEnum):
    ACKNOWLEDGE_CANCELLATION = "acknowledge_cancellation"
    CONFIRM_WITH_BILLING = "confirm_with_billing"


class SecondaryCta(StrEnum):
    REPORT_PROBLEM = "report_problem"


class CalculationCode(StrEnum):
    NON_BILLABLE_RETURN = "NON_BILLABLE_RETURN"
    FREE_CANCELLATION_WINDOW = "FREE_CANCELLATION_WINDOW"
    COMPANY_WAIVED = "COMPANY_WAIVED"
    POLICY_TIER_FEE = "POLICY_TIER_FEE"
    APPROACH_FEE = "APPROACH_FEE"
    FULL_OUTBOUND_FARE = "FULL_OUTBOUND_FARE"
    COMPANY_CUSTOM_FEE = "COMPANY_CUSTOM_FEE"


class CancellationRespondErrorCode(StrEnum):
    INTERRUPTION_REQUIRED = "INTERRUPTION_REQUIRED"
    BILLING_OUTCOME_REQUIRED = "BILLING_OUTCOME_REQUIRED"
    BILLING_OUTCOME_NOT_ALLOWED = "BILLING_OUTCOME_NOT_ALLOWED"
    FEE_AMOUNT_NOT_ALLOWED = "FEE_AMOUNT_NOT_ALLOWED"
    CUSTOM_FEE_AMOUNT_REQUIRED = "CUSTOM_FEE_AMOUNT_REQUIRED"
    CUSTOM_FEE_AMOUNT_INVALID = "CUSTOM_FEE_AMOUNT_INVALID"
    CUSTOM_FEE_AMOUNT_OUT_OF_RANGE = "CUSTOM_FEE_AMOUNT_OUT_OF_RANGE"
    BILLING_COMMENT_REQUIRED = "BILLING_COMMENT_REQUIRED"
    CANCELLATION_RESPONSE_CONTEXT_CHANGED = "CANCELLATION_RESPONSE_CONTEXT_CHANGED"


class ContextChangeReason(StrEnum):
    POLICY_CHANGED = "POLICY_CHANGED"
    TIME_WINDOW_CHANGED = "TIME_WINDOW_CHANGED"
    BOOKING_STATUS_CHANGED = "BOOKING_STATUS_CHANGED"
    SCOPE_CHANGED = "SCOPE_CHANGED"
    AMOUNT_CHANGED = "AMOUNT_CHANGED"
    AFFECTED_BOOKINGS_CHANGED = "AFFECTED_BOOKINGS_CHANGED"


class CancellationRespondError(Exception):
    """Erreur métier de réponse commerciale à une annulation."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 422,
        change_reason: str | None = None,
        respond_ui: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.change_reason = change_reason
        self.respond_ui = respond_ui
        self.extra = extra or {}


@dataclass(frozen=True, slots=True)
class FeeQuote:
    amount: Decimal
    currency: str = CURRENCY
    calculation_code: str = ""
    calculation_basis: dict[str, Any] = field(default_factory=dict)
    percent: int | None = None
    tier_id: str | None = None


@dataclass
class CancellationRespondContext:
    situation: str
    affected_booking_ids: list[int]
    cancelable_booking_ids: list[int]
    non_cancelable_booking_ids: list[int]
    non_billable_booking_ids: list[int]
    billing_eligible_booking_id: int | None
    billing_eligible_leg: str | None
    billing_scope: str
    suggested_outcome: dict[str, Any]
    allowed_outcomes: list[dict[str, Any]]
    primary_cta: str
    secondary_cta: str
    policy_version: str
    respond_context_version: int
    seconds_before_departure: int | None
    threshold_seconds: int
    departure_time_passed: bool
    has_in_progress: bool = False
    # objets internes (non sérialisés tels quels)
    affected_bookings: list[Booking] = field(default_factory=list, repr=False)
    cancelable_bookings: list[Booking] = field(default_factory=list, repr=False)
    billing_eligible_booking: Booking | None = field(default=None, repr=False)
    policy: dict[str, Any] | None = field(default=None, repr=False)


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status) or "").upper()


def _money_str(amount: Decimal) -> str:
    return f"{amount.quantize(_CENT):.2f}"


def resolve_trip_leg(booking: Booking) -> TripLeg:
    """OUTBOUND | RETURN — résolution centralisée."""
    if bool(getattr(booking, "is_return", False)):
        return TripLeg.RETURN
    if getattr(booking, "parent_booking_id", None):
        return TripLeg.RETURN
    if bool(getattr(booking, "is_return_stop", False)):
        return TripLeg.RETURN
    return TripLeg.OUTBOUND


def is_outbound_leg(booking: Booking) -> bool:
    return resolve_trip_leg(booking) == TripLeg.OUTBOUND


def clear_cancellation_billing(booking: Booking) -> None:
    """Remet à zéro les champs frais d'annulation (cancelables non éligibles)."""
    booking.is_cancellation_billable = False
    booking.cancellation_fee_amount = _ZERO
    booking.cancellation_fee_percent = None
    booking.cancellation_fee_tier_id = None


def resolve_affected_bookings(
    booking: Booking,
    action: BookingChangeRequest | None = None,
) -> list[Booking]:
    """Tous les bookings visés par la demande (source + liés + aller si initié depuis retour)."""
    from services.institutions.booking_change_service import _collect_linked_bookings

    by_id: dict[int, Booking] = {int(booking.id): booking}

    for linked in _collect_linked_bookings(booking):
        by_id[int(linked.id)] = linked

    # Action initiée depuis le retour : résoudre explicitement l'aller
    parent_id = getattr(booking, "parent_booking_id", None)
    if parent_id:
        from ext import db

        parent = db.session.get(Booking, int(parent_id))
        if parent is not None:
            by_id[int(parent.id)] = parent
            for linked in _collect_linked_bookings(parent):
                by_id[int(linked.id)] = linked

    # Bookings pointant vers la même action active
    action_id = getattr(action, "id", None) if action is not None else None
    if action_id:
        from ext import db

        for other in (
            Booking.query.filter(Booking.active_change_request_id == int(action_id)).all()
        ):
            by_id[int(other.id)] = other

    return list(by_id.values())


def get_cancellation_billing_booking(
    cancelable_bookings: list[Booking],
) -> Booking | None:
    """Unique aller annulable pouvant porter des frais."""
    outbounds = [b for b in cancelable_bookings if is_outbound_leg(b)]
    if not outbounds:
        return None
    # Préférer le plus bas route_sequence_number, sinon id croissant
    def _key(b: Booking) -> tuple:
        seq = getattr(b, "route_sequence_number", None)
        return (seq is None, int(seq or 0), int(b.id))

    return sorted(outbounds, key=_key)[0]


def _load_cancellation_policy(booking: Booking) -> tuple[dict[str, Any] | None, str]:
    from models.invoice import CompanyBillingSettings

    company_id = getattr(booking, "company_id", None) or getattr(
        booking, "executing_company_id", None
    )
    if not company_id:
        return None, "no-company"
    billing = CompanyBillingSettings.query.filter_by(company_id=int(company_id)).first()
    policy = getattr(billing, "cancellation_policy", None) if billing else None
    if not isinstance(policy, dict):
        policy = None
    version = f"company-{company_id}-cancellation-v{int(getattr(billing, 'id', 0) or 0) if billing else 0}"
    if policy and policy.get("version"):
        version = str(policy["version"])
    return policy, version


def _free_threshold_seconds(policy: dict[str, Any] | None) -> int:
    """Seuil au-delà duquel l'annulation est libre.

    Priorité : free_cancellation_threshold_hours, sinon le plus grand
    palier temps configuré, sinon 24 h.
    """
    hours = DEFAULT_FREE_THRESHOLD_HOURS
    if policy:
        raw = policy.get("free_cancellation_threshold_hours")
        if raw is not None:
            try:
                hours = max(0, int(raw))
            except (TypeError, ValueError):
                hours = DEFAULT_FREE_THRESHOLD_HOURS
        else:
            time_tiers = [
                t for t in (policy.get("tiers") or []) if t.get("type") == "time"
            ]
            if time_tiers:
                try:
                    hours = max(int(float(t.get("hours_before") or 0)) for t in time_tiers)
                except (TypeError, ValueError):
                    hours = DEFAULT_FREE_THRESHOLD_HOURS
    return int(hours) * 3600


def _seconds_before_departure(
    booking: Booking, *, now: datetime | None = None
) -> tuple[int | None, bool]:
    scheduled = getattr(booking, "scheduled_time", None)
    if scheduled is None:
        return None, False
    now_utc = now or datetime.now(UTC)
    aware = (
        scheduled
        if getattr(scheduled, "tzinfo", None)
        else scheduled.replace(tzinfo=UTC)
    )
    delta = int((aware - now_utc).total_seconds())
    return delta, delta < 0


def compute_full_cancellation_charge(outbound: Booking) -> FeeQuote:
    if not is_outbound_leg(outbound):
        raise CancellationRespondError(
            CancellationRespondErrorCode.BILLING_OUTCOME_NOT_ALLOWED,
            "FULL_FARE réservé à un trajet aller.",
        )
    amount = getattr(outbound, "amount", None)
    if amount is None or float(amount) <= 0:
        amount = getattr(outbound, "price_amount", None)
    fare = Decimal(str(amount or 0)).quantize(_CENT)
    if fare < _ZERO:
        fare = _ZERO
    return FeeQuote(
        amount=fare,
        calculation_code=CalculationCode.FULL_OUTBOUND_FARE,
        calculation_basis={"booking_id": int(outbound.id), "basis": "booking_amount"},
    )


def compute_approach_fee(
    outbound: Booking, policy: dict[str, Any] | None
) -> FeeQuote:
    if not is_outbound_leg(outbound):
        raise CancellationRespondError(
            CancellationRespondErrorCode.BILLING_OUTCOME_NOT_ALLOWED,
            "APPROACH_FEE réservé à un trajet aller.",
        )
    from application.bookings.cancellation_rules import compute_cancellation_fee

    fee = compute_cancellation_fee(
        outbound,
        status_at_cancel=BookingStatus.EN_ROUTE.value,
        cancelled_at=datetime.now(UTC),
        reason_code="CLIENT_REQUEST",
        policy=policy,
        ignore_assignment_gate=True,
    )
    amount = fee.fee_amount if fee.fee_amount is not None else _ZERO
    if amount <= _ZERO:
        full = compute_full_cancellation_charge(outbound).amount
        amount = (full * Decimal("0.5")).quantize(_CENT)
        return FeeQuote(
            amount=amount,
            calculation_code=CalculationCode.APPROACH_FEE,
            calculation_basis={
                "booking_id": int(outbound.id),
                "fallback": "half_outbound_fare",
                "fee_label": "Approche (50 %)",
                "percent": 50,
            },
            percent=50,
        )
    return FeeQuote(
        amount=amount.quantize(_CENT),
        calculation_code=CalculationCode.APPROACH_FEE,
        calculation_basis={
            "booking_id": int(outbound.id),
            "tier_id": fee.tier_id,
            "percent": fee.percent,
            "fee_label": fee.fee_label or "Chauffeur en route",
        },
        percent=fee.percent,
        tier_id=fee.tier_id,
    )


def compute_policy_fee(
    outbound: Booking, policy: dict[str, Any] | None, *, status_at_cancel: str
) -> FeeQuote:
    if not is_outbound_leg(outbound):
        raise CancellationRespondError(
            CancellationRespondErrorCode.BILLING_OUTCOME_NOT_ALLOWED,
            "POLICY_FEE réservé à un trajet aller.",
        )
    from application.bookings.cancellation_rules import compute_cancellation_fee

    fee = compute_cancellation_fee(
        outbound,
        status_at_cancel=status_at_cancel,
        cancelled_at=datetime.now(UTC),
        reason_code="CLIENT_REQUEST",
        policy=policy,
        ignore_assignment_gate=True,
    )
    amount = (fee.fee_amount if fee.fee_amount is not None else _ZERO).quantize(_CENT)
    basis: dict[str, Any] = {
        "booking_id": int(outbound.id),
        "tier_id": fee.tier_id,
        "percent": fee.percent,
        "fee_label": fee.fee_label,
    }
    return FeeQuote(
        amount=amount,
        calculation_code=CalculationCode.POLICY_TIER_FEE,
        calculation_basis=basis,
        percent=fee.percent,
        tier_id=fee.tier_id,
    )


def _outcome_dict(
    code: str,
    quote: FeeQuote,
    *,
    requires_amount: bool = False,
    requires_comment: bool = False,
) -> dict[str, Any]:
    return {
        "code": code,
        "amount": _money_str(quote.amount),
        "currency": quote.currency,
        "calculation_code": quote.calculation_code,
        "calculation_basis": quote.calculation_basis,
        "percent": quote.percent,
        "tier_id": quote.tier_id,
        "requires_amount": requires_amount,
        "requires_comment": requires_comment,
    }


def _compute_respond_context_version(parts: dict[str, Any]) -> int:
    raw = "|".join(f"{k}={parts[k]}" for k in sorted(parts.keys()))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_cancellation_respond_context(
    booking: Booking,
    action: BookingChangeRequest,
    *,
    now: datetime | None = None,
) -> CancellationRespondContext:
    """Calcule la policy complète (à appeler aussi sous verrou au POST accept)."""
    affected = resolve_affected_bookings(booking, action)
    cancelable: list[Booking] = []
    non_cancelable: list[Booking] = []
    has_in_progress = False

    for b in affected:
        st = _status_value(b.status)
        if st in {s.upper() for s in TERMINAL_STATUSES} or st in {
            "COMPLETED",
            "RETURN_COMPLETED",
            "CANCELED",
            "CANCELLED",
        }:
            non_cancelable.append(b)
        elif st == BookingStatus.IN_PROGRESS.value.upper() or st == "IN_PROGRESS":
            has_in_progress = True
            # IN_PROGRESS n'est pas cancelable via cette action
            non_cancelable.append(b)
        else:
            cancelable.append(b)

    eligible = get_cancellation_billing_booking(cancelable)
    policy, policy_version = _load_cancellation_policy(booking)
    threshold_seconds = _free_threshold_seconds(policy)

    non_billable_ids = [
        int(b.id) for b in cancelable if not is_outbound_leg(b) or (eligible and b.id != eligible.id)
    ]
    # Tous les non-éligibles parmi cancelables (retours + allers secondaires)
    if eligible:
        non_billable_ids = [int(b.id) for b in cancelable if int(b.id) != int(eligible.id)]
    else:
        non_billable_ids = [int(b.id) for b in cancelable]

    if eligible is None:
        situation = CancellationSituation.NON_BILLABLE_RETURN
        suggested = _outcome_dict(
            BillingOutcome.ZERO,
            FeeQuote(
                amount=_ZERO,
                calculation_code=CalculationCode.NON_BILLABLE_RETURN,
                calculation_basis={"reason": "RETURN_LEG_NEVER_BILLABLE"},
            ),
        )
        allowed = [suggested]
        primary = PrimaryCta.ACKNOWLEDGE_CANCELLATION
        billing_scope = BillingScope.NONE
        seconds_before = None
        departure_passed = False
        ref_status = ""
    else:
        ref_status = _status_value(eligible.status)
        seconds_before, departure_passed = _seconds_before_departure(eligible, now=now)
        if ref_status == BookingStatus.EN_ROUTE.value.upper() or ref_status == "EN_ROUTE":
            situation = CancellationSituation.EN_ROUTE
            full_q = compute_full_cancellation_charge(eligible)
            approach_q = compute_approach_fee(eligible, policy)
            waived = FeeQuote(
                amount=_ZERO,
                calculation_code=CalculationCode.COMPANY_WAIVED,
                calculation_basis={"reason": "COMPANY_CHOICE"},
            )
            suggested = _outcome_dict(BillingOutcome.APPROACH_FEE, approach_q)
            allowed = [
                _outcome_dict(BillingOutcome.ZERO, waived),
                _outcome_dict(BillingOutcome.APPROACH_FEE, approach_q),
                _outcome_dict(BillingOutcome.FULL_FARE, full_q),
                _outcome_dict(
                    BillingOutcome.CUSTOM,
                    FeeQuote(
                        amount=_ZERO,
                        calculation_code=CalculationCode.COMPANY_CUSTOM_FEE,
                        calculation_basis={"maximum_allowed": _money_str(full_q.amount)},
                    ),
                    requires_amount=True,
                    requires_comment=True,
                ),
            ]
            primary = PrimaryCta.CONFIRM_WITH_BILLING
            billing_scope = BillingScope.OUTBOUND_ONLY
        elif seconds_before is not None and seconds_before >= threshold_seconds:
            situation = CancellationSituation.FREE_WINDOW
            suggested = _outcome_dict(
                BillingOutcome.ZERO,
                FeeQuote(
                    amount=_ZERO,
                    calculation_code=CalculationCode.FREE_CANCELLATION_WINDOW,
                    calculation_basis={
                        "seconds_before_departure": seconds_before,
                        "threshold_seconds": threshold_seconds,
                    },
                ),
            )
            allowed = [suggested]
            primary = PrimaryCta.ACKNOWLEDGE_CANCELLATION
            billing_scope = BillingScope.OUTBOUND_ONLY
        else:
            situation = CancellationSituation.FEE_WINDOW
            status_for_fee = ref_status or BookingStatus.ASSIGNED.value
            policy_q = compute_policy_fee(
                eligible, policy, status_at_cancel=status_for_fee
            )
            full_q = compute_full_cancellation_charge(eligible)
            waived = FeeQuote(
                amount=_ZERO,
                calculation_code=CalculationCode.COMPANY_WAIVED,
                calculation_basis={"reason": "COMPANY_CHOICE"},
            )
            suggested = _outcome_dict(BillingOutcome.POLICY_FEE, policy_q)
            allowed = [
                _outcome_dict(BillingOutcome.ZERO, waived),
                _outcome_dict(BillingOutcome.POLICY_FEE, policy_q),
                _outcome_dict(
                    BillingOutcome.CUSTOM,
                    FeeQuote(
                        amount=_ZERO,
                        calculation_code=CalculationCode.COMPANY_CUSTOM_FEE,
                        calculation_basis={"maximum_allowed": _money_str(full_q.amount)},
                    ),
                    requires_amount=True,
                    requires_comment=True,
                ),
            ]
            primary = PrimaryCta.CONFIRM_WITH_BILLING
            billing_scope = BillingScope.OUTBOUND_ONLY

    affected_ids = sorted(int(b.id) for b in affected)
    cancelable_ids = sorted(int(b.id) for b in cancelable)
    non_cancelable_ids = sorted(int(b.id) for b in non_cancelable)
    suggested_amount = suggested.get("amount", "0.00")

    version_parts = {
        "situation": situation,
        "policy_version": policy_version,
        "affected": ",".join(map(str, affected_ids)),
        "cancelable": ",".join(map(str, cancelable_ids)),
        "eligible": str(eligible.id if eligible else None),
        "scope": billing_scope,
        "suggested": f"{suggested.get('code')}:{suggested_amount}",
        # Ne pas hasher seconds_before_departure : il change chaque seconde
        # et provoquerait des faux 409 TIME_WINDOW_CHANGED.
        "threshold": str(threshold_seconds),
        "allowed": ",".join(o["code"] for o in allowed),
        "suggested_calc": str(suggested.get("calculation_code") or ""),
        "suggested_pct": str(suggested.get("percent") if suggested.get("percent") is not None else ""),
    }
    respond_context_version = _compute_respond_context_version(version_parts)

    return CancellationRespondContext(
        situation=str(situation),
        affected_booking_ids=affected_ids,
        cancelable_booking_ids=cancelable_ids,
        non_cancelable_booking_ids=non_cancelable_ids,
        non_billable_booking_ids=sorted(non_billable_ids),
        billing_eligible_booking_id=int(eligible.id) if eligible else None,
        billing_eligible_leg=TripLeg.OUTBOUND if eligible else None,
        billing_scope=str(billing_scope),
        suggested_outcome=suggested,
        allowed_outcomes=allowed,
        primary_cta=str(primary),
        secondary_cta=str(SecondaryCta.REPORT_PROBLEM),
        policy_version=policy_version,
        respond_context_version=respond_context_version,
        seconds_before_departure=seconds_before,
        threshold_seconds=threshold_seconds,
        departure_time_passed=departure_passed,
        has_in_progress=has_in_progress,
        affected_bookings=affected,
        cancelable_bookings=cancelable,
        billing_eligible_booking=eligible,
        policy=policy,
    )


def serialize_respond_ui(ctx: CancellationRespondContext) -> dict[str, Any]:
    display_hours = None
    if ctx.seconds_before_departure is not None:
        display_hours = round(ctx.seconds_before_departure / 3600, 2)
    return {
        "situation": ctx.situation,
        "seconds_before_departure": ctx.seconds_before_departure,
        "threshold_seconds": ctx.threshold_seconds,
        "hours_before_departure_display": display_hours,
        "departure_time_passed": ctx.departure_time_passed,
        "policy_version": ctx.policy_version,
        "respond_context_version": ctx.respond_context_version,
        "billing_scope": ctx.billing_scope,
        "affected_booking_ids": ctx.affected_booking_ids,
        "cancelable_booking_ids": ctx.cancelable_booking_ids,
        "non_cancelable_booking_ids": ctx.non_cancelable_booking_ids,
        "billing_eligible_booking_id": ctx.billing_eligible_booking_id,
        "billing_eligible_leg": ctx.billing_eligible_leg,
        "non_billable_booking_ids": ctx.non_billable_booking_ids,
        "suggested_outcome": ctx.suggested_outcome,
        "allowed_outcomes": ctx.allowed_outcomes,
        "primary_cta": ctx.primary_cta,
        "secondary_cta": ctx.secondary_cta,
    }


def detect_context_change(
    previous: CancellationRespondContext,
    current: CancellationRespondContext,
) -> str | None:
    if previous.policy_version != current.policy_version:
        return ContextChangeReason.POLICY_CHANGED
    if previous.situation != current.situation:
        return ContextChangeReason.TIME_WINDOW_CHANGED
    if previous.affected_booking_ids != current.affected_booking_ids:
        return ContextChangeReason.AFFECTED_BOOKINGS_CHANGED
    if previous.cancelable_booking_ids != current.cancelable_booking_ids:
        return ContextChangeReason.BOOKING_STATUS_CHANGED
    if previous.billing_scope != current.billing_scope:
        return ContextChangeReason.SCOPE_CHANGED
    if previous.billing_eligible_booking_id != current.billing_eligible_booking_id:
        return ContextChangeReason.SCOPE_CHANGED
    prev_amt = (previous.suggested_outcome or {}).get("amount")
    cur_amt = (current.suggested_outcome or {}).get("amount")
    if prev_amt != cur_amt:
        return ContextChangeReason.AMOUNT_CHANGED
    if previous.respond_context_version != current.respond_context_version:
        return ContextChangeReason.AMOUNT_CHANGED
    return None


def parse_fee_amount(raw: Any) -> Decimal:
    if raw is None:
        raise CancellationRespondError(
            CancellationRespondErrorCode.CUSTOM_FEE_AMOUNT_REQUIRED,
            "fee_amount obligatoire pour CUSTOM.",
        )
    if not isinstance(raw, str):
        raise CancellationRespondError(
            CancellationRespondErrorCode.CUSTOM_FEE_AMOUNT_INVALID,
            "fee_amount doit être une chaîne décimale (ex. \"50.00\").",
        )
    text = raw.strip()
    if not text:
        raise CancellationRespondError(
            CancellationRespondErrorCode.CUSTOM_FEE_AMOUNT_REQUIRED,
            "fee_amount obligatoire pour CUSTOM.",
        )
    if "." in text and len(text.split(".", 1)[1]) > 2:
        raise CancellationRespondError(
            CancellationRespondErrorCode.CUSTOM_FEE_AMOUNT_INVALID,
            "fee_amount doit avoir au plus 2 décimales.",
        )
    try:
        value = Decimal(text).quantize(_CENT)
    except (InvalidOperation, ValueError) as exc:
        raise CancellationRespondError(
            CancellationRespondErrorCode.CUSTOM_FEE_AMOUNT_INVALID,
            "fee_amount invalide.",
        ) from exc
    if value < _ZERO:
        raise CancellationRespondError(
            CancellationRespondErrorCode.CUSTOM_FEE_AMOUNT_OUT_OF_RANGE,
            "fee_amount ne peut pas être négatif.",
        )
    return value


def resolve_selected_fee(
    ctx: CancellationRespondContext,
    *,
    billing_outcome: str | None,
    fee_amount: Any = None,
    billing_comment: str | None = None,
    body_provided: bool = False,
) -> tuple[str, FeeQuote, str | None]:
    """Valide le choix commercial et retourne (outcome, quote, comment)."""
    situation = ctx.situation
    allowed_codes = {o["code"] for o in ctx.allowed_outcomes}

    if situation in (
        CancellationSituation.NON_BILLABLE_RETURN,
        CancellationSituation.FREE_WINDOW,
    ):
        if not body_provided or not billing_outcome:
            outcome = BillingOutcome.ZERO
        else:
            outcome = str(billing_outcome).upper()
            if outcome not in allowed_codes:
                raise CancellationRespondError(
                    CancellationRespondErrorCode.BILLING_OUTCOME_NOT_ALLOWED,
                    f"Outcome {outcome} non autorisé pour {situation}.",
                )
        if fee_amount is not None:
            raise CancellationRespondError(
                CancellationRespondErrorCode.FEE_AMOUNT_NOT_ALLOWED,
                "fee_amount interdit hors CUSTOM.",
            )
        # suggested déjà ZERO
        suggested = ctx.suggested_outcome
        quote = FeeQuote(
            amount=Decimal(suggested["amount"]),
            calculation_code=suggested["calculation_code"],
            calculation_basis=dict(suggested.get("calculation_basis") or {}),
        )
        return str(outcome), quote, None

    # FEE_WINDOW / EN_ROUTE : choix explicite obligatoire
    if not billing_outcome:
        raise CancellationRespondError(
            CancellationRespondErrorCode.BILLING_OUTCOME_REQUIRED,
            "billing_outcome obligatoire pour cette situation commerciale.",
        )
    outcome = str(billing_outcome).upper()
    if outcome not in allowed_codes:
        raise CancellationRespondError(
            CancellationRespondErrorCode.BILLING_OUTCOME_NOT_ALLOWED,
            f"Outcome {outcome} non autorisé pour {situation}.",
        )

    if outcome != BillingOutcome.CUSTOM and fee_amount is not None:
        raise CancellationRespondError(
            CancellationRespondErrorCode.FEE_AMOUNT_NOT_ALLOWED,
            "fee_amount interdit hors CUSTOM.",
        )

    eligible = ctx.billing_eligible_booking
    if eligible is None:
        raise CancellationRespondError(
            CancellationRespondErrorCode.BILLING_OUTCOME_NOT_ALLOWED,
            "Aucun aller facturable.",
        )

    if outcome == BillingOutcome.ZERO:
        return (
            outcome,
            FeeQuote(
                amount=_ZERO,
                calculation_code=CalculationCode.COMPANY_WAIVED,
                calculation_basis={"reason": "COMPANY_CHOICE"},
            ),
            None,
        )
    if outcome == BillingOutcome.POLICY_FEE:
        return (
            outcome,
            compute_policy_fee(
                eligible,
                ctx.policy,
                status_at_cancel=_status_value(eligible.status),
            ),
            None,
        )
    if outcome == BillingOutcome.APPROACH_FEE:
        return outcome, compute_approach_fee(eligible, ctx.policy), None
    if outcome == BillingOutcome.FULL_FARE:
        return outcome, compute_full_cancellation_charge(eligible), None
    if outcome == BillingOutcome.CUSTOM:
        comment = (billing_comment or "").strip()
        if not comment:
            raise CancellationRespondError(
                CancellationRespondErrorCode.BILLING_COMMENT_REQUIRED,
                "billing_comment obligatoire pour CUSTOM.",
            )
        amount = parse_fee_amount(fee_amount)
        maximum = compute_full_cancellation_charge(eligible).amount
        if amount > maximum:
            raise CancellationRespondError(
                CancellationRespondErrorCode.CUSTOM_FEE_AMOUNT_OUT_OF_RANGE,
                f"fee_amount hors bornes 0..{ _money_str(maximum) }.",
                extra={"maximum_allowed": _money_str(maximum)},
            )
        return (
            outcome,
            FeeQuote(
                amount=amount,
                calculation_code=CalculationCode.COMPANY_CUSTOM_FEE,
                calculation_basis={"maximum_allowed": _money_str(maximum)},
            ),
            comment,
        )

    raise CancellationRespondError(
        CancellationRespondErrorCode.BILLING_OUTCOME_NOT_ALLOWED,
        f"Outcome inconnu: {outcome}",
    )


def build_commercial_terms(
    ctx: CancellationRespondContext,
    *,
    billing_outcome: str,
    quote: FeeQuote,
    billing_comment: str | None,
) -> dict[str, Any]:
    return {
        "billing_outcome": billing_outcome,
        "billing_eligible_booking_id": ctx.billing_eligible_booking_id,
        "billing_scope": ctx.billing_scope,
        "fee_amount": _money_str(quote.amount),
        "currency": quote.currency,
        "calculation_code": quote.calculation_code,
        "calculation_basis": quote.calculation_basis,
        "policy_version": ctx.policy_version,
        "respond_context_version": ctx.respond_context_version,
        "billing_comment": billing_comment,
        "percent": quote.percent,
        "tier_id": quote.tier_id,
    }


def persist_selected_cancellation_fee(
    booking: Booking,
    *,
    quote: FeeQuote,
    reason: str | None,
) -> None:
    """Persiste les frais sur l'aller éligible uniquement."""
    from application.bookings.cancellation_rules import get_cancellation_display_label

    booking.is_cancellation_billable = quote.amount > _ZERO
    booking.cancellation_fee_amount = quote.amount
    booking.cancellation_fee_percent = quote.percent
    booking.cancellation_fee_tier_id = quote.tier_id
    booking.cancellation_reason_code = "CLIENT_REQUEST"
    booking.cancellation_reason_text = reason or "Annulation confirmée"
    booking.cancellation_display_label = get_cancellation_display_label(
        "CLIENT_REQUEST", booking.cancellation_reason_text
    )


def attach_respond_ui_to_action(
    action: BookingChangeRequest,
    booking: Booking | None = None,
) -> dict[str, Any] | None:
    """Construit respond_ui pour sérialisation GET (affichage seul)."""
    if action.action_type != TransportActionType.CANCELLATION:
        return None
    if booking is None:
        from ext import db

        booking = db.session.get(Booking, action.booking_id)
    if booking is None:
        return None
    try:
        ctx = build_cancellation_respond_context(booking, action)
        return serialize_respond_ui(ctx)
    except Exception as exc:
        logger.warning("[CancellationRespond] respond_ui: %s", exc)
        return None
