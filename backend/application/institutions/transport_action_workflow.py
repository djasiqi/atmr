"""Moteur TransportActionWorkflow — intention → décision → EffectPlan → Completed.

Référence : docs/domain/transport-decision-workflow.md
Aucun publish legacy depuis ce module — uniquement LegacyCompatibilityAdapter.
"""

from __future__ import annotations

import contextlib
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ext import db
from models import Booking, BookingChangeRequest, TransportRequest
from models.booking_change_request import (
    TransportActionEffectStatus,
    TransportActionNextActor,
    TransportActionStatus,
    TransportActionType,
)
from models.enums import BookingStatus, RequestStatus
from models.transport_action_exchange import (
    TransportActionExchange,
    TransportActionExchangeDecision,
)

logger = logging.getLogger(__name__)

ALLOWED_STATUS_EFFECT: frozenset[tuple[str, str]] = frozenset(
    {
        (TransportActionStatus.REQUESTED, TransportActionEffectStatus.NONE),
        (TransportActionStatus.PENDING, TransportActionEffectStatus.NONE),
        (TransportActionStatus.COUNTER_PENDING, TransportActionEffectStatus.NONE),
        (TransportActionStatus.ACCEPTED, TransportActionEffectStatus.PENDING),
        (TransportActionStatus.ACCEPTED, TransportActionEffectStatus.FAILED),
        (TransportActionStatus.COMPLETED, TransportActionEffectStatus.COMPLETED),
        (TransportActionStatus.REJECTED, TransportActionEffectStatus.NONE),
        (TransportActionStatus.REFUSED, TransportActionEffectStatus.NONE),
        (TransportActionStatus.EXPIRED, TransportActionEffectStatus.NONE),
        (TransportActionStatus.CLOSED_REPLACED, TransportActionEffectStatus.NONE),
        (TransportActionStatus.SUPERSEDED, TransportActionEffectStatus.NONE),
        (
            TransportActionStatus.NEGOTIATION_LIMIT_REACHED,
            TransportActionEffectStatus.NONE,
        ),
        (TransportActionStatus.CONFLICTED, TransportActionEffectStatus.NONE),
    }
)


def is_counter_enabled() -> bool:
    """V1.2 feature flag — COUNTER désactivé par défaut (V1.1)."""
    return os.getenv("TRANSPORT_ACTION_COUNTER_ENABLED", "false").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def get_action_ttl_minutes() -> int:
    from application.institutions.transport_action_policies import (
        default_response_policy,
    )

    return default_response_policy().ttl_minutes_default


def classify_action_type(
    changed_fields: set[str] | dict, *, is_cancellation: bool = False
) -> str:
    if is_cancellation:
        return TransportActionType.CANCELLATION
    fields = (
        set(changed_fields.keys())
        if isinstance(changed_fields, dict)
        else set(changed_fields)
    )
    if fields & {"scheduled_time"} and not (
        fields & {"pickup_location", "dropoff_location"}
    ):
        # date vs time : scheduled_time porte souvent les deux
        return TransportActionType.CHANGE_TIME
    if fields & {"pickup_location", "pickup_lat", "pickup_lon"}:
        return TransportActionType.CHANGE_PICKUP_ADDRESS
    if fields & {"dropoff_location", "dropoff_lat", "dropoff_lon"}:
        return TransportActionType.CHANGE_DROPOFF_ADDRESS
    if fields & {"wheelchair_need", "wheelchair_client_has"}:
        return TransportActionType.CHANGE_PASSENGER_REQUIREMENTS
    if fields & {"mission_type", "delivery_description"}:
        return TransportActionType.CHANGE_OTHER
    return TransportActionType.CHANGE_OTHER


def assert_status_effect_combo(status: str, effect_status: str) -> None:
    if (status, effect_status) not in ALLOWED_STATUS_EFFECT:
        raise ValueError(
            f"Combinaison status/effect_status interdite: {status}+{effect_status}"
        )


@dataclass
class EffectPlan:
    transactional_steps: list[str] = field(default_factory=list)
    post_commit_events: list[dict[str, Any]] = field(default_factory=list)
    accepted_values: dict[str, Any] = field(default_factory=dict)
    is_cancellation: bool = False


def build_decision_context_snapshot(booking: Booking) -> dict[str, Any]:
    """Snapshot immuable minimal (privacy-aware)."""
    minutes = None
    try:
        st = getattr(booking, "scheduled_time", None)
        if st is not None:
            now = datetime.now(UTC)
            aware = st if getattr(st, "tzinfo", None) else st.replace(tzinfo=UTC)
            minutes = int((aware - now).total_seconds() // 60)
    except Exception:
        minutes = None
    return {
        "booking_id": booking.id,
        "status": str(getattr(booking.status, "value", booking.status)),
        "patient_display_name": getattr(booking, "customer_name", None),
        "driver_id": getattr(booking, "driver_id", None),
        "company_id": getattr(booking, "company_id", None)
        or getattr(booking, "executing_company_id", None),
        "minutes_to_departure": minutes,
        "pickup_summary": getattr(booking, "pickup_location", None),
        "dropoff_summary": getattr(booking, "dropoff_location", None),
        "edit_version": int(getattr(booking, "edit_version", 1) or 1),
        "captured_at": datetime.now(UTC).isoformat(),
    }


def _append_exchange(
    action: BookingChangeRequest,
    *,
    decision_type: str,
    actor_type: str,
    actor_id: int | None,
    values: dict[str, Any] | None = None,
    commercial_terms: dict[str, Any] | None = None,
    comment: str | None = None,
    created_from: str | None = "api",
    idempotency_key: str | None = None,
    snapshot: dict[str, Any] | None = None,
) -> TransportActionExchange:
    seq = 1
    if action.exchanges:
        seq = max(int(ex.sequence) for ex in action.exchanges) + 1
    else:
        # query si relation non chargée
        last = (
            TransportActionExchange.query.filter_by(transport_action_id=action.id)
            .order_by(TransportActionExchange.sequence.desc())
            .first()
        )
        if last:
            seq = int(last.sequence) + 1

    ex = TransportActionExchange()
    ex.transport_action_id = action.id
    ex.sequence = seq
    ex.actor_type = actor_type
    ex.actor_id = actor_id
    ex.decision_type = decision_type
    ex.values = values
    ex.commercial_terms = commercial_terms
    ex.comment = comment
    ex.created_from = created_from
    ex.idempotency_key = idempotency_key or str(uuid.uuid4())
    ex.decision_context_snapshot = snapshot
    db.session.add(ex)
    db.session.flush()
    action.active_exchange_id = ex.id
    return ex


def close_open_actions_as_replaced(
    booking: Booking, *, excluding_id: int | None = None
) -> list[int]:
    """Ferme les actions ouvertes (CLOSED_REPLACED) — V1.1 sans confirmation UI avancée."""
    q = BookingChangeRequest.query.filter(
        BookingChangeRequest.booking_id == booking.id,
        BookingChangeRequest.status.in_(list(TransportActionStatus.OPEN)),
    )
    if excluding_id:
        q = q.filter(BookingChangeRequest.id != excluding_id)
    closed: list[int] = []
    for action in q.all():
        action.status = TransportActionStatus.CLOSED_REPLACED
        action.effect_status = TransportActionEffectStatus.NONE
        action.next_actor_type = TransportActionNextActor.NONE
        action.version = int(action.version or 1) + 1
        assert_status_effect_combo(action.status, action.effect_status)
        closed.append(action.id)
    return closed


def create_transport_action_from_intention(
    *,
    booking: Booking,
    transport_request: TransportRequest | None,
    institution_id: int,
    action_type: str,
    proposed_patch: dict[str, Any],
    before_snapshot: dict[str, Any],
    after_snapshot: dict[str, Any],
    changed_fields: dict[str, Any],
    reason: str | None,
    actor_user_id: int | None,
    actor_role: str | None,
    action_scope: str = "BOOKING",
    created_from: str = "institution_portal",
) -> BookingChangeRequest:
    """Crée une TransportAction REQUESTED + exchange REQUEST. Commit laissé à l'appelant."""
    closed = close_open_actions_as_replaced(booking)

    action = BookingChangeRequest()
    action.booking_id = booking.id
    action.transport_request_id = transport_request.id if transport_request else None
    action.institution_id = institution_id
    action.status = TransportActionStatus.REQUESTED
    action.effect_status = TransportActionEffectStatus.NONE
    action.next_actor_type = TransportActionNextActor.COMPANY
    action.version = 1
    action.proposed_patch = proposed_patch or {}
    action.before_snapshot = before_snapshot
    action.after_snapshot = after_snapshot
    action.changed_fields = changed_fields
    action.reason = reason
    action.action_type = action_type
    action.action_scope = action_scope
    action.mission_version_at_request = int(getattr(booking, "edit_version", 1) or 1)
    action.requested_by_user_id = actor_user_id
    action.requested_by_role = actor_role
    action.expires_at = datetime.now(UTC) + timedelta(minutes=get_action_ttl_minutes())
    action.handling_status = "UNSEEN"
    db.session.add(action)
    db.session.flush()

    snapshot = build_decision_context_snapshot(booking)
    _append_exchange(
        action,
        decision_type=TransportActionExchangeDecision.REQUEST,
        actor_type="institution",
        actor_id=actor_user_id,
        values=proposed_patch or {},
        comment=reason,
        created_from=created_from,
        snapshot=snapshot,
    )

    booking.active_change_request_id = action.id
    booking.updated_at = datetime.now(UTC)

    # Parcours A/R ou multi-legs : la décision concerne toute la mission.
    scope = (action_scope or "BOOKING").upper()
    if scope in {"ROUND_TRIP", "ROUTE_GROUP", "MISSION"}:
        from application.institutions.cancellation_respond_policy import (
            resolve_affected_bookings,
        )

        for linked in resolve_affected_bookings(booking, action):
            if int(linked.id) == int(booking.id):
                continue
            linked.active_change_request_id = action.id
            linked.updated_at = datetime.now(UTC)

    assert_status_effect_combo(action.status, action.effect_status)

    if closed:
        logger.info(
            "[TransportAction] Remplacé actions=%s par action=%s booking=%s",
            closed,
            action.id,
            booking.id,
        )
    return action


def build_effect_plan(
    action: BookingChangeRequest,
    *,
    accepted_exchange: TransportActionExchange,
) -> EffectPlan:
    is_cancel = action.action_type == TransportActionType.CANCELLATION
    values = accepted_exchange.values or action.proposed_patch or {}
    return EffectPlan(
        accepted_values=dict(values),
        is_cancellation=is_cancel,
        transactional_steps=[
            "lock_and_validate",
            "mutate_mission" if not is_cancel else "cancel_mission",
            "timeline",
            "mark_completed",
            "outbox_completed",
        ],
        post_commit_events=[
            {
                "type": "TransportActionCompleted",
                "action_id": action.id,
                "booking_id": action.booking_id,
                "action_type": action.action_type,
                "accepted_exchange_id": accepted_exchange.id,
            }
        ],
    )


def _clear_driver_and_assignments(booking: Booking) -> None:
    booking.driver_id = None
    try:
        from models import Assignment

        for a in Assignment.query.filter_by(booking_id=booking.id).all():
            st = str(getattr(a.status, "value", a.status) or "").upper()
            if st in ("ACTIVE", "ASSIGNED", "PENDING", ""):
                if hasattr(a, "status"):
                    with contextlib.suppress(Exception):
                        a.status = "CANCELLED"
                if hasattr(a, "ended_at") and a.ended_at is None:
                    a.ended_at = datetime.now(UTC)
    except Exception as exc:
        logger.warning("[TransportAction] clear assignments: %s", exc)


def _apply_cancellation_effects(
    booking: Booking,
    action: BookingChangeRequest,
    *,
    reason: str | None,
    commercial_terms: dict[str, Any] | None = None,
    cancelable_booking_ids: list[int] | None = None,
    billing_eligible_booking_id: int | None = None,
    fee_quote: Any | None = None,
) -> None:
    """Applique les Cancellation Effect Invariants (§13).

    Ne mute que cancelable_booking_ids.
    Frais uniquement sur billing_eligible_booking_id (OUTBOUND).
    """
    from application.bookings.cancellation_rules import get_cancellation_display_label
    from application.institutions.cancellation_respond_policy import (
        clear_cancellation_billing,
        persist_selected_cancellation_fee,
        resolve_affected_bookings,
    )

    cancelled_at = datetime.now(UTC)
    reason_text = reason or action.reason or "Annulation confirmée"
    display = get_cancellation_display_label("CLIENT_REQUEST", reason_text)

    if cancelable_booking_ids is None:
        # Fallback legacy : tout le périmètre affecté non terminal
        affected = resolve_affected_bookings(booking, action)
        cancelable_ids = {
            int(b.id)
            for b in affected
            if str(getattr(b.status, "value", b.status)).upper()
            not in {"COMPLETED", "RETURN_COMPLETED", "CANCELED", "CANCELLED"}
        }
    else:
        cancelable_ids = {int(x) for x in cancelable_booking_ids}

    eligible_id = (
        int(billing_eligible_booking_id)
        if billing_eligible_booking_id is not None
        else None
    )
    if eligible_id is None and commercial_terms:
        raw = commercial_terms.get("billing_eligible_booking_id")
        eligible_id = int(raw) if raw is not None else None

    affected = resolve_affected_bookings(booking, action)

    for target in affected:
        if int(target.id) not in cancelable_ids:
            continue  # non_cancelable : intact (historique inclus)

        target.status = BookingStatus.CANCELED
        target.cancellation_reason_code = "CLIENT_REQUEST"
        target.cancellation_reason_text = reason_text
        target.cancellation_display_label = display
        if hasattr(target, "cancelled_at"):
            target.cancelled_at = cancelled_at
        if hasattr(target, "cancelled_by_role"):
            target.cancelled_by_role = "company"
        _clear_driver_and_assignments(target)

        if (
            eligible_id is not None
            and int(target.id) == eligible_id
            and fee_quote is not None
        ):
            persist_selected_cancellation_fee(
                target, quote=fee_quote, reason=reason_text
            )
        else:
            clear_cancellation_billing(target)

    # Sync transport_request
    tr = None
    if action.transport_request_id:
        tr = db.session.get(TransportRequest, action.transport_request_id)
    if tr is not None:
        tr.status = RequestStatus.CANCELLED
        if hasattr(tr, "cancelled_at"):
            tr.cancelled_at = cancelled_at


def clear_active_change_request_refs(action_id: int | None) -> None:
    """Retire active_change_request_id sur tous les bookings pointant vers l'action."""
    if not action_id:
        return
    from models.booking import Booking

    Booking.query.filter(Booking.active_change_request_id == int(action_id)).update(
        {Booking.active_change_request_id: None},
        synchronize_session=False,
    )


def complete_effects(
    *,
    booking: Booking,
    action: BookingChangeRequest,
    accepted_exchange: TransportActionExchange,
    actor_user_id: int | None,
    reason: str | None = None,
    cancellation_effect_args: dict[str, Any] | None = None,
) -> EffectPlan:
    """Applique EffectPlan transactionnel. Appelant gère commit.

    Préconditions : verrous déjà pris, versions vérifiées.
    """
    expected = int(action.mission_version_at_request or 0)
    current = int(getattr(booking, "edit_version", 1) or 1)
    if expected and current != expected:
        action.status = TransportActionStatus.CONFLICTED
        action.effect_status = TransportActionEffectStatus.NONE
        action.next_actor_type = TransportActionNextActor.NONE
        clear_active_change_request_refs(action.id)
        assert_status_effect_combo(action.status, action.effect_status)
        raise ConflictError("Version mission divergente — action CONFLICTED.")

    plan = build_effect_plan(action, accepted_exchange=accepted_exchange)

    action.status = TransportActionStatus.ACCEPTED
    action.effect_status = TransportActionEffectStatus.PENDING
    assert_status_effect_combo(action.status, action.effect_status)

    if plan.is_cancellation:
        args = cancellation_effect_args or {}
        _apply_cancellation_effects(
            booking,
            action,
            reason=reason,
            commercial_terms=args.get("commercial_terms"),
            cancelable_booking_ids=args.get("cancelable_booking_ids"),
            billing_eligible_booking_id=args.get("billing_eligible_booking_id"),
            fee_quote=args.get("fee_quote"),
        )
    else:
        from services.institutions.booking_change_service import (
            apply_operational_patch,
            bump_edit_version,
        )

        apply_operational_patch(booking, plan.accepted_values)
        bump_edit_version(booking)

    action.status = TransportActionStatus.COMPLETED
    action.effect_status = TransportActionEffectStatus.COMPLETED
    action.next_actor_type = TransportActionNextActor.NONE
    action.completed_at = datetime.now(UTC)
    action.responded_at = datetime.now(UTC)
    action.responded_by_user_id = actor_user_id
    action.responded_by_role = "company"
    action.version = int(action.version or 1) + 1
    clear_active_change_request_refs(action.id)
    assert_status_effect_combo(action.status, action.effect_status)

    # Outbox logique : stocké dans post_commit_events du plan (dispatch après commit)
    return plan


class ConflictError(Exception):
    """Version mission divergente."""


class NegotiationDisabledError(Exception):
    """COUNTER non activé (V1.1)."""


def dispatch_post_commit(plan: EffectPlan, *, action: BookingChangeRequest) -> None:
    """Effets post-commit : notifs + adapter legacy. Ne mute pas la mission."""
    from application.institutions.legacy_transport_action_adapter import (
        publish_legacy_after_transport_action_completed,
    )
    from domain.events.events import TransportActionCompletedEvent
    from shared.events.event_bus import publish_event

    try:
        publish_event(
            TransportActionCompletedEvent(
                action_id=action.id,
                booking_id=action.booking_id,
                action_type=action.action_type or "",
                accepted_exchange_id=action.active_exchange_id,
                is_cancellation=plan.is_cancellation,
            )
        )
    except Exception as exc:
        logger.warning("[TransportAction] publish Completed failed: %s", exc)

    try:
        publish_legacy_after_transport_action_completed(
            action=action, is_cancellation=plan.is_cancellation
        )
    except Exception as exc:
        logger.warning("[TransportAction] legacy adapter failed: %s", exc)

    try:
        from services.events.institution_events import persist_company_notification

        company_id = None
        booking = db.session.get(Booking, action.booking_id)
        if booking:
            company_id = booking.company_id or booking.executing_company_id
        if company_id and not plan.is_cancellation:
            persist_company_notification(
                company_id=int(company_id),
                event_type="transport_action_completed",
                title="Modification confirmée",
                message=f"La demande #{action.id} a été appliquée (course #{action.booking_id}).",
                metadata={
                    "booking_id": action.booking_id,
                    "action_id": action.id,
                    "action_type": action.action_type,
                },
                dedupe_key=f"ta_done_{action.id}",
            )
    except Exception as exc:
        logger.warning("[TransportAction] company notif completed: %s", exc)
