# application/institutions/respond_to_change_request.py
"""Réponse transporteur à une TransportAction (Accept / Reject / Counter).

V1.1 : ACCEPT applique EffectPlan ; REJECT = no-op mission (plus de redispatch).
V1.2 : COUNTER derrière TRANSPORT_ACTION_COUNTER_ENABLED.
CANCELLATION V1 : billing_outcome / commercial_terms / revalidation sous verrou.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from application.institutions.cancellation_respond_policy import (
    CancellationRespondError,
    CancellationRespondErrorCode,
    build_cancellation_respond_context,
    build_commercial_terms,
    resolve_selected_fee,
    serialize_respond_ui,
)
from application.institutions.transport_action_workflow import (
    ConflictError,
    _append_exchange,
    assert_status_effect_combo,
    build_decision_context_snapshot,
    complete_effects,
    dispatch_post_commit,
    is_counter_enabled,
)
from ext import db
from models import Booking, BookingChangeRequest
from models.booking_change_request import (
    TransportActionEffectStatus,
    TransportActionNextActor,
    TransportActionStatus,
    TransportActionType,
)
from models.transport_action_exchange import TransportActionExchangeDecision
from security.audit_log import AuditLogger
from services.institutions.booking_change_service import record_change_event
from services.institutions.transport_timeline_service import resolve_actor_name

logger = logging.getLogger(__name__)

ACTION_ACCEPT = "accept"
ACTION_REFUSE = "refuse"
ACTION_REJECT = "reject"
ACTION_COUNTER = "counter"


def _resolve_transport_request(change_request: BookingChangeRequest):
    if not change_request.transport_request_id:
        return None
    from models import TransportRequest

    return db.session.get(TransportRequest, change_request.transport_request_id)


@dataclass(frozen=True, slots=True)
class RespondToChangeRequestInput:
    booking_id: int
    change_request_id: int
    company_id: int
    user_id: int | None
    action: str
    version: int
    reason: str | None = None
    counter_values: dict[str, Any] | None = None
    commercial_terms: dict[str, Any] | None = None
    accepted_exchange_id: int | None = None
    idempotency_key: str | None = None
    # CANCELLATION commercial
    billing_outcome: str | None = None
    fee_amount: str | None = None
    billing_comment: str | None = None
    policy_version: str | None = None
    respond_context_version: int | None = None
    billing_body_provided: bool = False
    rejection_reason_code: str | None = None
    # Snapshot client pour détection d'obsolescence sémantique
    client_situation: str | None = None
    client_suggested_amount: str | None = None
    client_cancelable_booking_ids: list[int] | None = None


@dataclass(frozen=True, slots=True)
class RespondToChangeRequestResult:
    success: bool
    booking_id: int
    change_request_id: int
    status: str | None = None
    redispatched: bool = False
    error: str | None = None
    status_code: int = 200
    payload: dict[str, Any] | None = None


class RespondToChangeRequestUseCase:
    def execute(
        self, input_data: RespondToChangeRequestInput
    ) -> RespondToChangeRequestResult:
        action = (input_data.action or "").strip().lower()
        if action in (ACTION_REFUSE, ACTION_REJECT):
            action = ACTION_REJECT
        if action not in (ACTION_ACCEPT, ACTION_REJECT, ACTION_COUNTER):
            return RespondToChangeRequestResult(
                success=False,
                booking_id=input_data.booking_id,
                change_request_id=input_data.change_request_id,
                error="Action invalide (accept|reject|counter).",
                status_code=400,
            )
        if action == ACTION_COUNTER and not is_counter_enabled():
            return RespondToChangeRequestResult(
                success=False,
                booking_id=input_data.booking_id,
                change_request_id=input_data.change_request_id,
                error="Contre-proposition non activée (TRANSPORT_ACTION_COUNTER_ENABLED).",
                status_code=400,
            )

        try:
            booking = (
                db.session.query(Booking)
                .filter(Booking.id == input_data.booking_id)
                .with_for_update()
                .first()
            )
            if not booking:
                return RespondToChangeRequestResult(
                    success=False,
                    booking_id=input_data.booking_id,
                    change_request_id=input_data.change_request_id,
                    error="Course introuvable.",
                    status_code=404,
                )

            owner_id = cast(
                int | None, booking.company_id or booking.executing_company_id
            )
            if int(owner_id or 0) != int(input_data.company_id):
                return RespondToChangeRequestResult(
                    success=False,
                    booking_id=input_data.booking_id,
                    change_request_id=input_data.change_request_id,
                    error="Vous n'êtes pas le transporteur de cette course.",
                    status_code=403,
                )

            change_request = (
                db.session.query(BookingChangeRequest)
                .filter(BookingChangeRequest.id == input_data.change_request_id)
                .with_for_update()
                .first()
            )
            if not change_request or change_request.booking_id != booking.id:
                return RespondToChangeRequestResult(
                    success=False,
                    booking_id=input_data.booking_id,
                    change_request_id=input_data.change_request_id,
                    error="Demande introuvable.",
                    status_code=404,
                )

            if change_request.status not in TransportActionStatus.OPEN:
                return RespondToChangeRequestResult(
                    success=False,
                    booking_id=input_data.booking_id,
                    change_request_id=change_request.id,
                    error=(
                        "Cette demande n'est plus en attente "
                        f"(statut {change_request.status})."
                    ),
                    status_code=409,
                    payload={"current_status": change_request.status},
                )

            if int(booking.active_change_request_id or 0) != int(change_request.id):
                return RespondToChangeRequestResult(
                    success=False,
                    booking_id=input_data.booking_id,
                    change_request_id=change_request.id,
                    error="Cette demande a été remplacée par une plus récente.",
                    status_code=409,
                )

            if change_request.next_actor_type not in (
                TransportActionNextActor.COMPANY,
                None,
                "",
            ) and (
                change_request.next_actor_type == TransportActionNextActor.INSTITUTION
            ):
                return RespondToChangeRequestResult(
                    success=False,
                    booking_id=booking.id,
                    change_request_id=change_request.id,
                    error="C'est au tour de l'institution de répondre.",
                    status_code=409,
                )

            current_version = int(change_request.version or 1)
            if int(input_data.version) != current_version:
                return RespondToChangeRequestResult(
                    success=False,
                    booking_id=input_data.booking_id,
                    change_request_id=change_request.id,
                    error="Conflit de version : la demande a évolué entre-temps.",
                    status_code=409,
                    payload={"current_version": current_version},
                )

            if action == ACTION_ACCEPT:
                return self._accept(booking, change_request, input_data)
            if action == ACTION_COUNTER:
                return self._counter(booking, change_request, input_data)
            return self._reject(booking, change_request, input_data)

        except CancellationRespondError as e:
            db.session.rollback()
            payload: dict[str, Any] = {"code": e.code, "error": e.message}
            if e.change_reason:
                payload["change_reason"] = e.change_reason
            if e.respond_ui is not None:
                payload["respond_ui"] = e.respond_ui
            payload.update(e.extra)
            return RespondToChangeRequestResult(
                success=False,
                booking_id=input_data.booking_id,
                change_request_id=input_data.change_request_id,
                error=e.message,
                status_code=e.status_code,
                payload=payload,
            )
        except ConflictError as e:
            db.session.commit()
            return RespondToChangeRequestResult(
                success=False,
                booking_id=input_data.booking_id,
                change_request_id=input_data.change_request_id,
                error=str(e),
                status_code=409,
                payload={"status": TransportActionStatus.CONFLICTED},
            )
        except Exception as e:
            logger.exception(
                "[RespondToChangeRequest] Erreur change_request=%s booking=%s",
                input_data.change_request_id,
                input_data.booking_id,
            )
            db.session.rollback()
            return RespondToChangeRequestResult(
                success=False,
                booking_id=input_data.booking_id,
                change_request_id=input_data.change_request_id,
                error=f"Erreur inattendue: {e!s}",
                status_code=500,
            )

    def _lock_affected_bookings(
        self, booking: Booking, change_request: BookingChangeRequest
    ) -> None:
        from application.institutions.cancellation_respond_policy import (
            resolve_affected_bookings,
        )

        affected = resolve_affected_bookings(booking, change_request)
        ids = sorted({int(b.id) for b in affected})
        if not ids:
            return
        (db.session.query(Booking).filter(Booking.id.in_(ids)).with_for_update().all())

    def _accept_cancellation(
        self,
        booking: Booking,
        change_request: BookingChangeRequest,
        input_data: RespondToChangeRequestInput,
    ) -> RespondToChangeRequestResult:
        self._lock_affected_bookings(booking, change_request)
        db.session.refresh(booking)

        ctx = build_cancellation_respond_context(booking, change_request)
        respond_ui = serialize_respond_ui(ctx)

        if ctx.has_in_progress:
            raise CancellationRespondError(
                CancellationRespondErrorCode.INTERRUPTION_REQUIRED,
                "Un trajet concerné est en cours : interruption requise.",
                status_code=422,
                respond_ui=respond_ui,
            )

        # Obsolescence : comparaison sémantique (pas le hash opaque seul).
        # Le hash peut diverger après redéploiement / anciens clients.
        change_reason = None
        if (
            input_data.policy_version
            and input_data.policy_version != ctx.policy_version
        ):
            change_reason = "POLICY_CHANGED"
        elif (
            input_data.client_situation and input_data.client_situation != ctx.situation
        ):
            change_reason = "TIME_WINDOW_CHANGED"
        elif (
            input_data.client_suggested_amount is not None
            and ctx.suggested_outcome
            and str(input_data.client_suggested_amount)
            != str(ctx.suggested_outcome.get("amount"))
            and (
                not input_data.billing_outcome
                or str(input_data.billing_outcome).upper()
                in {"POLICY_FEE", "APPROACH_FEE", "FULL_FARE"}
            )
        ):
            change_reason = "AMOUNT_CHANGED"
        elif input_data.client_cancelable_booking_ids is not None and sorted(
            int(x) for x in input_data.client_cancelable_booking_ids
        ) != list(ctx.cancelable_booking_ids):
            change_reason = "BOOKING_STATUS_CHANGED"

        if change_reason:
            raise CancellationRespondError(
                CancellationRespondErrorCode.CANCELLATION_RESPONSE_CONTEXT_CHANGED,
                "Le contexte de réponse a évolué — veuillez confirmer à nouveau.",
                status_code=409,
                change_reason=change_reason,
                respond_ui=respond_ui,
            )

        outcome, quote, comment = resolve_selected_fee(
            ctx,
            billing_outcome=input_data.billing_outcome,
            fee_amount=input_data.fee_amount,
            billing_comment=input_data.billing_comment or input_data.reason,
            body_provided=input_data.billing_body_provided,
        )
        commercial_terms = build_commercial_terms(
            ctx,
            billing_outcome=outcome,
            quote=quote,
            billing_comment=comment,
        )

        accepted_ex = _append_exchange(
            change_request,
            decision_type=TransportActionExchangeDecision.ACCEPT,
            actor_type="company",
            actor_id=input_data.user_id,
            values=change_request.proposed_patch or {},
            commercial_terms=commercial_terms,
            comment=comment or input_data.reason,
            created_from="company_portal",
            idempotency_key=input_data.idempotency_key,
            snapshot=build_decision_context_snapshot(booking),
        )

        plan = complete_effects(
            booking=booking,
            action=change_request,
            accepted_exchange=accepted_ex,
            actor_user_id=input_data.user_id,
            reason=input_data.reason or change_request.reason,
            cancellation_effect_args={
                "commercial_terms": commercial_terms,
                "cancelable_booking_ids": ctx.cancelable_booking_ids,
                "billing_eligible_booking_id": ctx.billing_eligible_booking_id,
                "fee_quote": quote,
            },
        )

        record_change_event(
            booking=booking,
            transport_request=_resolve_transport_request(change_request),
            institution_id=change_request.institution_id,
            actor_user_id=input_data.user_id,
            actor_role="company",
            actor_type="company",
            actor_display_name=resolve_actor_name(input_data.user_id),
            action_type="cancelled",
            change_scope="cancellation",
            source="company_portal",
            before_snapshot=change_request.before_snapshot,
            after_snapshot=change_request.after_snapshot,
            reason=input_data.reason or change_request.reason,
            change_class="major",
            severity="INFO",
            ack_required=False,
            operational_impact={
                "transport_action_accepted": True,
                "commercial_terms": commercial_terms,
            },
        )

        self._record_response_timeline(
            change_request=change_request,
            event_type="change_accepted_by_company",
            company_id=input_data.company_id,
            user_id=input_data.user_id,
        )

        db.session.commit()
        dispatch_post_commit(plan, action=change_request)
        self._audit(input_data, change_request, accepted=True)

        return RespondToChangeRequestResult(
            success=True,
            booking_id=booking.id,
            change_request_id=change_request.id,
            status=change_request.status,
            redispatched=False,
            payload={
                "edit_version": int(booking.edit_version or 1),
                "change_request": change_request.serialize(),
                "effect_status": change_request.effect_status,
                "commercial_terms": commercial_terms,
            },
        )

    def _accept(
        self,
        booking: Booking,
        change_request: BookingChangeRequest,
        input_data: RespondToChangeRequestInput,
    ) -> RespondToChangeRequestResult:
        if change_request.action_type == TransportActionType.CANCELLATION:
            return self._accept_cancellation(booking, change_request, input_data)

        exchange_id = (
            input_data.accepted_exchange_id or change_request.active_exchange_id
        )
        accepted_ex = None
        if exchange_id:
            from models.transport_action_exchange import TransportActionExchange

            accepted_ex = db.session.get(TransportActionExchange, int(exchange_id))
        if accepted_ex is None or accepted_ex.transport_action_id != change_request.id:
            accepted_ex = _append_exchange(
                change_request,
                decision_type=TransportActionExchangeDecision.ACCEPT,
                actor_type="company",
                actor_id=input_data.user_id,
                values=change_request.proposed_patch or {},
                comment=input_data.reason,
                created_from="company_portal",
                idempotency_key=input_data.idempotency_key,
                snapshot=build_decision_context_snapshot(booking),
            )
        else:
            accepted_ex = _append_exchange(
                change_request,
                decision_type=TransportActionExchangeDecision.ACCEPT,
                actor_type="company",
                actor_id=input_data.user_id,
                values=accepted_ex.values or change_request.proposed_patch or {},
                comment=input_data.reason,
                created_from="company_portal",
                idempotency_key=input_data.idempotency_key,
                snapshot=build_decision_context_snapshot(booking),
            )

        try:
            plan = complete_effects(
                booking=booking,
                action=change_request,
                accepted_exchange=accepted_ex,
                actor_user_id=input_data.user_id,
                reason=input_data.reason or change_request.reason,
            )
        except ConflictError:
            raise
        except Exception:
            db.session.rollback()
            failed_request = (
                db.session.query(BookingChangeRequest)
                .filter_by(id=input_data.change_request_id)
                .with_for_update()
                .first()
            )
            if failed_request is not None:
                failed_request.status = TransportActionStatus.ACCEPTED
                failed_request.effect_status = TransportActionEffectStatus.FAILED
                assert_status_effect_combo(
                    failed_request.status, failed_request.effect_status
                )
                db.session.commit()
            raise

        record_change_event(
            booking=booking,
            transport_request=_resolve_transport_request(change_request),
            institution_id=change_request.institution_id,
            actor_user_id=input_data.user_id,
            actor_role="company",
            actor_type="company",
            actor_display_name=resolve_actor_name(input_data.user_id),
            action_type="field_updated",
            change_scope="operational",
            source="company_portal",
            before_snapshot=change_request.before_snapshot,
            after_snapshot=change_request.after_snapshot,
            reason=input_data.reason or change_request.reason,
            change_class="major",
            severity="INFO",
            ack_required=False,
            operational_impact={"transport_action_accepted": True},
        )

        self._record_response_timeline(
            change_request=change_request,
            event_type="change_accepted_by_company",
            company_id=input_data.company_id,
            user_id=input_data.user_id,
        )

        db.session.commit()
        dispatch_post_commit(plan, action=change_request)
        self._audit(input_data, change_request, accepted=True)

        return RespondToChangeRequestResult(
            success=True,
            booking_id=booking.id,
            change_request_id=change_request.id,
            status=change_request.status,
            redispatched=False,
            payload={
                "edit_version": int(booking.edit_version or 1),
                "change_request": change_request.serialize(),
                "effect_status": change_request.effect_status,
            },
        )

    def _reject(
        self,
        booking: Booking,
        change_request: BookingChangeRequest,
        input_data: RespondToChangeRequestInput,
    ) -> RespondToChangeRequestResult:
        reason = (input_data.reason or "").strip() or "Signalé comme problématique"
        if change_request.action_type == TransportActionType.CANCELLATION:
            reason = (
                input_data.reason or ""
            ).strip() or "Problème signalé par le transporteur"

        _append_exchange(
            change_request,
            decision_type=TransportActionExchangeDecision.REJECT,
            actor_type="company",
            actor_id=input_data.user_id,
            values=(
                {"rejection_reason_code": input_data.rejection_reason_code}
                if input_data.rejection_reason_code
                else None
            ),
            comment=reason,
            created_from="company_portal",
            idempotency_key=input_data.idempotency_key,
            snapshot=build_decision_context_snapshot(booking),
        )

        change_request.status = TransportActionStatus.REJECTED
        change_request.effect_status = TransportActionEffectStatus.NONE
        change_request.next_actor_type = TransportActionNextActor.NONE
        change_request.rejection_reason = reason
        change_request.responded_by_user_id = input_data.user_id
        change_request.responded_by_role = "company"
        change_request.responded_at = datetime.now(UTC)
        change_request.version = int(change_request.version or 1) + 1
        from application.institutions.transport_action_workflow import (
            clear_active_change_request_refs,
        )

        clear_active_change_request_refs(change_request.id)
        assert_status_effect_combo(change_request.status, change_request.effect_status)

        record_change_event(
            booking=booking,
            transport_request=_resolve_transport_request(change_request),
            institution_id=change_request.institution_id,
            actor_user_id=input_data.user_id,
            actor_role="company",
            actor_type="company",
            actor_display_name=resolve_actor_name(input_data.user_id),
            action_type="change_request_refused",
            change_scope=(
                "cancellation"
                if change_request.action_type == TransportActionType.CANCELLATION
                else "operational"
            ),
            source="company_portal",
            before_snapshot=change_request.before_snapshot,
            after_snapshot=change_request.before_snapshot,
            reason=reason,
            change_class="major",
            severity="INFO",
            ack_required=False,
            operational_impact={
                "transport_action_rejected": True,
                "mission_unchanged": True,
            },
        )

        self._record_response_timeline(
            change_request=change_request,
            event_type="change_refused_by_company",
            company_id=input_data.company_id,
            user_id=input_data.user_id,
        )

        db.session.commit()
        self._audit(input_data, change_request, accepted=False)

        try:
            from domain.events.events import TransportActionRejectedEvent
            from shared.events.event_bus import publish_event

            publish_event(
                TransportActionRejectedEvent(
                    action_id=change_request.id,
                    booking_id=booking.id,
                    action_type=change_request.action_type or "",
                    rejection_reason=reason,
                )
            )
        except Exception as exc:
            logger.warning("[RespondToChangeRequest] reject event: %s", exc)

        return RespondToChangeRequestResult(
            success=True,
            booking_id=booking.id,
            change_request_id=change_request.id,
            status=change_request.status,
            redispatched=False,
            payload={
                "mission_unchanged": True,
                "change_request": change_request.serialize(),
            },
        )

    def _counter(
        self,
        booking: Booking,
        change_request: BookingChangeRequest,
        input_data: RespondToChangeRequestInput,
    ) -> RespondToChangeRequestResult:
        values = input_data.counter_values or {}
        if not values:
            return RespondToChangeRequestResult(
                success=False,
                booking_id=booking.id,
                change_request_id=change_request.id,
                error="counter_values obligatoire.",
                status_code=400,
            )

        from application.institutions.transport_action_policies import (
            default_negotiation_policy,
        )

        allowed = default_negotiation_policy().allowed_counter_fields(
            change_request.action_type or ""
        )
        if allowed:
            rejected = set(values.keys()) - set(allowed)
            if rejected:
                return RespondToChangeRequestResult(
                    success=False,
                    booking_id=booking.id,
                    change_request_id=change_request.id,
                    error=f"Champs non négociables: {sorted(rejected)}",
                    status_code=400,
                )

        _append_exchange(
            change_request,
            decision_type=TransportActionExchangeDecision.COUNTER,
            actor_type="company",
            actor_id=input_data.user_id,
            values=values,
            commercial_terms=input_data.commercial_terms,
            comment=input_data.reason,
            created_from="company_portal",
            idempotency_key=input_data.idempotency_key,
            snapshot=build_decision_context_snapshot(booking),
        )
        change_request.status = TransportActionStatus.COUNTER_PENDING
        change_request.effect_status = TransportActionEffectStatus.NONE
        change_request.next_actor_type = TransportActionNextActor.INSTITUTION
        change_request.proposed_patch = {
            **(change_request.proposed_patch or {}),
            **values,
        }
        change_request.version = int(change_request.version or 1) + 1
        assert_status_effect_combo(change_request.status, change_request.effect_status)
        db.session.commit()

        return RespondToChangeRequestResult(
            success=True,
            booking_id=booking.id,
            change_request_id=change_request.id,
            status=change_request.status,
            payload={
                "change_request": change_request.serialize(),
                "next_actor_type": "INSTITUTION",
            },
        )

    def _record_response_timeline(
        self,
        *,
        change_request: BookingChangeRequest,
        event_type: str,
        company_id: int,
        user_id: int | None,
    ) -> None:
        try:
            from services.institutions.transport_timeline_service import (
                TimelineActor,
                record_event,
            )

            record_event(
                event_type,
                institution_id=change_request.institution_id,
                transport_request_id=change_request.transport_request_id,
                booking_id=change_request.booking_id,
                actor=TimelineActor(actor_type="company", actor_user_id=user_id),
                payload={
                    "company_id": company_id,
                    "change_request_id": change_request.id,
                    "action_type": change_request.action_type,
                    "actor_name": resolve_actor_name(user_id),
                    "proposed_patch": change_request.proposed_patch,
                    "changed_fields": change_request.changed_fields,
                },
            )
        except Exception as exc:
            logger.warning("[RespondToChangeRequest] timeline: %s", exc)

    def _audit(
        self,
        input_data: RespondToChangeRequestInput,
        change_request: BookingChangeRequest,
        *,
        accepted: bool,
    ) -> None:
        with contextlib.suppress(Exception):
            AuditLogger.log(
                action=(
                    "transport_action_accepted"
                    if accepted
                    else "transport_action_rejected"
                ),
                user_id=input_data.user_id,
                details={
                    "booking_id": input_data.booking_id,
                    "change_request_id": change_request.id,
                    "action_type": change_request.action_type,
                },
            )
