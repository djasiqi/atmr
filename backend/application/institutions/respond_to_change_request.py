# application/institutions/respond_to_change_request.py
"""Use case: Réponse transporteur à une demande de validation de modification.

Le transporteur (entreprise) accepte ou refuse une modification critique
demandée par l'institution sur une course déjà acceptée/assignée.

- accept : applique le patch, conserve le transporteur, clôt la BCR (accepted)
- refuse : applique le patch (la modification institution fait foi), libère la
  course et la remet en diffusion (redispatch)

Verrou optimiste : la version fournie par le client doit correspondre à la
version courante de la BCR, et la BCR doit être la demande active du booking
(active_change_request_id). Sinon 409 (concurrence / superseded).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ext import db
from models import Booking, BookingChangeRequest
from models.booking_change_request import BookingChangeRequestStatus
from security.audit_log import AuditLogger
from services.institutions.booking_change_service import (
    _booking_operational_snapshot,
    apply_operational_patch,
    bump_edit_version,
    record_change_event,
)

logger = logging.getLogger(__name__)

ACTION_ACCEPT = "accept"
ACTION_REFUSE = "refuse"


@dataclass(frozen=True, slots=True)
class RespondToChangeRequestInput:
    """Input pour la réponse transporteur."""

    booking_id: int
    change_request_id: int
    company_id: int
    user_id: int | None
    action: str  # "accept" | "refuse"
    version: int
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RespondToChangeRequestResult:
    """Résultat de la réponse transporteur."""

    success: bool
    booking_id: int
    change_request_id: int
    status: str | None = None
    redispatched: bool = False
    error: str | None = None
    status_code: int = 200
    payload: dict[str, Any] | None = None


class RespondToChangeRequestUseCase:
    """Use case: accepter / refuser une demande de validation de modification."""

    def execute(
        self, input_data: RespondToChangeRequestInput
    ) -> RespondToChangeRequestResult:
        action = (input_data.action or "").strip().lower()
        if action not in (ACTION_ACCEPT, ACTION_REFUSE):
            return RespondToChangeRequestResult(
                success=False,
                booking_id=input_data.booking_id,
                change_request_id=input_data.change_request_id,
                error="Action invalide (accept|refuse).",
                status_code=400,
            )

        try:
            # Verrou pessimiste sur le booking (sérialise les réponses concurrentes)
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

            # Vérifier l'appartenance (entreprise propriétaire ou exécutante)
            owner_id = booking.company_id or booking.executing_company_id
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
                    error="Demande de modification introuvable.",
                    status_code=404,
                )

            # Statut de la BCR
            if change_request.status != BookingChangeRequestStatus.PENDING:
                return RespondToChangeRequestResult(
                    success=False,
                    booking_id=input_data.booking_id,
                    change_request_id=change_request.id,
                    error=(
                        "Cette demande de modification n'est plus en attente "
                        f"(statut {change_request.status})."
                    ),
                    status_code=409,
                    payload={"current_status": change_request.status},
                )

            # La BCR doit être la demande active du booking (sinon superseded)
            if int(booking.active_change_request_id or 0) != int(change_request.id):
                return RespondToChangeRequestResult(
                    success=False,
                    booking_id=input_data.booking_id,
                    change_request_id=change_request.id,
                    error="Cette demande a été remplacée par une plus récente.",
                    status_code=409,
                    payload={
                        "active_change_request_id": booking.active_change_request_id
                    },
                )

            # Verrou optimiste de version
            current_version = int(change_request.version or 1)
            if int(input_data.version) != current_version:
                return RespondToChangeRequestResult(
                    success=False,
                    booking_id=input_data.booking_id,
                    change_request_id=change_request.id,
                    error=(
                        "Conflit de version : la demande a évolué entre-temps."
                    ),
                    status_code=409,
                    payload={"current_version": current_version},
                )

            if action == ACTION_ACCEPT:
                return self._accept(booking, change_request, input_data)
            return self._refuse(booking, change_request, input_data)

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

    def _accept(
        self,
        booking: Booking,
        change_request: BookingChangeRequest,
        input_data: RespondToChangeRequestInput,
    ) -> RespondToChangeRequestResult:
        before = _booking_operational_snapshot(booking)
        try:
            updated_fields = apply_operational_patch(
                booking, change_request.proposed_patch or {}
            )
        except ValueError as e:
            db.session.rollback()
            return RespondToChangeRequestResult(
                success=False,
                booking_id=booking.id,
                change_request_id=change_request.id,
                error=str(e),
                status_code=400,
            )

        bump_edit_version(booking)
        after = _booking_operational_snapshot(booking)

        change_request.status = BookingChangeRequestStatus.ACCEPTED
        change_request.responded_by_user_id = input_data.user_id
        change_request.responded_by_role = "company"
        change_request.responded_at = datetime.now(UTC)
        change_request.version = int(change_request.version or 1) + 1
        change_request.after_snapshot = after
        booking.active_change_request_id = None

        record_change_event(
            booking=booking,
            transport_request=None,
            institution_id=change_request.institution_id,
            actor_user_id=input_data.user_id,
            actor_role="company",
            actor_type="company",
            actor_display_name=None,
            action_type="field_updated",
            change_scope="operational",
            source="company_portal",
            before_snapshot=before,
            after_snapshot=after,
            reason=input_data.reason or change_request.reason,
            change_class="major",
            severity="INFO",
            ack_required=False,
            operational_impact={"revalidation_accepted": True},
        )

        self._record_response_timeline(
            change_request=change_request,
            event_type="change_accepted_by_company",
            company_id=input_data.company_id,
            user_id=input_data.user_id,
        )

        db.session.commit()

        self._audit(input_data, change_request, accepted=True, redispatched=False)

        return RespondToChangeRequestResult(
            success=True,
            booking_id=booking.id,
            change_request_id=change_request.id,
            status=change_request.status,
            redispatched=False,
            payload={
                "updated_fields": updated_fields,
                "edit_version": int(booking.edit_version or 1),
                "change_request": change_request.serialize(),
            },
        )

    def _refuse(
        self,
        booking: Booking,
        change_request: BookingChangeRequest,
        input_data: RespondToChangeRequestInput,
    ) -> RespondToChangeRequestResult:
        before = _booking_operational_snapshot(booking)

        # La modification institution fait foi : on applique le patch demandé
        # puis on libère la course pour rediffusion (le transporteur refuse).
        try:
            apply_operational_patch(booking, change_request.proposed_patch or {})
        except ValueError as e:
            db.session.rollback()
            return RespondToChangeRequestResult(
                success=False,
                booking_id=booking.id,
                change_request_id=change_request.id,
                error=str(e),
                status_code=400,
            )

        bump_edit_version(booking)
        after = _booking_operational_snapshot(booking)

        change_request.status = BookingChangeRequestStatus.REFUSED
        change_request.responded_by_user_id = input_data.user_id
        change_request.responded_by_role = "company"
        change_request.responded_at = datetime.now(UTC)
        change_request.version = int(change_request.version or 1) + 1
        change_request.after_snapshot = after
        booking.active_change_request_id = None

        record_change_event(
            booking=booking,
            transport_request=None,
            institution_id=change_request.institution_id,
            actor_user_id=input_data.user_id,
            actor_role="company",
            actor_type="company",
            actor_display_name=None,
            action_type="field_updated",
            change_scope="operational",
            source="company_portal",
            before_snapshot=before,
            after_snapshot=after,
            reason=input_data.reason or change_request.reason,
            change_class="major",
            severity="WARNING",
            ack_required=False,
            operational_impact={"revalidation_refused": True},
        )

        self._record_response_timeline(
            change_request=change_request,
            event_type="change_refused_by_company",
            company_id=input_data.company_id,
            user_id=input_data.user_id,
        )

        db.session.flush()

        # Libération + rediffusion (use case dédié)
        redispatched = False
        try:
            from application.institutions.release_booking_for_redispatch import (
                ReleaseBookingForRedispatchInput,
                ReleaseBookingForRedispatchUseCase,
            )

            release_result = ReleaseBookingForRedispatchUseCase().execute(
                ReleaseBookingForRedispatchInput(
                    booking_id=booking.id,
                    institution_id=change_request.institution_id,
                    reason=input_data.reason or "Refus modification transporteur",
                    previous_company_id=int(input_data.company_id),
                    actor_user_id=input_data.user_id,
                    trigger_redispatch=True,
                )
            )
            redispatched = release_result.redispatched
        except Exception as release_err:
            logger.warning(
                "[RespondToChangeRequest] release/redispatch failed booking=%s: %s",
                booking.id,
                release_err,
            )

        db.session.commit()

        self._audit(
            input_data, change_request, accepted=False, redispatched=redispatched
        )

        return RespondToChangeRequestResult(
            success=True,
            booking_id=booking.id,
            change_request_id=change_request.id,
            status=change_request.status,
            redispatched=redispatched,
            payload={
                "edit_version": int(booking.edit_version or 1),
                "change_request": change_request.serialize(),
            },
        )

    @staticmethod
    def _record_response_timeline(
        *,
        change_request: BookingChangeRequest,
        event_type: str,
        company_id: int,
        user_id: int | None,
    ) -> None:
        try:
            from services.institutions.transport_timeline_service import (
                TimelineActor,
                find_latest_event,
                record_event,
            )

            source = find_latest_event(
                booking_id=change_request.booking_id,
                event_type="change_confirmation_requested",
            )
            record_event(
                event_type,
                institution_id=change_request.institution_id,
                transport_request_id=change_request.transport_request_id,
                booking_id=change_request.booking_id,
                actor=TimelineActor(
                    actor_type="company",
                    actor_user_id=user_id,
                    company_id=company_id,
                ),
                payload={"change_request_id": change_request.id},
                correlation_id=f"{event_type}:{change_request.id}",
                source_event_id=source.id if source else None,
            )
        except Exception as timeline_err:
            logger.warning(
                "[RespondToChangeRequest] timeline %s failed: %s",
                event_type,
                timeline_err,
            )

    @staticmethod
    def _audit(
        input_data: RespondToChangeRequestInput,
        change_request: BookingChangeRequest,
        *,
        accepted: bool,
        redispatched: bool,
    ) -> None:
        try:
            AuditLogger.log_action(
                action_type=(
                    "change_request_accepted"
                    if accepted
                    else "change_request_refused"
                ),
                action_category="institution",
                user_id=input_data.user_id,
                user_type="company",
                company_id=input_data.company_id,
                institution_id=change_request.institution_id,
                result_status="success",
                action_details={
                    "booking_id": input_data.booking_id,
                    "change_request_id": change_request.id,
                    "redispatched": redispatched,
                },
            )
        except Exception as audit_err:
            logger.warning("[RespondToChangeRequest] audit failed: %s", audit_err)
