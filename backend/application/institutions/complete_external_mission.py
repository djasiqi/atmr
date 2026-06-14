# application/institutions/complete_external_mission.py
"""Use case: Déclarer une mission externe comme réalisée par l'institution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from ext import db
from models import RequestStatus, TransportRequest, User
from models.enums import CarrierSource
from security.audit_log import AuditLogger

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompleteExternalMissionInput:
    """Entrée pour la déclaration de réalisation externe."""

    transport_request_id: int
    institution_id: int
    user_id: int
    executed_at: datetime | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class CompleteExternalMissionResult:
    """Résultat de la déclaration externe."""

    success: bool
    transport_request_id: int
    error: str | None = None
    status_code: int = 200


class CompleteExternalMissionUseCase:
    """Use case: déclarer une mission externe comme réalisée."""

    def execute(
        self, input_data: CompleteExternalMissionInput
    ) -> CompleteExternalMissionResult:
        try:
            transport_request = (
                db.session.query(TransportRequest)
                .filter(
                    TransportRequest.id == input_data.transport_request_id,
                    TransportRequest.institution_id == input_data.institution_id,
                )
                .with_for_update()
                .first()
            )
            if not transport_request:
                return CompleteExternalMissionResult(
                    success=False,
                    transport_request_id=input_data.transport_request_id,
                    error="Demande de transport introuvable",
                    status_code=404,
                )

            if transport_request.carrier_source != CarrierSource.EXTERNAL.value:
                return CompleteExternalMissionResult(
                    success=False,
                    transport_request_id=transport_request.id,
                    error="Cette demande n'est pas en mode transporteur externe",
                    status_code=409,
                )

            if transport_request.status != RequestStatus.EXTERNAL_ASSIGNED.value:
                return CompleteExternalMissionResult(
                    success=False,
                    transport_request_id=transport_request.id,
                    error=(
                        f"Demande en statut {transport_request.status}, "
                        "déclaration impossible"
                    ),
                    status_code=409,
                )

            executed_at = input_data.executed_at or datetime.now(UTC)
            notes = (input_data.notes or "").strip() or None
            declared_by = self._user_display_name(input_data.user_id)
            carrier_name = transport_request.external_carrier_name or "Transporteur externe"

            transport_request.executed_externally_at = executed_at
            transport_request.executed_externally_by_user_id = input_data.user_id
            transport_request.external_execution_notes = notes
            transport_request.status = RequestStatus.EXTERNAL_DECLARED_COMPLETED.value

            self._record_timeline(
                transport_request=transport_request,
                user_id=input_data.user_id,
                carrier_name=carrier_name,
                declared_by=declared_by,
                declared_at=executed_at,
                notes=notes,
            )

            db.session.commit()

            try:
                AuditLogger.log_action(
                    action_type="external_mission_completed",
                    action_category="institution",
                    user_id=input_data.user_id,
                    user_type="institution",
                    institution_id=input_data.institution_id,
                    result_status="success",
                    action_details={
                        "transport_request_id": transport_request.id,
                        "carrier_name": carrier_name,
                        "executed_at": executed_at.isoformat(),
                    },
                )
            except Exception as audit_err:
                logger.warning("[CompleteExternalMission] Audit log error: %s", audit_err)

            return CompleteExternalMissionResult(
                success=True,
                transport_request_id=transport_request.id,
            )
        except Exception as exc:
            logger.exception(
                "Erreur lors de la déclaration externe request=%s",
                input_data.transport_request_id,
            )
            db.session.rollback()
            return CompleteExternalMissionResult(
                success=False,
                transport_request_id=input_data.transport_request_id,
                error=f"Erreur inattendue: {exc!s}",
                status_code=500,
            )

    @staticmethod
    def _user_display_name(user_id: int) -> str | None:
        user = User.query.get(user_id)
        if not user:
            return None
        first = getattr(user, "first_name", "") or ""
        last = getattr(user, "last_name", "") or ""
        name = f"{first} {last}".strip()
        return name or getattr(user, "username", None)

    @staticmethod
    def _record_timeline(
        *,
        transport_request: TransportRequest,
        user_id: int,
        carrier_name: str,
        declared_by: str | None,
        declared_at: datetime,
        notes: str | None,
    ) -> None:
        try:
            from services.institutions.transport_timeline_service import (
                TimelineActor,
                record_event,
            )

            record_event(
                "external_mission_completed",
                institution_id=transport_request.institution_id,
                transport_request_id=transport_request.id,
                actor=TimelineActor(
                    actor_type="institution",
                    actor_user_id=user_id,
                ),
                payload={
                    "carrier_name": carrier_name,
                    "declared_by": declared_by,
                    "declared_at": declared_at.isoformat(),
                    "notes": notes,
                },
                correlation_id=f"external_mission_completed:{transport_request.id}:{user_id}",
            )
        except Exception as timeline_err:
            logger.warning(
                "[CompleteExternalMission] Timeline recording failed: %s",
                timeline_err,
            )
