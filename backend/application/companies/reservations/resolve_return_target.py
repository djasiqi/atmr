"""Résolution autoritaire de la cible retour pour trigger-return.

Ordre de résolution :
1. booking courant = retour classique → modify_current
2. TransportRequest avec legs → return leg (is_return_stop) → modify_leg_return
3. enfant classique is_return=true → modify_existing_classic_return
4. aucune topologie retour → create_new
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from repositories.booking_repository import BookingRepository

ReturnTargetAction = Literal[
    "modify_current",
    "modify_leg_return",
    "modify_existing_classic_return",
    "create_new",
]

ReturnTargetSource = Literal[
    "current_return",
    "institution_return_leg",
    "classic_child_return",
    "none",
]


@dataclass(frozen=True, slots=True)
class ReturnTargetResolution:
    action: ReturnTargetAction
    source: ReturnTargetSource
    target_booking: Any | None = None


class ReturnTopologyError(Exception):
    """Topologie retour institution incohérente ou ambiguë — pas de création de secours."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "return_topology_inconsistent",
        http_status: int = 409,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.http_status = http_status
        self.details = details or {}

    def to_payload(self) -> tuple[dict[str, Any], int]:
        body: dict[str, Any] = {
            "error": self.error_code,
            "message": self.message,
        }
        if self.details:
            body["details"] = self.details
        return body, self.http_status


def _find_transport_request_for_booking(booking: Any) -> Any | None:
    """Localise la TransportRequest source sans heuristique route_sequence MAX."""
    from models.transport_request import TransportRequest
    from models.transport_request_leg import TransportRequestLeg

    leg = TransportRequestLeg.query.filter_by(booking_id=booking.id).first()
    if leg is not None:
        return TransportRequest.query.get(leg.transport_request_id)

    tr = TransportRequest.query.filter_by(booking_id=booking.id).first()
    if tr is not None:
        return tr

    reqs = getattr(booking, "source_request", None)
    if reqs:
        return reqs[0] if isinstance(reqs, list) else reqs

    route_group_id = getattr(booking, "route_group_id", None)
    if route_group_id:
        candidates = TransportRequest.query.filter_by(
            route_group_id=route_group_id
        ).all()
        if len(candidates) > 1:
            raise ReturnTopologyError(
                "Plusieurs demandes institutionnelles partagent le même route_group_id.",
                details={"route_group_id": route_group_id},
            )
        if len(candidates) == 1:
            return candidates[0]

    return None


def resolve_existing_return_target(
    booking: Any,
    *,
    company_id: int,
    booking_repo: BookingRepository | None = None,
) -> ReturnTargetResolution:
    """Détermine quelle réservation retour doit être modifiée ou créée."""
    repo = booking_repo or BookingRepository()

    if bool(getattr(booking, "is_return", False)):
        return ReturnTargetResolution(
            action="modify_current",
            source="current_return",
            target_booking=booking,
        )

    transport_request = _find_transport_request_for_booking(booking)
    legs = sorted(
        getattr(transport_request, "legs", None) or [],
        key=lambda leg: getattr(leg, "sequence_index", 0),
    )
    if transport_request is not None and legs:
        return_stops = [
            leg for leg in legs if bool(getattr(leg, "is_return_stop", False))
        ]
        if len(return_stops) > 1:
            raise ReturnTopologyError(
                "Topologie retour institution ambiguë : plusieurs legs is_return_stop.",
                details={"transport_request_id": transport_request.id},
            )
        if len(return_stops) == 1:
            return_leg = return_stops[0]
            return_booking_id = getattr(return_leg, "booking_id", None)
            if return_booking_id is None:
                raise ReturnTopologyError(
                    "Le leg retour institution existe mais n'a pas de réservation associée.",
                    details={
                        "transport_request_id": transport_request.id,
                        "return_leg_id": return_leg.id,
                    },
                )
            target = repo.find_model_by_id_with_visibility(
                return_booking_id, company_id
            )
            if target is None:
                raise ReturnTopologyError(
                    "La réservation du leg retour est inaccessible pour cette entreprise.",
                    details={
                        "transport_request_id": transport_request.id,
                        "return_leg_id": return_leg.id,
                        "return_booking_id": return_booking_id,
                    },
                )
            if target.id == booking.id:
                return ReturnTargetResolution(
                    action="modify_current",
                    source="institution_return_leg",
                    target_booking=target,
                )
            return ReturnTargetResolution(
                action="modify_leg_return",
                source="institution_return_leg",
                target_booking=target,
            )

    existing = repo.find_model_by_parent_booking_id_and_company(
        booking.id,
        company_id,
        is_return=True,
    )
    if existing is not None:
        return ReturnTargetResolution(
            action="modify_existing_classic_return",
            source="classic_child_return",
            target_booking=existing,
        )

    return ReturnTargetResolution(
        action="create_new",
        source="none",
        target_booking=None,
    )
