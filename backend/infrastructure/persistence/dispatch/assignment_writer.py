from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from ext import db
from models import Assignment, AssignmentStatus, DispatchRun
from models.enums import DispatchStatus as DispatchStatusEnum
from repositories.assignment_repository import AssignmentRepository
from repositories.dispatch_run_repository import DispatchRunRepository
from shared.time_utils import to_geneva_local


class _BookingLike(Protocol):
    id: int | None
    scheduled_time: Any


@dataclass(frozen=True, slots=True)
class SqlAlchemyAssignmentWriter:
    """Adaptateur Infrastructure: garantit un Assignment + DispatchRun cohérent."""

    dispatch_run_repo: DispatchRunRepository
    assignment_repo: AssignmentRepository

    def ensure_assignment_for_booking(
        self, *, company_id: int, booking: _BookingLike, driver_id: int
    ) -> None:
        booking_id_obj = getattr(booking, "id", None)
        if booking_id_obj is None:
            raise ValueError("booking.id manquant")
        booking_id = int(booking_id_obj)

        st = getattr(booking, "scheduled_time", None)
        if st is None:
            day_local = datetime.now(UTC).date()
        else:
            dt_local_any = to_geneva_local(st)
            # certains stubs typent to_geneva_local -> Optional[datetime]
            day_local = st.date() if dt_local_any is None else dt_local_any.date()

        dispatch_run = self.dispatch_run_repo.find_model_by_company_and_day(
            company_id, day_local
        )
        if not dispatch_run:
            dispatch_run = DispatchRun()
            dispatch_run.company_id = company_id
            dispatch_run.day = day_local
            dispatch_run.status = DispatchStatusEnum.COMPLETED
            db.session.add(dispatch_run)
            db.session.flush()

        assignment = self.assignment_repo.find_model_by_booking_id(booking_id)
        if not assignment:
            assignment = Assignment()
            assignment.booking_id = booking_id
            assignment.driver_id = driver_id
            assignment.dispatch_run_id = cast("Any", dispatch_run).id
            assignment.status = AssignmentStatus.SCHEDULED
            db.session.add(assignment)
        else:
            try:
                same_driver = assignment.driver_id is not None and int(
                    cast("Any", assignment.driver_id)
                ) == int(driver_id)
            except (TypeError, ValueError):
                same_driver = False
            assignment.dispatch_run_id = cast("Any", dispatch_run).id
            if not same_driver:
                # Nouveau chauffeur = nouveau cycle opérationnel.
                assignment.driver_id = driver_id
                assignment.status = AssignmentStatus.SCHEDULED
                try:
                    assignment.revision = (
                        int(getattr(assignment, "revision", 0) or 0) + 1
                    )
                except (TypeError, ValueError):
                    assignment.revision = 1
            # Même chauffeur : ne JAMAIS régresser la progression
            # (EN_ROUTE_PICKUP/ARRIVED_PICKUP/ONBOARD/…) via un simple ensure.
