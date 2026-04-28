"""Repository pour l'accès aux données TripTracking."""

from ext import db
from models import TripTracking
from services.data_plane.read_write_routing import ReadContext, data_plane_router

logger = __import__("logging").getLogger(__name__)


class TripTrackingRepository:
    """Repository pour l'accès aux données TripTracking."""

    def find_models_by_assignment_id(self, assignment_id: int) -> list[TripTracking]:
        """Trouve les positions de tracking par assignment_id.

        Args:
            assignment_id: ID de l'assignment

        Returns:
            Liste de TripTracking triés par timestamp ascendant
        """
        bind_key = data_plane_router.read_bind(
            ReadContext(
                critical_post_write=False,
                analytical=True,
            )
        )
        stmt = (
            db.select(TripTracking)
            .where(TripTracking.assignment_id == assignment_id)
            .order_by(TripTracking.timestamp.asc())
            .execution_options(bind_key=bind_key)
        )
        return list(db.session.execute(stmt).scalars().all())
