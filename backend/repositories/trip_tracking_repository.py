"""Repository pour l'accès aux données TripTracking."""

from models import TripTracking

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
        return (
            TripTracking.query.filter_by(assignment_id=assignment_id)
            .order_by(TripTracking.timestamp.asc())
            .all()
        )
