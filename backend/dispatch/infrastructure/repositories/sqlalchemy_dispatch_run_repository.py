"""Implémentation SQLAlchemy du repository DispatchRun."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from dispatch.domain.dispatch_run import DispatchRun
from dispatch.domain.dispatch_run_id import DispatchRunId

if TYPE_CHECKING:
    from models import DispatchRun as SQLAlchemyDispatchRun
else:
    SQLAlchemyDispatchRun = Any

logger = __import__("logging").getLogger(__name__)


class SqlAlchemyDispatchRunRepository:
    """Implémentation SQLAlchemy du repository DispatchRun.

    Adapte les modèles SQLAlchemy vers les agrégats du domaine.
    """

    def _to_aggregate(self, sa_run: SQLAlchemyDispatchRun) -> DispatchRun:
        """Convertit un modèle SQLAlchemy en agrégat DispatchRun."""
        from dispatch.domain.value_objects import DispatchMetrics, DispatchStatus

        # Construire DispatchStatus
        status = DispatchStatus(str(sa_run.status.value))

        # Construire DispatchMetrics si présent
        metrics = None
        if sa_run.metrics:
            metrics_data = sa_run.metrics
            metrics = DispatchMetrics(
                assignments_count=metrics_data.get("assignments_count", 0),
                unassigned_count=metrics_data.get("unassigned_count", 0),
                total_distance_km=metrics_data.get("total_distance_km", 0.0),
                total_duration_minutes=metrics_data.get("total_duration_minutes", 0),
                average_wait_time_minutes=metrics_data.get(
                    "average_wait_time_minutes", 0.0
                ),
            )

        return DispatchRun(
            id=DispatchRunId(sa_run.id),
            company_id=sa_run.company_id,
            day=sa_run.day,
            status=status,
            started_at=sa_run.started_at,
            completed_at=sa_run.completed_at,
            created_at=sa_run.created_at,
            config=dict(sa_run.config) if sa_run.config else None,
            metrics=metrics,
        )

    def _from_aggregate(self, dispatch_run: DispatchRun) -> dict[str, Any]:
        """Convertit un agrégat DispatchRun en dictionnaire pour SQLAlchemy."""
        data: dict[str, Any] = {
            "id": dispatch_run.id.value,
            "company_id": dispatch_run.company_id,
            "day": dispatch_run.day,
            "status": dispatch_run.status.value,
            "started_at": dispatch_run.started_at,
            "completed_at": dispatch_run.completed_at,
            "created_at": dispatch_run.created_at,
            "config": dispatch_run.config,
        }

        # Convertir DispatchMetrics en dict pour metrics
        if dispatch_run.metrics:
            data["metrics"] = {
                "assignments_count": dispatch_run.metrics.assignments_count,
                "unassigned_count": dispatch_run.metrics.unassigned_count,
                "total_distance_km": dispatch_run.metrics.total_distance_km,
                "total_duration_minutes": dispatch_run.metrics.total_duration_minutes,
                "average_wait_time_minutes": dispatch_run.metrics.average_wait_time_minutes,
            }
        else:
            data["metrics"] = None

        return data

    def save(self, dispatch_run: DispatchRun) -> None:
        """Sauvegarde un dispatch run."""
        from ext import db
        from models import DispatchRun as SQLAlchemyDispatchRun
        from models.enums import DispatchStatus as SQLAlchemyDispatchStatus

        data = self._from_aggregate(dispatch_run)
        run_id = data.pop("id")

        sa_run = SQLAlchemyDispatchRun.query.get(run_id)
        if sa_run:
            # Update
            for key, value in data.items():
                if key == "status":
                    # Convertir string en enum
                    status_enum = SQLAlchemyDispatchStatus(value)
                    setattr(sa_run, key, status_enum)
                else:
                    setattr(sa_run, key, value)
        else:
            # Create
            # Convertir status en enum
            status_enum = SQLAlchemyDispatchStatus(data["status"])
            data["status"] = status_enum
            sa_run = SQLAlchemyDispatchRun(**data)
            db.session.add(sa_run)

        db.session.commit()

    def find_by_id(self, run_id: DispatchRunId) -> DispatchRun | None:
        """Trouve un dispatch run par ID."""
        from models import DispatchRun as SQLAlchemyDispatchRun

        sa_run = SQLAlchemyDispatchRun.query.get(run_id.value)
        if sa_run is None:
            return None
        return self._to_aggregate(sa_run)

    def find_by_company_and_day(self, company_id: int, day: date) -> DispatchRun | None:
        """Trouve un dispatch run par entreprise et jour."""
        from models import DispatchRun as SQLAlchemyDispatchRun

        sa_run = SQLAlchemyDispatchRun.query.filter_by(
            company_id=company_id, day=day
        ).first()
        if sa_run is None:
            return None
        return self._to_aggregate(sa_run)
