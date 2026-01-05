"""Repository pour l'accès aux données DispatchRun avec conversion DTO.

Ce repository découple les services de l'implémentation SQLAlchemy
en retournant des DTOs au lieu de modèles SQLAlchemy directs.
"""

from datetime import date
from typing import Any

from domain.dispatch_run_dto import DispatchRunDTO
from models import DispatchRun

logger = __import__("logging").getLogger(__name__)


class DispatchRunRepository:
    """Repository pour l'accès aux données DispatchRun avec conversion DTO."""

    def _to_dto(self, dispatch_run: DispatchRun) -> DispatchRunDTO:
        """Convertit un modèle DispatchRun SQLAlchemy en DTO.

        Args:
            dispatch_run: Modèle DispatchRun SQLAlchemy

        Returns:
            DispatchRunDTO correspondant
        """
        return DispatchRunDTO(
            id=dispatch_run.id,
            company_id=dispatch_run.company_id,
            day=dispatch_run.day,
            status=dispatch_run.status,
            started_at=dispatch_run.started_at,
            completed_at=dispatch_run.completed_at,
            created_at=dispatch_run.created_at,
            config=(dict(dispatch_run.config) if dispatch_run.config else None),
            metrics=(dict(dispatch_run.metrics) if dispatch_run.metrics else None),
        )

    def find_by_id(self, dispatch_run_id: int) -> DispatchRunDTO | None:
        """Trouve un dispatch run par son ID.

        Args:
            dispatch_run_id: ID du dispatch run

        Returns:
            DispatchRunDTO ou None si non trouvé
        """
        dispatch_run = DispatchRun.query.get(dispatch_run_id)
        if dispatch_run is None:
            return None
        return self._to_dto(dispatch_run)

    def find_by_company_and_day(
        self, company_id: int, day: date
    ) -> DispatchRunDTO | None:
        """Trouve un dispatch run par entreprise et jour.

        Args:
            company_id: ID de l'entreprise
            day: Date du jour

        Returns:
            DispatchRunDTO ou None si non trouvé
        """
        dispatch_run = DispatchRun.query.filter_by(
            company_id=company_id, day=day
        ).first()
        if dispatch_run is None:
            return None
        return self._to_dto(dispatch_run)

    def find_by_company_id(
        self, company_id: int, limit: int | None = None
    ) -> list[DispatchRunDTO]:
        """Trouve les dispatch runs d'une entreprise.

        Args:
            company_id: ID de l'entreprise
            limit: Nombre maximum de résultats (optionnel)

        Returns:
            Liste de DispatchRunDTO
        """
        query = DispatchRun.query.filter_by(company_id=company_id).order_by(
            DispatchRun.created_at.desc()
        )
        if limit:
            query = query.limit(limit)
        dispatch_runs = query.all()
        return [self._to_dto(dr) for dr in dispatch_runs]

    def find_model_by_company_and_day(
        self, company_id: int, day: date
    ) -> DispatchRun | None:
        """Trouve un dispatch run par entreprise et jour (retourne le modèle SQLAlchemy).

        Args:
            company_id: ID de l'entreprise
            day: Date du jour

        Returns:
            DispatchRun ou None si non trouvé
        """
        return DispatchRun.query.filter_by(company_id=company_id, day=day).first()

    def find_model_by_id_and_company(
        self, dispatch_run_id: int, company_id: int
    ) -> DispatchRun | None:
        """Trouve un dispatch run par ID et company_id (retourne le modèle SQLAlchemy).

        Args:
            dispatch_run_id: ID du dispatch run
            company_id: ID de l'entreprise

        Returns:
            DispatchRun ou None si non trouvé
        """
        return DispatchRun.query.filter_by(
            id=dispatch_run_id, company_id=company_id
        ).first()

    def find_model_by_company_id_query(self, company_id: int):
        """Retourne une query DispatchRun filtrée par company_id (pour compatibilité avec code existant).

        Args:
            company_id: ID de l'entreprise

        Returns:
            Query SQLAlchemy filtrée
        """
        return DispatchRun.query.filter_by(company_id=company_id)

    def find_model_by_company_and_day_ordered(
        self, company_id: int, day: date
    ) -> DispatchRun | None:
        """Trouve un dispatch run par entreprise et jour, trié par created_at desc (retourne le modèle SQLAlchemy).

        Args:
            company_id: ID de l'entreprise
            day: Date du jour

        Returns:
            DispatchRun ou None si non trouvé
        """
        return (
            DispatchRun.query.filter_by(company_id=company_id, day=day)
            .order_by(DispatchRun.created_at.desc())
            .first()
        )

    def find_models_by_company_and_date_range(
        self, company_id: int, start_date: date, limit: int | None = None
    ) -> list[DispatchRun]:
        """Trouve les dispatch runs d'une entreprise à partir d'une date (retourne les modèles SQLAlchemy).

        Args:
            company_id: ID de l'entreprise
            start_date: Date de début (inclusive)
            limit: Nombre maximum de résultats (optionnel)

        Returns:
            Liste de DispatchRun
        """
        query = DispatchRun.query.filter(
            DispatchRun.company_id == company_id,
            DispatchRun.created_at >= start_date,
        ).order_by(DispatchRun.created_at.desc())
        if limit:
            query = query.limit(limit)
        return query.all()

    def find_models_by_company_with_custom_order(
        self,
        company_id: int,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[DispatchRun]:
        """Trouve les dispatch runs d'une entreprise avec tri personnalisé (retourne les modèles SQLAlchemy).

        Args:
            company_id: ID de l'entreprise
            limit: Nombre maximum de résultats (optionnel)
            offset: Nombre de résultats à sauter (optionnel)

        Returns:
            Liste de DispatchRun triés par ordre de priorité (completed_at > started_at > day > created_at > id)
        """
        query = DispatchRun.query.filter_by(company_id=company_id)

        # Fallback de tri: completed_at > started_at > day > created_at > id
        order_cols = []
        if hasattr(DispatchRun, "completed_at"):
            order_cols.append(DispatchRun.completed_at.desc())
        if hasattr(DispatchRun, "started_at"):
            order_cols.append(DispatchRun.started_at.desc())
        if hasattr(DispatchRun, "day"):
            order_cols.append(DispatchRun.day.desc())
        if hasattr(DispatchRun, "created_at"):
            order_cols.append(DispatchRun.created_at.desc())
        order_cols.append(DispatchRun.id.desc())

        query = query.order_by(*order_cols)

        if limit:
            query = query.limit(limit)
        if offset:
            query = query.offset(offset)

        return query.all()

    def find_models_by_company_and_date_from(
        self, company_id: int, start_date: Any, limit: int | None = None
    ) -> list[DispatchRun]:
        """Trouve les dispatch runs d'une entreprise à partir d'une date (retourne les modèles SQLAlchemy).

        Args:
            company_id: ID de l'entreprise
            start_date: Date de début (inclusive)
            limit: Nombre maximum de résultats (optionnel)

        Returns:
            Liste de DispatchRun triés par created_at desc
        """
        query = DispatchRun.query.filter(
            DispatchRun.company_id == company_id,
            DispatchRun.created_at >= start_date,
        ).order_by(DispatchRun.created_at.desc())
        if limit:
            query = query.limit(limit)
        return query.all()
