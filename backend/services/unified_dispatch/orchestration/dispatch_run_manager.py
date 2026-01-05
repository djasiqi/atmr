# backend/services/unified_dispatch/orchestration/dispatch_run_manager.py
"""Gestionnaire de DispatchRun.

Ce module gère la création, la réutilisation et la mise à jour des DispatchRun.
Il est responsable de :
- La création de nouveaux DispatchRun avec gestion des race conditions
- La réutilisation de DispatchRun existants
- La mise à jour du statut des DispatchRun
- La finalisation des DispatchRun (marquage comme COMPLETED)

Side-effects:
    - Accès DB (lecture/écriture DispatchRun via DispatchRunRepository)
    - Transactions DB (commit/rollback)
    - Métriques: Tracking des IntegrityError (race conditions)
"""

from __future__ import annotations  # noqa: I001

import logging
from datetime import UTC, datetime
from typing import Any, Dict, cast

from ext import db
from models import Company, DispatchRun, DispatchStatus
from repositories.dispatch_run_repository import DispatchRunRepository
from services.unified_dispatch.error_metrics import track_integrity_error
from services.unified_dispatch.orchestration.utils import safe_int, to_date_ymd
from services.unified_dispatch.transaction_helpers import _begin_tx
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError

logger = logging.getLogger(__name__)


class DispatchRunManager:
    """Gestionnaire de DispatchRun pour le dispatch.

    Cette classe centralise la logique de gestion des DispatchRun :
    - Création avec gestion des race conditions (IntegrityError)
    - Réutilisation de DispatchRun existants
    - Mise à jour du statut (RUNNING, COMPLETED, FAILED)
    - Finalisation avec compteurs d'assignations

    Exemple:
        >>> manager = DispatchRunManager()
        >>> dispatch_run, error = manager.create_or_reuse(
        ...     company=company,
        ...     company_id=1,
        ...     day_str="2025-01-14",
        ...     mode="auto",
        ...     regular_first=True,
        ...     allow_emg=True,
        ...     for_date="2025-01-14",
        ...     existing_id=None
        ... )
        >>> if dispatch_run:
        ...     manager.update_status(dispatch_run, DispatchStatus.RUNNING)
        ...     # ... dispatch logic ...
        ...     manager.finalize(dispatch_run, assignments_count=42, unassigned_count=5)
    """

    def create_or_reuse(
        self,
        company: Company,
        company_id: int,
        day_str: str,
        mode: str,
        regular_first: bool,
        allow_emg: bool,
        for_date: str | None,
        existing_id: int | None,
    ) -> tuple[DispatchRun | None, Dict[str, Any] | None]:
        """Crée ou réutilise un DispatchRun.

        Crée un nouveau DispatchRun ou réutilise un existant selon les critères :
        - Si `existing_id` est fourni, réutilise ce DispatchRun (avec validation)
        - Sinon, cherche un DispatchRun existant pour company_id + day
        - Si aucun trouvé, crée un nouveau DispatchRun

        Gère les race conditions : si deux threads créent simultanément le même
        DispatchRun, l'un recevra IntegrityError et récupérera l'existant.

        Args:
            _company: Objet Company (pour validation)
            _company_id: ID de l'entreprise
            _day_str: Date au format YYYY-MM-DD
            _mode: Mode de dispatch
            _regular_first: Prioriser les courses régulières
            _allow_emg: Autoriser les courses d'urgence
            _for_date: Date du dispatch (pour contexte)
            _existing_id: ID d'un DispatchRun existant à réutiliser (optionnel)

        Returns:
            Tuple (dispatch_run, error_result) où :
            - dispatch_run: Objet DispatchRun si succès, None sinon
            - error_result: Dict avec résultat d'erreur structuré si échec,
              None si succès

        Raises:
            IntegrityError: Si échec de création et aucun DispatchRun existant trouvé
            ValueError: Si company est None lors de la création
            DBAPIError, OperationalError: Erreurs DB lors du commit

        Side-effects:
            - Accès DB (lecture/écriture DispatchRun)
            - Transactions DB (commit)
            - Métriques: Track IntegrityError si race condition
            - Logging: Contexte dispatch_run_id si disponible
        """
        # Convertir day_str en date
        try:
            day_date = to_date_ymd(day_str)
        except (ValueError, TypeError) as e:
            # Erreurs de parsing de date attendues
            logger.warning(
                "[DispatchRunManager] Invalid day_str=%r (validation error: %s), fallback to today",
                day_str,
                e,
            )
            day_date = datetime.now(UTC).date()
        except Exception as e:
            # Erreur inattendue : logger et utiliser fallback
            logger.warning(
                "[DispatchRunManager] Unexpected error parsing day_str=%r: %s, fallback to today",
                day_str,
                e,
            )
            day_date = datetime.now(UTC).date()

        logger.info(
            "[DispatchRunManager] Using day_date: %s for dispatch run", day_date
        )

        cfg = {
            "mode": mode,
            "regular_first": bool(regular_first),
            "allow_emergency": bool(allow_emg),
            "for_date": for_date,
        }

        # ✅ Utilisation du repository pour découpler de SQLAlchemy
        dispatch_run_repo = DispatchRunRepository()

        # ✅ Si existing_id est fourni, réutiliser le DispatchRun existant
        dispatch_run: DispatchRun | None = None
        if existing_id:
            dispatch_run_dto = dispatch_run_repo.find_by_id(existing_id)
            if dispatch_run_dto:
                # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
                dispatch_run = DispatchRun.query.get(dispatch_run_dto.id)
                if dispatch_run:
                    # Vérifier que le DispatchRun correspond à la company et à la date
                    if dispatch_run.company_id != company_id:
                        logger.warning(
                            "[DispatchRunManager] DispatchRun id=%s company_id=%s doesn't match requested company_id=%s, creating new",
                            existing_id,
                            dispatch_run.company_id,
                            company_id,
                        )
                        dispatch_run = None
                    elif dispatch_run.day != day_date:
                        logger.warning(
                            "[DispatchRunManager] DispatchRun id=%s day=%s doesn't match requested day=%s, creating new",
                            existing_id,
                            dispatch_run.day,
                            day_date,
                        )
                        dispatch_run = None
                    else:
                        logger.info(
                            "[DispatchRunManager] Reusing existing DispatchRun id=%s for company=%s day=%s",
                            existing_id,
                            company_id,
                            day_str,
                        )
                else:
                    dispatch_run = None
            else:
                logger.warning(
                    "[DispatchRunManager] DispatchRun id=%s not found, will create new",
                    existing_id,
                )
                dispatch_run = None
        else:
            # Comportement par défaut : chercher par company_id+day
            dispatch_run_dto = dispatch_run_repo.find_by_company_and_day(
                company_id, day_date
            )
            if dispatch_run_dto:
                # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
                dispatch_run = DispatchRun.query.get(dispatch_run_dto.id)
            else:
                dispatch_run = None

        # ✅ Définir le contexte logging pour dispatch_run_id
        dispatch_run_id_val = (
            safe_int(getattr(dispatch_run, "id", None)) if dispatch_run else None
        )
        if dispatch_run_id_val:
            try:
                from shared.dispatch_logging import set_dispatch_context

                set_dispatch_context(
                    dispatch_run_id=dispatch_run_id_val, company_id=company_id
                )
            except ImportError:
                pass  # Module optionnel

        if dispatch_run is None:
            # TX courte de création ; en cas de race → IntegrityError
            # ✅ FIX: Vérifier que la company existe avant de créer DispatchRun
            # Note: company ne devrait jamais être None ici car elle est validée avant,
            # mais on garde cette vérification pour la sécurité
            if not company:
                logger.error(
                    "[DispatchRunManager] Cannot create DispatchRun: company_id=%s does not exist",
                    company_id,
                )
                error_result = {
                    "assignments": [],
                    "unassigned": [],
                    "bookings": [],
                    "drivers": [],
                    "meta": {"reason": "company_not_found"},
                    "debug": {
                        "reason": "company_not_found",
                        "company_id": company_id,
                    },
                }
                return None, error_result
            try:
                with _begin_tx():
                    dr_any: Any = DispatchRun()
                    dr_any.company_id = int(company_id)
                    dr_any.day = day_date
                    dr_any.status = DispatchStatus.RUNNING
                    dr_any.started_at = datetime.now(UTC)
                    dr_any.created_at = datetime.now(UTC)
                    dr_any.config = cfg
                    db.session.add(dr_any)
                    db.session.flush()
                    dispatch_run = cast("DispatchRun", dr_any)
                    # ✅ FIX: Vérifier que l'ID est disponible après flush
                    # (éviter assert en production)
                    if getattr(dispatch_run, "id", None) is None:
                        error_msg = "DispatchRun ID should be available after flush"
                        logger.error("[DispatchRunManager] %s", error_msg)
                        raise ValueError(error_msg)
                logger.info(
                    "[DispatchRunManager] Created DispatchRun id=%s for company=%s day=%s",
                    dispatch_run.id,
                    company_id,
                    day_str,
                )
            except IntegrityError as e:
                # ✅ P2.2: Track métrique IntegrityError (race condition)
                error_code = (
                    getattr(e.orig, "pgcode", None) if hasattr(e, "orig") else None
                )
                track_integrity_error(
                    error_code=str(error_code) if error_code else "unknown",
                    company_id=company_id,
                    dispatch_run_id=None,
                )

                # Un autre thread l'a créé entre-temps → récupère l'existant
                # puis MAJ sous TX courte
                db.session.rollback()
                # ✅ Utilisation du repository pour découpler de SQLAlchemy
                dispatch_run_dto = dispatch_run_repo.find_by_company_and_day(
                    company_id, day_date
                )
                if dispatch_run_dto:
                    dispatch_run = DispatchRun.query.get(dispatch_run_dto.id)
                else:
                    dispatch_run = None
                if dispatch_run is None:
                    raise
                with _begin_tx():
                    dr2any: Any = dispatch_run
                    dr2any.status = DispatchStatus.RUNNING
                    dr2any.started_at = datetime.now(UTC)
                    dr2any.completed_at = None
                    dr2any.config = cfg
                    db.session.add(dr2any)
        else:
            # Reuse : MAJ sous TX courte (mettre à jour le statut à RUNNING
            # si nécessaire)
            with _begin_tx():
                dr3any: Any = dispatch_run
                # Ne mettre à jour le statut que s'il n'est pas déjà RUNNING
                if dr3any.status != DispatchStatus.RUNNING:
                    dr3any.status = DispatchStatus.RUNNING
                    dr3any.started_at = datetime.now(UTC)
                    dr3any.completed_at = None
                    dr3any.config = cfg
                    db.session.add(dr3any)

        # 3) Commit du DispatchRun pour qu'il soit visible dans les prochaines
        # transactions
        try:
            db.session.commit()
            logger.info(
                "[DispatchRunManager] DispatchRun id=%s committed successfully",
                dispatch_run.id,
            )
        except (IntegrityError, OperationalError, DBAPIError) as e:
            # Erreurs DB attendues : contraintes violées, connexion, timeout
            logger.error(
                "[DispatchRunManager] Failed to commit DispatchRun (DB error: %s): %s",
                type(e).__name__,
                e,
            )
            db.session.rollback()
            raise
        except Exception:
            # Erreur inattendue : logger avec trace complète et re-lever
            logger.exception(
                "[DispatchRunManager] Failed to commit DispatchRun (unexpected error)"
            )
            db.session.rollback()
            raise

        return dispatch_run, None

    def update_status(self, dispatch_run: DispatchRun, status: DispatchStatus) -> None:
        """Met à jour le statut d'un DispatchRun.

        Met à jour le statut du DispatchRun sous transaction courte.
        Utile pour marquer un DispatchRun comme RUNNING, FAILED, etc.

        Args:
            dispatch_run: Objet DispatchRun à mettre à jour
            status: Nouveau statut (DispatchStatus)

        Side-effects:
            - Accès DB (écriture DispatchRun)
            - Transaction DB (commit)
        """
        try:
            with _begin_tx():
                dispatch_run.status = status
                if status == DispatchStatus.RUNNING:
                    dispatch_run.started_at = datetime.now(UTC)
                db.session.add(dispatch_run)
            db.session.commit()
            logger.info(
                "[DispatchRunManager] Updated DispatchRun id=%s status to %s",
                dispatch_run.id,
                status,
            )
        except (IntegrityError, OperationalError, DBAPIError) as e:
            # Erreurs DB attendues : contraintes violées, connexion, timeout
            logger.error(
                "[DispatchRunManager] Failed to update DispatchRun status (DB error: %s): %s",
                type(e).__name__,
                e,
            )
            db.session.rollback()
            raise
        except Exception:
            # Erreur inattendue : logger avec trace complète et re-lever
            logger.exception(
                "[DispatchRunManager] Failed to update DispatchRun status (unexpected error)"
            )
            db.session.rollback()
            raise

    def finalize(
        self,
        dispatch_run: DispatchRun,
        assignments_count: int,
        unassigned_count: int,
    ) -> None:
        """Finalise un DispatchRun.

        Marque un DispatchRun comme COMPLETED avec les compteurs finaux
        d'assignations et de non-assignés.

        Args:
            dispatch_run: Objet DispatchRun à finaliser
            assignments_count: Nombre d'assignations créées
            unassigned_count: Nombre de bookings non assignés

        Side-effects:
            - Accès DB (écriture DispatchRun)
            - Transaction DB (commit)
        """
        try:
            with _begin_tx():
                dispatch_run.status = DispatchStatus.COMPLETED
                dispatch_run.completed_at = datetime.now(UTC)
                # Optionnel: enregistrer les compteurs si le modèle le supporte
                # (à adapter selon le modèle DispatchRun)
                db.session.add(dispatch_run)
            db.session.commit()
            logger.info(
                "[DispatchRunManager] Finalized DispatchRun id=%s with %d assignments, %d unassigned",
                dispatch_run.id,
                assignments_count,
                unassigned_count,
            )
        except (IntegrityError, OperationalError, DBAPIError) as e:
            # Erreurs DB attendues : contraintes violées, connexion, timeout
            logger.error(
                "[DispatchRunManager] Failed to finalize DispatchRun (DB error: %s): %s",
                type(e).__name__,
                e,
            )
            db.session.rollback()
            raise
        except Exception:
            # Erreur inattendue : logger avec trace complète et re-lever
            logger.exception(
                "[DispatchRunManager] Failed to finalize DispatchRun (unexpected error)"
            )
            db.session.rollback()
            raise
