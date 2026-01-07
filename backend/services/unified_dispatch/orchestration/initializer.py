# backend/services/unified_dispatch/orchestration/initializer.py
"""Initialisation et validation pour le dispatch.

Ce module gère l'initialisation et la validation de l'entreprise ainsi que
la configuration des settings de dispatch. Il est responsable de :
- La recherche et validation de l'entreprise (Company)
- La configuration des settings avec gestion des overrides
- La détection du mode rapide (fast_mode)

Side-effects:
    - Accès DB (lecture Company via CompanyRepository)
    - Métriques: Tracking des erreurs CompanyNotFoundError
"""

from __future__ import annotations

import inspect
import logging
import traceback
from typing import Any, Dict

from ext import db
from models import Company
from repositories.company_repository import CompanyRepository
from services.unified_dispatch.core import settings as ud_settings
from services.unified_dispatch.core.exceptions import CompanyNotFoundError
from services.unified_dispatch.core.types import DispatchResult
from services.unified_dispatch.error_metrics import track_company_not_found

logger = logging.getLogger(__name__)


class DispatchInitializer:
    """Gestionnaire d'initialisation et de configuration pour le dispatch.

    Cette classe centralise la logique d'initialisation du dispatch :
    - Recherche et validation de l'entreprise
    - Configuration des settings avec support des overrides
    - Détection et activation du mode rapide

    Exemple:
        >>> initializer = DispatchInitializer()
        >>> company, error = initializer.find_and_validate_company(
        ...     company_id=1,
        ...     for_date="2025-01-14",
        ...     mode="auto",
        ...     raise_on_not_found=False
        ... )
        >>> if company:
        ...     settings, mode, allow_emg, is_fast = initializer.configure_settings(
        ...         company=company,
        ...         custom_settings=None,
        ...         overrides={"fast_mode": True},
        ...         allow_emergency=True,
        ...         mode="auto"
        ...     )
    """

    def find_and_validate_company(
        self,
        company_id: int,
        for_date: str | None,
        mode: str,
        raise_on_company_not_found: bool,
    ) -> tuple[Company | None, Dict[str, Any] | None]:
        """Trouve et valide la Company.

        Recherche l'entreprise dans la base de données en utilisant le repository.
        Gère les cas où l'objet peut être flushé ou expiré dans la session SQLAlchemy.
        En cas d'échec, peut soit lever une exception soit retourner un résultat
        d'erreur structuré selon le paramètre `raise_on_company_not_found`.

        Args:
            company_id: ID de l'entreprise à rechercher
            for_date: Date du dispatch (pour contexte d'erreur)
            mode: Mode de dispatch (pour contexte d'erreur)
            raise_on_company_not_found: Si True, lève CompanyNotFoundError si introuvable

        Returns:
            Tuple (company, error_result) où :
            - company: Objet Company si trouvé, None sinon
            - error_result: Dict avec résultat d'erreur structuré si company introuvable,
              None si company trouvée

        Raises:
            CompanyNotFoundError: Si `raise_on_company_not_found=True` et company introuvable

        Side-effects:
            - Accès DB (lecture via CompanyRepository)
            - Métriques: Track CompanyNotFoundError si company introuvable
            - Logging: Erreurs et stack traces en mode DEBUG
        """
        # ✅ FIX RC3: Améliorer la recherche de Company pour gérer les objets flushés
        # ✅ Utilisation du repository pour découpler de SQLAlchemy
        company_repo = CompanyRepository()
        company: Company | None = None
        company_dto = company_repo.find_by_id(company_id)
        if company_dto:
            # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
            try:
                company = db.session.get(Company, company_dto.id)
            except AttributeError:
                # Fallback pour SQLAlchemy < 2.0
                company = Company.query.get(company_dto.id)

        # ✅ FIX RC3: Si pas trouvé, flush et réessayer
        if not company:
            db.session.flush()
            company_dto = company_repo.find_by_id(company_id)
            if company_dto:
                try:
                    company = db.session.get(Company, company_dto.id)
                except AttributeError:
                    company = Company.query.get(company_dto.id)

        # ✅ FIX RC3: Si toujours pas trouvé, essayer avec expire_all()
        # pour forcer le rechargement
        if not company:
            db.session.expire_all()
            company_dto = company_repo.find_by_id(company_id)
            if company_dto:
                try:
                    company = db.session.get(Company, company_dto.id)
                except AttributeError:
                    company = Company.query.get(company_dto.id)

        if not company:
            # ✅ FIX P1.2: Améliorer la gestion d'erreurs - Company introuvable
            # Récupérer le contexte de l'appelant pour améliorer le logging
            caller_frame = inspect.currentframe()
            caller_info = None
            if caller_frame and caller_frame.f_back:
                caller_frame = caller_frame.f_back
                caller_info = {
                    "file": caller_frame.f_code.co_filename,
                    "line": caller_frame.f_lineno,
                    "function": caller_frame.f_code.co_name,
                }

            # Construire le message d'erreur avec contexte
            error_msg = (
                f"[Initializer] ❌ Company {company_id} introuvable - dispatch "
                "impossible. "
                "Vérifier que la Company existe en DB et est commitée avant "
                "d'appeler engine.run()"
            )

            # Ajouter les informations du caller si disponibles
            if caller_info:
                file_info = caller_info.get("file", "")
                file_str = str(file_info).split("/")[-1] if file_info else "unknown"
                error_msg += (
                    f" | Appelé depuis: {caller_info.get('function', 'unknown')}() "
                    f"({file_str}:{caller_info.get('line', '?')})"
                )

            # Logger avec stack trace en mode DEBUG
            logger.error(
                error_msg, extra={"company_id": company_id, "caller": caller_info}
            )
            logger.debug(
                "[Initializer] Stack trace pour CompanyNotFoundError:\n%s",
                "".join(traceback.format_stack()[:-1]),  # Exclure la ligne actuelle
                extra={"company_id": company_id},
            )

            # ⚠️ Ne pas créer DispatchRun avec company_id invalide (violation FK)
            # Le DispatchRun nécessite une Company valide en DB

            # ✅ P2.2: Track métrique CompanyNotFoundError
            track_company_not_found(company_id, dispatch_run_id=None)

            # ✅ Option A: Lever une exception si demandé
            if raise_on_company_not_found:
                raise CompanyNotFoundError(
                    company_id=company_id,
                    caller=caller_info,
                    for_date=for_date,
                    mode=mode,
                )

            # ✅ Option par défaut: Retourner un résultat structuré pour traçabilité
            error_result = DispatchResult(
                dispatch_run_id=None,  # Pas de DispatchRun créé (Company n'existe pas)
                assignments=[],
                unassigned=[],
                bookings=[],
                drivers=[],
                meta={
                    "reason": "company_not_found",
                    "error": f"Company {company_id} introuvable en DB",
                    "dispatch_run_id": None,
                    "caller": caller_info,  # ✅ Ajout du contexte du caller
                },
                debug={
                    "reason": "company_not_found",
                    "error": f"Company {company_id} introuvable en DB",
                    "dispatch_run_id": None,
                    "caller": caller_info,  # ✅ Ajout du contexte du caller
                    "stack_trace": traceback.format_stack()[:-1]
                    if logger.isEnabledFor(logging.DEBUG)
                    else None,
                },
            ).to_dict()
            return None, error_result

        return company, None

    def configure_settings(
        self,
        company: Company,
        custom_settings: Any | None,
        overrides: dict[str, Any] | None,
        allow_emergency: bool | None,
        mode: str,
    ) -> tuple[Any, str, bool, bool]:
        """Configure les settings et détecte le mode rapide.

        Configure les settings de dispatch en combinant :
        - Les settings par défaut de l'entreprise
        - Les settings personnalisés (si fournis)
        - Les overrides de configuration
        - Le flag allow_emergency

        Détecte et active automatiquement le mode rapide si `fast_mode=True`
        dans les overrides. Le mode rapide force `heuristic_only` et désactive
        les optimisations lourdes (solver, RL).

        Args:
            company: Objet Company pour récupérer les settings par défaut
            custom_settings: Settings personnalisés (optionnel, prioritaire sur defaults)
            overrides: Overrides de configuration à merger avec les settings
            allow_emergency: Autoriser les courses d'urgence (override du setting)
            mode: Mode de dispatch initial (peut être modifié par fast_mode)

        Returns:
            Tuple (settings, mode, allow_emg, is_fast_mode) où :
            - settings: Objet Settings configuré
            - mode: Mode de dispatch (peut être modifié si fast_mode activé)
            - allow_emg: Bool indiquant si les courses d'urgence sont autorisées
            - is_fast_mode: Bool indiquant si le mode rapide est activé

        Side-effects:
            - Logging: Informations sur les overrides appliqués
            - Modification des settings: Désactivation solver/RL si fast_mode
        """
        # 1) Configuration
        s = custom_settings or ud_settings.for_company(company)

        # ✅ Gérer l'override de mode depuis overrides (priorité sur fast_mode)
        if overrides and "mode" in overrides:
            mode = overrides["mode"]
            logger.info("[Initializer] Mode override depuis overrides: %s", mode)

        # ⚡ Détecter le mode rapide depuis overrides
        is_fast_mode: bool = bool(overrides and overrides.get("fast_mode") is True)
        if is_fast_mode:
            # ⚡ Mode rapide : forcer heuristic_only et désactiver optimisations lourdes
            mode = "heuristic_only"
            logger.info(
                "[Initializer] ⚡ Mode RAPIDE détecté : heuristic_only, optimisations désactivées"
            )
            # Désactiver solver et RL pour garantir < 1 minute
            if not hasattr(s, "features"):
                s.features = ud_settings.FeatureFlags()
            s.features.enable_solver = False
            s.features.enable_rl = False
            s.features.enable_parallel_heuristics = (
                True  # Activer parallélisme pour vitesse
            )
            # Limiter le solver à 10s max si jamais appelé (sécurité)
            s.solver.time_limit_sec = 10

        if overrides:
            logger.info("[Initializer] Applying overrides: %s", list(overrides.keys()))
            logger.info(
                "[Initializer] 📋 Overrides détaillés: reset_existing=%s, preferred_driver_id=%s, fast_mode=%s",
                overrides.get("reset_existing"),
                overrides.get("preferred_driver_id"),
                overrides.get("fast_mode"),
            )

            # ✅ Logger les paramètres demandés avant merge
            logger.info("[Initializer] 📥 Overrides demandés: %s", overrides)

            # Capturer les valeurs avant merge pour comparaison
            fairness_weight_before = (
                getattr(getattr(s, "fairness", None), "fairness_weight", None)
                if hasattr(s, "fairness")
                else None
            )
            driver_load_before = (
                getattr(getattr(s, "heuristic", None), "driver_load_balance", None)
                if hasattr(s, "heuristic")
                else None
            )
            proximity_before = (
                getattr(getattr(s, "heuristic", None), "proximity", None)
                if hasattr(s, "heuristic")
                else None
            )

            try:
                s = ud_settings.merge_overrides(s, overrides)

                # ✅ Logger les paramètres appliqués vs demandés
                # (comparaison avant/après)
                if hasattr(s, "heuristic") and not isinstance(
                    getattr(s, "heuristic", None), str
                ):
                    driver_load_after = getattr(
                        s.heuristic, "driver_load_balance", None
                    )
                    proximity_after = getattr(s.heuristic, "proximity", None)
                    heuristic_override = (
                        overrides.get("heuristic", {})
                        if isinstance(overrides.get("heuristic"), dict)
                        else {}
                    )
                    logger.info(
                        "[Initializer] ✅ After merge - heuristic.driver_load_balance: %s → %s (demandé: %s)",
                        driver_load_before,
                        driver_load_after,
                        heuristic_override.get("driver_load_balance", "N/A"),
                    )
                    logger.info(
                        "[Initializer] ✅ After merge - heuristic.proximity: %s → %s (demandé: %s)",
                        proximity_before,
                        proximity_after,
                        heuristic_override.get("proximity", "N/A"),
                    )
                if hasattr(s, "fairness"):
                    fairness_weight_after = s.fairness.fairness_weight
                    fairness_override = (
                        overrides.get("fairness", {})
                        if isinstance(overrides.get("fairness"), dict)
                        else {}
                    )
                    logger.info(
                        "[Initializer] ✅ After merge - fairness.fairness_weight: %s → %s (demandé: %s)",
                        fairness_weight_before,
                        fairness_weight_after,
                        fairness_override.get("fairness_weight", "N/A"),
                    )
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                # Erreurs de validation/parsing attendues dans les overrides
                logger.warning(
                    "[Initializer] merge_overrides failed (validation error): %s. Using base settings.",
                    e,
                )
            except Exception:
                # Erreur inattendue : logger et re-lever
                logger.exception(
                    "[Initializer] Erreur inattendue lors du merge des overrides"
                )
                raise
        if allow_emergency is not None:
            try:
                s.emergency.allow_emergency_drivers = bool(allow_emergency)
            except (AttributeError, TypeError):
                # Erreurs attendues : attribut manquant, type incorrect
                logger.debug(
                    "[Initializer] Failed to set allow_emergency_drivers (expected error)"
                )
            except Exception:
                # Erreur inattendue : logger mais continuer (non-critique)
                logger.debug(
                    "[Initializer] Unexpected error setting allow_emergency_drivers"
                )
        allow_emg = bool(
            getattr(getattr(s, "emergency", None), "allow_emergency_drivers", True)
        )

        return s, mode, allow_emg, is_fast_mode
