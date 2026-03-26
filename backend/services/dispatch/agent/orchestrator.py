"""Orchestrateur principal de l'agent dispatch.

Fonctionne en boucle continue (tick) qui :
1. Lit l'état actuel (get_state)
2. Vérifie la santé OSRM (osrm_health)
3. Identifie les urgences (non assignées, ETA > TW)
4. Déclenche ré-optimisation si nécessaire
5. Applique les assignations avec validation
6. Log toutes les actions
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, cast
from zoneinfo import ZoneInfo

from flask import current_app

from ext import db
from models import Company
from repositories.assignment_repository import AssignmentRepository
from repositories.booking_repository import BookingRepository
from repositories.company_repository import CompanyRepository
from repositories.driver_repository import DriverRepository
from services.dispatch.agent.reporting import generate_daily_report
from services.dispatch.agent.safety_policy import SafetyPolicy
from services.dispatch.agent.tools import AgentTools
from shared.time_utils import now_local

TZ = ZoneInfo("Europe/Zurich")
logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """État de l'agent."""

    company_id: int
    running: bool = False
    last_tick: datetime | None = None
    actions_today: int = 0
    actions_last_hour: int = 0
    last_report: datetime | None = None
    current_plan: dict[str, Any] | None = None
    osrm_health: dict[str, Any] | None = None
    # ✅ Mémorisation de l'état précédent pour détecter les changements
    last_known_booking_ids: set[int] | None = None
    last_known_driver_ids: set[int] | None = None
    last_known_unassigned_count: int = 0
    # ✅ Mémorisation de la configuration précédente pour détecter les changements
    last_known_preferred_driver_id: int | None = None
    # ✅ Mémorisation des corrections d'urgent déjà effectuées (pour éviter répétitions)
    emergency_corrections_done: set[int] | None = None


class AgentOrchestrator:
    """Orchestrateur principal de l'agent dispatch.

    Fonctionne en boucle continue (tick) :
    1. Lit l'état actuel (get_state)
    2. Vérifie la santé OSRM (osrm_health)
    3. Identifie les urgences (non assignées, ETA > TW)
    4. Déclenche ré-optimisation si nécessaire
    5. Applique les assignations avec validation
    6. Log toutes les actions
    """

    def __init__(self, company_id: int, app=None):
        """Initialise l'orchestrateur.

        Args:
            company_id: ID de l'entreprise
            app: Instance Flask app (pour le contexte dans le thread)

        Raises:
            ValueError: Si l'entreprise n'existe pas

        """
        super().__init__()
        self.company_id = company_id
        # ✅ Utilisation du repository pour découpler de SQLAlchemy
        company_repo = CompanyRepository()
        company_dto = company_repo.find_by_id(company_id)
        if not company_dto:
            msg = f"Company {company_id} not found"
            raise ValueError(msg)
        # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
        self.company = Company.query.get(company_dto.id)

        self.tools = AgentTools(company_id)
        self.safety = SafetyPolicy(company_id)
        # Initialiser last_known_preferred_driver_id depuis la config actuelle
        initial_preferred_driver_id = None
        if self.company:
            autonomous_config = self.company.get_autonomous_config()
            dispatch_overrides = autonomous_config.get("dispatch_overrides", {})
            if "preferred_driver_id" in dispatch_overrides:
                preferred_id = dispatch_overrides["preferred_driver_id"]
                if preferred_id:
                    with contextlib.suppress(ValueError, TypeError):
                        initial_preferred_driver_id = int(preferred_id)

        self.state = AgentState(
            company_id=company_id,
            last_known_booking_ids=None,
            last_known_driver_ids=None,
            last_known_unassigned_count=0,
            last_known_preferred_driver_id=initial_preferred_driver_id,
            emergency_corrections_done=set(),
        )
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # current_app est un LocalProxy, _get_current_object() existe mais
        # pyright ne le reconnaît pas
        self._app = (
            app or getattr(current_app, "_get_current_object", lambda: current_app)()
        )

        logger.info("[AgentOrchestrator] Initialized for company %s", company_id)

    def start(self) -> None:
        """Démarre l'agent en mode continu."""
        with self._lock:
            if self._running:
                logger.warning(
                    "[AgentOrchestrator] Already running for company %s",
                    self.company_id,
                )
                return

            self._running = True
            self.state.running = True
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=False,
                name=f"AgentDispatch-{self.company_id}",
            )
            self._thread.start()
            logger.info(
                "[AgentOrchestrator] ✅ Started for company %s", self.company_id
            )

    def stop(self) -> None:
        """Arrête l'agent."""
        with self._lock:
            self._running = False
            self.state.running = False
        logger.info("[AgentOrchestrator] ⏸️ Stopped for company %s", self.company_id)

    def _run_loop(self) -> None:
        """Boucle principale de l'agent."""
        with self._app.app_context():
            while self._running:
                try:
                    self._tick()
                    # Tick toutes les 2 minutes
                    time.sleep(120)
                except Exception as e:
                    logger.exception("[AgentOrchestrator] Error in tick: %s", e)
                    time.sleep(60)  # Attendre 1 min avant de réessayer

    def _tick(self) -> None:
        """Un cycle de décision de l'agent avec logique progressive.

        Règles:
        1. Situation normale (tout assigné, pas de retard, pas d'urgence)
           → Surveillance uniquement
        2. Nouvelle course → Décision progressive
           (assignation simple → réorganisation ciblée → dispatch complet)
        3. Retard détecté → Optimiseur différé (1h avant chaque course)
        """
        now = now_local()
        self.state.last_tick = now
        logger.info("[AgentOrchestrator] ⏰ Tick démarré à %s", now.isoformat())

        # 1. Vérifier santé OSRM
        health = self.tools.osrm_health()
        self.state.osrm_health = health
        self.tools.log_action(
            kind="tick",
            payload={"osrm": health},
            reasoning_brief="Tick horaire + état OSRM vérifié.",
        )

        # 2. Lire l'état actuel (fenêtre étendue: 48h pour détecter toutes les courses)
        window_start = now
        window_end = now + timedelta(hours=48)
        state = self.tools.get_state(window_start=window_start, window_end=window_end)

        all_jobs = state.get("jobs", [])
        all_drivers = state.get("drivers", [])

        # ✅ IMPORTANT: Recharger l'objet Company depuis la DB à chaque tick
        # pour détecter les changements
        # L'objet SQLAlchemy peut être en cache et ne pas refléter
        # les changements récents
        db.session.expire_all()  # Expirer tous les objets en cache
        # ✅ Utilisation du repository pour découpler de SQLAlchemy
        company_repo = CompanyRepository()
        company_dto = company_repo.find_by_id(self.company_id)
        if not company_dto:
            logger.error("[AgentOrchestrator] Company %s non trouvée", self.company_id)
            return
        # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
        current_company = Company.query.get(company_dto.id)
        if not current_company:
            logger.error("[AgentOrchestrator] Company %s non trouvée", self.company_id)
            return

        # ✅ Détecter les changements de configuration
        # (chauffeur préféré)
        current_preferred_driver_id = None
        if current_company:
            autonomous_config = current_company.get_autonomous_config()
            dispatch_overrides = autonomous_config.get("dispatch_overrides", {})
            if "preferred_driver_id" in dispatch_overrides:
                preferred_id = dispatch_overrides["preferred_driver_id"]
                if preferred_id:
                    with contextlib.suppress(ValueError, TypeError):
                        current_preferred_driver_id = int(preferred_id)
                        logger.debug(
                            (
                                "[AgentOrchestrator] 🔍 Chauffeur préféré lu depuis DB:"
                                " %s (last_known: %s)"
                            ),
                            current_preferred_driver_id,
                            self.state.last_known_preferred_driver_id,
                        )

        preferred_driver_changed = (
            current_preferred_driver_id != self.state.last_known_preferred_driver_id
        )
        if preferred_driver_changed:
            logger.info(
                (
                    "[AgentOrchestrator] 🔄 Changement de chauffeur préféré détecté: "
                    "%s → %s"
                ),
                self.state.last_known_preferred_driver_id,
                current_preferred_driver_id,
            )
            self.state.last_known_preferred_driver_id = current_preferred_driver_id
        elif (
            current_preferred_driver_id
            and self.state.last_known_preferred_driver_id is None
        ):
            # Premier tick avec chauffeur préféré configuré
            logger.info(
                (
                    "[AgentOrchestrator] 🎯 Chauffeur préféré configuré "
                    "au premier tick: #%s"
                ),
                current_preferred_driver_id,
            )
            self.state.last_known_preferred_driver_id = current_preferred_driver_id

        logger.debug(
            (
                "[AgentOrchestrator] État récupéré: %d jobs, %d drivers, "
                "chauffeur préféré: %s"
            ),
            len(all_jobs),
            len(all_drivers),
            current_preferred_driver_id or "aucun",
        )

        # 3. ✅ DÉTECTION DES CHANGEMENTS : Comparer avec l'état précédent
        current_booking_ids = {j.get("job_id") for j in all_jobs if j.get("job_id")}
        current_driver_ids = {
            d.get("driver_id") for d in all_drivers if d.get("driver_id")
        }
        unassigned_jobs = [j for j in all_jobs if j.get("status") == "unassigned"]
        unassigned_count = len(unassigned_jobs)

        # Variable pour suivre si c'est le premier tick
        is_first_tick = self.state.last_known_booking_ids is None

        # ✅ INITIALISATION : Si c'est le premier tick, initialiser avec l'état actuel
        # (pas de détection de changements)
        if is_first_tick:
            logger.info(
                (
                    "[AgentOrchestrator] 🔄 Premier tick - Initialisation de l'état "
                    "mémorisé: %d courses, %d drivers, %d non assignées, "
                    "chauffeur préféré: %s"
                ),
                len(current_booking_ids),
                len(current_driver_ids),
                unassigned_count,
                current_preferred_driver_id or "aucun",
            )
            self.state.last_known_booking_ids = current_booking_ids.copy()
            self.state.last_known_driver_ids = current_driver_ids.copy()
            self.state.last_known_unassigned_count = unassigned_count
            # Au premier tick, on ne détecte pas de changements
            # (tout est considéré comme état initial)
            # EXCEPTION: Si un chauffeur préféré est configuré ET qu'il y a
            # des courses non assignées, on doit agir pour appliquer la préférence
            new_bookings = set()
            drivers_became_unavailable = set()
            unassigned_increased = False

            # Si des courses sont non assignées au premier tick ET qu'un chauffeur
            # préféré est configuré, on doit agir pour appliquer la préférence
            if unassigned_jobs and current_preferred_driver_id:
                logger.info(
                    (
                        "[AgentOrchestrator] 🎯 Premier tick: "
                        "%d course(s) non assignée(s) + chauffeur préféré "
                        "configuré (#%s) → Action pour appliquer préférence"
                    ),
                    unassigned_count,
                    current_preferred_driver_id,
                )
                # On va passer à la logique de décision progressive pour assigner
                # les courses non assignées avec le chauffeur préféré
                # Ne pas return ici, continuer avec la logique normale
            elif unassigned_jobs:
                logger.info(
                    (
                        "[AgentOrchestrator] ⏸️ Premier tick: "
                        "%d course(s) non assignée(s) détectée(s) mais pas d'action "
                        "(attente événement réel ou configuration préférence)"
                    ),
                    unassigned_count,
                )
                # Vérifier les retards potentiels (optimiseur différé)
                self._check_delayed_optimizer(now, all_jobs, state)
                return
        else:
            # Détecter nouvelles courses (seulement celles qui n'étaient pas
            # dans l'état précédent)
            new_bookings = current_booking_ids - (
                self.state.last_known_booking_ids or set()
            )
            # Détecter chauffeurs devenus indisponibles
            drivers_became_unavailable = (
                self.state.last_known_driver_ids or set()
            ) - current_driver_ids
            # Détecter changement dans le nombre de courses non assignées
            unassigned_increased = (
                unassigned_count > self.state.last_known_unassigned_count
            )

            logger.debug(
                (
                    "[AgentOrchestrator] 🔍 Détection changements: "
                    "%d nouvelles courses, %d chauffeurs indisponibles, "
                    "non assignées: %d (était %d)"
                ),
                len(new_bookings),
                len(drivers_became_unavailable),
                unassigned_count,
                self.state.last_known_unassigned_count,
            )

        # Mettre à jour l'état mémorisé
        self.state.last_known_booking_ids = current_booking_ids.copy()
        self.state.last_known_driver_ids = current_driver_ids.copy()
        self.state.last_known_unassigned_count = unassigned_count

        # 4. ✅ RÈGLE 1 : Situation normale → Pas d'action
        # Si toutes les courses sont assignées, aucun changement détecté,
        # et pas de retard
        # EXCEPTION: Si le chauffeur préféré a changé, on doit réoptimiser
        # même si tout est assigné
        if (
            not unassigned_jobs
            and not new_bookings
            and not drivers_became_unavailable
            and not preferred_driver_changed
        ):
            logger.info(
                "[AgentOrchestrator] ✅ Situation normale - Toutes les courses assignées (%d), aucun changement détecté. Surveillance uniquement.",
                len(all_jobs),
            )
            # Vérifier les retards potentiels (optimiseur différé)
            self._check_delayed_optimizer(now, all_jobs, state)
            return

        # 5. ✅ RÈGLE 2 : Nouvelle course ou changement détecté → Décision progressive
        # Si le chauffeur préféré a changé, on doit réoptimiser même si tout est assigné
        # Si c'est le premier tick ET qu'un chauffeur préféré est configuré ET qu'il y a
        # des courses non assignées, on doit agir
        should_act = (
            new_bookings
            or drivers_became_unavailable
            or unassigned_increased
            or preferred_driver_changed
            or (is_first_tick and unassigned_jobs and current_preferred_driver_id)
        )
        if should_act:
            if preferred_driver_changed:
                logger.info(
                    "[AgentOrchestrator] 🔄 Changement de chauffeur préféré détecté → Réoptimisation pour appliquer nouvelle préférence"
                )
            elif is_first_tick and unassigned_jobs and current_preferred_driver_id:
                logger.info(
                    "[AgentOrchestrator] 🎯 Premier tick avec chauffeur préféré configuré (#%s) → Réoptimisation pour appliquer préférence aux %d course(s) non assignée(s)",
                    current_preferred_driver_id,
                    unassigned_count,
                )
            else:
                logger.info(
                    "[AgentOrchestrator] 🔄 Changement détecté: %d nouvelle(s) course(s), %d chauffeur(s) indisponible(s), %d course(s) non assignée(s)",
                    len(new_bookings),
                    len(drivers_became_unavailable),
                    unassigned_count,
                )

            # Décision progressive
            self._handle_progressive_decision(
                unassigned_jobs=unassigned_jobs,
                new_bookings=new_bookings,
                drivers_became_unavailable=drivers_became_unavailable,
                state=state,
                health=health,
                now=now,
                preferred_driver_changed=preferred_driver_changed,
            )
            return

        # 6. ✅ RÈGLE 3 : Optimiseur différé (vérifier 1h avant chaque course)
        self._check_delayed_optimizer(now, all_jobs, state)

        # 7. Générer rapport périodique (toutes les 2h et à 23:00)
        REPORT_HOUR_23 = 23
        REPORT_MINUTE_THRESHOLD = 5
        if now.hour == REPORT_HOUR_23 or (
            now.hour % 2 == 0 and now.minute < REPORT_MINUTE_THRESHOLD
        ):
            self._generate_periodic_report()

    def _handle_progressive_decision(
        self,
        unassigned_jobs: list[Dict[str, Any]],
        new_bookings: set[int],
        drivers_became_unavailable: set[int],
        state: Dict[str, Any],
        health: Dict[str, Any],
        now: datetime,
        preferred_driver_changed: bool = False,
    ) -> None:
        """Décision progressive :
        assignation simple → réorganisation ciblée → dispatch complet.

        Étape 1 : Essayer assignation simple pour chaque nouvelle course
        Étape 2 : Si conflit local → réorganisation ciblée
            (seulement les courses impactées)
        Étape 3 : Si aucune solution locale → dispatch complet

        Si preferred_driver_changed=True, on doit réoptimiser même si tout est assigné.
        """
        # Si le chauffeur préféré a changé, on doit réoptimiser même si tout est assigné
        if preferred_driver_changed and not unassigned_jobs:
            logger.info(
                "[AgentOrchestrator] 🔄 Changement de chauffeur préféré détecté → Réoptimisation complète pour appliquer nouvelle préférence"
            )
            # Extraire la date la plus fréquente des courses assignées
            all_jobs = state.get("jobs", [])
            job_dates = []
            for job in all_jobs:
                scheduled_time = job.get("scheduled_time")
                if scheduled_time:
                    try:
                        if scheduled_time.endswith("Z"):
                            scheduled_time = scheduled_time.replace("Z", "+00:00")
                        dt = datetime.fromisoformat(scheduled_time)
                        job_dates.append(dt.strftime("%Y-%m-%d"))
                    except Exception:
                        pass

            if job_dates:
                most_common_date = Counter(job_dates).most_common(1)[0][0]
                logger.info(
                    "[AgentOrchestrator] 📅 Réoptimisation pour date: %s (changement chauffeur préféré)",
                    most_common_date,
                )
                plan = self.tools.reoptimize(
                    scope="all",
                    strategy="full",
                    for_date=most_common_date,
                    force_reassign=True,  # ⚡ Forcer la réassignation pour appliquer
                    # le nouveau preferred_driver
                )
                if plan and plan.get("plan"):
                    success = self._apply_plan_with_validation(
                        plan.get("plan", []), "full", now, preferred_driver_changed=True
                    )
                    if not success:
                        logger.warning(
                            "[AgentOrchestrator] ⚠️ Plan initial rejeté, ré-optimisation avec contraintes renforcées"
                        )
                        # Ré-optimiser avec contraintes plus strictes
                        plan_retry = self.tools.reoptimize(
                            scope="all",
                            strategy="full",
                            for_date=most_common_date,
                            force_reassign=True,  # ⚡ Forcer la réassignation
                            # pour appliquer le nouveau preferred_driver
                        )
                        if plan_retry and plan_retry.get("plan"):
                            self._apply_plan_with_validation(
                                plan_retry.get("plan", []), "full_retry", now
                            )
            return

        if not unassigned_jobs:
            logger.info(
                "[AgentOrchestrator] Aucune course non assignée, pas d'action nécessaire"
            )
            return

        # Étape 1 : Essayer assignation simple pour chaque nouvelle course
        # (seulement les nouvelles courses, pas toutes les non assignées)
        new_unassigned_jobs = [
            j for j in unassigned_jobs if j.get("job_id") in new_bookings
        ]

        if new_unassigned_jobs and len(new_unassigned_jobs) == 1:
            # Une seule nouvelle course → essayer assignation simple
            job = new_unassigned_jobs[0]
            logger.info(
                "[AgentOrchestrator] 🎯 Nouvelle course unique détectée (#%s), tentative assignation simple",
                job.get("job_id"),
            )

            # Trouver le meilleur chauffeur disponible sans impact
            # sur ses autres courses
            best_driver = self._find_best_driver_simple(job, state.get("drivers", []))
            job_id = job.get("job_id")

            if best_driver and job_id:
                result = self.tools.assign(
                    job_id=int(job_id),
                    driver_id=best_driver,
                    note=f"Assignation simple nouvelle course {now.isoformat()}",
                )
                if result.get("ok"):
                    logger.info(
                        "[AgentOrchestrator] ✅ Assignation simple réussie: job %s → driver %s",
                        job.get("job_id"),
                        best_driver,
                    )
                    self.state.actions_today += 1
                    self.state.actions_last_hour += 1
                    return

                logger.warning(
                    "[AgentOrchestrator] ⚠️ Assignation simple échouée: %s, passage à réorganisation ciblée",
                    result.get("error"),
                )

        # Étape 2 : Réorganisation ciblée (seulement les courses impactées)
        MAX_JOBS_FOR_TARGETED_REORG = 3
        if (
            drivers_became_unavailable
            or len(unassigned_jobs) <= MAX_JOBS_FOR_TARGETED_REORG
        ):
            logger.info(
                "[AgentOrchestrator] 🔄 Réorganisation ciblée: %d course(s) non assignée(s) ou chauffeur indisponible",
                len(unassigned_jobs),
            )

            # Extraire la date la plus fréquente
            job_dates = []
            for job in unassigned_jobs:
                scheduled_time = job.get("scheduled_time")
                if scheduled_time:
                    try:
                        if scheduled_time.endswith("Z"):
                            scheduled_time = scheduled_time.replace("Z", "+00:00")
                        dt = datetime.fromisoformat(scheduled_time)
                        job_dates.append(dt.strftime("%Y-%m-%d"))
                    except Exception:
                        pass

            most_common_date = (
                Counter(job_dates).most_common(1)[0][0]
                if job_dates
                else now.strftime("%Y-%m-%d")
            )

            strategy = (
                "full" if health.get("state") == "CLOSED" else "degraded_proximity"
            )
            plan = self.tools.reoptimize(
                scope="window",
                strategy=strategy,
                overrides=state.get("overrides", {}),
                for_date=most_common_date,
            )

            if plan and plan.get("plan"):
                success = self._apply_plan_with_validation(plan["plan"], strategy, now)
                if success:
                    return
                logger.warning(
                    "[AgentOrchestrator] ⚠️ Plan ciblé rejeté, passage au dispatch complet"
                )

        # Étape 3 : Dispatch complet (dernier recours)
        logger.info(
            "[AgentOrchestrator] 🚀 Dispatch complet nécessaire: %d course(s) non assignée(s)",
            len(unassigned_jobs),
        )

        job_dates = []
        for job in unassigned_jobs:
            scheduled_time = job.get("scheduled_time")
            if scheduled_time:
                try:
                    if scheduled_time.endswith("Z"):
                        scheduled_time = scheduled_time.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(scheduled_time)
                    job_dates.append(dt.strftime("%Y-%m-%d"))
                except Exception:
                    pass

        most_common_date = (
            Counter(job_dates).most_common(1)[0][0]
            if job_dates
            else now.strftime("%Y-%m-%d")
        )
        strategy = "full" if health.get("state") == "CLOSED" else "degraded_proximity"

        plan = self.tools.reoptimize(
            scope="window",
            strategy=strategy,
            overrides=state.get("overrides", {}),
            for_date=most_common_date,
        )

        if plan and plan.get("plan"):
            success = self._apply_plan_with_validation(plan["plan"], strategy, now)
            if not success:
                logger.error(
                    "[AgentOrchestrator] ❌ Échec application plan complet après %d tentatives",
                    3,
                )

    def _find_best_driver_simple(
        self, job: Dict[str, Any], drivers: list[Dict[str, Any]]
    ) -> int | None:
        """Trouve le meilleur chauffeur pour une assignation simple sans impact.

        Retourne None si aucun chauffeur disponible sans conflit.
        """
        from models import Assignment, AssignmentStatus, Driver

        job_id = job.get("job_id")
        if not job_id:
            return None

        # ✅ Utilisation du repository pour découpler de SQLAlchemy
        booking_repo = BookingRepository()
        booking_dto = booking_repo.find_by_id(job_id)
        if not booking_dto or not booking_dto.scheduled_time:
            return None
        # Récupérer le modèle SQLAlchemy depuis le repository pour la compatibilité
        booking = booking_repo.find_model_by_id(booking_dto.id)
        if not booking:
            return None

        MIN_TIME_GAP_MINUTES = 30  # Minimum 30 minutes entre deux courses

        for driver_info in drivers:
            driver_id = driver_info.get("driver_id")
            if not driver_id or not driver_info.get("available"):
                continue

            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            driver_repo = DriverRepository()
            driver_dto = driver_repo.find_by_id(driver_id)
            if not driver_dto:
                continue
            # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
            driver = Driver.query.get(driver_dto.id)
            if not driver:
                continue

            # Vérifier conflits temporels avec les courses existantes du chauffeur
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            assignment_repo = AssignmentRepository()
            assignment_dtos = assignment_repo.find_by_driver_id(driver_id)
            # Filtrer par statuts en mémoire
            active_statuses = [
                AssignmentStatus.SCHEDULED,
                AssignmentStatus.EN_ROUTE_PICKUP,
                AssignmentStatus.ARRIVED_PICKUP,
                AssignmentStatus.ONBOARD,
                AssignmentStatus.EN_ROUTE_DROPOFF,
            ]
            filtered_assignment_dtos = [
                dto for dto in assignment_dtos if dto.status in active_statuses
            ]
            # Récupérer les modèles SQLAlchemy depuis les IDs des DTOs pour la compatibilité
            assignment_ids = [dto.id for dto in filtered_assignment_dtos]
            driver_assignments = (
                Assignment.query.filter(Assignment.id.in_(assignment_ids)).all()
                if assignment_ids
                else []
            )
            has_conflict = False

            for existing_assignment in driver_assignments:
                existing_booking = existing_assignment.booking
                if not existing_booking or not existing_booking.scheduled_time:
                    continue

                # Vérifier si les deux courses sont trop proches temporellement
                time_diff = abs(
                    (
                        booking.scheduled_time - existing_booking.scheduled_time
                    ).total_seconds()
                    / 60
                )
                if time_diff < MIN_TIME_GAP_MINUTES:
                    has_conflict = True
                    break

            if has_conflict:
                continue

            # Calculer score simple (distance estimée)
            # Pour l'instant, on prend le premier chauffeur sans conflit
            # TODO: Améliorer avec calcul de distance réel
            return int(driver_id) if driver_id is not None else None

        return None

    def _find_regular_driver_for_booking(
        self, booking_id: int, state: Dict[str, Any]
    ) -> int | None:
        """Trouve un chauffeur régulier disponible pour une course
        actuellement assignée à l'urgent.

        Retourne None si aucun régulier disponible sans conflit.
        """
        from models import Assignment, AssignmentStatus, Booking, Driver

        # ✅ Utilisation du repository pour découpler de SQLAlchemy
        booking_repo = BookingRepository()
        booking_dto = booking_repo.find_by_id(booking_id)
        if not booking_dto or not booking_dto.scheduled_time:
            return None
        # Récupérer le modèle SQLAlchemy depuis le repository pour la compatibilité
        booking = booking_repo.find_model_by_id(booking_dto.id)
        if not booking:
            return None

        # Récupérer tous les chauffeurs réguliers disponibles
        all_drivers = state.get("drivers", [])
        regular_drivers = []

        for driver_info in all_drivers:
            driver_id = driver_info.get("driver_id")
            if not driver_id or not driver_info.get("available"):
                continue

            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            driver_repo = DriverRepository()
            driver_dto = driver_repo.find_by_id(driver_id)
            if not driver_dto:
                continue
            # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
            driver = Driver.query.get(driver_dto.id)
            if not driver:
                continue

            # Vérifier si c'est un régulier (pas un urgent)
            driver_type = getattr(driver, "driver_type", None)
            if driver_type:
                driver_type_str = str(driver_type).strip().upper()
                if "." in driver_type_str:
                    driver_type_str = driver_type_str.split(".")[-1]
                if driver_type_str != "EMERGENCY":
                    regular_drivers.append((driver_id, driver))

        if not regular_drivers:
            return None

        MIN_TIME_GAP_MINUTES = 30  # Minimum 30 minutes entre deux courses

        # Tester chaque régulier pour trouver le meilleur (sans conflit)
        for driver_id, _driver in regular_drivers:
            # Vérifier conflits temporels avec les courses existantes du chauffeur
            # ⚠️ TODO: Refactorer pour utiliser AssignmentRepository.find_active_by_driver_with_bookings()
            # quand cette méthode sera ajoutée au repository pour gérer les joins complexes avec Booking.
            # Pour l'instant, on utilise encore la requête directe pour cette validation complexe.
            has_conflict = False
            driver_assignments = (
                Assignment.query.filter_by(driver_id=driver_id)
                .filter(
                    Assignment.status.in_(
                        [
                            AssignmentStatus.SCHEDULED,
                            AssignmentStatus.EN_ROUTE_PICKUP,
                            AssignmentStatus.ARRIVED_PICKUP,
                            AssignmentStatus.ONBOARD,
                            AssignmentStatus.EN_ROUTE_DROPOFF,
                        ]
                    )
                )
                .join(Booking)
                .filter(Booking.scheduled_time.isnot(None))
                .all()
            )

            for existing_assignment in driver_assignments:
                existing_booking = existing_assignment.booking
                if not existing_booking or not existing_booking.scheduled_time:
                    continue

                # Vérifier si les deux courses sont trop proches temporellement
                time_diff = abs(
                    (
                        booking.scheduled_time - existing_booking.scheduled_time
                    ).total_seconds()
                    / 60
                )
                if time_diff < MIN_TIME_GAP_MINUTES:
                    has_conflict = True
                    break

            if has_conflict:
                continue

            # Calculer score simple (distance estimée depuis le bureau
            # ou dernière course)
            # Pour l'instant, on prend le premier chauffeur sans conflit
            # TODO: Améliorer avec calcul de distance réel
            return int(driver_id) if driver_id is not None else None

        return None

    def _apply_plan_with_validation(
        self,
        plan: list[Dict[str, Any]],
        strategy: str,
        now: datetime,
        max_retries: int = 3,
        preferred_driver_changed: bool = False,  # noqa: ARG002
    ) -> bool:
        """Applique un plan avec validation et ré-optimisation si nécessaire.

        Args:
            plan: Liste des assignations à appliquer
            strategy: Stratégie utilisée
            now: Timestamp actuel
            max_retries: Nombre maximum de tentatives
            preferred_driver_changed: Si True, permet plus de retries
                pour changement de préférence

        Returns:
            True si appliqué avec succès, False si conflits persistants
        """
        retry_count = 0
        current_plan = plan

        while retry_count <= max_retries:
            if retry_count > 0:
                logger.info(
                    "[AgentOrchestrator] 🔄 Tentative %d/%d de ré-optimisation",
                    retry_count,
                    max_retries,
                )

                # Ré-optimiser avec contraintes plus strictes
                from collections import Counter

                job_dates = []
                for step in current_plan:
                    job_id = step.get("job_id")
                    if job_id:
                        # ✅ Utilisation du repository pour découpler de SQLAlchemy
                        booking_repo = BookingRepository()
                        booking_dto = booking_repo.find_by_id(job_id)
                        if booking_dto and booking_dto.scheduled_time:
                            # Récupérer le modèle SQLAlchemy depuis le repository pour la compatibilité
                            booking = booking_repo.find_model_by_id(booking_dto.id)
                            if booking and booking.scheduled_time:
                                job_dates.append(
                                    booking.scheduled_time.strftime("%Y-%m-%d")
                                )

                most_common_date = (
                    Counter(job_dates).most_common(1)[0][0]
                    if job_dates
                    else now.strftime("%Y-%m-%d")
                )

                # Utiliser une stratégie plus stricte
                retry_strategy = "full"  # Toujours utiliser "full" pour les retries
                current_plan_result = self.tools.reoptimize(
                    scope="all",
                    strategy=retry_strategy,
                    for_date=most_common_date,
                )

                if not current_plan_result or not current_plan_result.get("plan"):
                    logger.warning(
                        "[AgentOrchestrator] ⚠️ Ré-optimisation n'a pas généré de plan"
                    )
                    break

                current_plan = current_plan_result.get("plan", [])

            # Appliquer le plan avec validation
            success = self._apply_plan(current_plan, strategy, now)

            if success:
                return True

            retry_count += 1

        return False

    def _apply_plan(
        self, plan: list[Dict[str, Any]], strategy: str, now: datetime
    ) -> bool:
        """Applique un plan d'assignations avec validations et ré-optimisation
        si nécessaire.

        Args:
            plan: Liste des assignations à appliquer
            strategy: Stratégie utilisée (pour logging)
            now: Timestamp actuel
            max_retries: Nombre maximum de tentatives de ré-optimisation

        Returns:
            True si le plan a été appliqué avec succès, False si des conflits persistent
        """
        from models import Assignment, AssignmentStatus

        filtered_plan = []
        for step in plan:
            job_id = step.get("job_id")
            if not job_id:
                continue

            # Vérifier si déjà assignée
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            booking_repo = BookingRepository()
            booking_dto = booking_repo.find_by_id(job_id)
            if booking_dto:
                # Récupérer le modèle SQLAlchemy depuis le repository pour la compatibilité
                booking = booking_repo.find_model_by_id(booking_dto.id)
                if booking:
                    # ✅ Utilisation du repository pour découpler de SQLAlchemy
                    assignment_repo = AssignmentRepository()
                    assignment_dtos = assignment_repo.find_by_booking_id(job_id)
                    # Filtrer par statuts en mémoire
                    active_statuses = [
                        AssignmentStatus.SCHEDULED,
                        AssignmentStatus.EN_ROUTE_PICKUP,
                        AssignmentStatus.ARRIVED_PICKUP,
                        AssignmentStatus.ONBOARD,
                        AssignmentStatus.EN_ROUTE_DROPOFF,
                    ]
                    filtered_assignment_dtos = [
                        dto for dto in assignment_dtos if dto.status in active_statuses
                    ]
                    # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
                    existing_assignment = (
                        Assignment.query.get(filtered_assignment_dtos[0].id)
                        if filtered_assignment_dtos
                        else None
                    )
                    if existing_assignment:
                        logger.debug(
                            "[AgentOrchestrator] ⏭️ Job %s déjà assigné, skip", job_id
                        )
                        continue

            filtered_plan.append(step)

        logger.info(
            "[AgentOrchestrator] Plan à appliquer: %d étapes (sur %d initiales)",
            len(filtered_plan),
            len(plan),
        )

        # ✅ VALIDATION AVANT APPLICATION avec calculs de temps réels
        # Simuler les assignations pour valider
        simulated_assignments = []
        for step in filtered_plan:
            job_id = step.get("job_id")
            if job_id:
                # ✅ Utilisation du repository pour découpler de SQLAlchemy
                booking_repo = BookingRepository()
                booking_dto = booking_repo.find_by_id(job_id)
                if booking_dto and booking_dto.scheduled_time:
                    # Récupérer le modèle SQLAlchemy depuis le repository pour la compatibilité
                    booking = booking_repo.find_model_by_id(booking_dto.id)
                    if booking and booking.scheduled_time:
                        simulated_assignments.append(
                            {
                                "booking_id": step.get("job_id"),
                                "driver_id": step.get("driver_id"),
                                "scheduled_time": booking.scheduled_time.isoformat(),
                            }
                        )

        # Valider le plan simulé avec validation améliorée
        if simulated_assignments:
            # ✅ Utiliser une validation améliorée qui calcule les temps réels
            has_conflicts = self._validate_plan_with_real_times(filtered_plan)

            if has_conflicts:
                logger.warning(
                    "[AgentOrchestrator] ⚠️ Conflits temporels détectés dans le plan, ré-optimisation nécessaire"
                )
                return False

        # Appliquer les assignations
        applied_count = 0
        failed_count = 0

        for step in filtered_plan:
            can_proceed, reason = self.safety.check_action(
                action_type="assign", context=step
            )

            if not can_proceed:
                logger.warning("[AgentOrchestrator] ⚠️ Action bloquée: %s", reason)
                failed_count += 1
                continue

            result = self.tools.assign(
                job_id=step["job_id"],
                driver_id=step["driver_id"],
                note=f"{strategy} {now.isoformat()}",
            )

            if result.get("ok"):
                self.state.actions_today += 1
                self.state.actions_last_hour += 1
                applied_count += 1
                logger.info(
                    "[AgentOrchestrator] ✅ Assigné job %s → driver %s",
                    step["job_id"],
                    step["driver_id"],
                )
            else:
                failed_count += 1
                logger.error(
                    "[AgentOrchestrator] ❌ Échec assignation: %s",
                    result.get("error"),
                )

        logger.info(
            "[AgentOrchestrator] Plan appliqué: %d réussies, %d échouées",
            applied_count,
            failed_count,
        )

        return applied_count > 0

    def _validate_plan_with_real_times(self, plan: list[Dict[str, Any]]) -> bool:
        """Valide un plan en calculant les temps réels entre courses.

        Args:
            plan: Liste des assignations à valider

        Returns:
            True si des conflits sont détectés, False sinon
        """
        from models import Company
        from services.unified_dispatch import settings as ud_settings
        from shared.geo_utils import haversine_minutes

        # Récupérer les paramètres configurables
        # ✅ Utilisation du repository pour découpler de SQLAlchemy
        company_repo = CompanyRepository()
        company_dto = company_repo.find_by_id(self.company_id)
        if not company_dto:
            return False
        # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
        company = Company.query.get(company_dto.id)
        if not company:
            return False

        dispatch_settings = ud_settings.for_company(company)
        pickup_service_min = dispatch_settings.service_times.pickup_service_min
        dropoff_service_min = dispatch_settings.service_times.dropoff_service_min
        min_transition_margin_min = (
            dispatch_settings.service_times.min_transition_margin_min
        )

        # Grouper par chauffeur
        by_driver: Dict[int, List[Dict[str, Any]]] = {}
        for step in plan:
            driver_id = step.get("driver_id")
            if driver_id:
                if driver_id not in by_driver:
                    by_driver[driver_id] = []
                by_driver[driver_id].append(step)

        # Vérifier chaque chauffeur
        MIN_STEPS_FOR_CONFLICT = 2
        for driver_steps in by_driver.values():
            if len(driver_steps) < MIN_STEPS_FOR_CONFLICT:
                continue

            # Trier par scheduled_time
            def get_scheduled_time(step: Dict[str, Any]) -> datetime:
                """Extrait le scheduled_time d'un step pour le tri."""
                # ✅ Utilisation du repository pour découpler de SQLAlchemy
                job_id = step.get("job_id")
                if job_id:
                    booking_repo = BookingRepository()
                    booking_dto = booking_repo.find_by_id(job_id)
                    if booking_dto and booking_dto.scheduled_time:
                        # Récupérer le modèle SQLAlchemy depuis le repository pour la compatibilité
                        booking = booking_repo.find_model_by_id(booking_dto.id)
                        if booking:
                            # booking.scheduled_time est une colonne SQLAlchemy,
                            # convertir en datetime
                            return cast(datetime, booking.scheduled_time)
                return datetime.min

            sorted_steps = sorted(driver_steps, key=get_scheduled_time)

            # Vérifier chaque paire consécutive
            for i in range(len(sorted_steps) - 1):
                current_step = sorted_steps[i]
                next_step = sorted_steps[i + 1]

                # ✅ Utilisation du repository pour découpler de SQLAlchemy
                current_job_id = current_step.get("job_id")
                next_job_id = next_step.get("job_id")
                if not current_job_id or not next_job_id:
                    continue
                booking_repo = BookingRepository()
                current_booking_dto = booking_repo.find_by_id(current_job_id)
                next_booking_dto = booking_repo.find_by_id(next_job_id)
                # Récupérer les modèles SQLAlchemy depuis le repository pour la compatibilité
                current_booking = (
                    booking_repo.find_model_by_id(current_booking_dto.id)
                    if current_booking_dto
                    else None
                )
                next_booking = (
                    booking_repo.find_model_by_id(next_booking_dto.id)
                    if next_booking_dto
                    else None
                )

                if not current_booking or not next_booking:
                    continue
                if (
                    not current_booking.scheduled_time
                    or not next_booking.scheduled_time
                ):
                    continue

                # Calculer temps de trajet course actuelle
                current_pickup_lat = getattr(current_booking, "pickup_lat", None)
                current_pickup_lon = getattr(current_booking, "pickup_lon", None)
                current_dropoff_lat = getattr(current_booking, "dropoff_lat", None)
                current_dropoff_lon = getattr(current_booking, "dropoff_lon", None)

                # Calculer temps de transition
                next_pickup_lat = getattr(next_booking, "pickup_lat", None)
                next_pickup_lon = getattr(next_booking, "pickup_lon", None)

                # Temps de trajet course actuelle
                if (
                    current_pickup_lat
                    and current_pickup_lon
                    and current_dropoff_lat
                    and current_dropoff_lon
                ):
                    trip_time_min = haversine_minutes(
                        current_pickup_lat,
                        current_pickup_lon,
                        current_dropoff_lat,
                        current_dropoff_lon,
                        avg_speed_kmh=25,
                    )
                else:
                    trip_time_min = 20  # Estimation par défaut

                # Temps de transition
                if (
                    current_dropoff_lat
                    and current_dropoff_lon
                    and next_pickup_lat
                    and next_pickup_lon
                ):
                    transition_time_min = haversine_minutes(
                        current_dropoff_lat,
                        current_dropoff_lon,
                        next_pickup_lat,
                        next_pickup_lon,
                        avg_speed_kmh=25,
                    )
                else:
                    transition_time_min = 15  # Estimation par défaut

                # Temps total nécessaire
                total_time_needed = (
                    trip_time_min
                    + dropoff_service_min
                    + transition_time_min
                    + pickup_service_min
                    + min_transition_margin_min
                )

                # Heure de fin estimée
                current_end_time = current_booking.scheduled_time + timedelta(
                    minutes=trip_time_min + pickup_service_min + dropoff_service_min
                )

                # Heure de début nécessaire
                required_start_time = next_booking.scheduled_time - timedelta(
                    minutes=transition_time_min
                    + pickup_service_min
                    + min_transition_margin_min
                )

                # Vérifier conflit
                if current_end_time > required_start_time:
                    time_gap = (
                        required_start_time - current_end_time
                    ).total_seconds() / 60
                    logger.warning(
                        (
                            "[AgentOrchestrator] ⚠️ Conflit temporel détecté: "
                            "Course #%s (fin %s) et #%s (début %s) → temps nécessaire: "
                            "%dmin, écart disponible: %.1fmin"
                        ),
                        current_booking.id,
                        current_end_time.strftime("%H:%M"),
                        next_booking.id,
                        next_booking.scheduled_time.strftime("%H:%M"),
                        total_time_needed,
                        time_gap,
                    )
                    return True

        return False

    def _check_delayed_optimizer(
        self, now: datetime, all_jobs: list[Dict[str, Any]], state: Dict[str, Any]
    ) -> None:
        """Optimiseur différé : vérifie 1h avant chaque course pour détecter
        les retards.

        Ne réorganise que si une meilleure solution réduit réellement les retards.
        Aussi vérifie les assignations inappropriées à l'urgent
        (seulement si régulier disponible).
        """
        from models import Driver

        # ✅ DÉTECTION ET CORRECTION : Courses assignées à l'urgent qui pourraient
        # être assignées à un régulier
        # (correction unique - une seule fois par course)
        MAX_EMERGENCY_ASSIGNMENTS_TO_CHECK = 3
        emergency_assignments_to_check = []

        # Initialiser emergency_corrections_done si None
        if self.state.emergency_corrections_done is None:
            self.state.emergency_corrections_done = set()

        for job in all_jobs:
            if job.get("status") != "assigned":
                continue

            job_id = job.get("job_id")
            driver_id = job.get("driver_id")

            if not job_id or not driver_id:
                continue

            # ✅ Ignorer si on a déjà tenté une correction pour cette course
            if job_id in self.state.emergency_corrections_done:
                continue

            # Vérifier si c'est un chauffeur d'urgence
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            driver_repo = DriverRepository()
            driver_dto = driver_repo.find_by_id(driver_id)
            if not driver_dto:
                continue
            # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
            driver = Driver.query.get(driver_dto.id)
            if not driver:
                continue

            driver_type = getattr(driver, "driver_type", None)
            if driver_type:
                driver_type_str = str(driver_type).strip().upper()
                if "." in driver_type_str:
                    driver_type_str = driver_type_str.split(".")[-1]
                if driver_type_str == "EMERGENCY":
                    # Course assignée à l'urgent → vérifier si un régulier
                    # pourrait la prendre
                    emergency_assignments_to_check.append((job_id, driver_id))

        # Si des courses sont assignées à l'urgent, vérifier si on peut
        # les réassigner à un régulier
        if (
            emergency_assignments_to_check
            and len(emergency_assignments_to_check)
            <= MAX_EMERGENCY_ASSIGNMENTS_TO_CHECK
        ):
            logger.info(
                (
                    "[AgentOrchestrator] 🔍 Détection: "
                    "%d course(s) assignée(s) à l'urgent, "
                    "vérification si réassignation possible"
                ),
                len(emergency_assignments_to_check),
            )

            # Pour chaque course assignée à l'urgent, chercher un régulier disponible
            for job_id, emergency_driver_id in emergency_assignments_to_check:
                # Marquer comme "déjà vérifié" pour éviter répétitions
                self.state.emergency_corrections_done.add(job_id)

                # Trouver un régulier disponible pour cette course
                best_regular_driver = self._find_regular_driver_for_booking(
                    job_id, state
                )

                if best_regular_driver:
                    logger.info(
                        (
                            "[AgentOrchestrator] 🔄 Correction: Réassignation "
                            "course #%s de l'urgent #%s vers régulier #%s"
                        ),
                        job_id,
                        emergency_driver_id,
                        best_regular_driver,
                    )

                    # Vérifier garde-fous
                    can_proceed, reason = self.safety.check_action(
                        action_type="assign",
                        context={
                            "job_id": job_id,
                            "driver_id": best_regular_driver,
                            "reason": "correction_urgent",
                        },
                    )

                    if can_proceed:
                        result = self.tools.assign(
                            job_id=job_id,
                            driver_id=best_regular_driver,
                            note=(
                                f"Correction assignation urgente → régulier "
                                f"{now.isoformat()}"
                            ),
                        )

                        if result.get("ok"):
                            self.state.actions_today += 1
                            self.state.actions_last_hour += 1
                            logger.info(
                                (
                                    "[AgentOrchestrator] ✅ Correction réussie: "
                                    "course #%s réassignée de l'urgent "
                                    "vers régulier #%s"
                                ),
                                job_id,
                                best_regular_driver,
                            )
                        else:
                            logger.warning(
                                (
                                    "[AgentOrchestrator] ⚠️ Correction échouée pour "
                                    "course #%s: %s"
                                ),
                                job_id,
                                result.get("error"),
                            )
                    else:
                        logger.warning(
                            (
                                "[AgentOrchestrator] ⚠️ Correction bloquée par safety "
                                "pour course #%s: %s"
                            ),
                            job_id,
                            reason,
                        )
                else:
                    logger.debug(
                        (
                            "[AgentOrchestrator] ℹ️ Aucun régulier disponible pour "
                            "course #%s (urgent nécessaire)"
                        ),
                        job_id,
                    )

        # Vérifier les courses dans la prochaine heure pour détecter les retards
        one_hour_later = now + timedelta(hours=1)

        jobs_to_check = []
        for job in all_jobs:
            if job.get("status") != "assigned":
                continue

            scheduled_time_str = job.get("scheduled_time")
            if not scheduled_time_str:
                continue

            try:
                if scheduled_time_str.endswith("Z"):
                    scheduled_time_str = scheduled_time_str.replace("Z", "+00:00")
                scheduled_dt = datetime.fromisoformat(scheduled_time_str)

                # Vérifier si la course est dans la prochaine heure
                if now <= scheduled_dt <= one_hour_later:
                    jobs_to_check.append((job, scheduled_dt))
            except Exception:
                continue

        if not jobs_to_check:
            return

        logger.debug(
            (
                "[AgentOrchestrator] 🔍 Optimiseur différé: vérification "
                "de %d course(s) dans la prochaine heure"
            ),
            len(jobs_to_check),
        )

        # Pour chaque course, vérifier si un retard est prévu
        # TODO: Implémenter calcul ETA réel et détection de retard
        # Pour l'instant, on ne fait que logger
        for job, scheduled_dt in jobs_to_check:
            job_id = job.get("job_id")
            driver_id = job.get("driver_id")

            if not job_id or not driver_id:
                continue

            # Vérifier si un retard est prévu (simplifié pour l'instant)
            # TODO: Calculer ETA réel et comparer avec scheduled_time
            logger.debug(
                (
                    "[AgentOrchestrator] ⏰ Course #%s à %s (driver %s) - "
                    "vérification retard"
                ),
                job_id,
                scheduled_dt.strftime("%H:%M"),
                driver_id,
            )

            # Si retard détecté, proposer réorganisation seulement si meilleure solution
            # (à implémenter avec calcul ETA réel)

    def _generate_periodic_report(self) -> None:
        """Génère un rapport périodique."""
        try:
            report = generate_daily_report(self.company_id)
            # Envoyer via notification (email)
            company_email = (
                getattr(self.company, "email", None) if self.company else None
            )
            self.tools.notify(
                channel="email",
                to=company_email or "admin@atmr.com",
                template_id="daily_dispatch_report",
                vars=report,
            )
            self.state.last_report = now_local()
            logger.info("[AgentOrchestrator] 📊 Rapport quotidien généré")
        except Exception as e:
            logger.exception("[AgentOrchestrator] Erreur génération rapport: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Retourne l'état actuel de l'agent.

        Returns:
            Dict avec running, last_tick, actions_today, osrm_health, etc.

        """
        return {
            "running": self.state.running,
            "last_tick": self.state.last_tick.isoformat()
            if self.state.last_tick
            else None,
            "actions_today": self.state.actions_today,
            "actions_last_hour": self.state.actions_last_hour,
            "osrm_health": self.state.osrm_health,
            "current_plan": self.state.current_plan,
            "last_report": self.state.last_report.isoformat()
            if self.state.last_report
            else None,
        }


# Singleton pour gérer les agents par entreprise
_active_agents: Dict[int, AgentOrchestrator] = {}
_agents_lock = threading.Lock()


def get_agent_for_company(company_id: int, app=None) -> AgentOrchestrator:
    """Récupère ou crée un agent pour une entreprise.

    Args:
        company_id: ID de l'entreprise
        app: Instance Flask app (optionnel)

    Returns:
        Instance AgentOrchestrator

    """
    with _agents_lock:
        if company_id not in _active_agents:
            agent = AgentOrchestrator(company_id, app=app)
            _active_agents[company_id] = agent
            logger.info(
                "[AgentOrchestrator] Created new agent for company %s", company_id
            )
        else:
            agent = _active_agents[company_id]
            logger.debug(
                "[AgentOrchestrator] Reusing existing agent for company %s",
                company_id,
            )

        return agent


def stop_agent_for_company(company_id: int) -> None:
    """Arrête l'agent d'une entreprise.

    Args:
        company_id: ID de l'entreprise

    """
    with _agents_lock:
        agent = _active_agents.pop(company_id, None)
        if agent:
            agent.stop()
            logger.info("[AgentOrchestrator] Stopped agent for company %s", company_id)
