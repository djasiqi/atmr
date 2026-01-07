# backend/services/unified_dispatch/realtime_optimizer.py

# Constantes pour éviter les valeurs magiques
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Tuple
from typing import cast as tcast

from cachetools import TTLCache  # pyright: ignore[reportMissingModuleSource]
from flask import current_app  # pyright: ignore[reportMissingImports]

from ext import db
from models import Assignment, Booking, BookingStatus, DelayEvent, Driver
from repositories.assignment_repository import AssignmentRepository
from repositories.booking_repository import BookingRepository
from repositories.driver_repository import DriverRepository
from services.dispatch.auto_reassignment import get_auto_reassignment_service
from services.ml.models.eta_delay import get_eta_delay_model
from services.notifications.core import notify_dispatcher_optimization_opportunity
from services.osrm_client import _table
from services.unified_dispatch.data import calculate_eta
from services.unified_dispatch.delay_predictor import DelayPredictor
from services.unified_dispatch.utils.suggestions import Suggestion, SuggestionEngine
from shared.time_utils import day_local_bounds, now_local

TIME_DIFFERENCE_ZERO = 0
TIME_DIFFERENCE_THRESHOLD = 300
DELAY_MINUTES_THRESHOLD = 5
ABS_DELAY_THRESHOLD = 10
MIN_DETECTION_THRESHOLD = 5
OVERLOADED_DRIVER_THRESHOLD = 2
DEFAULT_CONFIDENCE_THRESHOLD = (
    0.6  # Epic 4.1 - Seuil P(retard) pour notification/reassign
)
# ✅ 3.5.1: Constantes pour inférence cause retard
HIGH_DELAY_TRAFFIC_THRESHOLD = 30  # Retard > 30 min = probablement trafic
BOOKING_TIME_DIFF_THRESHOLD = (
    5  # Booking créé/modifié dans les ±5 min = probablement retard booking
)

# ✅ P1: Constantes pour optimisations performance
REALTIME_OPTIMIZER_TIME_WINDOW_HOURS = 2  # Fenêtre temporelle ±2h pour requêtes DB
DELAY_CALCULATION_CACHE_TTL_SEC = 300  # TTL cache calculs de délai (5 minutes)
DELAY_CALCULATION_CACHE_MAX_SIZE = 1000  # Taille max cache (1000 entrées)
REALTIME_OPTIMIZER_BATCH_OSRM_THRESHOLD = (
    3  # Seuil minimum d'assignations pour utiliser batch OSRM
)

# ✅ P1: Cache pour résultats de calculs de délai (TTL 5 minutes)
_DELAY_CALCULATION_CACHE: TTLCache[str, int] = TTLCache(
    maxsize=DELAY_CALCULATION_CACHE_MAX_SIZE, ttl=DELAY_CALCULATION_CACHE_TTL_SEC
)
_DELAY_CACHE_LOCK = threading.Lock()

"""Système d'optimisation en temps réel pour le dispatch.
Surveille en continu les assignations et propose des ajustements automatiques.
"""
# date.today() utilisé volontairement pour comparaisons de dates locales


logger = logging.getLogger(__name__)


@dataclass
class OptimizationOpportunity:
    """Opportunité d'optimisation détectée."""

    assignment_id: int
    booking_id: int
    driver_id: int
    current_delay_minutes: int
    severity: str  # "low", "medium", "high", "critical"
    suggestions: List[Suggestion]
    detected_at: datetime
    auto_applicable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "booking_id": self.booking_id,
            "driver_id": self.driver_id,
            "current_delay_minutes": self.current_delay_minutes,
            "severity": self.severity,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "detected_at": self.detected_at.isoformat(),
            "auto_applicable": self.auto_applicable,
        }


class RealtimeOptimizer:
    """Monitore en continu les assignations et propose des optimisations.
    Peut fonctionner en mode manuel (sur demande) ou automatique (background).
    """

    def __init__(self, company_id: int, check_interval_seconds: int = 120, app=None):
        """Args:
        company_id: ID de l'entreprise à monitorer
        check_interval_seconds: Intervalle entre chaque vérification (défaut: 2 min)
        app: Instance Flask app (pour le contexte dans le thread).

        """
        super().__init__()
        self.company_id = company_id
        self.check_interval = check_interval_seconds
        self.suggestion_engine = SuggestionEngine()
        self.delay_predictor = DelayPredictor()
        self.eta_delay_model = (
            get_eta_delay_model()
        )  # Epic 4.1 - Modèle ML prédiction retard
        self.auto_reassignment_service = get_auto_reassignment_service()  # ✅ 3.4.1
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_check: datetime | None = None
        self._opportunities: List[OptimizationOpportunity] = []
        self._lock = threading.Lock()
        # current_app est un LocalProxy, _get_current_object() existe mais
        # pyright ne le reconnaît pas
        self._app = (
            app or getattr(current_app, "_get_current_object", lambda: current_app)()
        )

    def start_monitoring(self) -> None:
        """Démarre le monitoring en arrière-plan."""
        if self._running:
            logger.warning(
                "[RealtimeOptimizer] Already running for company %s", self.company_id
            )
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=False,  # ⭐ NON-DAEMON : le thread persiste même si
            # la requête HTTP se termine
            name=f"RealtimeOptimizer-{self.company_id}",
        )
        self._thread.start()
        logger.info(
            "[RealtimeOptimizer] Started PERSISTENT monitoring for company %s",
            self.company_id,
        )

    def stop_monitoring(self) -> None:
        """Arrête le monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(
            "[RealtimeOptimizer] Stopped monitoring for company %s", self.company_id
        )

    def _monitoring_loop(self) -> None:
        """Boucle principale de monitoring."""
        while self._running:
            try:
                # ⭐ IMPORTANT : Utiliser le contexte Flask dans le thread
                with self._app.app_context():
                    # Vérifier les assignations du jour
                    opportunities = self.check_current_assignments()

                    # Notifier si opportunités critiques
                    if opportunities:
                        self._notify_opportunities(opportunities)

                    # Mettre à jour le cache
                    with self._lock:
                        self._opportunities = opportunities
                        self._last_check = now_local()

            except Exception as e:
                logger.exception(
                    "[RealtimeOptimizer] Error in monitoring loop for company %s: %s",
                    self.company_id,
                    e,
                )

            # Pause avant la prochaine vérification
            time.sleep(self.check_interval)

    def check_current_assignments(
        self, for_date: str | None = None
    ) -> List[OptimizationOpportunity]:
        """Vérifie toutes les assignations actives et détecte
        les opportunités d'optimisation.

        Args:
            for_date: Date à vérifier (format YYYY-MM-DD), par défaut aujourd'hui

        Returns:
            Liste d'opportunités d'optimisation détectées

        """
        if for_date is None:
            for_date = date.today().strftime("%Y-%m-%d")

        try:
            d0, d1 = day_local_bounds(for_date)
        except Exception:
            logger.warning("[RealtimeOptimizer] Invalid date %s, using today", for_date)
            d0, d1 = day_local_bounds(date.today().strftime("%Y-%m-%d"))

        opportunities: List[OptimizationOpportunity] = []

        try:
            # ✅ P1: Limiter la requête aux assignations proches de l'heure actuelle (±2h)
            # Évite de charger toutes les assignations de la journée
            current_time = now_local()
            time_window_start = current_time - timedelta(
                hours=REALTIME_OPTIMIZER_TIME_WINDOW_HOURS
            )
            time_window_end = current_time + timedelta(
                hours=REALTIME_OPTIMIZER_TIME_WINDOW_HOURS
            )

            # ✅ Utilisation des repositories pour découpler de SQLAlchemy
            # Récupérer d'abord les bookings dans la fenêtre temporelle
            booking_repo = BookingRepository()
            booking_dtos = booking_repo.find_for_day(self.company_id, for_date)

            # Filtrer par fenêtre temporelle et statuts
            valid_statuses = [BookingStatus.ACCEPTED, BookingStatus.ASSIGNED]
            filtered_booking_dtos = [
                dto
                for dto in booking_dtos
                if dto.scheduled_time
                and d0 <= dto.scheduled_time < d1
                and time_window_start <= dto.scheduled_time <= time_window_end
                and dto.status in valid_statuses
            ]

            # Récupérer les assignments pour ces bookings
            assignment_repo = AssignmentRepository()
            booking_ids = [dto.id for dto in filtered_booking_dtos]
            assignment_dtos = assignment_repo.find_by_booking_ids(booking_ids)

            # Récupérer les modèles SQLAlchemy depuis les IDs des DTOs pour la compatibilité
            # (nécessaire pour eager loading avec joinedload)
            assignment_ids = [dto.id for dto in assignment_dtos]
            if assignment_ids:
                from sqlalchemy.orm import joinedload

                assignments = (
                    Assignment.query.options(
                        joinedload(Assignment.booking), joinedload(Assignment.driver)
                    )
                    .filter(Assignment.id.in_(assignment_ids))
                    .all()
                )
            else:
                assignments = []

            logger.debug(
                (
                    "[RealtimeOptimizer] Checking %d assignments for company %s "
                    "(time window: ±%d hours around %s)"
                ),
                len(assignments),
                self.company_id,
                REALTIME_OPTIMIZER_TIME_WINDOW_HOURS,
                current_time.isoformat()[:16],
            )

            # ✅ P1: Batch OSRM pour calculs ETA si plusieurs assignations avec GPS
            if len(assignments) >= REALTIME_OPTIMIZER_BATCH_OSRM_THRESHOLD:
                try:
                    self._calculate_delays_batch(assignments, current_time)
                except Exception as e:
                    logger.warning(
                        "[RealtimeOptimizer] Batch OSRM failed, falling back to individual: %s",
                        e,
                    )
                    # Continue avec calculs individuels en cas d'échec

            # Analyser chaque assignation
            for assignment in assignments:
                opportunity = self._analyze_assignment(assignment)
                if opportunity:
                    opportunities.append(opportunity)

            # 🆕 DÉTECTION INTELLIGENTE : Chauffeurs surchargés avec multiples retards
            opportunities.extend(self._detect_overloaded_drivers(assignments))

            # Trier par sévérité
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            opportunities.sort(
                key=lambda o: (
                    priority_order.get(o.severity, 99),
                    -abs(o.current_delay_minutes),
                )
            )

            logger.info(
                (
                    "[RealtimeOptimizer] Found %d optimization opportunities "
                    "for company %s"
                ),
                len(opportunities),
                self.company_id,
            )

        except Exception as e:
            logger.exception(
                "[RealtimeOptimizer] Failed to check assignments for company %s: %s",
                self.company_id,
                e,
            )

        return opportunities

    def _analyze_assignment(
        self, assignment: Assignment
    ) -> OptimizationOpportunity | None:
        """Analyse une assignation pour détecter des opportunités d'optimisation.

        Returns:
            OptimizationOpportunity si une optimisation est possible, None sinon

        """
        try:
            booking_id_val = int(assignment.booking_id)  # type: ignore[reportArgumentType]
            booking = db.session.get(Booking, booking_id_val)
            if not booking:
                return None

            driver_id_val = int(assignment.driver_id) if assignment.driver_id else None  # type: ignore[reportArgumentType,reportGeneralTypeIssues]
            driver = db.session.get(Driver, driver_id_val) if driver_id_val else None
            if not driver:
                return None

            # Calculer le retard en temps réel
            delay_minutes = self._calculate_realtime_delay(assignment, booking, driver)

            # Epic 4.1 - Utiliser prédiction ML si P(retard) > seuil
            ml_prediction = self.eta_delay_model.predict(booking, driver, now_local())
            if (
                ml_prediction.probability_delay > DEFAULT_CONFIDENCE_THRESHOLD
                and ml_prediction.predicted_delay_minutes > delay_minutes
            ):
                # Mettre à jour delay_minutes avec prédiction ML
                delay_minutes = int(ml_prediction.predicted_delay_minutes)
                logger.info(
                    (
                        "[RealtimeOptimizer] Prédiction ML: booking %s, "
                        "P(retard)=%.2f, delay=%d min"
                    ),
                    booking.id,
                    ml_prediction.probability_delay,
                    delay_minutes,
                )

            # Seuil de détection : au moins 5 min de retard (plus sensible)
            if abs(delay_minutes) < MIN_DETECTION_THRESHOLD:
                return None

            # Générer des suggestions
            suggestions = self.suggestion_engine.generate_suggestions_for_assignment(
                assignment, delay_minutes, self.company_id
            )

            if not suggestions:
                return None

            # Déterminer la sévérité
            severity = self._determine_severity(delay_minutes, booking)

            # ✅ 3.5.1: Logger événement retard dans table delay_events
            try:
                # Créer instance DelayEvent avec attributs (SQLAlchemy accepte attributs directement)
                delay_event = DelayEvent()
                delay_event.assignment_id = int(tcast("Any", assignment.id))
                delay_event.booking_id = int(tcast("Any", booking.id))
                delay_event.delay_minutes = delay_minutes
                delay_event.severity = severity
                delay_event.detected_at = now_local()
                delay_event.cause = self._infer_delay_cause(
                    assignment, booking, driver, delay_minutes
                )
                db.session.add(delay_event)
                db.session.commit()
                logger.debug(
                    "[RealtimeOptimizer] Logged delay event %s for assignment %s",
                    delay_event.id,
                    assignment.id,
                )
            except Exception as e:
                logger.warning(
                    "[RealtimeOptimizer] Failed to log delay event for assignment %s: %s",
                    assignment.id,
                    e,
                )
                db.session.rollback()

            # Vérifier si auto-applicable (toutes les suggestions sont
            # auto-applicables)
            auto_applicable = all(s.auto_applicable for s in suggestions)

            return OptimizationOpportunity(
                assignment_id=int(tcast("Any", assignment.id)),
                booking_id=int(tcast("Any", booking.id)),
                driver_id=int(tcast("Any", driver.id)),
                current_delay_minutes=delay_minutes,
                severity=severity,
                suggestions=suggestions,
                detected_at=now_local(),
                auto_applicable=auto_applicable,
            )

        except Exception as e:
            logger.warning(
                "[RealtimeOptimizer] Failed to analyze assignment %s: %s",
                getattr(assignment, "id", None),
                e,
            )
            return None

    def _calculate_realtime_delay(
        self, assignment: Assignment, booking: Booking, driver: Driver
    ) -> int:
        """Calcule le retard en temps réel basé sur la position actuelle du chauffeur.

        ✅ P1: Utilise un cache TTL pour éviter recalculs répétés.

        Returns:
            Retard en minutes (positif = retard, négatif = avance)

        """
        try:
            # Temps prévu
            scheduled_time = getattr(booking, "scheduled_time", None)
            if not scheduled_time:
                return 0

            current_time = now_local()

            # ✅ P1: Créer clé de cache pour ce calcul
            assignment_id = int(tcast("Any", getattr(assignment, "id", 0)) or 0)
            booking_id = int(tcast("Any", getattr(booking, "id", 0)) or 0)
            driver_id = int(tcast("Any", getattr(driver, "id", 0)) or 0)

            # Position actuelle du chauffeur
            driver_pos = (
                getattr(driver, "current_lat", getattr(driver, "latitude", None)),
                getattr(driver, "current_lon", getattr(driver, "longitude", None)),
            )

            # Position du pickup
            pickup_pos = (
                getattr(booking, "pickup_lat", None),
                getattr(booking, "pickup_lon", None),
            )

            # ✅ P1: Créer clé de cache basée sur assignment + positions + temps
            # (arrondi à la minute pour éviter cache trop granulaire)
            cache_key_parts = [
                f"a:{assignment_id}",
                f"b:{booking_id}",
                f"d:{driver_id}",
                f"st:{scheduled_time.isoformat()[:16]}",  # Arrondi à la minute
            ]
            if all(driver_pos):
                cache_key_parts.append(
                    f"dp:{driver_pos[0]:.4f},{driver_pos[1]:.4f}"
                )  # Arrondi à 4 décimales
            if all(pickup_pos):
                cache_key_parts.append(
                    f"pp:{pickup_pos[0]:.4f},{pickup_pos[1]:.4f}"
                )  # Arrondi à 4 décimales

            cache_key = "|".join(cache_key_parts)
            cache_key_hash = hashlib.md5(
                cache_key.encode("utf-8"), usedforsecurity=False
            ).hexdigest()

            # ✅ P1: Vérifier cache avant calcul
            with _DELAY_CACHE_LOCK:
                if cache_key_hash in _DELAY_CALCULATION_CACHE:
                    cached_delay = _DELAY_CALCULATION_CACHE[cache_key_hash]
                    logger.debug(
                        "[RealtimeOptimizer] ✅ Delay cache hit for assignment %s: %d min",
                        assignment_id,
                        cached_delay,
                    )
                    return cached_delay

            # ⭐ CAS 1 : GPS disponible → Calcul ETA précis
            if all(driver_pos) and all(pickup_pos):
                try:
                    # Cast pour typage strict (déjà validé par all())
                    driver_pos_valid = tcast("tuple[float, float]", driver_pos)
                    pickup_pos_valid = tcast("tuple[float, float]", pickup_pos)
                    eta_seconds = calculate_eta(driver_pos_valid, pickup_pos_valid)
                    current_eta = current_time + timedelta(seconds=eta_seconds)
                    delay_seconds = (current_eta - scheduled_time).total_seconds()
                    delay_minutes = int(delay_seconds / 60)

                    logger.debug(
                        (
                            "[RealtimeOptimizer] Assignment %s: "
                            "ETA-based delay = %d min (GPS: %s → %s)"
                        ),
                        assignment.id,
                        delay_minutes,
                        driver_pos,
                        pickup_pos,
                    )

                    # ✅ P1: Mettre en cache le résultat
                    with _DELAY_CACHE_LOCK:
                        _DELAY_CALCULATION_CACHE[cache_key_hash] = delay_minutes

                    return delay_minutes
                except Exception as e:
                    logger.warning(
                        (
                            "[RealtimeOptimizer] GPS calculation failed "
                            "for assignment %s: %s"
                        ),
                        assignment.id,
                        e,
                    )
                    # Fallback au cas 2

            # ⭐ CAS 2 : Pas de GPS → Comparer simplement l'heure actuelle
            # vs heure prévue
            # Si l'heure actuelle est déjà après l'heure prévue, c'est un
            # retard
            time_difference = (current_time - scheduled_time).total_seconds()

            # Si l'heure est déjà passée et que le statut n'est pas en route,
            # c'est un retard
            if time_difference > TIME_DIFFERENCE_ZERO:
                delay_minutes = int(time_difference / 60)

                # Ajouter un buffer de temps de trajet estimé (ex: 15 min)
                # Le chauffeur devrait être parti 15 min avant l'heure prévue
                buffer_minutes = 15
                total_delay = (
                    delay_minutes + buffer_minutes
                    if time_difference > TIME_DIFFERENCE_THRESHOLD
                    else delay_minutes
                )

                logger.debug(
                    (
                        "[RealtimeOptimizer] Assignment %s: "
                        "Time-based delay = %d min (no GPS, time diff: %.1f min)"
                    ),
                    assignment.id,
                    total_delay,
                    time_difference / 60,
                )

                # ✅ P1: Mettre en cache le résultat
                with _DELAY_CACHE_LOCK:
                    _DELAY_CALCULATION_CACHE[cache_key_hash] = total_delay

                return total_delay

            # Pas encore de retard
            # ✅ P1: Mettre en cache même les résultats 0 (évite recalculs)
            with _DELAY_CACHE_LOCK:
                _DELAY_CALCULATION_CACHE[cache_key_hash] = 0

            return 0

        except Exception as e:
            logger.warning(
                "[RealtimeOptimizer] Failed to calculate delay for assignment %s: %s",
                getattr(assignment, "id", None),
                e,
            )
            return 0

    def _calculate_delays_batch(
        self, assignments: List[Assignment], current_time: datetime
    ) -> None:
        """✅ P1: Calcule les délais en batch via OSRM pour toutes les assignations.

        Collecte toutes les assignations avec GPS disponibles, construit une matrice
        de distance OSRM, et remplit le cache avec les résultats.

        Args:
            assignments: Liste des assignations à traiter
            current_time: Heure actuelle pour calculs de délai
        """
        try:
            # Collecter toutes les assignations avec GPS disponibles
            batch_items: List[Dict[str, Any]] = []
            coords_list: List[Tuple[float, float]] = []
            coord_to_item: Dict[int, Dict[str, Any]] = {}

            for assignment in assignments:
                try:
                    booking_id_val = int(assignment.booking_id)  # type: ignore[reportArgumentType]
                    booking = db.session.get(Booking, booking_id_val)
                    if not booking:
                        continue

                    driver_id_val = (
                        int(assignment.driver_id) if assignment.driver_id else None  # type: ignore[reportArgumentType,reportGeneralTypeIssues]
                    )
                    driver = (
                        db.session.get(Driver, driver_id_val) if driver_id_val else None
                    )
                    if not driver:
                        continue

                    scheduled_time = getattr(booking, "scheduled_time", None)
                    if not scheduled_time:
                        continue

                    # Position actuelle du chauffeur
                    driver_pos = (
                        getattr(
                            driver, "current_lat", getattr(driver, "latitude", None)
                        ),
                        getattr(
                            driver, "current_lon", getattr(driver, "longitude", None)
                        ),
                    )

                    # Position du pickup
                    pickup_pos = (
                        getattr(booking, "pickup_lat", None),
                        getattr(booking, "pickup_lon", None),
                    )

                    # ✅ Uniquement les assignations avec GPS complet
                    if not all(driver_pos) or not all(pickup_pos):
                        continue

                    driver_pos_valid = tcast("tuple[float, float]", driver_pos)
                    pickup_pos_valid = tcast("tuple[float, float]", pickup_pos)

                    # Ajouter les coordonnées à la liste (driver puis pickup)
                    driver_idx = len(coords_list)
                    coords_list.append(driver_pos_valid)
                    pickup_idx = len(coords_list)
                    coords_list.append(pickup_pos_valid)

                    batch_items.append(
                        {
                            "assignment": assignment,
                            "booking": booking,
                            "driver": driver,
                            "scheduled_time": scheduled_time,
                            "driver_idx": driver_idx,
                            "pickup_idx": pickup_idx,
                        }
                    )

                    coord_to_item[driver_idx] = {
                        "type": "driver",
                        "item": batch_items[-1],
                    }
                    coord_to_item[pickup_idx] = {
                        "type": "pickup",
                        "item": batch_items[-1],
                    }

                except Exception as e:
                    logger.debug(
                        "[RealtimeOptimizer] Skipping assignment %s for batch: %s",
                        getattr(assignment, "id", None),
                        e,
                    )
                    continue

            MIN_COORDS_FOR_BATCH = 2
            if not batch_items or len(coords_list) < MIN_COORDS_FOR_BATCH:
                logger.debug(
                    "[RealtimeOptimizer] Not enough GPS-enabled assignments for batch OSRM"
                )
                return

            logger.info(
                "[RealtimeOptimizer] Batch OSRM: %d assignments, %d coordinates",
                len(batch_items),
                len(coords_list),
            )

            # ✅ Construire matrice de distance OSRM en une seule requête
            try:
                # Utiliser OSRM pour calculer la matrice de distance
                # sources = tous les drivers, destinations = tous les pickups
                sources = [item["driver_idx"] for item in batch_items]
                destinations = [item["pickup_idx"] for item in batch_items]

                # Appel batch OSRM via _table (supporte sources/destinations)
                base_url = os.getenv("OSRM_BASE_URL", "http://osrm:5000")
                profile = "driving"
                timeout = 30  # Timeout 30s pour batch

                table_result = _table(
                    base_url=base_url,
                    profile=profile,
                    coords=coords_list,
                    sources=sources,
                    destinations=destinations,
                    timeout=timeout,
                )

                # Extraire la matrice de durées depuis la réponse OSRM
                durations_matrix = table_result.get("durations", [])
                if not durations_matrix:
                    logger.warning(
                        "[RealtimeOptimizer] Batch OSRM: no durations in response"
                    )
                    return

                distance_matrix_seconds = durations_matrix

                # ✅ Mapper les résultats aux assignations et remplir le cache
                for item in batch_items:
                    try:
                        driver_idx = item["driver_idx"]
                        pickup_idx = item["pickup_idx"]
                        assignment = item["assignment"]
                        booking = item["booking"]
                        driver = item["driver"]
                        scheduled_time = item["scheduled_time"]

                        # Récupérer ETA depuis la matrice
                        if driver_idx < len(
                            distance_matrix_seconds
                        ) and pickup_idx < len(distance_matrix_seconds[driver_idx]):
                            eta_seconds = int(
                                distance_matrix_seconds[driver_idx][pickup_idx]
                            )
                        else:
                            logger.warning(
                                (
                                    "[RealtimeOptimizer] Invalid matrix indices: "
                                    "driver_idx=%d, pickup_idx=%d"
                                ),
                                driver_idx,
                                pickup_idx,
                            )
                            continue

                        # Calculer le délai
                        current_eta = current_time + timedelta(seconds=eta_seconds)
                        delay_seconds = (current_eta - scheduled_time).total_seconds()
                        delay_minutes = int(delay_seconds / 60)

                        # ✅ Créer clé de cache (identique à _calculate_realtime_delay)
                        assignment_id = int(
                            tcast("Any", getattr(assignment, "id", 0)) or 0
                        )
                        booking_id = int(tcast("Any", getattr(booking, "id", 0)) or 0)
                        driver_id = int(tcast("Any", getattr(driver, "id", 0)) or 0)

                        driver_pos = (
                            getattr(
                                driver, "current_lat", getattr(driver, "latitude", None)
                            ),
                            getattr(
                                driver,
                                "current_lon",
                                getattr(driver, "longitude", None),
                            ),
                        )
                        pickup_pos = (
                            getattr(booking, "pickup_lat", None),
                            getattr(booking, "pickup_lon", None),
                        )

                        cache_key_parts = [
                            f"a:{assignment_id}",
                            f"b:{booking_id}",
                            f"d:{driver_id}",
                            f"st:{scheduled_time.isoformat()[:16]}",
                        ]
                        if all(driver_pos):
                            cache_key_parts.append(
                                f"dp:{driver_pos[0]:.4f},{driver_pos[1]:.4f}"
                            )
                        if all(pickup_pos):
                            cache_key_parts.append(
                                f"pp:{pickup_pos[0]:.4f},{pickup_pos[1]:.4f}"
                            )

                        cache_key = "|".join(cache_key_parts)
                        cache_key_hash = hashlib.md5(
                            cache_key.encode("utf-8"), usedforsecurity=False
                        ).hexdigest()

                        # ✅ Remplir le cache avec le résultat
                        with _DELAY_CACHE_LOCK:
                            _DELAY_CALCULATION_CACHE[cache_key_hash] = delay_minutes

                        logger.debug(
                            (
                                "[RealtimeOptimizer] Batch OSRM: assignment %s → "
                                "delay %d min (cached)"
                            ),
                            assignment_id,
                            delay_minutes,
                        )

                    except Exception as e:
                        logger.warning(
                            "[RealtimeOptimizer] Failed to process batch item: %s", e
                        )
                        continue

                logger.info(
                    (
                        "[RealtimeOptimizer] ✅ Batch OSRM completed: %d ETAs "
                        "calculated in 1 request"
                    ),
                    len(batch_items),
                )

            except Exception as e:
                logger.warning(
                    "[RealtimeOptimizer] Batch OSRM matrix calculation failed: %s", e
                )
                # Ne pas lever l'exception : fallback vers calculs individuels

        except Exception as e:
            logger.warning("[RealtimeOptimizer] Batch OSRM preparation failed: %s", e)
            # Ne pas lever l'exception : fallback vers calculs individuels

    def _detect_overloaded_drivers(
        self, assignments: List[Assignment]
    ) -> List[OptimizationOpportunity]:
        """Détecte les chauffeurs surchargés avec plusieurs courses en retard.
        Suggère de répartir les courses sur plusieurs chauffeurs.
        """
        opportunities = []

        try:
            # ✅ PERF: Charger tous les bookings et drivers en une seule query
            # chacun (évite N+1)
            # ✅ P1: Utiliser set pour vérifications d'appartenance O(1)
            booking_ids = {int(a.booking_id) for a in assignments if a.booking_id}  # type: ignore[reportArgumentType,reportGeneralTypeIssues]
            driver_ids = {int(a.driver_id) for a in assignments if a.driver_id}  # type: ignore[reportArgumentType,reportGeneralTypeIssues]

            # ✅ Utilisation des repositories pour découpler de SQLAlchemy
            bookings_map = {}
            if booking_ids:
                booking_repo = BookingRepository()
                # Convertir set en list pour find_by_ids
                booking_dtos = booking_repo.find_by_ids(list(booking_ids))
                # Récupérer les modèles SQLAlchemy depuis les IDs des DTOs
                booking_model_ids = [dto.id for dto in booking_dtos]
                bookings = Booking.query.filter(Booking.id.in_(booking_model_ids)).all()
                bookings_map = {b.id: b for b in bookings}

            drivers_map = {}
            if driver_ids:
                # ✅ Utilisation du repository pour découpler de SQLAlchemy
                driver_repo = DriverRepository()
                driver_dtos = driver_repo.find_by_ids(list(driver_ids))
                # Récupérer les modèles SQLAlchemy depuis les IDs des DTOs pour la compatibilité
                driver_model_ids = [dto.id for dto in driver_dtos]
                drivers = Driver.query.filter(Driver.id.in_(driver_model_ids)).all()
                drivers_map = {d.id: d for d in drivers}

            # Grouper les assignations par chauffeur
            driver_delays: dict[int, list[dict[str, Any]]] = {}
            for assignment in assignments:
                driver_id_val = (
                    int(assignment.driver_id) if assignment.driver_id else None  # type: ignore[reportArgumentType,reportGeneralTypeIssues]
                )
                if not driver_id_val:
                    continue

                booking = bookings_map.get(int(assignment.booking_id))  # type: ignore[reportArgumentType]
                if not booking:
                    continue

                driver = drivers_map.get(driver_id_val)
                if not driver:
                    continue

                # Calculer le retard pour cette assignation
                delay_minutes = self._calculate_realtime_delay(
                    assignment, booking, driver
                )

                # Stocker si retard significatif (> 5 min)
                if delay_minutes > DELAY_MINUTES_THRESHOLD:
                    if driver_id_val not in driver_delays:
                        driver_delays[driver_id_val] = []
                    driver_delays[driver_id_val].append(
                        {
                            "assignment": assignment,
                            "booking": booking,
                            "delay": delay_minutes,
                        }
                    )

            # Détecter les chauffeurs avec 2+ courses en retard
            for driver_id, delayed_trips in driver_delays.items():
                if len(delayed_trips) >= OVERLOADED_DRIVER_THRESHOLD:
                    total_delay = sum(trip["delay"] for trip in delayed_trips)

                    # Créer une opportunité pour répartir les courses (driver
                    # déjà chargé)
                    driver = drivers_map.get(driver_id)
                    driver_name = (
                        f"{driver.user.first_name} {driver.user.last_name}"
                        if driver and driver.user
                        else f"#{driver_id}"
                    )

                    # Générer suggestion de répartition
                    suggestions = [
                        Suggestion(
                            action="redistribute",
                            priority="critical",
                            message=(
                                f"🚨 URGENT : {driver_name} a "
                                f"{len(delayed_trips)} courses en retard "
                                f"(retard total: {total_delay} min). "
                                f"Recommandation : Répartir sur "
                                f"{len(delayed_trips)} chauffeurs différents."
                            ),
                            driver_id=driver_id,
                            additional_data={
                                "delayed_trips_count": len(delayed_trips),
                                "total_delay": total_delay,
                                "booking_ids": [
                                    trip["booking"].id for trip in delayed_trips
                                ],
                                "driver_name": driver_name,
                            },
                            auto_applicable=False,
                        )
                    ]

                    # Utiliser la première course pour créer l'opportunité
                    first_trip = delayed_trips[0]
                    opportunities.append(
                        OptimizationOpportunity(
                            assignment_id=first_trip["assignment"].id,
                            booking_id=first_trip["booking"].id,
                            driver_id=driver_id,
                            current_delay_minutes=total_delay,
                            severity="critical",
                            suggestions=suggestions,
                            detected_at=now_local(),
                            auto_applicable=False,
                        )
                    )

                    logger.warning(
                        (
                            "[RealtimeOptimizer] 🚨 Driver %s is overloaded: "
                            "%d trips delayed (total: %d min)"
                        ),
                        driver_name,
                        len(delayed_trips),
                        total_delay,
                    )

        except Exception as e:
            logger.exception(
                "[RealtimeOptimizer] Failed to detect overloaded drivers: %s", e
            )

        return opportunities

    def _infer_delay_cause(
        self,
        assignment: Assignment,
        booking: Booking,
        driver: Driver,  # noqa: ARG002
        delay_minutes: int,
    ) -> str | None:
        """Infère la cause probable du retard.

        Args:
            assignment: Assignment concerné
            booking: Booking concerné
            driver: Driver concerné
            delay_minutes: Retard en minutes

        Returns:
            Cause probable ("traffic", "driver_late", "booking_delay", etc.) ou None
        """
        # Heuristique simple : si le retard est très élevé, probablement trafic
        if delay_minutes > HIGH_DELAY_TRAFFIC_THRESHOLD:
            return "traffic"

        # Si le chauffeur a plusieurs assignments en retard, probablement surchargé
        # (vérification simplifiée, pourrait être améliorée)
        if hasattr(assignment, "driver_id") and bool(assignment.driver_id):
            try:
                # ✅ Utilisation du repository pour découpler de SQLAlchemy
                assignment_repo = AssignmentRepository()
                other_assignments_dtos = assignment_repo.find_by_driver_id(
                    int(assignment.driver_id)  # type: ignore[reportArgumentType]
                )
                # Filtrer pour exclure l'assignation courante et ne garder que les statuts actifs
                other_assignments_count = sum(
                    1
                    for dto in other_assignments_dtos
                    if dto.id != assignment.id
                    and dto.status in ["assigned", "in_progress"]
                )
                if other_assignments_count >= OVERLOADED_DRIVER_THRESHOLD:
                    return "driver_overloaded"
            except Exception:
                pass

        # Si l'heure prévue est très proche de maintenant, probablement retard booking
        if booking.scheduled_time:
            time_diff = (booking.scheduled_time - now_local()).total_seconds() / 60
            if (
                -BOOKING_TIME_DIFF_THRESHOLD <= time_diff <= BOOKING_TIME_DIFF_THRESHOLD
            ):  # Booking créé/modifié très récemment
                return "booking_delay"

        # Par défaut, considérer comme retard chauffeur
        if delay_minutes > 0:
            return "driver_late"

        return None

    def _determine_severity(self, delay_minutes: int, booking: Booking) -> str:
        """Détermine la sévérité basée sur le retard et le type de booking."""
        abs_delay = abs(delay_minutes)

        # Retard critique si booking urgent ou médical
        is_urgent = getattr(booking, "is_urgent", False)
        is_medical = bool(getattr(booking, "medical_facility", None))

        if is_urgent or is_medical:
            if abs_delay >= ABS_DELAY_THRESHOLD:
                return "critical"
            if abs_delay >= ABS_DELAY_THRESHOLD:
                return "high"

        # Sévérité normale
        if abs_delay >= ABS_DELAY_THRESHOLD:
            return "critical"
        if abs_delay >= ABS_DELAY_THRESHOLD:
            return "high"
        if abs_delay >= ABS_DELAY_THRESHOLD:
            return "medium"
        return "low"

    def _notify_opportunities(
        self, opportunities: List[OptimizationOpportunity]
    ) -> None:
        """Envoie des notifications pour les opportunités critiques."""
        # Filtrer les opportunités critiques
        critical_opportunities = [
            o for o in opportunities if o.severity in ("critical", "high")
        ]

        if not critical_opportunities:
            return

        try:
            for opportunity in critical_opportunities:
                notify_dispatcher_optimization_opportunity(
                    {
                        "company_id": self.company_id,
                        "assignment_id": opportunity.assignment_id,
                        "booking_id": opportunity.booking_id,
                        "driver_id": opportunity.driver_id,
                        "current_delay": opportunity.current_delay_minutes,
                        "severity": opportunity.severity,
                        "suggestions": [s.to_dict() for s in opportunity.suggestions],
                        "auto_apply": opportunity.auto_applicable,
                    }
                )

                logger.info(
                    (
                        "[RealtimeOptimizer] Notified %s opportunity "
                        "for assignment %s (delay: %d min)"
                    ),
                    opportunity.severity,
                    opportunity.assignment_id,
                    opportunity.current_delay_minutes,
                )

        except Exception as e:
            logger.warning("[RealtimeOptimizer] Failed to notify opportunities: %s", e)

    def get_current_opportunities(self) -> List[OptimizationOpportunity]:
        """Récupère les opportunités détectées lors du dernier check.
        Thread-safe.
        """
        with self._lock:
            return list(self._opportunities)

    def get_status(self) -> Dict[str, Any]:
        """Récupère le statut du monitoring."""
        with self._lock:
            return {
                "running": self._running,
                "company_id": self.company_id,
                "last_check": self._last_check.isoformat()
                if self._last_check
                else None,
                "opportunities_count": len(self._opportunities),
                "critical_count": len(
                    [o for o in self._opportunities if o.severity == "critical"]
                ),
                "check_interval_seconds": self.check_interval,
            }


# Singleton pour gérer les optimizers par entreprise
_active_optimizers: Dict[int, RealtimeOptimizer] = {}
_optimizers_lock = threading.Lock()


def start_optimizer_for_company(
    company_id: int, check_interval: int = 120, app=None
) -> RealtimeOptimizer:
    """Démarre un optimizer pour une entreprise (ou récupère l'existant).

    Args:
        company_id: ID de l'entreprise
        check_interval: Intervalle de vérification en secondes
        app: Instance Flask app (optionnel, récupéré automatiquement si None)

    Returns:
        L'instance RealtimeOptimizer

    """
    with _optimizers_lock:
        if company_id not in _active_optimizers:
            # Passer l'app Flask au RealtimeOptimizer
            optimizer = RealtimeOptimizer(company_id, check_interval, app=app)
            optimizer.start_monitoring()
            _active_optimizers[company_id] = optimizer
            logger.info(
                "[RealtimeOptimizer] Started optimizer for company %s", company_id
            )
        else:
            optimizer = _active_optimizers[company_id]
            logger.debug(
                "[RealtimeOptimizer] Reusing existing optimizer for company %s",
                company_id,
            )

        return optimizer


def stop_optimizer_for_company(company_id: int) -> None:
    """Arrête l'optimizer d'une entreprise."""
    with _optimizers_lock:
        optimizer = _active_optimizers.pop(company_id, None)
        if optimizer:
            optimizer.stop_monitoring()
            logger.info(
                "[RealtimeOptimizer] Stopped optimizer for company %s", company_id
            )


def get_optimizer_for_company(company_id: int) -> RealtimeOptimizer | None:
    """Récupère l'optimizer d'une entreprise (sans le démarrer)."""
    with _optimizers_lock:
        return _active_optimizers.get(company_id)


def check_opportunities_manual(
    company_id: int, for_date: str | None = None, app=None
) -> List[OptimizationOpportunity]:
    """Vérifie manuellement les opportunités d'optimisation (sans monitoring continu).

    Args:
        company_id: ID de l'entreprise
        for_date: Date à vérifier (format YYYY-MM-DD)
        app: Instance Flask app (optionnel)

    Returns:
        Liste d'opportunités d'optimisation

    """
    optimizer = RealtimeOptimizer(company_id, app=app)
    return optimizer.check_current_assignments(for_date)
