"""Tools exposés à l'agent (function calls).

Interface stricte avec validation côté backend.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from flask import current_app
from sqlalchemy import and_, or_
from sqlalchemy.orm import joinedload

from ext import db
from models import Assignment, Booking, Driver
from models.autonomous_action import AutonomousAction
from models.enums import AssignmentStatus, BookingStatus
from services.unified_dispatch.validation import check_existing_assignment_conflict

logger = logging.getLogger(__name__)


class AgentTools:
    """Tools pour l'agent dispatch."""

    def __init__(self, company_id: int):
        """Initialise les tools.

        Args:
            company_id: ID de l'entreprise

        """
        super().__init__()
        self.company_id = company_id

    def get_state(
        self, window_start: datetime, window_end: datetime
    ) -> Dict[str, Any]:
        """Récupère l'état actuel du dispatch.

        Args:
            window_start: Début de la fenêtre temporelle
            window_end: Fin de la fenêtre temporelle

        Returns:
            {
                "drivers": [...],
                "jobs": [...],
                "constraints": {...},
                "slas": {...}
            }

        """
        with current_app.app_context():
            # Récupérer drivers disponibles
            drivers = (
                Driver.query.filter(
                    Driver.company_id == self.company_id,
                    Driver.is_available == True,  # noqa: E712
                )
                .all()
            )

            # Récupérer bookings dans la fenêtre
            # Inclure ACCEPTED, ASSIGNED pour détecter toutes les courses à assigner
            # Si scheduled_time est NULL, inclure quand même (pour courses sans heure définie)
            bookings = (
                Booking.query.filter(
                    Booking.company_id == self.company_id,
                    or_(
                        Booking.scheduled_time.is_(None),
                        and_(
                            Booking.scheduled_time >= window_start,
                            Booking.scheduled_time <= window_end,
                        ),
                    ),
                    Booking.status.in_([
                        BookingStatus.ACCEPTED,
                        BookingStatus.ASSIGNED,
                    ]),
                )
                .options(joinedload(Booking.assignments))
                .all()
            )

            # Récupérer assignments actifs (pour référence future)
            # assignments = (
            #     Assignment.query.join(Booking)
            #     .filter(
            #         Booking.company_id == self.company_id,
            #         Assignment.status.in_(
            #             [
            #                 AssignmentStatus.SCHEDULED,
            #                 AssignmentStatus.EN_ROUTE_PICKUP,
            #                 AssignmentStatus.ARRIVED_PICKUP,
            #                 AssignmentStatus.ONBOARD,
            #             ]
            #         ),
            #     )
            #     .all()
            # )

            # Construire état
            jobs = []
            logger.info(
                "[AgentTools] 🔍 Récupéré %d bookings pour company %s (fenêtre: %s -> %s)",
                len(bookings),
                self.company_id,
                window_start.isoformat(),
                window_end.isoformat(),
            )
            for booking in bookings:
                assignment = next(
                    (
                        a
                        for a in booking.assignments
                        if a.status != AssignmentStatus.COMPLETED
                    ),
                    None,
                )

                # Calculer ETA risk (simplifié pour Phase 1)
                # Si pas d'assignation, c'est une urgence
                eta_risk = "HIGH" if not assignment else "LOW"

                jobs.append(
                    {
                        "job_id": int(booking.id),
                        "booking_id": int(booking.id),
                        "status": "unassigned" if not assignment else "assigned",
                        "driver_id": int(assignment.driver_id)
                        if assignment and assignment.driver_id
                        else None,
                        "scheduled_time": booking.scheduled_time.isoformat()
                        if booking.scheduled_time
                        else None,
                        "pickup_location": {
                            "lat": float(booking.pickup_lat)
                            if booking.pickup_lat
                            else None,
                            "lon": float(booking.pickup_lon)
                            if booking.pickup_lon
                            else None,
                        },
                        "dropoff_location": {
                            "lat": float(booking.dropoff_lat)
                            if booking.dropoff_lat
                            else None,
                            "lon": float(booking.dropoff_lon)
                            if booking.dropoff_lon
                            else None,
                        },
                        "time_window": {
                            "start": None,  # Booking model n'a pas pickup_time_window_start
                            "end": None,  # Booking model n'a pas pickup_time_window_end
                        },
                        "eta_risk": eta_risk,
                    }
                )

            unassigned_count = len([j for j in jobs if j.get("status") == "unassigned"])
            logger.info(
                "[AgentTools] État construit: %d jobs (%d non assignés), %d drivers disponibles",
                len(jobs),
                unassigned_count,
                len(drivers),
            )
            
            return {
                "drivers": [
                    {
                        "driver_id": int(d.id),
                        "name": d.user.full_name if d.user else "Unknown",
                        "available": d.is_available,
                        "current_location": {
                            "lat": float(d.latitude) if d.latitude is not None else None,
                            "lon": float(d.longitude) if d.longitude is not None else None,
                        },
                    }
                    for d in drivers
                ],
                "jobs": jobs,
                "constraints": {
                    "time_windows_enabled": True,
                    "capacity": 1,
                    "preferred_driver_enabled": True,
                },
                "slas": {
                    "max_delay_minutes": 15,
                    "max_wait_minutes": 10,
                },
            }

    def osrm_health(self) -> Dict[str, Any]:
        """Vérifie la santé OSRM en effectuant un test réel.

        Returns:
            {"state": "CLOSED|OPEN|HALF_OPEN", "latency_ms": int, "fail_ratio": float}

        """
        try:
            import os

            import requests

            from services.osrm_client import _osrm_circuit_breaker

            # Récupérer état circuit breaker
            cb_state = _osrm_circuit_breaker.state
            cb_failures = _osrm_circuit_breaker.failure_count

            # Test réel OSRM : faire un appel simple pour mesurer la latence
            osrm_url = os.getenv("UD_OSRM_URL", "http://osrm:5000")
            test_coords = "6.12486,46.20896;6.14296,46.19603"  # Coordonnées de test (Genève)
            
            start = time.time()
            latency_ms = -1
            test_successful = False
            
            HTTP_OK = 200
            try:
                # Test simple : appel table OSRM avec 2 points
                response = requests.get(
                    f"{osrm_url}/table/v1/car/{test_coords}",
                    params={"annotations": "duration"},
                    timeout=3,
                )
                if response.status_code == HTTP_OK:
                    latency_ms = int((time.time() - start) * 1000)
                    test_successful = True
                    logger.debug("[AgentTools] OSRM health check OK: %dms", latency_ms)
                else:
                    logger.warning("[AgentTools] OSRM health check returned status %d", response.status_code)
            except requests.exceptions.Timeout:
                logger.warning("[AgentTools] OSRM health check timeout (>3s)")
                latency_ms = -1
            except requests.exceptions.ConnectionError:
                logger.warning("[AgentTools] OSRM health check connection error")
                latency_ms = -1
            except Exception as e:
                logger.warning("[AgentTools] OSRM health check error: %s", e)
                latency_ms = -1

            # Si le test échoue et que le circuit breaker est CLOSED, on le considère comme suspect
            if not test_successful and cb_state == "CLOSED":
                logger.warning("[AgentTools] OSRM test failed but circuit breaker is CLOSED - possible issue")

            return {
                "state": cb_state,
                "latency_ms": latency_ms,
                "fail_ratio": cb_failures / 10.0 if cb_failures > 0 else 0.0,
                "failure_count": cb_failures,
                "test_successful": test_successful,
            }
        except Exception as e:
            logger.warning("[AgentTools] OSRM health check failed: %s", e)
            return {
                "state": "OPEN",
                "latency_ms": -1,
                "fail_ratio": 1.0,
                "failure_count": 999,
                "test_successful": False,
            }

    def log_action(
        self, kind: str, payload: Dict[str, Any], reasoning_brief: str
    ) -> Dict[str, Any]:
        """Log une action pour audit.

        Args:
            kind: Type d'action (tick, assign, reoptimize, etc.)
            payload: Données de l'action
            reasoning_brief: Explication brève (≤50 mots)

        Returns:
            {"event_id": str}

        """
        with current_app.app_context():
            try:
                event_id = f"evt_{datetime.now().isoformat()}"

                action = AutonomousAction()
                action.company_id = self.company_id
                action.action_type = kind
                action.action_description = reasoning_brief
                action.action_data = json.dumps(payload)
                action.success = True
                action.trigger_source = "agent_dispatch"

                db.session.add(action)
                db.session.commit()

                MAX_REASONING_LENGTH = 50
                reasoning_len = len(reasoning_brief)
                truncated_reasoning = (
                    reasoning_brief[:MAX_REASONING_LENGTH]
                    if reasoning_len > MAX_REASONING_LENGTH
                    else reasoning_brief
                )
                logger.info(
                    "[AgentTools] 📝 Logged action: %s - %s",
                    kind,
                    truncated_reasoning,
                )

                return {"event_id": event_id}
            except Exception as e:
                logger.exception("[AgentTools] Error logging action: %s", e)
                db.session.rollback()
                return {"event_id": None, "error": str(e)}

    def assign(
        self, job_id: int, driver_id: int, note: str = ""  # noqa: ARG002
    ) -> Dict[str, Any]:
        """Assigne un job à un driver.

        Args:
            job_id: ID du booking (job)
            driver_id: ID du driver
            note: Note optionnelle pour l'assignation

        Returns:
            {"ok": bool, "conflict": bool, "error": str, "diff": {...}}

        """
        with current_app.app_context():
            try:
                booking = Booking.query.get(job_id)
                if not booking:
                    return {"ok": False, "error": f"Booking {job_id} not found"}

                driver = Driver.query.get(driver_id)
                if not driver:
                    return {"ok": False, "error": f"Driver {driver_id} not found"}

                # Vérifier que le driver est disponible
                if not driver.is_available:
                    return {
                        "ok": False,
                        "conflict": True,
                        "error": f"Driver {driver_id} is not available",
                    }
                
                # Les chauffeurs d'urgence sont utilisés uniquement en dernier recours
                # (quand aucune autre solution n'est viable)
                # Cette vérification est faite au niveau de l'orchestrateur, pas ici
                # Ici on accepte l'assignation si elle est proposée

                # Vérifier contraintes (TW, capacité)
                conflict = self._check_conflicts(booking, driver)
                if conflict:
                    return {"ok": False, "conflict": True, "error": conflict}

                # Créer/modifier assignment
                existing = (
                    Assignment.query.filter_by(
                        booking_id=job_id,
                    )
                    .filter(
                        Assignment.status.in_(
                            [
                                AssignmentStatus.SCHEDULED,
                                AssignmentStatus.EN_ROUTE_PICKUP,
                                AssignmentStatus.ARRIVED_PICKUP,
                                AssignmentStatus.ONBOARD,
                            ]
                        )
                    )
                    .first()
                )

                old_driver_id = existing.driver_id if existing else None

                if existing:
                    existing.driver_id = driver_id
                    # Note: Assignment n'a pas de champ notes, utiliser decision_explanation si nécessaire
                else:
                    existing = Assignment()
                    existing.booking_id = job_id
                    existing.driver_id = driver_id
                    existing.status = AssignmentStatus.SCHEDULED
                    db.session.add(existing)

                db.session.commit()

                return {
                    "ok": True,
                    "conflict": False,
                    "diff": {
                        "booking_id": job_id,
                        "old_driver_id": old_driver_id,
                        "new_driver_id": driver_id,
                        "action": "assigned" if not old_driver_id else "reassigned",
                    },
                }
            except Exception as e:
                db.session.rollback()
                logger.exception("[AgentTools] Error in assign: %s", e)
                return {"ok": False, "error": str(e)}

    def _check_conflicts(
        self, booking: Booking, driver: Driver
    ) -> Optional[str]:
        """Vérifie les conflits (TW, capacité) avec calculs de temps réels.

        Args:
            booking: Booking à assigner
            driver: Driver candidat

        Returns:
            Message d'erreur si conflit, None sinon

        """
        if not booking.scheduled_time:
            return "Booking has no scheduled_time"

        # Récupérer les paramètres configurables
        from models import Company
        from services.unified_dispatch import settings as ud_settings
        
        company = Company.query.get(self.company_id)
        if company:
            dispatch_settings = ud_settings.for_company(company)
            pickup_service_min = dispatch_settings.service_times.pickup_service_min
            dropoff_service_min = dispatch_settings.service_times.dropoff_service_min
            min_transition_margin_min = dispatch_settings.service_times.min_transition_margin_min
        else:
            # Valeurs par défaut si company non trouvée
            pickup_service_min = 5
            dropoff_service_min = 10
            min_transition_margin_min = 15

        # Vérifier conflit temporel avec autres assignments du driver
        # Utiliser une tolérance calculée : pickup + dropoff + marge transition
        tolerance_minutes = pickup_service_min + dropoff_service_min + min_transition_margin_min
        
        has_conflict, conflict_msg = check_existing_assignment_conflict(
            driver_id=int(driver.id),
            scheduled_time=booking.scheduled_time,
            booking_id=int(booking.id),
            tolerance_minutes=tolerance_minutes,
        )

        if has_conflict:
            return conflict_msg or "Temporal conflict detected"

        # Vérification supplémentaire : calculer le temps réel entre les courses
        # Récupérer les assignments existants du driver
        from models import Assignment, AssignmentStatus
        existing_assignments = (
            Assignment.query.join(Booking)
            .filter(
                Assignment.driver_id == driver.id,
                Assignment.booking_id != booking.id,
                Assignment.status.in_([
                    AssignmentStatus.SCHEDULED,
                    AssignmentStatus.EN_ROUTE_PICKUP,
                    AssignmentStatus.ARRIVED_PICKUP,
                    AssignmentStatus.ONBOARD,
                    AssignmentStatus.EN_ROUTE_DROPOFF,
                ])
            )
            .order_by(Booking.scheduled_time)
            .all()
        )

        # Vérifier chaque assignment existant
        for existing_assignment in existing_assignments:
            existing_booking = existing_assignment.booking
            if not existing_booking or not existing_booking.scheduled_time:
                continue

            # Calculer le temps nécessaire entre les deux courses
            # Temps de trajet de la course précédente (pickup → dropoff)
            # + Temps de trajet entre dropoff précédent et pickup suivant
            # + Temps de pickup + dropoff + marge transition
            
            from shared.geo_utils import haversine_distance
            
            # Temps de trajet de la course précédente (estimation)
            existing_pickup_lat = getattr(existing_booking, "pickup_lat", None)
            existing_pickup_lon = getattr(existing_booking, "pickup_lon", None)
            existing_dropoff_lat = getattr(existing_booking, "dropoff_lat", None)
            existing_dropoff_lon = getattr(existing_booking, "dropoff_lon", None)
            booking_pickup_lat = getattr(booking, "pickup_lat", None)
            booking_pickup_lon = getattr(booking, "pickup_lon", None)
            
            if existing_pickup_lat and existing_pickup_lon and \
               existing_dropoff_lat and existing_dropoff_lon:
                trip_distance_km = haversine_distance(
                    float(existing_pickup_lat), float(existing_pickup_lon),
                    float(existing_dropoff_lat), float(existing_dropoff_lon)
                )
                # Vitesse moyenne 25 km/h en ville
                trip_time_min = int((trip_distance_km / 25) * 60)
            else:
                trip_time_min = 20  # Estimation par défaut
            
            # Temps de trajet entre dropoff précédent et pickup suivant
            if existing_dropoff_lat and existing_dropoff_lon and \
               booking_pickup_lat and booking_pickup_lon:
                transition_distance_km = haversine_distance(
                    float(existing_dropoff_lat), float(existing_dropoff_lon),
                    float(booking_pickup_lat), float(booking_pickup_lon)
                )
                transition_time_min = int((transition_distance_km / 25) * 60)
            else:
                transition_time_min = 15  # Estimation par défaut
            
            # Temps total nécessaire
            total_time_needed = (
                trip_time_min +  # Temps de trajet course précédente
                dropoff_service_min +  # Temps de dropoff
                transition_time_min +  # Temps de trajet entre courses
                pickup_service_min +  # Temps de pickup
                min_transition_margin_min  # Marge de sécurité
            )
            
            # Calculer l'heure de fin estimée de la course précédente
            from datetime import timedelta
            existing_end_time = existing_booking.scheduled_time + timedelta(
                minutes=trip_time_min + pickup_service_min + dropoff_service_min
            )
            
            # Calculer l'heure de début nécessaire pour la nouvelle course
            required_start_time = booking.scheduled_time - timedelta(
                minutes=transition_time_min + pickup_service_min + min_transition_margin_min
            )
            
            # Vérifier si on a assez de temps
            if existing_end_time > required_start_time:
                time_gap = (required_start_time - existing_end_time).total_seconds() / 60
                return (
                    f"Conflit temporel avec course #{existing_booking.id} à {existing_booking.scheduled_time:%H:%M}. "
                    f"Temps nécessaire: {total_time_needed}min, écart disponible: {time_gap:.1f}min"
                )

        return None

    def reoptimize(
        self,
        scope: str,  # noqa: ARG002
        strategy: str,
        overrides: Optional[Dict[str, Any]] = None,
        for_date: Optional[str] = None,  # ✅ Nouveau paramètre pour spécifier la date
        force_reassign: bool = False,  # ✅ Si True, réassigne même les bookings déjà assignés aux réguliers
    ) -> Dict[str, Any]:
        """Ré-optimise le dispatch.

        Args:
            scope: "window" | "driver" | "all"
            strategy: "full" | "degraded_proximity"
            overrides: Paramètres de surcharge
            for_date: Date au format YYYY-MM-DD (optionnel, utilise aujourd'hui par défaut)

        Returns:
            {"plan": [...], "gains": {...}}

        """
        with current_app.app_context():
            try:
                from services.unified_dispatch.engine import run as dispatch_run
                from shared.time_utils import now_local

                # Déterminer for_date (utiliser celle fournie ou aujourd'hui)
                if not for_date:
                    for_date = now_local().strftime("%Y-%m-%d")

                # Ajuster overrides selon stratégie
                final_overrides = overrides or {}
                
                # ✅ Récupérer preferred_driver_id et driver_load_multipliers depuis les paramètres de la company
                from models import Company, Driver
                company = Company.query.get(self.company_id)
                if company:
                    autonomous_config = company.get_autonomous_config()
                    dispatch_overrides = autonomous_config.get("dispatch_overrides", {})
                    
                    logger.info(
                        "[AgentTools] 🔍 Récupération config dispatch: dispatch_overrides clés disponibles: %s",
                        list(dispatch_overrides.keys()) if dispatch_overrides else []
                    )
                    
                    # Récupérer preferred_driver_id depuis dispatch_overrides
                    if "preferred_driver_id" in dispatch_overrides:
                        preferred_driver_id = dispatch_overrides["preferred_driver_id"]
                        if preferred_driver_id:
                            # S'assurer que c'est un entier
                            try:
                                preferred_driver_id = int(preferred_driver_id)
                                # Vérifier que le chauffeur existe et appartient à la company
                                driver = Driver.query.filter(
                                    Driver.id == preferred_driver_id,
                                    Driver.company_id == self.company_id
                                ).first()
                                if driver:
                                    final_overrides["preferred_driver_id"] = preferred_driver_id
                                    driver_name = getattr(driver.user, "full_name", None) or getattr(driver, "name", None) or f"Chauffeur #{preferred_driver_id}"
                                    logger.info(
                                        "[AgentTools] 🎯 Chauffeur préféré DÉTECTÉ et ACTIVÉ: %s (%s) - sera priorisé dans les assignations",
                                        preferred_driver_id,
                                        driver_name
                                    )
                                else:
                                    logger.warning(
                                        "[AgentTools] ⚠️ Chauffeur préféré #%s non trouvé ou n'appartient pas à la company %s",
                                        preferred_driver_id,
                                        self.company_id
                                    )
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    "[AgentTools] ⚠️ preferred_driver_id invalide: %s (erreur: %s)",
                                    preferred_driver_id,
                                    e
                                )
                        else:
                            logger.info("[AgentTools] ℹ️ preferred_driver_id est None/null - équité stricte sera appliquée")
                    else:
                        logger.info("[AgentTools] ℹ️ Aucun preferred_driver_id configuré dans dispatch_overrides - équité stricte sera appliquée")
                    
                    # Récupérer driver_load_multipliers depuis dispatch_overrides
                    if "driver_load_multipliers" in dispatch_overrides:
                        driver_load_multipliers = dispatch_overrides["driver_load_multipliers"]
                        if driver_load_multipliers:
                            # S'assurer que les clés sont des entiers et les valeurs des floats
                            try:
                                if isinstance(driver_load_multipliers, dict):
                                    normalized_multipliers = {
                                        int(k): float(v) for k, v in driver_load_multipliers.items()
                                    }
                                    final_overrides["driver_load_multipliers"] = normalized_multipliers
                                    logger.info(
                                        "[AgentTools] ⚖️ Multiplicateurs de charge DÉTECTÉS et ACTIVÉS: %s",
                                        normalized_multipliers
                                    )
                                else:
                                    logger.warning(
                                        "[AgentTools] ⚠️ driver_load_multipliers n'est pas un dict: %s (type: %s)",
                                        driver_load_multipliers,
                                        type(driver_load_multipliers).__name__
                                    )
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    "[AgentTools] ⚠️ Erreur normalisation driver_load_multipliers: %s (erreur: %s)",
                                    driver_load_multipliers,
                                    e
                                )
                        else:
                            logger.debug("[AgentTools] ℹ️ driver_load_multipliers est vide/None")
                    else:
                        logger.debug("[AgentTools] ℹ️ Aucun driver_load_multipliers dans dispatch_overrides")
                
                # ⚠️ IMPORTANT: Ne PAS utiliser reset_existing=True pour éviter de réassigner toutes les courses
                # L'agent doit seulement assigner les courses non assignées
                if "reset_existing" in final_overrides:
                    logger.warning(
                        "[AgentTools] ⚠️ reset_existing ignoré dans overrides (agent ne doit pas réassigner toutes les courses)"
                    )
                    final_overrides = {k: v for k, v in final_overrides.items() if k != "reset_existing"}
                
                # 🚨 IMPORTANT: Les chauffeurs d'urgence ne sont utilisés qu'en dernier recours
                # Le système engine.run() gère déjà les deux passes automatiquement :
                # - Pass 1: Chauffeurs réguliers uniquement (si regular_first=True)
                # - Pass 2: Ajout des chauffeurs d'urgence pour les courses non assignées (si allow_emergency=True)
                # 
                # ⚠️ CRITIQUE: Exclure les courses déjà assignées aux réguliers pour éviter les réassignations inutiles
                # On filtre les bookings avant de lancer le dispatch
                from sqlalchemy.orm import joinedload

                from models import Assignment, AssignmentStatus
                
                # Récupérer les bookings pour la date
                if for_date:
                    from services.unified_dispatch.data import get_bookings_for_day
                    all_bookings = get_bookings_for_day(self.company_id, for_date)
                else:
                    from services.unified_dispatch.data import get_bookings_for_dispatch
                    all_bookings = get_bookings_for_dispatch(self.company_id, 1440)  # 24h
                
                # Récupérer les assignments existants pour ces bookings
                booking_ids = [b.id for b in all_bookings]
                existing_assignments = {}
                if booking_ids:
                    assignments = (
                        Assignment.query
                        .filter(Assignment.booking_id.in_(booking_ids))
                        .filter(
                            Assignment.status.in_([
                                AssignmentStatus.SCHEDULED,
                                AssignmentStatus.EN_ROUTE_PICKUP,
                                AssignmentStatus.ARRIVED_PICKUP,
                                AssignmentStatus.ONBOARD,
                                AssignmentStatus.EN_ROUTE_DROPOFF,
                            ])
                        )
                        .options(joinedload(Assignment.driver))
                        .all()
                    )
                    
                    for assignment in assignments:
                        existing_assignments[assignment.booking_id] = assignment
                
                # Identifier les courses déjà assignées aux réguliers (à ne PAS réassigner)
                # et les courses non assignées ou assignées aux urgences (à réassigner si nécessaire)
                # ⚡ EXCEPTION: Si force_reassign=True, on inclut TOUTES les courses pour réassignation
                bookings_to_dispatch = []
                already_assigned_to_regular = []
                
                for booking in all_bookings:
                    assignment = existing_assignments.get(booking.id)
                    if assignment:
                        driver = assignment.driver
                        # Vérifier si le driver est un régulier (pas un urgent)
                        # Utiliser driver_type au lieu de is_emergency
                        is_emergency_driver = False
                        if driver:
                            driver_type = getattr(driver, "driver_type", None)
                            # Normaliser le type (support Enum et string)
                            if driver_type:
                                driver_type_str = str(driver_type).strip().upper()
                                if "." in driver_type_str:
                                    driver_type_str = driver_type_str.split(".")[-1]
                                is_emergency_driver = (driver_type_str == "EMERGENCY")
                        
                        # ⚡ Si force_reassign=True, on inclut TOUTES les courses pour réassignation
                        if force_reassign:
                            logger.info(
                                "[AgentTools] 🔄 Booking %s inclus pour réassignation (force_reassign=True, actuellement assigné à %s)",
                                booking.id,
                                driver.id if driver else "unknown"
                            )
                            bookings_to_dispatch.append(booking)
                            continue
                        
                        # Si assignée à un régulier, ne pas inclure dans le dispatch (sauf si force_reassign)
                        if driver and not is_emergency_driver:
                            already_assigned_to_regular.append(booking.id)
                            logger.debug(
                                "[AgentTools] ⏭️ Booking %s déjà assigné au régulier %s (type: %s), exclu du dispatch",
                                booking.id,
                                driver.id,
                                getattr(driver, "driver_type", "UNKNOWN")
                            )
                            continue
                        # Si assignée à un urgent, on peut la réassigner si nécessaire
                        logger.debug(
                            "[AgentTools] 🔄 Booking %s assigné à l'urgent %s (type: %s), inclus pour réassignation possible",
                            booking.id,
                            driver.id if driver else "unknown",
                            getattr(driver, "driver_type", "UNKNOWN") if driver else "UNKNOWN"
                        )
                    bookings_to_dispatch.append(booking)
                
                logger.info(
                    "[AgentTools] 📋 Dispatch: %d bookings à traiter (%d déjà assignés aux réguliers exclus)",
                    len(bookings_to_dispatch),
                    len(already_assigned_to_regular)
                )
                
                # Si aucune course à traiter, retourner vide
                if not bookings_to_dispatch:
                    logger.info("[AgentTools] ✅ Toutes les courses sont déjà assignées aux réguliers, aucun dispatch nécessaire")
                    return {
                        "plan": [],
                        "gains": {},
                    }
                
                # S'assurer que regular_first est activé pour prioriser les réguliers
                final_overrides = {
                    **final_overrides,
                    "regular_first": True,  # Toujours prioriser les réguliers
                    "allow_emergency": True,  # Autoriser les urgences en dernier recours
                    # ⚠️ IMPORTANT: Exclure les bookings déjà assignés aux réguliers (sauf si force_reassign)
                    # Si force_reassign=True, on veut réassigner même les bookings déjà assignés
                    "exclude_booking_ids": already_assigned_to_regular if not force_reassign else [],
                }
                
                if strategy == "degraded_proximity":
                    # Mode dégradé : désactiver OSRM, utiliser heuristiques seulement
                    final_overrides = {
                        **final_overrides,
                        "features": {
                            **final_overrides.get("features", {}),
                            "enable_solver": False,
                            "enable_heuristics": True,
                        },
                        "matrix": {
                            **final_overrides.get("matrix", {}),
                            "use_osrm": False,
                        },
                    }

                # Appeler engine.run() avec regular_first=True et allow_emergency=True
                # Le système gère automatiquement les deux passes :
                # - Pass 1: Réguliers uniquement
                # - Pass 2: Urgences pour les courses non assignées (seulement celles vraiment non assignées)
                result = dispatch_run(
                    company_id=self.company_id,
                    mode="auto",
                    regular_first=True,  # Prioriser les réguliers
                    allow_emergency=True,  # Autoriser urgences en dernier recours
                    overrides=final_overrides,
                    for_date=for_date,
                )

                # Construire plan depuis assignments
                assignments_raw = result.get("assignments", [])
                logger.info(
                    "[AgentTools] Reoptimize retourné %d assignments (type: %s)",
                    len(assignments_raw),
                    type(assignments_raw).__name__ if assignments_raw else "None",
                )
                
                # ✅ FILTRER les assignations pour exclure celles déjà assignées aux réguliers
                # (double vérification de sécurité même si exclude_booking_ids est appliqué)
                from models import Assignment, AssignmentStatus
                plan = []
                for assignment in assignments_raw:
                    # Essayer différents formats d'assignments
                    booking_id = (
                        assignment.get("booking_id")
                        or assignment.get("booking", {}).get("id")
                        or (assignment.get("booking") if isinstance(assignment.get("booking"), int) else None)
                    )
                    driver_id = (
                        assignment.get("driver_id")
                        or assignment.get("driver", {}).get("id")
                        or (assignment.get("driver") if isinstance(assignment.get("driver"), int) else None)
                    )
                    
                    logger.debug(
                        "[AgentTools] Assignment: booking_id=%s, driver_id=%s, assignment=%s",
                        booking_id,
                        driver_id,
                        type(assignment).__name__ if assignment else "None",
                    )

                    if not booking_id or not driver_id:
                        logger.warning(
                            "[AgentTools] Assignment ignoré: booking_id=%s, driver_id=%s",
                            booking_id,
                            driver_id,
                        )
                        continue
                    
                    # ✅ Vérifier si cette course est déjà assignée à un régulier
                    # (ne pas réassigner même si engine.run() l'a retournée)
                    existing_assignment = (
                        Assignment.query.filter_by(booking_id=int(booking_id))
                        .filter(
                            Assignment.status.in_([
                                AssignmentStatus.SCHEDULED,
                                AssignmentStatus.EN_ROUTE_PICKUP,
                                AssignmentStatus.ARRIVED_PICKUP,
                                AssignmentStatus.ONBOARD,
                                AssignmentStatus.EN_ROUTE_DROPOFF,
                            ])
                        )
                        .first()
                    )
                    
                    if existing_assignment:
                        # Vérifier si c'est un régulier (pas un urgent)
                        driver = existing_assignment.driver
                        is_emergency_driver = False
                        if driver:
                            driver_type = getattr(driver, "driver_type", None)
                            if driver_type:
                                driver_type_str = str(driver_type).strip().upper()
                                if "." in driver_type_str:
                                    driver_type_str = driver_type_str.split(".")[-1]
                                is_emergency_driver = (driver_type_str == "EMERGENCY")
                        
                        # Si déjà assignée à un régulier, ne pas inclure dans le plan
                        if driver and not is_emergency_driver:
                            logger.debug(
                                "[AgentTools] ⏭️ Booking %s déjà assigné au régulier %s, exclu du plan",
                                booking_id,
                                driver.id,
                            )
                            continue
                        # Si assignée à un urgent, on peut la réassigner
                        logger.debug(
                            "[AgentTools] 🔄 Booking %s assigné à l'urgent %s, inclus pour réassignation",
                            booking_id,
                            driver.id if driver else "unknown",
                        )
                    
                    plan.append(
                        {
                            "job_id": int(booking_id),
                            "driver_id": int(driver_id),
                            "reasoning_brief": f"Reoptimize {strategy} - ETA: {assignment.get('eta_minutes', 'N/A')} min",
                        }
                    )

                logger.info(
                    "[AgentTools] Plan construit: %d étapes (assignments reçus: %d)",
                    len(plan),
                    len(assignments_raw),
                )

                return {
                    "plan": plan,
                    "gains": {
                        "total_gain_minutes": result.get("meta", {}).get(
                            "total_gain_minutes", 0
                        ),
                        "fairness_improved": True,
                    },
                }
            except Exception as e:
                logger.exception("[AgentTools] Error in reoptimize: %s", e)
                return {"plan": [], "gains": {}, "error": str(e)}

    def notify(
        self,
        channel: str,
        to: str,
        template_id: str,
        vars: Dict[str, Any],  # noqa: A002
    ) -> Dict[str, Any]:
        """Envoie une notification.

        Args:
            channel: Canal de notification ('email', 'slack', 'sms')
            to: Destinataire (email, slack channel, phone)
            template_id: ID du template de notification
            vars: Variables pour le template

        Returns:
            {"ok": bool, "id": str}

        """
        try:
            # Intégrer avec système notifications existant
            # Pour l'instant, log seulement
            logger.info(
                "[AgentTools] 📧 Notification: %s → %s (%s) vars_keys=%s",
                channel,
                to,
                template_id,
                list(vars.keys()) if vars else [],
            )

            # TODO: Intégrer avec services/notification_service.py
            # from services.notification_service import send_notification
            # send_notification(channel, to, template_id, vars)

            return {"ok": True, "id": f"notif_{datetime.now().isoformat()}"}
        except Exception as e:
            logger.exception("[AgentTools] Error in notify: %s", e)
            return {"ok": False, "error": str(e)}

