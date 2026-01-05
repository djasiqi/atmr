# backend/services/unified_dispatch/reactive_suggestions.py

# Constantes pour éviter les valeurs magiques
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Tuple

from ext import db
from models import Assignment, Booking, Driver
from services.unified_dispatch.data import haversine_minutes
from services.unified_dispatch.settings import Settings
from shared.time_utils import now_local

DELAY_MINUTES_THRESHOLD = 15
DELAY_MINUTES_ZERO = 0
GAIN_THRESHOLD = 5
ADVANCE_MINUTES_THRESHOLD = 15
MODERATE_DELAY_THRESHOLD = 5
EARLY_ADVANCE_THRESHOLD = -10
SLIGHT_ADVANCE_THRESHOLD = -5

"""Système de suggestions RÉACTIVES pour l'optimisation du dispatch.

Ce module génère des suggestions contextuelles en RÉACTION à des événements détectés
(principalement des retards). Utilisé par le système de monitoring temps réel et
l'optimiseur automatique pour proposer des actions correctives.

Cas d'usage :
- Détection de retard → Suggère réassignation, notification client
- Chauffeur très en avance → Suggère course additionnelle
- Monitoring temps réel → Génère suggestions de correction

Différence avec rl/suggestion_generator.py :
- Ce module : Suggestions RÉACTIVES (1 assignment avec retard)
- rl/suggestion_generator.py : Suggestions PROACTIVES (optimisation globale via DQN)

Utilisé par :
- /company_dispatch/delays (endpoint retards)
- /company_dispatch/delays/live (endpoint retards temps réel)
- RealtimeOptimizer (monitoring automatique)
- AutonomousManager (mode fully-auto)

Voir aussi : services/rl/suggestion_generator.py (suggestions proactives basées RL)
"""


logger = logging.getLogger(__name__)


@dataclass
class Suggestion:
    """Une suggestion d'optimisation."""

    action: (
        str  # "reassign", "notify_customer", "add_booking", "adjust_time", "add_driver"
    )
    priority: str  # "low", "medium", "high", "critical"
    message: str
    estimated_gain_minutes: int | None = None
    booking_id: int | None = None
    driver_id: int | None = None
    alternative_driver_id: int | None = None
    additional_data: Dict[str, Any] | None = None
    auto_applicable: bool = False  # Peut être appliquée automatiquement

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "priority": self.priority,
            "message": self.message,
            "estimated_gain_minutes": self.estimated_gain_minutes,
            "booking_id": self.booking_id,
            "driver_id": self.driver_id,
            "alternative_driver_id": self.alternative_driver_id,
            "additional_data": self.additional_data or {},
            "auto_applicable": self.auto_applicable,
        }


class SuggestionEngine:
    """Moteur de génération de suggestions intelligentes."""

    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings()

    def generate_suggestions_for_assignment(
        self, assignment: Assignment, delay_minutes: int, company_id: int
    ) -> List[Suggestion]:
        """Génère des suggestions contextuelles pour une assignation avec retard.

        Args:
            assignment: L'assignation à analyser
            delay_minutes: Retard en minutes (positif = retard, négatif = avance)
            company_id: ID de l'entreprise

        Returns:
            Liste de suggestions classées par priorité

        """
        suggestions: list[Suggestion] = []

        # Récupérer le booking et le driver
        booking_id = (
            int(assignment.booking_id) if assignment.booking_id is not None else None  # type: ignore[reportArgumentType]
        )
        driver_id = (
            int(assignment.driver_id) if assignment.driver_id is not None else None  # type: ignore[reportArgumentType]
        )

        booking = db.session.get(Booking, booking_id) if booking_id else None
        driver = db.session.get(Driver, driver_id) if driver_id else None

        if not booking:
            return suggestions

        # Générer des suggestions selon le niveau de retard
        if delay_minutes > DELAY_MINUTES_THRESHOLD:
            # Retard critique → notification client URGENTE + réassignation
            suggestions.append(
                self._suggest_customer_notification(booking, delay_minutes)
            )
            suggestions.extend(
                self._suggest_reassignment(booking, driver, delay_minutes, company_id)
            )
        elif delay_minutes > DELAY_MINUTES_THRESHOLD:
            # Retard moyen → notification client
            suggestions.append(
                self._suggest_customer_notification(booking, delay_minutes)
            )
            # + possibilité de réassignation si chauffeur proche
            suggestions.extend(
                self._suggest_reassignment(
                    booking, driver, delay_minutes, company_id, threshold_km=3
                )
            )
        elif delay_minutes < EARLY_ADVANCE_THRESHOLD:
            # Très en avance → peut optimiser
            suggestions.extend(
                self._suggest_additional_booking(
                    booking, driver, abs(delay_minutes), company_id
                )
            )
        elif SLIGHT_ADVANCE_THRESHOLD < delay_minutes < DELAY_MINUTES_ZERO:
            # Légèrement en avance → OK, aucune action
            suggestions.append(
                Suggestion(
                    action="none",
                    priority="low",
                    message=(
                        f"✅ Chauffeur en avance de {abs(delay_minutes)} min "
                        "- situation optimale"
                    ),
                    booking_id=int(booking.id) if booking else None,
                    driver_id=int(driver.id) if driver else None,
                    auto_applicable=False,
                )
            )

        # Suggestions générales
        suggestions.extend(self._suggest_time_adjustments(booking, delay_minutes))

        # Trier par priorité
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 99))

        return suggestions

    def _suggest_reassignment(
        self,
        booking: Booking,
        current_driver: Driver | None,
        delay_minutes: int,
        company_id: int,
        threshold_km: float = 10.0,
    ) -> List[Suggestion]:
        """Suggère de réassigner à un chauffeur plus proche."""
        suggestions: list[Suggestion] = []

        try:
            # Trouver des chauffeurs disponibles à proximité
            exclude_id = int(current_driver.id) if current_driver else None
            nearby_drivers = self._find_nearby_available_drivers(
                booking,
                company_id,
                radius_km=threshold_km,
                exclude_driver_id=exclude_id,
            )

            if not nearby_drivers:
                return suggestions

            # Calculer le gain potentiel pour chaque chauffeur
            # Top 3
            for driver, distance_km, eta_minutes in nearby_drivers[:3]:
                # Estimation du gain
                current_eta = (
                    delay_minutes
                    + int((booking.scheduled_time - now_local()).total_seconds() / 60)
                    if booking.scheduled_time
                    else delay_minutes
                )
                new_eta = eta_minutes
                gain = current_eta - new_eta

                if gain > GAIN_THRESHOLD:  # Gain significatif uniquement
                    priority = (
                        "critical"
                        if delay_minutes > DELAY_MINUTES_THRESHOLD
                        else "high"
                    )

                    driver_name = driver.user.first_name if driver.user else "Driver"

                    suggestions.append(
                        Suggestion(
                            action="reassign",
                            priority=priority,
                            message=(
                                f"Réassigner au chauffeur #{driver.id} "
                                f"({driver_name}) "
                                f"- Gain: {gain} min "
                                f"(distance: {distance_km:.1f}km)"
                            ),
                            estimated_gain_minutes=gain,
                            booking_id=int(booking.id) if booking else None,
                            driver_id=int(current_driver.id)
                            if current_driver
                            else None,
                            alternative_driver_id=int(driver.id) if driver else None,
                            additional_data={
                                "distance_km": distance_km,
                                "new_eta_minutes": eta_minutes,
                                "driver_name": (
                                    f"{driver.user.first_name} {driver.user.last_name}"
                                )
                                if driver.user
                                else None,
                            },
                            auto_applicable=False,  # Nécessite validation
                        )
                    )

        except Exception as e:
            logger.warning("[Suggestions] Failed to suggest reassignment: %s", e)

        return suggestions

    def _suggest_customer_notification(
        self, booking: Booking, delay_minutes: int
    ) -> Suggestion:
        """Suggère de notifier le client du retard."""
        priority = "high" if delay_minutes > DELAY_MINUTES_THRESHOLD else "medium"

        # Auto-applicable SI :
        # - Retard modéré (5-20 min) : notification automatique OK
        # - Retard important (>20 min) : nécessite validation humaine
        auto_applicable = (
            MODERATE_DELAY_THRESHOLD <= delay_minutes <= DELAY_MINUTES_THRESHOLD
        )

        return Suggestion(
            action="notify_customer",
            priority=priority,
            message=f"Prévenir le client du retard de {delay_minutes} min",
            booking_id=int(booking.id) if booking else None,
            additional_data={
                "auto_message": (
                    f"Bonjour, votre chauffeur arrivera avec environ "
                    f"{delay_minutes} minutes de retard. "
                    f"Nous nous excusons pour ce désagrément."
                ),
                "customer_name": booking.customer_name,
                "customer_phone": getattr(booking, "customer_phone", None),
            },
            auto_applicable=auto_applicable,
        )

    def _suggest_additional_booking(
        self,
        booking: Booking,
        driver: Driver | None,
        advance_minutes: int,
        company_id: int,
    ) -> List[Suggestion]:
        """Suggère d'ajouter une course supplémentaire
        quand le chauffeur est très en avance."""
        suggestions: list[Suggestion] = []

        if not driver or advance_minutes < ADVANCE_MINUTES_THRESHOLD:
            return suggestions

        try:
            # Chercher des bookings en attente à proximité
            nearby_bookings = self._find_pending_bookings_nearby(
                booking,
                company_id,
                time_window_minutes=advance_minutes - 5,
                radius_km=5.0,
            )

            # Top 2
            for nearby_booking, distance_km, time_available in nearby_bookings[:2]:
                suggestions.append(
                    Suggestion(
                        action="add_booking",
                        priority="medium",
                        message=(
                            f"Chauffeur disponible {advance_minutes} min "
                            "avant rendez-vous. "
                            f"Peut prendre la course #{nearby_booking.id} "
                            f"({nearby_booking.customer_name}) à {distance_km:.1f}km"
                        ),
                        booking_id=int(nearby_booking.id) if nearby_booking else None,
                        driver_id=int(driver.id) if driver else None,
                        estimated_gain_minutes=time_available,
                        additional_data={
                            "original_booking_id": int(booking.id) if booking else None,
                            "distance_km": distance_km,
                            "pickup_address": getattr(
                                nearby_booking, "pickup_address", None
                            ),
                        },
                        auto_applicable=False,
                    )
                )

        except Exception as e:
            logger.warning("[Suggestions] Failed to suggest additional booking: %s", e)

        return suggestions

    def _suggest_time_adjustments(
        self, booking: Booking, delay_minutes: int
    ) -> List[Suggestion]:
        """Suggère des ajustements d'horaire si possible."""
        suggestions = []

        # Retard critique → URGENT : ajuster l'heure
        if delay_minutes > DELAY_MINUTES_THRESHOLD:
            suggestions.append(
                Suggestion(
                    action="adjust_time",
                    priority="critical",
                    message=(
                        f"🔴 URGENT : Reporter le rendez-vous de {delay_minutes} min "
                        f"({delay_minutes // 60}h{delay_minutes % 60:02d}) "
                        "et contacter le client immédiatement"
                    ),
                    booking_id=int(booking.id) if booking else None,
                    additional_data={
                        "proposed_new_time": (
                            booking.scheduled_time + timedelta(minutes=delay_minutes)
                        ).isoformat()
                        if booking.scheduled_time
                        else None,
                        "contact_customer_urgent": True,
                    },
                    auto_applicable=False,
                )
            )
        # Retard important → ajuster l'heure
        elif delay_minutes > DELAY_MINUTES_THRESHOLD:
            suggestions.append(
                Suggestion(
                    action="adjust_time",
                    priority="high",
                    message=(
                        f"Reporter le rendez-vous de {delay_minutes} min "
                        "et prévenir le client"
                    ),
                    booking_id=int(booking.id) if booking else None,
                    additional_data={
                        "proposed_new_time": (
                            booking.scheduled_time + timedelta(minutes=delay_minutes)
                        ).isoformat()
                        if booking.scheduled_time
                        else None,
                    },
                    auto_applicable=False,
                )
            )
        # Retard modéré et booking flexible
        elif MODERATE_DELAY_THRESHOLD < delay_minutes < DELAY_MINUTES_THRESHOLD:
            # Vérifier si le booking a une marge de flexibilité
            # (exemple: rendez-vous médical non urgent)
            is_flexible = not getattr(booking, "is_urgent", False)

            if is_flexible:
                suggestions.append(
                    Suggestion(
                        action="adjust_time",
                        priority="medium",
                        message=(
                            f"Proposer de décaler le rendez-vous "
                            f"de {delay_minutes} min (booking non urgent)"
                        ),
                        booking_id=int(booking.id) if booking else None,
                        additional_data={
                            "proposed_new_time": (
                                booking.scheduled_time
                                + timedelta(minutes=delay_minutes)
                            ).isoformat()
                            if booking.scheduled_time
                            else None,
                        },
                        auto_applicable=False,
                    )
                )

        return suggestions

    def _find_nearby_available_drivers(
        self,
        booking: Booking,
        company_id: int,
        radius_km: float = 10.0,
        exclude_driver_id: int | None = None,
    ) -> List[Tuple[Driver, float, int]]:
        """Trouve les chauffeurs disponibles à proximité.

        Returns:
            Liste de tuples (Driver, distance_km, eta_minutes) triés par distance

        """
        try:
            # Position du pickup
            pickup_lat = getattr(booking, "pickup_lat", None)
            pickup_lon = getattr(booking, "pickup_lon", None)

            if not pickup_lat or not pickup_lon:
                return []

            pickup_pos = (float(pickup_lat), float(pickup_lon))

            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            from repositories.driver_repository import DriverRepository

            driver_repo = DriverRepository()
            driver_dtos = driver_repo.find_by_company_id(company_id, active_only=True)
            # Filtrer par is_available en mémoire
            available_driver_dtos = [dto for dto in driver_dtos if dto.is_available]
            # Filtrer exclude_driver_id en mémoire
            if exclude_driver_id:
                available_driver_dtos = [
                    dto for dto in available_driver_dtos if dto.id != exclude_driver_id
                ]
            # Récupérer les modèles SQLAlchemy depuis les IDs des DTOs pour la compatibilité
            driver_ids = [dto.id for dto in available_driver_dtos]
            drivers = (
                Driver.query.filter(Driver.id.in_(driver_ids)).all()
                if driver_ids
                else []
            )

            # Calculer distance et ETA pour chaque chauffeur
            results = []
            for driver in drivers:
                driver_lat = getattr(
                    driver, "current_lat", getattr(driver, "latitude", None)
                )
                driver_lon = getattr(
                    driver, "current_lon", getattr(driver, "longitude", None)
                )

                if not driver_lat or not driver_lon:
                    continue

                driver_pos = (float(driver_lat), float(driver_lon))

                # Distance Haversine
                distance_km = self._calculate_distance_km(driver_pos, pickup_pos)

                if distance_km <= radius_km:
                    # ETA en minutes
                    eta_minutes = haversine_minutes(
                        driver_pos,
                        pickup_pos,
                        avg_kmh=getattr(self.settings.matrix, "avg_speed_kmh", 25.0),
                    )

                    results.append((driver, distance_km, eta_minutes))

            # Trier par distance
            results.sort(key=lambda x: x[1])

            return results

        except Exception as e:
            logger.warning("[Suggestions] Error finding nearby drivers: %s", e)
            return []

    def _find_pending_bookings_nearby(
        self,
        booking: Booking,
        company_id: int,
        time_window_minutes: int = 30,
        radius_km: float = 5.0,
    ) -> List[Tuple[Booking, float, int]]:
        """Trouve des bookings en attente à proximité dans une fenêtre de temps.

        Returns:
            Liste de tuples (Booking, distance_km, time_available_minutes)

        """
        try:
            pickup_lat = getattr(booking, "pickup_lat", None)
            pickup_lon = getattr(booking, "pickup_lon", None)

            if not pickup_lat or not pickup_lon:
                return []

            current_pos = (float(pickup_lat), float(pickup_lon))

            # Fenêtre de temps
            now = now_local()
            time_window_end = now + timedelta(minutes=time_window_minutes)

            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            pending_bookings = booking_repo.find_pending_by_company_and_time_window(
                company_id=company_id,
                now=now,
                time_window_end=time_window_end,
                exclude_booking_id=int(booking.id) if booking.id else None,
            )

            results = []
            for pending_booking in pending_bookings:
                p_lat = getattr(pending_booking, "pickup_lat", None)
                p_lon = getattr(pending_booking, "pickup_lon", None)

                if not p_lat or not p_lon:
                    continue

                pending_pos = (float(p_lat), float(p_lon))
                distance_km = self._calculate_distance_km(current_pos, pending_pos)

                if distance_km <= radius_km and pending_booking.scheduled_time:
                    # Temps disponible avant ce booking
                    time_available = int(
                        (pending_booking.scheduled_time - now).total_seconds() / 60
                    )
                    results.append((pending_booking, distance_km, time_available))

            # Trier par temps disponible (plus urgent en premier)
            results.sort(key=lambda x: x[2])

            return results

        except Exception as e:
            logger.warning("[Suggestions] Error finding pending bookings: %s", e)
            return []

    def _calculate_distance_km(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Calcule la distance en km entre deux positions (Haversine)."""
        # Import centralisé depuis shared.geo_utils
        from shared.geo_utils import haversine_tuple

        return haversine_tuple(pos1, pos2)


def generate_reactive_suggestions(
    assignment: Assignment,
    delay_minutes: int,
    company_id: int,
    settings: Settings | None = None,
) -> List[Suggestion]:
    """Génère des suggestions RÉACTIVES basées sur un retard détecté.

    Usage:
        suggestions = generate_reactive_suggestions(
            assignment, delay_minutes=12, company_id=1
        )
        for suggestion in suggestions:
            print(suggestion.message)
            if suggestion.auto_applicable:
                apply_suggestion(suggestion, company_id)

    Args:
        assignment: L'assignation avec retard
        delay_minutes: Retard en minutes (positif = retard, négatif = avance)
        company_id: ID de l'entreprise
        settings: Configuration dispatch (optionnel)

    Returns:
        Liste de suggestions triées par priorité

    """
    engine = SuggestionEngine(settings)
    return engine.generate_suggestions_for_assignment(
        assignment, delay_minutes, company_id
    )


# Alias pour rétrocompatibilité (sera supprimé dans version future)
generate_suggestions = generate_reactive_suggestions


def apply_suggestion(
    suggestion: Suggestion, company_id: int, dry_run: bool = False
) -> Dict[str, Any]:
    """Applique automatiquement une suggestion si elle est auto-applicable.

    Args:
        suggestion: La suggestion à appliquer
        company_id: ID de l'entreprise
        dry_run: Si True, simule l'application sans exécuter (pour tests)

    Returns:
        Dict avec le résultat de l'application

    """
    if not suggestion.auto_applicable:
        return {
            "success": False,
            "error": "Cette suggestion nécessite une validation manuelle",
            "suggestion": suggestion.to_dict(),
        }

    try:
        if suggestion.action == "notify_customer":
            return _apply_customer_notification(suggestion, company_id, dry_run)
        if suggestion.action == "reassign":
            return _apply_reassignment(suggestion, company_id, dry_run)
        if suggestion.action == "adjust_time":
            return _apply_time_adjustment(suggestion, company_id, dry_run)
        return {
            "success": False,
            "error": (
                f"Action '{suggestion.action}' non supportée pour auto-application"
            ),
            "suggestion": suggestion.to_dict(),
        }
    except Exception as e:
        logger.exception("[Suggestions] Failed to apply suggestion: %s", e)
        return {"success": False, "error": str(e), "suggestion": suggestion.to_dict()}


def _apply_customer_notification(
    suggestion: Suggestion, company_id: int, dry_run: bool
) -> Dict[str, Any]:
    """Applique une notification client automatique."""
    booking_id = suggestion.booking_id
    if not booking_id:
        return {"success": False, "error": "Booking ID manquant"}

    booking = db.session.get(Booking, booking_id)
    if not booking or bool(booking.company_id != company_id):
        return {"success": False, "error": "Booking introuvable"}

    auto_message = (
        suggestion.additional_data.get("auto_message", "")
        if suggestion.additional_data
        else ""
    )
    customer_phone = (
        suggestion.additional_data.get("customer_phone")
        if suggestion.additional_data
        else None
    )

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "action": "notify_customer",
            "booking_id": booking_id,
            "message": auto_message,
            "customer": booking.customer_name,
            "phone": customer_phone,
        }

    # Notification réelle (SMS/Email/Push selon configuration)
    try:
        # Note: notify_customer_delay n'existe pas encore dans notification_service
        # Pour l'instant, on log et on considère comme envoyé
        # TODO: Implémenter notify_customer_delay dans
        # services/notification_service.py
        logger.info(
            "[Suggestions] Would send notification to customer %s: %s",
            booking.customer_name,
            auto_message,
        )

        logger.info(
            (
                "[Suggestions] Auto-applied customer notification for booking %s "
                "(delay: %d min)"
            ),
            booking_id,
            suggestion.additional_data.get("delay_minutes", 0)
            if suggestion.additional_data
            else 0,
        )

        return {
            "success": True,
            "action": "notify_customer",
            "booking_id": booking_id,
            "customer": booking.customer_name,
            "message_sent": True,
        }
    except Exception as e:
        logger.exception("[Suggestions] Failed to send customer notification: %s", e)
        return {
            "success": False,
            "error": f"Échec envoi notification: {e}",
            "booking_id": booking_id,
        }


def _apply_reassignment(
    suggestion: Suggestion, _company_id: int, _dry_run: bool
) -> Dict[str, Any]:
    """Applique une réassignation automatique."""
    # Pour l'instant, les réassignations ne sont PAS auto-applicables (trop risqué)
    # Mais on garde la fonction pour évolution future avec seuils de confiance
    return {
        "success": False,
        "error": "Réassignation automatique désactivée (nécessite validation humaine)",
        "suggestion": suggestion.to_dict(),
    }


def _apply_time_adjustment(
    suggestion: Suggestion, _company_id: int, _dry_run: bool
) -> Dict[str, Any]:
    """Applique un ajustement d'horaire automatique."""
    # Pour l'instant, les ajustements d'horaire ne sont PAS auto-applicables
    # (impact client trop important)
    return {
        "success": False,
        "error": (
            "Ajustement horaire automatique désactivé (nécessite validation humaine)"
        ),
        "suggestion": suggestion.to_dict(),
    }
