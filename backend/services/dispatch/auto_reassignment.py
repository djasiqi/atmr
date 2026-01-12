# backend/services/auto_reassignment_service.py

"""✅ 3.4.1: Service de réassignation automatique si retard détecté.

Ce service centralise la logique de réassignation automatique :
- Détection de retard projeté
- Recherche meilleur chauffeur alternatif
- Notification chauffeur actuel (Socket.IO)
- Réassignation avec confirmation ou timeout
- Émission d'events Socket.IO
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from models import Assignment, AssignmentStatus, Booking, Driver
from repositories.assignment_repository import AssignmentRepository
from repositories.booking_repository import BookingRepository
from repositories.driver_repository import DriverRepository
from services.unified_dispatch.optimization.assignment_applier import (
    apply_assignments,
)

# ✅ FIX: Import lazy de calculate_eta pour éviter import circulaire
# from services.unified_dispatch.data import calculate_eta  # ❌ Cause import circulaire
from services.unified_dispatch.utils.suggestions import SuggestionEngine
from shared.time_utils import now_local

logger = logging.getLogger(__name__)

# Constantes
DEFAULT_DELAY_THRESHOLD_MINUTES = int(
    os.getenv("AUTO_REASSIGNMENT_DELAY_THRESHOLD_MINUTES", "10")
)
DEFAULT_CONFIRMATION_TIMEOUT_SECONDS = int(
    os.getenv("AUTO_REASSIGNMENT_CONFIRMATION_TIMEOUT_SECONDS", "30")
)
DEFAULT_MIN_GAIN_MINUTES = int(os.getenv("AUTO_REASSIGNMENT_MIN_GAIN_MINUTES", "5"))


@dataclass
class ReassignmentResult:
    """Résultat d'une tentative de réassignation automatique."""

    success: bool
    reassigned: bool  # True si réassignation effectuée
    assignment_id: int
    old_driver_id: int | None
    new_driver_id: int | None
    delay_minutes: int
    projected_delay_minutes: int
    gain_minutes: int | None  # Gain estimé avec nouveau chauffeur
    reason: str  # Raison de la réassignation ou échec
    requires_confirmation: bool = False  # Si True, attend confirmation chauffeur
    confirmation_timeout_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation."""
        return {
            "success": self.success,
            "reassigned": self.reassigned,
            "assignment_id": self.assignment_id,
            "old_driver_id": self.old_driver_id,
            "new_driver_id": self.new_driver_id,
            "delay_minutes": self.delay_minutes,
            "projected_delay_minutes": self.projected_delay_minutes,
            "gain_minutes": self.gain_minutes,
            "reason": self.reason,
            "requires_confirmation": self.requires_confirmation,
            "confirmation_timeout_at": (
                self.confirmation_timeout_at.isoformat()
                if self.confirmation_timeout_at
                else None
            ),
        }


class AutoReassignmentService:
    """Service de réassignation automatique si retard détecté."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        delay_threshold_minutes: int = DEFAULT_DELAY_THRESHOLD_MINUTES,
        confirmation_timeout_seconds: int = DEFAULT_CONFIRMATION_TIMEOUT_SECONDS,
        min_gain_minutes: int = DEFAULT_MIN_GAIN_MINUTES,
        enabled: bool = True,
    ):
        """Initialise le service de réassignation automatique.

        Args:
            delay_threshold_minutes: Seuil de retard (minutes) pour déclencher
                réassignation
            confirmation_timeout_seconds: Timeout pour confirmation chauffeur (secondes)
            min_gain_minutes: Gain minimum requis (minutes) pour réassigner
            enabled: Activer/désactiver le service (feature flag)
        """
        self.delay_threshold_minutes = delay_threshold_minutes
        self.confirmation_timeout_seconds = confirmation_timeout_seconds
        self.min_gain_minutes = min_gain_minutes
        self.enabled = enabled
        self.suggestion_engine = SuggestionEngine()

    def check_and_reassign(  # noqa: PLR0911
        self,
        assignment_id: int,
        delay_threshold_minutes: int | None = None,
        auto_apply: bool = False,
    ) -> ReassignmentResult | None:
        """Vérifie retard et réassigne si nécessaire.

        Args:
            assignment_id: ID de l'assignment à vérifier
            delay_threshold_minutes: Seuil de retard (override config)
            auto_apply: Si True, réassigne automatiquement sans confirmation

        Returns:
            ReassignmentResult si réassignation tentée, None sinon
        """
        if not self.enabled:
            return None

        threshold = delay_threshold_minutes or self.delay_threshold_minutes

        try:
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            assignment_repo = AssignmentRepository()
            assignment_dto = assignment_repo.find_by_id(assignment_id)
            if not assignment_dto:
                assignment = None
            else:
                # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
                assignment = Assignment.query.get(assignment_dto.id)
            if not assignment:
                return ReassignmentResult(
                    success=False,
                    reassigned=False,
                    assignment_id=assignment_id,
                    old_driver_id=None,
                    new_driver_id=None,
                    delay_minutes=0,
                    projected_delay_minutes=0,
                    gain_minutes=None,
                    reason="Assignment introuvable",
                )

            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            booking_repo = BookingRepository()
            booking_dto = booking_repo.find_by_id(assignment.booking_id)
            # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
            booking = Booking.query.get(booking_dto.id) if booking_dto else None
            if not booking:
                return ReassignmentResult(
                    success=False,
                    reassigned=False,
                    assignment_id=assignment_id,
                    old_driver_id=assignment.driver_id,
                    new_driver_id=None,
                    delay_minutes=0,
                    projected_delay_minutes=0,
                    gain_minutes=None,
                    reason="Booking introuvable",
                )

            # Vérifier que l'assignment est actif
            if assignment.status not in [
                AssignmentStatus.SCHEDULED,
                AssignmentStatus.EN_ROUTE_PICKUP,
            ]:
                return None  # Assignment déjà en cours ou terminé

            # Calculer retard projeté
            projected_delay = self._calculate_projected_delay(assignment, booking)

            if projected_delay < threshold:
                # Retard insuffisant pour réassigner
                return None

            # Trouver meilleur chauffeur alternatif
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            driver_repo = DriverRepository()
            current_driver = None
            if assignment.driver_id:
                driver_dto = driver_repo.find_by_id(assignment.driver_id)
                if driver_dto:
                    # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
                    current_driver = Driver.query.get(driver_dto.id)
            best_alternative = self._find_best_alternative_driver(
                booking, current_driver, booking.company_id
            )

            if not best_alternative:
                return ReassignmentResult(
                    success=False,
                    reassigned=False,
                    assignment_id=assignment_id,
                    old_driver_id=assignment.driver_id,
                    new_driver_id=None,
                    delay_minutes=projected_delay,
                    projected_delay_minutes=projected_delay,
                    gain_minutes=None,
                    reason="Aucun chauffeur alternatif disponible",
                )

            new_driver, _distance_km, new_eta_minutes = best_alternative

            # Calculer gain estimé
            current_eta_minutes = projected_delay + int(
                (booking.scheduled_time - now_local()).total_seconds() / 60
                if booking.scheduled_time
                else 0
            )
            gain_minutes = current_eta_minutes - new_eta_minutes

            if gain_minutes < self.min_gain_minutes:
                return ReassignmentResult(
                    success=False,
                    reassigned=False,
                    assignment_id=assignment_id,
                    old_driver_id=assignment.driver_id,
                    new_driver_id=new_driver.id,
                    delay_minutes=projected_delay,
                    projected_delay_minutes=projected_delay,
                    gain_minutes=gain_minutes,
                    reason=(
                        f"Gain insuffisant ({gain_minutes} min < "
                        f"{self.min_gain_minutes} min)"
                    ),
                )

            # Notifier chauffeur actuel si auto_apply=False
            if not auto_apply and current_driver:
                # Émettre event Socket.IO pour notification
                self._notify_driver_reassignment_pending(
                    assignment_id=assignment_id,
                    driver_id=current_driver.id,
                    new_driver_id=new_driver.id,
                    reason=f"Retard projeté: {projected_delay} min",
                    timeout_seconds=self.confirmation_timeout_seconds,
                )

                return ReassignmentResult(
                    success=True,
                    reassigned=False,
                    assignment_id=assignment_id,
                    old_driver_id=assignment.driver_id,
                    new_driver_id=new_driver.id,
                    delay_minutes=projected_delay,
                    projected_delay_minutes=projected_delay,
                    gain_minutes=gain_minutes,
                    reason="Notification envoyée, en attente de confirmation",
                    requires_confirmation=True,
                    confirmation_timeout_at=datetime.now(UTC)
                    + timedelta(seconds=self.confirmation_timeout_seconds),
                )

            # Réassigner automatiquement
            return self._perform_reassignment(
                assignment=assignment,
                booking=booking,
                new_driver_id=new_driver.id,
                gain_minutes=gain_minutes,
            )

        except Exception as e:
            logger.exception(
                "[AutoReassignment] Failed to check and reassign assignment %d: %s",
                assignment_id,
                e,
            )
            return ReassignmentResult(
                success=False,
                reassigned=False,
                assignment_id=assignment_id,
                old_driver_id=None,
                new_driver_id=None,
                delay_minutes=0,
                projected_delay_minutes=0,
                gain_minutes=None,
                reason=f"Erreur: {e!s}",
            )

    def _calculate_projected_delay(
        self, assignment: Assignment, booking: Booking
    ) -> int:
        """Calcule le retard projeté pour une assignation.

        Args:
            assignment: Assignment à analyser
            booking: Booking associé

        Returns:
            Retard projeté en minutes
        """
        if not booking.scheduled_time:
            return 0

        current_time = now_local()

        # Si ETA pickup disponible, l'utiliser
        if assignment.eta_pickup_at:
            delay_seconds = (
                assignment.eta_pickup_at - booking.scheduled_time
            ).total_seconds()
            return int(delay_seconds / 60)

        # Sinon, calculer ETA depuis position actuelle du chauffeur
        if bool(assignment.driver_id):
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            driver_repo = DriverRepository()
            driver_dto = (
                driver_repo.find_by_id(int(assignment.driver_id))  # type: ignore[reportArgumentType]
                if bool(assignment.driver_id)
                else None
            )
            driver = Driver.query.get(driver_dto.id) if driver_dto else None
            if (
                driver
                and bool(driver.latitude)
                and bool(driver.longitude)
                and bool(booking.pickup_lat)
                and bool(booking.pickup_lon)
            ):
                try:
                    # ✅ FIX: Import lazy pour éviter import circulaire
                    from services.unified_dispatch.data import calculate_eta

                    driver_position = (float(driver.latitude), float(driver.longitude))
                    pickup_position = (
                        float(booking.pickup_lat),  # type: ignore[reportArgumentType]
                        float(booking.pickup_lon),  # type: ignore[reportArgumentType]
                    )
                    eta_seconds = calculate_eta(
                        driver_position=driver_position,
                        destination=pickup_position,
                    )
                    arrival_time = current_time + timedelta(seconds=eta_seconds)
                    delay_seconds = (
                        arrival_time - booking.scheduled_time
                    ).total_seconds()
                    return int(delay_seconds / 60)
                except Exception as e:
                    logger.debug(
                        "[AutoReassignment] ETA calculation failed: %s", str(e)
                    )

        # Fallback: retard basé sur heure actuelle
        delay_seconds = (current_time - booking.scheduled_time).total_seconds()
        return int(delay_seconds / 60)

    def _find_best_alternative_driver(
        self,
        booking: Booking,
        current_driver: Driver | None,
        company_id: int,
        radius_km: float = 10.0,
    ) -> tuple[Driver, float, int] | None:
        """Trouve le meilleur chauffeur alternatif pour une booking.

        Args:
            booking: Booking à réassigner
            current_driver: Chauffeur actuel (à exclure)
            company_id: ID de l'entreprise
            radius_km: Rayon de recherche (km)

        Returns:
            Tuple (Driver, distance_km, eta_minutes) ou None
        """
        try:
            exclude_id = int(current_driver.id) if current_driver else None
            nearby_drivers = self.suggestion_engine._find_nearby_available_drivers(
                booking=booking,
                company_id=company_id,
                radius_km=radius_km,
                exclude_driver_id=exclude_id,
            )

            if not nearby_drivers:
                return None

            # Retourner le meilleur (premier de la liste triée par distance)
            return nearby_drivers[0]

        except Exception as e:
            logger.debug(
                "[AutoReassignment] Failed to find alternative driver: %s", str(e)
            )
            return None

    def _notify_driver_reassignment_pending(
        self,
        assignment_id: int,
        driver_id: int,
        new_driver_id: int,
        reason: str,
        timeout_seconds: int,
    ) -> None:
        """Notifie le chauffeur qu'une réassignation est en attente.

        Args:
            assignment_id: ID de l'assignment
            driver_id: ID du chauffeur actuel
            new_driver_id: ID du nouveau chauffeur proposé
            reason: Raison de la réassignation
            timeout_seconds: Timeout pour confirmation
        """
        try:
            from app import socketio

            # Émettre event Socket.IO vers la room du chauffeur
            driver_room = f"driver_{driver_id}"
            socketio.emit(
                "driver:reassignment_pending",
                {
                    "assignment_id": assignment_id,
                    "current_driver_id": driver_id,
                    "new_driver_id": new_driver_id,
                    "reason": reason,
                    "timeout_seconds": timeout_seconds,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                to=driver_room,
            )

            logger.info(
                "[AutoReassignment] Notification sent to driver %d (room: %s)",
                driver_id,
                driver_room,
            )

        except Exception as e:
            logger.warning("[AutoReassignment] Failed to notify driver: %s", str(e))

    def _perform_reassignment(
        self,
        assignment: Assignment,
        booking: Booking,
        new_driver_id: int,
        gain_minutes: int,
    ) -> ReassignmentResult:
        """Effectue la réassignation.

        Args:
            assignment: Assignment à réassigner
            booking: Booking associé
            new_driver_id: ID du nouveau chauffeur
            gain_minutes: Gain estimé (minutes)

        Returns:
            ReassignmentResult avec succès/échec
        """
        try:
            old_driver_id = int(assignment.driver_id) if assignment.driver_id else None  # type: ignore[reportArgumentType,reportGeneralTypeIssues]

            # Créer nouvelle assignation avec nouveau chauffeur
            new_assignment = {
                "booking_id": booking.id,
                "driver_id": new_driver_id,
            }

            # Appliquer la réassignation
            result = apply_assignments(
                company_id=int(booking.company_id),  # type: ignore[reportArgumentType]
                assignments=[new_assignment],
                allow_reassign=True,
                respect_existing=False,  # Forcer réassignation
            )

            if result.get("applied") and len(result["applied"]) > 0:
                # Réassignation réussie
                # Émettre event Socket.IO
                self._emit_reassignment_event(
                    assignment_id=assignment.id,
                    old_driver_id=old_driver_id,
                    new_driver_id=new_driver_id,
                    booking_id=booking.id,
                )

                return ReassignmentResult(
                    success=True,
                    reassigned=True,
                    assignment_id=assignment.id,
                    old_driver_id=old_driver_id,
                    new_driver_id=new_driver_id,
                    delay_minutes=0,  # Sera calculé après réassignation
                    projected_delay_minutes=0,
                    gain_minutes=gain_minutes,
                    reason="Réassignation effectuée avec succès",
                )
            # Échec réassignation
            error_msg = result.get("error", "Erreur inconnue")
            return ReassignmentResult(
                success=False,
                reassigned=False,
                assignment_id=assignment.id,
                old_driver_id=old_driver_id,
                new_driver_id=new_driver_id,
                delay_minutes=0,
                projected_delay_minutes=0,
                gain_minutes=gain_minutes,
                reason=f"Échec réassignation: {error_msg}",
            )

        except Exception as e:
            logger.exception("[AutoReassignment] Failed to perform reassignment: %s", e)
            return ReassignmentResult(
                success=False,
                reassigned=False,
                assignment_id=assignment.id,
                old_driver_id=int(assignment.driver_id)  # type: ignore[reportArgumentType]
                if bool(assignment.driver_id)
                else None,
                new_driver_id=new_driver_id,
                delay_minutes=0,
                projected_delay_minutes=0,
                gain_minutes=gain_minutes,
                reason=f"Exception: {e!s}",
            )

    def _emit_reassignment_event(
        self,
        assignment_id: int,
        old_driver_id: int | None,
        new_driver_id: int,
        booking_id: int,
    ) -> None:
        """Émet un event Socket.IO pour notifier la réassignation.

        Args:
            assignment_id: ID de l'assignment
            old_driver_id: ID de l'ancien chauffeur
            new_driver_id: ID du nouveau chauffeur
            booking_id: ID de la booking
        """
        try:
            from app import socketio

            # Émettre vers la room de l'entreprise
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            booking_repo = BookingRepository()
            booking_dto = booking_repo.find_by_id(booking_id)
            booking = Booking.query.get(booking_dto.id) if booking_dto else None
            if booking:
                company_room = f"company_{booking.company_id}"
                # ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence avec mobile
                socketio.emit(
                    "dispatch_assignment_reassigned",
                    {
                        "assignment_id": assignment_id,
                        "old_driver_id": old_driver_id,
                        "new_driver_id": new_driver_id,
                        "booking_id": booking_id,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                    to=company_room,
                )

                logger.info(
                    "[AutoReassignment] Reassignment event emitted to company %d",
                    booking.company_id,
                )

        except Exception as e:
            logger.warning(
                "[AutoReassignment] Failed to emit reassignment event: %s", str(e)
            )


# Instance globale (singleton)
_auto_reassignment_service_instance: AutoReassignmentService | None = None


def get_auto_reassignment_service() -> AutoReassignmentService:
    """Retourne l'instance singleton du AutoReassignmentService."""
    global _auto_reassignment_service_instance  # noqa: PLW0603
    if _auto_reassignment_service_instance is None:
        enabled = os.getenv("AUTO_REASSIGNMENT_ENABLED", "false").lower() == "true"
        _auto_reassignment_service_instance = AutoReassignmentService(enabled=enabled)
    return _auto_reassignment_service_instance
