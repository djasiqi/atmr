# backend/services/unified_dispatch/suggestions.py
"""
Système de suggestions intelligentes pour l'optimisation du dispatch.
Génère des recommandations contextuelles basées sur la situation actuelle.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from sqlalchemy import or_
from models import Booking, Driver, Assignment, BookingStatus
from services.unified_dispatch.data import haversine_minutes
from services.unified_dispatch.settings import Settings
from shared.time_utils import now_local
from ext import db

logger = logging.getLogger(__name__)


@dataclass
class Suggestion:
    """Une suggestion d'optimisation"""
    action: str  # "reassign", "notify_customer", "add_booking", "adjust_time", "add_driver"
    priority: str  # "low", "medium", "high", "critical"
    message: str
    estimated_gain_minutes: Optional[int] = None
    booking_id: Optional[int] = None
    driver_id: Optional[int] = None
    alternative_driver_id: Optional[int] = None
    additional_data: Optional[Dict[str, Any]] = None
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
    """
    Moteur de génération de suggestions intelligentes
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
    
    def generate_suggestions_for_assignment(
        self,
        assignment: Assignment,
        delay_minutes: int,
        company_id: int
    ) -> List[Suggestion]:
        """
        Génère des suggestions contextuelles pour une assignation avec retard.
        
        Args:
            assignment: L'assignation à analyser
            delay_minutes: Retard en minutes (positif = retard, négatif = avance)
            company_id: ID de l'entreprise
        
        Returns:
            Liste de suggestions classées par priorité
        """
        suggestions = []
        
        # Récupérer le booking et le driver
        booking = db.session.get(Booking, assignment.booking_id)
        driver = db.session.get(Driver, assignment.driver_id) if assignment.driver_id else None
        
        if not booking:
            return suggestions
        
        # Générer des suggestions selon le niveau de retard
        if delay_minutes > 15:
            # Retard critique → notification client URGENTE + réassignation
            suggestions.append(
                self._suggest_customer_notification(booking, delay_minutes)
            )
            suggestions.extend(
                self._suggest_reassignment(booking, driver, delay_minutes, company_id)
            )
        elif delay_minutes > 5:
            # Retard moyen → notification client
            suggestions.append(
                self._suggest_customer_notification(booking, delay_minutes)
            )
            # + possibilité de réassignation si chauffeur proche
            suggestions.extend(
                self._suggest_reassignment(booking, driver, delay_minutes, company_id, threshold_km=3)
            )
        elif delay_minutes < -10:
            # Très en avance → peut optimiser
            suggestions.extend(
                self._suggest_additional_booking(booking, driver, abs(delay_minutes), company_id)
            )
        elif -5 < delay_minutes < 0:
            # Légèrement en avance → OK, aucune action
            suggestions.append(Suggestion(
                action="none",
                priority="low",
                message=f"✅ Chauffeur en avance de {abs(delay_minutes)} min - situation optimale",
                booking_id=booking.id,
                driver_id=driver.id if driver else None,
                auto_applicable=False
            ))
        
        # Suggestions générales
        suggestions.extend(
            self._suggest_time_adjustments(booking, delay_minutes)
        )
        
        # Trier par priorité
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 99))
        
        return suggestions
    
    def _suggest_reassignment(
        self,
        booking: Booking,
        current_driver: Optional[Driver],
        delay_minutes: int,
        company_id: int,
        threshold_km: float = 10.0
    ) -> List[Suggestion]:
        """Suggère de réassigner à un chauffeur plus proche"""
        
        suggestions = []
        
        try:
            # Trouver des chauffeurs disponibles à proximité
            nearby_drivers = self._find_nearby_available_drivers(
                booking,
                company_id,
                radius_km=threshold_km,
                exclude_driver_id=current_driver.id if current_driver else None
            )
            
            if not nearby_drivers:
                return suggestions
            
            # Calculer le gain potentiel pour chaque chauffeur
            for driver, distance_km, eta_minutes in nearby_drivers[:3]:  # Top 3
                # Estimation du gain
                current_eta = delay_minutes + int(
                    (booking.scheduled_time - now_local()).total_seconds() / 60
                )
                new_eta = eta_minutes
                gain = current_eta - new_eta
                
                if gain > 5:  # Gain significatif uniquement
                    priority = "critical" if delay_minutes > 20 else "high"
                    
                    suggestions.append(Suggestion(
                        action="reassign",
                        priority=priority,
                        message=(
                            f"Réassigner au chauffeur #{driver.id} "
                            f"({driver.user.first_name if driver.user else 'Driver'}) "
                            f"- Gain: {gain} min (distance: {distance_km:.1f} km)"
                        ),
                        estimated_gain_minutes=gain,
                        booking_id=booking.id,
                        driver_id=current_driver.id if current_driver else None,
                        alternative_driver_id=driver.id,
                        additional_data={
                            "distance_km": distance_km,
                            "new_eta_minutes": eta_minutes,
                            "driver_name": f"{driver.user.first_name} {driver.user.last_name}" if driver.user else None
                        },
                        auto_applicable=False  # Nécessite validation
                    ))
        
        except Exception as e:
            logger.warning("[Suggestions] Failed to suggest reassignment: %s", e)
        
        return suggestions
    
    def _suggest_customer_notification(
        self,
        booking: Booking,
        delay_minutes: int
    ) -> Suggestion:
        """Suggère de notifier le client du retard"""
        
        priority = "high" if delay_minutes > 10 else "medium"
        
        return Suggestion(
            action="notify_customer",
            priority=priority,
            message=f"Prévenir le client du retard de {delay_minutes} min",
            booking_id=booking.id,
            additional_data={
                "auto_message": (
                    f"Bonjour, votre chauffeur arrivera avec environ {delay_minutes} minutes de retard. "
                    f"Nous nous excusons pour ce désagrément."
                ),
                "customer_name": booking.customer_name,
                "customer_phone": getattr(booking, "customer_phone", None),
            },
            auto_applicable=True  # Peut être automatique
        )
    
    def _suggest_additional_booking(
        self,
        booking: Booking,
        driver: Optional[Driver],
        advance_minutes: int,
        company_id: int
    ) -> List[Suggestion]:
        """Suggère d'ajouter une course supplémentaire quand le chauffeur est très en avance"""
        
        suggestions = []
        
        if not driver or advance_minutes < 15:
            return suggestions
        
        try:
            # Chercher des bookings en attente à proximité
            nearby_bookings = self._find_pending_bookings_nearby(
                booking,
                company_id,
                time_window_minutes=advance_minutes - 5,
                radius_km=5.0
            )
            
            for nearby_booking, distance_km, time_available in nearby_bookings[:2]:  # Top 2
                suggestions.append(Suggestion(
                    action="add_booking",
                    priority="medium",
                    message=(
                        f"Chauffeur disponible {advance_minutes} min avant rendez-vous. "
                        f"Peut prendre la course #{nearby_booking.id} "
                        f"({nearby_booking.customer_name}) à {distance_km:.1f} km"
                    ),
                    booking_id=nearby_booking.id,
                    driver_id=driver.id,
                    estimated_gain_minutes=time_available,
                    additional_data={
                        "original_booking_id": booking.id,
                        "distance_km": distance_km,
                        "pickup_address": getattr(nearby_booking, "pickup_address", None),
                    },
                    auto_applicable=False
                ))
        
        except Exception as e:
            logger.warning("[Suggestions] Failed to suggest additional booking: %s", e)
        
        return suggestions
    
    def _suggest_time_adjustments(
        self,
        booking: Booking,
        delay_minutes: int
    ) -> List[Suggestion]:
        """Suggère des ajustements d'horaire si possible"""
        
        suggestions = []
        
        # Retard critique → URGENT : ajuster l'heure
        if delay_minutes > 30:
            suggestions.append(Suggestion(
                action="adjust_time",
                priority="critical",
                message=(
                    f"🔴 URGENT : Reporter le rendez-vous de {delay_minutes} min "
                    f"({delay_minutes // 60}h{delay_minutes % 60:02d}) et contacter le client immédiatement"
                ),
                booking_id=booking.id,
                additional_data={
                    "proposed_new_time": (
                        booking.scheduled_time + timedelta(minutes=delay_minutes)
                    ).isoformat() if booking.scheduled_time else None,
                    "contact_customer_urgent": True,
                },
                auto_applicable=False
            ))
        # Retard important → ajuster l'heure
        elif delay_minutes > 15:
            suggestions.append(Suggestion(
                action="adjust_time",
                priority="high",
                message=(
                    f"Reporter le rendez-vous de {delay_minutes} min et prévenir le client"
                ),
                booking_id=booking.id,
                additional_data={
                    "proposed_new_time": (
                        booking.scheduled_time + timedelta(minutes=delay_minutes)
                    ).isoformat() if booking.scheduled_time else None,
                },
                auto_applicable=False
            ))
        # Retard modéré et booking flexible
        elif 5 < delay_minutes < 15:
            # Vérifier si le booking a une marge de flexibilité
            # (exemple: rendez-vous médical non urgent)
            is_flexible = not getattr(booking, "is_urgent", False)
            
            if is_flexible:
                suggestions.append(Suggestion(
                    action="adjust_time",
                    priority="medium",
                    message=(
                        f"Proposer de décaler le rendez-vous de {delay_minutes} min "
                        f"(booking non urgent)"
                    ),
                    booking_id=booking.id,
                    additional_data={
                        "proposed_new_time": (
                            booking.scheduled_time + timedelta(minutes=delay_minutes)
                        ).isoformat() if booking.scheduled_time else None,
                    },
                    auto_applicable=False
                ))
        
        return suggestions
    
    def _find_nearby_available_drivers(
        self,
        booking: Booking,
        company_id: int,
        radius_km: float = 10.0,
        exclude_driver_id: Optional[int] = None
    ) -> List[Tuple[Driver, float, int]]:
        """
        Trouve les chauffeurs disponibles à proximité.
        
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
            
            # Récupérer les chauffeurs actifs et disponibles
            query = Driver.query.filter(
                Driver.company_id == company_id,
                Driver.is_active == True,
                Driver.is_available == True
            )
            
            if exclude_driver_id:
                query = query.filter(Driver.id != exclude_driver_id)
            
            drivers = query.all()
            
            # Calculer distance et ETA pour chaque chauffeur
            results = []
            for driver in drivers:
                driver_lat = getattr(driver, "current_lat", getattr(driver, "latitude", None))
                driver_lon = getattr(driver, "current_lon", getattr(driver, "longitude", None))
                
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
                        avg_kmh=getattr(self.settings.matrix, "avg_speed_kmh", 25.0)
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
        radius_km: float = 5.0
    ) -> List[Tuple[Booking, float, int]]:
        """
        Trouve des bookings en attente à proximité dans une fenêtre de temps.
        
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
            
            # Bookings en attente
            pending_bookings = Booking.query.filter(
                Booking.company_id == company_id,
                or_(
                    Booking.status == BookingStatus.PENDING,
                    Booking.status == BookingStatus.ACCEPTED
                ),
                Booking.scheduled_time >= now,
                Booking.scheduled_time <= time_window_end,
                Booking.id != booking.id  # Exclure le booking actuel
            ).all()
            
            results = []
            for pending_booking in pending_bookings:
                p_lat = getattr(pending_booking, "pickup_lat", None)
                p_lon = getattr(pending_booking, "pickup_lon", None)
                
                if not p_lat or not p_lon:
                    continue
                
                pending_pos = (float(p_lat), float(p_lon))
                distance_km = self._calculate_distance_km(current_pos, pending_pos)
                
                if distance_km <= radius_km:
                    # Temps disponible avant ce booking
                    if pending_booking.scheduled_time:
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
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> float:
        """Calcule la distance en km entre deux positions (Haversine)"""
        import math
        
        lat1, lon1 = pos1
        lat2, lon2 = pos2
        
        R = 6371.0  # Rayon de la Terre en km
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        a = (
            math.sin(dphi / 2) ** 2 +
            math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c


def generate_suggestions(
    assignment: Assignment,
    delay_minutes: int,
    company_id: int,
    settings: Optional[Settings] = None
) -> List[Suggestion]:
    """
    Fonction helper pour générer des suggestions.
    
    Usage:
        suggestions = generate_suggestions(assignment, delay_minutes=12, company_id=1)
        for suggestion in suggestions:
            print(suggestion.message)
    """
    engine = SuggestionEngine(settings)
    return engine.generate_suggestions_for_assignment(assignment, delay_minutes, company_id)

