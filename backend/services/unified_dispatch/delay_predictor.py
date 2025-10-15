# backend/services/unified_dispatch/delay_predictor.py
"""
Module de prédiction et analyse des retards pour le système de dispatch.
Anticipe les retards AVANT l'assignation finale et propose des ajustements.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

from models import Booking, Driver
from services.unified_dispatch.data import calculate_eta
from services.unified_dispatch.settings import Settings
from shared.time_utils import now_local

logger = logging.getLogger(__name__)


@dataclass
class DelayPrediction:
    """Prédiction de retard pour une assignation"""
    booking_id: int
    driver_id: int
    predicted_delay_minutes: int
    severity: str  # "low", "medium", "high", "critical"
    confidence: float  # 0.0 - 1.0
    scheduled_time: datetime
    estimated_arrival: datetime
    current_eta: datetime | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "booking_id": self.booking_id,
            "driver_id": self.driver_id,
            "predicted_delay_minutes": self.predicted_delay_minutes,
            "severity": self.severity,
            "confidence": self.confidence,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "estimated_arrival": self.estimated_arrival.isoformat() if self.estimated_arrival else None,
            "current_eta": self.current_eta.isoformat() if self.current_eta else None,
        }


@dataclass
class DelayAnalysis:
    """Analyse complète des retards pour un ensemble d'assignations"""
    predictions: List[DelayPrediction]
    total_assignments: int
    on_time_count: int
    delayed_count: int
    average_delay: float
    max_delay: int
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "summary": {
                "total_assignments": self.total_assignments,
                "on_time_count": self.on_time_count,
                "delayed_count": self.delayed_count,
                "average_delay": round(self.average_delay, 2),
                "max_delay": self.max_delay,
            },
            "recommendations": self.recommendations,
        }


class DelayPredictor:
    """
    Prédit les retards potentiels pour les assignations
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        # Seuils de retard
        self.delay_thresholds = {
            "low": 5,      # < 5 min
            "medium": 10,  # 5-10 min
            "high": 15,    # 10-15 min
            "critical": 20 # > 15 min
        }

    def predict_delays_before_dispatch(
        self,
        problem: Dict[str, Any],
        assignments: List[Any],
    ) -> DelayAnalysis:
        """
        Calcule les retards AVANT l'assignation finale et propose des ajustements.
        
        Args:
            problem: Données du problème de dispatch
            assignments: Liste des assignations proposées
        
        Returns:
            DelayAnalysis avec prédictions et recommandations
        """
        if not assignments:
            return DelayAnalysis(
                predictions=[],
                total_assignments=0,
                on_time_count=0,
                delayed_count=0,
                average_delay=0.0,
                max_delay=0,
                recommendations=["Aucune assignation à analyser"]
            )

        bookings_map = {b.id: b for b in problem.get("bookings", [])}
        drivers_map = {d.id: d for d in problem.get("drivers", [])}

        predictions: List[DelayPrediction] = []

        for assignment in assignments:
            booking_id = getattr(assignment, "booking_id", None)
            driver_id = getattr(assignment, "driver_id", None)

            if booking_id is None or driver_id is None:
                continue

            booking = bookings_map.get(booking_id)
            driver = drivers_map.get(driver_id)

            if not booking or not driver:
                continue

            # Calculer la prédiction de retard
            prediction = self._predict_single_delay(booking, driver, assignment, problem)

            if prediction:
                predictions.append(prediction)

        # Analyser les résultats
        analysis = self._analyze_predictions(predictions)

        # Générer des recommandations
        analysis.recommendations = self._generate_recommendations(analysis, problem)

        logger.info(
            "[DelayPredictor] Analyzed %d assignments: %d on-time, %d delayed (avg: %.1f min)",
            analysis.total_assignments,
            analysis.on_time_count,
            analysis.delayed_count,
            analysis.average_delay
        )

        return analysis

    def _predict_single_delay(
        self,
        booking: Booking,
        driver: Driver,
        assignment: Any,
        problem: Dict[str, Any]
    ) -> DelayPrediction | None:
        """Prédit le retard pour une seule assignation"""

        try:
            # Temps prévu
            scheduled_time = getattr(booking, "scheduled_time", None)
            if not scheduled_time:
                return None

            # ETA estimé
            estimated_pickup = getattr(assignment, "estimated_pickup_arrival", None)

            if not estimated_pickup:
                # Fallback: calculer depuis les coordonnées
                driver_pos = (
                    getattr(driver, "current_lat", getattr(driver, "latitude", 46.2044)),
                    getattr(driver, "current_lon", getattr(driver, "longitude", 6.1432))
                )
                pickup_pos = (
                    getattr(booking, "pickup_lat", 46.2044),
                    getattr(booking, "pickup_lon", 6.1432)
                )

                eta_seconds = calculate_eta(driver_pos, pickup_pos, settings=self.settings)
                estimated_pickup = now_local() + timedelta(seconds=eta_seconds)

            # Calculer le retard
            if isinstance(scheduled_time, datetime):
                scheduled_dt = scheduled_time
            else:
                scheduled_dt = now_local()

            if isinstance(estimated_pickup, datetime):
                estimated_dt = estimated_pickup
            else:
                estimated_dt = now_local()

            delay_seconds = (estimated_dt - scheduled_dt).total_seconds()
            delay_minutes = int(delay_seconds / 60)

            # Déterminer la sévérité
            severity = self._calculate_severity(delay_minutes)

            # Calculer la confiance (basée sur la qualité des données)
            confidence = self._calculate_confidence(booking, driver, problem)

            return DelayPrediction(
                booking_id=booking.id,
                driver_id=driver.id,
                predicted_delay_minutes=delay_minutes,
                severity=severity,
                confidence=confidence,
                scheduled_time=scheduled_dt,
                estimated_arrival=estimated_dt,
            )

        except Exception as e:
            logger.warning(
                "[DelayPredictor] Failed to predict delay for booking %s: %s",
                getattr(booking, "id", None), e
            )
            return None

    def _calculate_severity(self, delay_minutes: int) -> str:
        """Détermine la sévérité du retard"""
        abs_delay = abs(delay_minutes)

        if abs_delay < self.delay_thresholds["low"]:
            return "low"
        elif abs_delay < self.delay_thresholds["medium"]:
            return "medium"
        elif abs_delay < self.delay_thresholds["high"]:
            return "high"
        else:
            return "critical"

    def _calculate_confidence(
        self,
        booking: Booking,
        driver: Driver,
        problem: Dict[str, Any]
    ) -> float:
        """
        Calcule la confiance de la prédiction basée sur la qualité des données.
        Retourne un score entre 0.0 et 1.0.
        """
        confidence = 1.0

        # Réduire si coordonnées manquantes ou par défaut
        if not getattr(booking, "pickup_lat", None) or not getattr(booking, "pickup_lon", None):
            confidence -= 0.3

        if not getattr(driver, "latitude", None) or not getattr(driver, "longitude", None):
            confidence -= 0.2

        # Position du chauffeur récente ?
        last_update = getattr(driver, "last_position_update", None)
        if last_update:
            age_minutes = (now_local() - last_update).total_seconds() / 60
            if age_minutes > 10:
                confidence -= 0.1
        else:
            confidence -= 0.2

        # Matrice de distance utilisée
        matrix_provider = problem.get("matrix_provider", "haversine")
        if matrix_provider == "haversine":
            confidence -= 0.1  # OSRM est plus précis

        return max(0.0, min(1.0, confidence))

    def _analyze_predictions(self, predictions: List[DelayPrediction]) -> DelayAnalysis:
        """Analyse les prédictions et génère des statistiques"""

        if not predictions:
            return DelayAnalysis(
                predictions=[],
                total_assignments=0,
                on_time_count=0,
                delayed_count=0,
                average_delay=0.0,
                max_delay=0,
                recommendations=[]
            )

        total = len(predictions)
        delayed = [p for p in predictions if p.predicted_delay_minutes > 5]
        on_time = total - len(delayed)

        delay_values = [p.predicted_delay_minutes for p in predictions]
        avg_delay = sum(delay_values) / len(delay_values) if delay_values else 0.0
        max_delay = max(delay_values) if delay_values else 0

        return DelayAnalysis(
            predictions=predictions,
            total_assignments=total,
            on_time_count=on_time,
            delayed_count=len(delayed),
            average_delay=avg_delay,
            max_delay=max_delay,
            recommendations=[]  # Sera rempli par _generate_recommendations
        )

    def _generate_recommendations(
        self,
        analysis: DelayAnalysis,
        problem: Dict[str, Any]
    ) -> List[str]:
        """Génère des recommandations basées sur l'analyse"""

        recommendations = []

        # Taux de retard élevé
        if analysis.total_assignments > 0:
            delay_rate = analysis.delayed_count / analysis.total_assignments

            if delay_rate > 0.5:
                recommendations.append(
                    f"⚠️ Taux de retard élevé ({int(delay_rate * 100)}%). "
                    "Envisagez d'ajouter des chauffeurs ou d'activer les chauffeurs d'urgence."
                )
            elif delay_rate > 0.3:
                recommendations.append(
                    f"⚠️ Taux de retard modéré ({int(delay_rate * 100)}%). "
                    "Surveillez la situation et anticipez les besoins."
                )

        # Retard moyen élevé
        if analysis.average_delay > 15:
            recommendations.append(
                f"⏰ Retard moyen élevé ({analysis.average_delay:.1f} min). "
                "Vérifiez les fenêtres horaires et la capacité des chauffeurs."
            )

        # Retards critiques
        critical_delays = [p for p in analysis.predictions if p.severity == "critical"]
        if critical_delays:
            recommendations.append(
                f"🚨 {len(critical_delays)} assignation(s) avec retard critique (>15 min). "
                "Réassignation urgente recommandée."
            )

        # Recommandations spécifiques
        drivers_count = len(problem.get("drivers", []))
        bookings_count = len(problem.get("bookings", []))

        if drivers_count > 0:
            ratio = bookings_count / drivers_count
            if ratio > 5:
                recommendations.append(
                    f"📊 Ratio courses/chauffeur élevé ({ratio:.1f}). "
                    "Ajoutez des chauffeurs pour améliorer les délais."
                )

        if not recommendations:
            recommendations.append("✅ Planning optimal, aucune action requise.")

        return recommendations


def predict_delays_for_dispatch(
    problem: Dict[str, Any],
    assignments: List[Any],
    settings: Settings | None = None
) -> DelayAnalysis:
    """
    Fonction helper pour prédire les retards d'un ensemble d'assignations.
    
    Usage:
        analysis = predict_delays_for_dispatch(problem, assignments)
        if analysis.delayed_count > 0:
            print(f"Attention: {analysis.delayed_count} retards prévus!")
    """
    predictor = DelayPredictor(settings)
    return predictor.predict_delays_before_dispatch(problem, assignments)

