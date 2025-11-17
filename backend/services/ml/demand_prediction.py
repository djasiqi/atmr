"""Epic 4.2 - Prédiction de demande avec séries temporelles.

Ce module implémente:
- Prédiction de demande par zone géographique
- Séries temporelles: day-of-week, trend, vacances
- Usage: Planifier emergency drivers dynamiquement
- Critère: -10% utilisation "urgence" aux heures pleines, même SLA
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import dynamique
try:
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    logger.error("[DemandPrediction] scikit-learn requis. Installer avec: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# statsmodels optionnel - pas utilisé dans cette implémentation
STATSMODELS_AVAILABLE = False

# Constantes
DEMAND_PREDICTION_WINDOW_HOURS = 24
HOLIDAY_MULTIPLIER = 1.5
WEEKEND_MULTIPLIER = 1.2
RUSH_HOUR_MULTIPLIER = 1.3

# Constantes pour les valeurs magiques
MIN_HISTORICAL_DATA_POINTS = 7
MIN_WEEKDAY_DATA = 5
WEEKEND_DAY_START = 5  # Samedi
MORNING_RUSH_START = 7
MORNING_RUSH_END = 9
EVENING_RUSH_START = 17
EVENING_RUSH_END = 19


@dataclass
class ZoneDemandPrediction:
    """Prédiction de demande pour une zone."""

    zone_id: str
    predicted_demand: float  # Nombre de courses prévues
    confidence: float  # 0.0 - 1.0
    peak_hours: List[int]  # Heures de pointe
    factors: Dict[str, float]  # Facteurs contributifs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "predicted_demand": round(self.predicted_demand, 1),
            "confidence": round(self.confidence, 3),
            "peak_hours": self.peak_hours,
            "factors": self.factors,
        }


@dataclass
class EmergencyDriverPlanning:
    """Planification de drivers d'urgence."""

    zone_id: str
    recommended_emergency_drivers: int
    predicted_peak_demand: float
    current_driver_count: int
    urgency_score: float  # 0.0 - 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "recommended_emergency_drivers": self.recommended_emergency_drivers,
            "predicted_peak_demand": round(self.predicted_peak_demand, 1),
            "current_driver_count": self.current_driver_count,
            "urgency_score": round(self.urgency_score, 3),
        }


class DemandPredictor:
    """Prédicteur de demande avec séries temporelles.

    Prédit la demande future par zone géographique en utilisant:
    - Patterns hebdomadaires (day-of-week)
    - Trends saisonniers
    - Effets vacances/jours fériés
    - Heures de pointe
    """

    def __init__(self):
        super().__init__()
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()  # type: ignore[call-arg]
        else:
            self.scaler = None
        self.holiday_service = None
        self._historical_data: Dict[str, pd.DataFrame] = {}  # zone_id -> DataFrame

    def predict_demand_for_zone(self, zone_id: str, target_date: datetime) -> ZoneDemandPrediction:
        """Prédit la demande pour une zone à une date donnée.

        Args:
            zone_id: ID de la zone
            target_date: Date cible

        Returns:
            ZoneDemandPrediction
        """
        try:
            # Charger données historiques de la zone
            historical_data = self._get_historical_data_for_zone(zone_id)

            if historical_data is None or len(historical_data) < MIN_HISTORICAL_DATA_POINTS:
                # Fallback: prédiction basée sur patterns de base
                return self._fallback_prediction(zone_id, target_date)

            # Analyser les patterns
            day_of_week = target_date.weekday()
            hour = target_date.hour

            # Moyenne historique par jour de semaine
            avg_demand_by_weekday = historical_data.groupby(historical_data.index.weekday).mean()["demand"]

            # Moyenne historique par heure
            avg_demand_by_hour = historical_data.groupby(historical_data.index.hour).mean()["demand"]

            # Base prediction
            base_demand_value = avg_demand_by_weekday.get(day_of_week, None)
            if base_demand_value is None:
                avg_mean = avg_demand_by_weekday.mean()
                base_demand = float(avg_mean if isinstance(avg_mean, (int, float)) else avg_mean.iloc[0])
            else:
                base_demand = float(base_demand_value)

            # Ajustements
            predictions = []
            avg_hour_mean = avg_demand_by_hour.mean()

            # Calculer facteurs pour target_date (pas dans la boucle)
            is_weekend_init = target_date.weekday() >= WEEKEND_DAY_START
            holiday_factor_init = self._get_holiday_multiplier(target_date)

            for h in range(24):
                future_hour = (hour + h) % 24
                future_date = target_date + timedelta(hours=h)

                # Facteur horaire
                future_hour_value = avg_demand_by_hour.get(future_hour, None)
                if future_hour_value is None or avg_hour_mean == 0:
                    hourly_demand_factor = 1.0
                else:
                    hourly_demand_factor = float(future_hour_value) / float(avg_hour_mean)

                # Facteur week-end
                is_weekend = future_date.weekday() >= WEEKEND_DAY_START
                weekend_factor = WEEKEND_MULTIPLIER if is_weekend else 1.0

                # Facteur vacances
                holiday_factor = self._get_holiday_multiplier(future_date)

                # Facteur rush hour
                is_rush = (
                    MORNING_RUSH_START <= future_hour <= MORNING_RUSH_END
                    or EVENING_RUSH_START <= future_hour <= EVENING_RUSH_END
                )
                rush_factor = RUSH_HOUR_MULTIPLIER if is_rush else 1.0

                # Prédiction par heure
                predicted_demand = base_demand * hourly_demand_factor * weekend_factor * holiday_factor * rush_factor
                predictions.append(predicted_demand)

            # Demande totale prédite
            total_predicted_demand = sum(predictions)

            # Heures de pointe (top 3)
            peak_hours = sorted(range(24), key=lambda h: predictions[h], reverse=True)[:3]

            # Confiance (basée sur quantité de données historiques)
            days_of_data = len(historical_data) / 24
            confidence = min(1.0, days_of_data / 30)  # Au moins 30 jours pour confiance maximale

            # Facteurs contributifs (utiliser valeurs initiales)
            is_rush_init = (
                MORNING_RUSH_START <= target_date.hour <= MORNING_RUSH_END
                or EVENING_RUSH_START <= target_date.hour <= EVENING_RUSH_END
            )
            factors = {
                "weekend": 1.0 if is_weekend_init else 0.0,
                "rush_hour": 1.0 if is_rush_init else 0.0,
                "holiday": 1.0 if holiday_factor_init > 1.0 else 0.0,
                "seasonal_trend": 1.0,  # TODO: calculer depuis décomposition saisonnière
            }

            return ZoneDemandPrediction(
                zone_id=zone_id,
                predicted_demand=total_predicted_demand,
                confidence=confidence,
                peak_hours=peak_hours,
                factors=factors,
            )

        except Exception as e:
            logger.error("[DemandPrediction] Erreur prédiction zone %s: %s", zone_id, e)
            return self._fallback_prediction(zone_id, target_date)

    def _get_historical_data_for_zone(self, _zone_id: str) -> pd.DataFrame | None:
        """Récupère les données historiques pour une zone.

        Args:
            _zone_id: ID de la zone (non utilisé actuellement, TODO: implémenter filtrage)

        Returns:
            DataFrame avec colonnes 'timestamp', 'demand'
        """
        try:
            from datetime import timedelta

            from models import Booking, BookingStatus

            # Charger 90 derniers jours
            cutoff_date = datetime.now() - timedelta(days=90)

            from sqlalchemy import and_

            bookings = Booking.query.filter(
                and_(
                    Booking.status == BookingStatus.COMPLETED,  # type: ignore[arg-type]
                    Booking.completed_at >= cutoff_date,
                )
            ).all()

            if not bookings:
                return None

            # Filtrer par zone (TODO: implémenter mapping zone_id)
            # Pour l'instant, utiliser tous les bookings

            # Grouper par heure
            df_data = []
            for booking in bookings:
                completed_at = getattr(booking, "completed_at", None)
                if completed_at:
                    # Round à l'heure
                    hour = completed_at.replace(minute=0, second=0, microsecond=0)
                    df_data.append({"timestamp": hour, "demand": 1})

            if not df_data:
                return None

            df = pd.DataFrame(df_data)
            df_grouped = df.groupby("timestamp").count().reset_index()
            df_grouped.rename(columns={"demand": "demand"}, inplace=True)

            # Index par timestamp
            df_grouped["timestamp"] = pd.to_datetime(df_grouped["timestamp"])
            df_grouped.set_index("timestamp", inplace=True)

            # Remplir les heures manquantes avec 0
            return df_grouped.resample("H").sum().fillna(0)

        except Exception as e:
            logger.warning("[DemandPrediction] Erreur chargement historique: %s", e)
            return None

    def _get_holiday_multiplier(self, date: datetime) -> float:
        """Retourne le multiplicateur pour vacances/jours fériés."""
        try:
            from services.holidays_service import is_holiday

            if is_holiday(date):
                return HOLIDAY_MULTIPLIER
        except Exception:
            pass

        return 1.0

    def _fallback_prediction(self, zone_id: str, target_date: datetime) -> ZoneDemandPrediction:
        """Prédiction de fallback."""
        is_weekend_value = target_date.weekday() >= WEEKEND_DAY_START
        is_rush_value = MORNING_RUSH_START <= target_date.hour <= EVENING_RUSH_END

        base_demand = 20.0  # Demande moyenne

        if is_weekend_value:
            base_demand *= WEEKEND_MULTIPLIER
        if is_rush_value:
            base_demand *= RUSH_HOUR_MULTIPLIER

        return ZoneDemandPrediction(
            zone_id=zone_id,
            predicted_demand=base_demand,
            confidence=0.5,
            peak_hours=[8, 18],
            factors={
                "weekend": 1.0 if is_weekend_value else 0.0,
                "rush_hour": 1.0 if is_rush_value else 0.0,
            },
        )

    def plan_emergency_drivers(
        self, zone_id: str, current_driver_count: int, target_date: datetime | None = None
    ) -> EmergencyDriverPlanning:
        """Recommande le nombre de emergency drivers à activer.

        Args:
            zone_id: ID de la zone
            current_driver_count: Nombre actuel de drivers
            target_date: Date cible (par défaut: maintenant)

        Returns:
            EmergencyDriverPlanning
        """
        if target_date is None:
            target_date = datetime.now()

        # Prédire demande
        demand_prediction = self.predict_demand_for_zone(zone_id, target_date)
        predicted_peak_demand = demand_prediction.predicted_demand

        # Ratio drivers/demand idéal (ex: 1 driver pour 3 courses/heure)
        optimal_ratio = 0.33  # 1 driver pour 3 courses/heure

        # Drivers nécessaires
        required_drivers = int(np.ceil(predicted_peak_demand * optimal_ratio))

        # Drivers d'urgence à activer
        if current_driver_count >= required_drivers:
            recommended_emergency = 0
            urgency_score = 0.0
        else:
            recommended_emergency = required_drivers - current_driver_count
            # Score d'urgence: 0-1
            urgency_score = min(1.0, predicted_peak_demand / (current_driver_count * 3))

        return EmergencyDriverPlanning(
            zone_id=zone_id,
            recommended_emergency_drivers=recommended_emergency,
            predicted_peak_demand=predicted_peak_demand,
            current_driver_count=current_driver_count,
            urgency_score=urgency_score,
        )


# Instance globale
_global_demand_predictor: DemandPredictor | None = None


def get_demand_predictor() -> DemandPredictor:
    """Récupère l'instance globale du prédicteur de demande."""
    global _global_demand_predictor  # noqa: PLW0603

    if _global_demand_predictor is None:
        _global_demand_predictor = DemandPredictor()

    return _global_demand_predictor
