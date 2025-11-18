# backend/services/unified_dispatch/ml_predictor.py

# Constantes pour éviter les valeurs magiques
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

DELAY_THRESHOLD = 5
SI_THRESHOLD = 10
RETARD_THRESHOLD = 5
DELAY_MINUTES_THRESHOLD = 5
TOTAL_COUNT_ZERO = 0
DAY_OF_WEEK_THRESHOLD = 5
HOUR_THRESHOLD = 9
ABS_DELAY_THRESHOLD = 5
MORNING_RUSH_START = 7
MORNING_RUSH_END = 9
EVENING_RUSH_START = 17
EVENING_RUSH_END = 19
MIN_IMPACT_THRESHOLD = 0.1

"""Module de Machine Learning pour la prédiction avancée des retards.
Apprend des patterns historiques pour améliorer les prédictions futures.

Note: Nécessite scikit-learn pour l'entraînement du modèle.
Installation: pip install scikit-learn pandas
"""


logger = logging.getLogger(__name__)

# Vérifier si scikit-learn est disponible
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = False
    logger.warning("[MLPredictor] scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class DelayPrediction:
    """Prédiction de retard par ML."""

    booking_id: int
    predicted_delay_minutes: float
    confidence: float  # 0 - 1
    risk_level: str  # "low", "medium", "high"
    contributing_factors: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "booking_id": self.booking_id,
            "predicted_delay_minutes": round(self.predicted_delay_minutes, 2),
            "confidence": round(self.confidence, 3),
            "risk_level": self.risk_level,
            "contributing_factors": self.contributing_factors,
        }


class DelayMLPredictor(object):
    """Prédicteur ML de retards basé sur l'historique.
    Utilise Random Forest pour la régression.
    """

    def __init__(self, model_path: str | None = None):
        """Args:
        model_path: Chemin vers le modèle sauvegardé (optionnel).

        """
        super().__init__()
        self.model_path = model_path or "data/ml/models/delay_predictor.pkl"
        self.model: RandomForestRegressor | None = None
        self.scaler: StandardScaler | None = None
        self.scaler_params: Dict[str, Any] | None = None
        self.feature_names: List[str] = []
        self.is_trained = False

        # Charger le modèle existant si disponible
        if Path(self.model_path).exists():
            self.load_model()
        else:
            logger.warning("[MLPredictor] Model not found at %s", self.model_path)

    def extract_features(self, booking: Any, driver: Any, current_time: datetime | None = None) -> Dict[str, float]:
        """Extrait les features pour la prédiction.
        Features utilisées:
        - time_of_day: Heure de la journée (0-23)
        - day_of_week: Jour de la semaine (0-6)
        - distance_km: Distance estimée
        - is_medical: Booking médical (0/1)
        - is_urgent: Booking urgent (0/1)
        - driver_punctuality_score: Score de ponctualité du chauffeur (0-1)
        - booking_priority: Priorité du booking (0-1)
        - weather_factor: Facteur météo (placeholder pour l'instant)
        - traffic_density: Densité du trafic estimée (0-1).
        """
        if current_time is None:
            current_time = datetime.now()

        scheduled_time = getattr(booking, "scheduled_time", current_time)

        # Time features
        time_of_day = scheduled_time.hour if scheduled_time else 12
        day_of_week = scheduled_time.weekday() if scheduled_time else 0

        # Distance (estimation haversine)
        distance_km = self._estimate_distance(booking)

        # Booking characteristics
        is_medical = 1 if getattr(booking, "medical_facility", None) else 0
        is_urgent = 1 if getattr(booking, "is_urgent", False) else 0

        # Driver features
        driver_score = self._calculate_driver_punctuality(driver)

        # Priority (exemple simplifié)
        booking_priority = 0.8 if is_medical or is_urgent else 0.5

        # Traffic density (estimation basée sur l'heure)
        traffic_density = self._estimate_traffic_density(time_of_day, day_of_week)

        # Weather (placeholder - pourrait être intégré avec une API météo)
        weather_factor = 0.5  # Neutre par défaut

        return {
            "time_of_day": float(time_of_day),
            "day_of_week": float(day_of_week),
            "distance_km": float(distance_km),
            "is_medical": float(is_medical),
            "is_urgent": float(is_urgent),
            "driver_punctuality_score": float(driver_score),
            "booking_priority": float(booking_priority),
            "traffic_density": float(traffic_density),
            "weather_factor": float(weather_factor),
        }

    def _estimate_distance(self, booking: Any) -> float:
        """Estime la distance en km (Haversine)."""
        try:
            lat1 = float(getattr(booking, "pickup_lat", 46.2044))
            lon1 = float(getattr(booking, "pickup_lon", 6.1432))
            lat2 = float(getattr(booking, "dropoff_lat", 46.2044))
            lon2 = float(getattr(booking, "dropoff_lon", 6.1432))

            # Import centralisé depuis shared.geo_utils
            from shared.geo_utils import haversine_distance

            return haversine_distance(lat1, lon1, lat2, lon2)
        except Exception:
            return 5  # Distance par défaut

    def _calculate_driver_punctuality(self, driver: Any) -> float:
        """Calcule un score de ponctualité du chauffeur (0-1) basé sur l'historique réel.
        Méthode :
        - Récupère les 50 dernières courses du chauffeur
        - Calcule le % de courses terminées à temps (delay <= DELAY_THRESHOLD min)
        - Retourne 0.75 par défaut si < SI_THRESHOLD courses (pas assez de données).

        Returns:
            Score entre 0 (toujours en retard) et 1 (toujours à l'heure)

        """
        try:
            from datetime import datetime, timedelta

            from sqlalchemy import and_

            from models import Booking, BookingStatus

            driver_id = getattr(driver, "id", None)
            if not driver_id:
                return 0.75  # Valeur par défaut si driver inconnu

            # Récupérer les 50 dernières courses terminées dans les 90 derniers jours
            cutoff_date = datetime.now(datetime.timezone.utc) - timedelta(days=90)
            recent_bookings = (
                Booking.query.filter(
                    and_(
                        Booking.driver_id == driver_id,
                        Booking.status == BookingStatus.COMPLETED,  # type: ignore[arg-type]
                        Booking.completed_at >= cutoff_date,
                    )
                )
                .order_by(Booking.completed_at.desc())
                .limit(50)
                .all()
            )

            # Minimum SI_THRESHOLD courses pour avoir des statistiques significatives
            if len(recent_bookings) < SI_THRESHOLD:
                logger.debug(
                    "[MLPredictor] Driver #%s : seulement %s courses, score par défaut 0.75",
                    driver_id,
                    len(recent_bookings),
                )
                return 0.75

            # Calculer combien de courses étaient à l'heure
            on_time_count = 0
            total_count = 0

            for booking in recent_bookings:
                # Comparer scheduled_time vs actual_pickup_time (ou completed_at si pas de pickup_time)
                scheduled = getattr(booking, "scheduled_time", None)
                actual_pickup = getattr(booking, "actual_pickup_time", None)

                if not scheduled:
                    continue

                # Si pas de actual_pickup_time, utiliser completed_at comme proxy
                if not actual_pickup:
                    actual_pickup = getattr(booking, "completed_at", None)

                if not actual_pickup:
                    continue

                # Calculer le retard en minutes
                delay_seconds = (actual_pickup - scheduled).total_seconds()
                delay_minutes = delay_seconds / 60

                total_count += 1

                # Considérer "à l'heure" si retard <= RETARD_THRESHOLD minutes
                if delay_minutes <= DELAY_MINUTES_THRESHOLD:
                    on_time_count += 1

            if total_count == TOTAL_COUNT_ZERO:
                return 0.75

            # Score = % de courses à l'heure
            punctuality_score = on_time_count / total_count

            logger.debug(
                "[MLPredictor] Driver #%s : %s/%s courses à l'heure = %.2f",
                driver_id,
                on_time_count,
                total_count,
                punctuality_score,
            )

            return punctuality_score

        except Exception as e:
            # En cas d'erreur (DB non accessible, etc.), retourner la valeur par défaut
            logger.warning("[MLPredictor] Erreur calcul ponctualité : %s", e)
            return 0.75

    def _estimate_traffic_density(self, hour: int, day_of_week: int) -> float:
        """Estime la densité du trafic (0-1) basée sur l'heure et le jour."""
        # Heures de pointe: 7-9h et 17-19h en semaine
        is_weekday = day_of_week < DAY_OF_WEEK_THRESHOLD
        is_morning_rush = MORNING_RUSH_START <= hour <= HOUR_THRESHOLD
        is_evening_rush = EVENING_RUSH_START <= hour <= EVENING_RUSH_END

        if is_weekday and (is_morning_rush or is_evening_rush):
            return 0.8  # Trafic dense
        if is_weekday and MORNING_RUSH_END < hour < HOUR_THRESHOLD:
            return 0.5  # Trafic moyen
        return 0.3  # Trafic faible

    def train_on_historical_data(
        self, historical_data: List[Dict[str, Any]], save_model: bool = True
    ) -> Dict[str, Any]:
        """Entraîne le modèle sur des données historiques.

        Args:
            historical_data: Liste de dicts avec features + actual_delay_minutes
            save_model: Sauvegarder le modèle après entraînement
        Returns:
            Métriques d'entraînement

        """
        if not SKLEARN_AVAILABLE:
            msg = "scikit-learn is required for training. Install with: pip install scikit-learn"
            raise ImportError(msg)

        if not historical_data:
            msg = "No historical data provided for training"
            raise ValueError(msg)

        # Préparer les données
        X = []
        y = []

        for record in historical_data:
            features = record.get("features", {})
            actual_delay = record.get("actual_delay_minutes", 0)

            if not features:
                continue

            # Ordonner les features de manière cohérente
            if not self.feature_names:
                self.feature_names = sorted(features.keys())

            feature_vector = [features.get(f, 0) for f in self.feature_names]
            X.append(feature_vector)
            y.append(actual_delay)

        if not X or not y:
            msg = "No valid training data after preprocessing"
            raise ValueError(msg)

        X_array = np.array(X)
        y_array = np.array(y)

        logger.info("[MLPredictor] Training on %s samples with %s features", len(X), len(self.feature_names))

        # Standardiser les features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_array)

        # Entraîner le modèle
        self.model = RandomForestRegressor(
            n_estimators=0.100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1
        )

        self.model.fit(X_scaled, y_array)
        self.is_trained = True

        # Métriques
        train_score = self.model.score(X_scaled, y_array)

        metrics = {
            "samples_count": len(X),
            "features_count": len(self.feature_names),
            "r2score": float(train_score),
            "feature_importance": {
                name: float(importance)
                for name, importance in zip(self.feature_names, self.model.feature_importances_, strict=False)
            },
        }

        logger.info("[MLPredictor] Training complete. R² score: %s", train_score)

        # Sauvegarder
        if save_model:
            self.save_model()

        return metrics

    def predict_delay(self, booking: Any, driver: Any, current_time: datetime | None = None) -> DelayPrediction:
        """Prédit le retard pour une assignation avec le modèle entraîné.

        Returns:
            DelayPrediction avec la prédiction et la confiance

        """
        if not self.is_trained or self.model is None:
            logger.warning("[MLPredictor] Model not trained, using fallback heuristic")
            _ = current_time  # Utiliser le paramètre pour éviter l'avertissement
            # Fallback: estimation simple basée sur distance
            pickup_lat = float(getattr(booking, "pickup_lat", 0) or 0)
            pickup_lon = float(getattr(booking, "pickup_lon", 0) or 0)
            dropoff_lat = float(getattr(booking, "dropoff_lat", 0) or 0)
            dropoff_lon = float(getattr(booking, "dropoff_lon", 0) or 0)

            if all([pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]):
                from shared.geo_utils import haversine_distance

                distance_km = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
                predicted_delay = distance_km * 0.5  # Simple heuristique
            else:
                predicted_delay = 3

            return DelayPrediction(
                booking_id=getattr(booking, "id", 0),
                predicted_delay_minutes=predicted_delay,
                confidence=0.3,  # Faible confiance (fallback)
                risk_level="medium",
                contributing_factors={"heuristic": 1},
            )

        if not SKLEARN_AVAILABLE:
            msg = "scikit-learn is required for prediction"
            raise ImportError(msg)

        try:
            # Utiliser le nouveau pipeline de feature engineering
            from services.ml_features import (
                engineer_features,
                features_to_dataframe,
                normalize_features,
            )

            # 1. Feature engineering complet
            features = engineer_features(booking, driver)

            # 2. Normaliser
            if self.scaler_params:
                features = normalize_features(features, self.scaler_params)

            # 3. Convertir en DataFrame avec bon ordre de colonnes
            feature_df = features_to_dataframe(features, self.feature_names)

            # 4. Prédire
            predicted_delay = float(self.model.predict(feature_df)[0])

            # 5. Calculer la confiance (basée sur variance des arbres)
            tree_predictions = [tree.predict(feature_df)[0] for tree in self.model.estimators_]
            std = float(np.std(tree_predictions))
            confidence = max(0, min(1, 1 - (std / 10)))

            # 6. Niveau de risque
            abs_delay = abs(predicted_delay)
            if abs_delay < ABS_DELAY_THRESHOLD:
                risk_level = "low"
            elif abs_delay < ABS_DELAY_THRESHOLD:
                risk_level = "medium"
            else:
                risk_level = "high"

            # 7. Top 5 facteurs contributifs
            feature_importances = self.model.feature_importances_
            top_factors = {}
            for i, name in enumerate(self.feature_names):
                if i < len(feature_importances):
                    impact = float(features.get(name, 0) * feature_importances[i])
                    if abs(impact) > MIN_IMPACT_THRESHOLD:  # Seulement facteurs significatifs
                        top_factors[name] = impact

            # Garder top 5
            sorted_factors = sorted(top_factors.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            contributing_factors = dict(sorted_factors)

            logger.info("[MLPredictor] Prediction for booking %s: %.2fmin", getattr(booking, "id", 0), predicted_delay)

            return DelayPrediction(
                booking_id=getattr(booking, "id", 0),
                predicted_delay_minutes=predicted_delay,
                confidence=confidence,
                risk_level=risk_level,
                contributing_factors=contributing_factors,
            )

        except Exception as e:
            logger.error("[MLPredictor] Prediction failed: %s", e)
            # Fallback en cas d'erreur
            return DelayPrediction(
                booking_id=getattr(booking, "id", 0),
                predicted_delay_minutes=5,
                confidence=0.2,
                risk_level="medium",
                contributing_factors={"error": 1},
            )

    def save_model(self) -> None:
        """Sauvegarde le modèle sur disque."""
        if not self.model or not self.scaler:
            msg = "No model to save"
            raise RuntimeError(msg)

        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "trained_at": datetime.now().isoformat(),
        }

        with Path(self.model_path).open("wb") as f:
            pickle.dump(model_data, f)

        logger.info("[MLPredictor] Model saved to %s", self.model_path)

    def load_model(self) -> None:
        """Charge le modèle depuis le disque."""
        if not Path(self.model_path).exists():
            msg = f"Model file not found: {self.model_path}"
            raise FileNotFoundError(msg)

        try:
            with Path(self.model_path).open("rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.feature_names = model_data["feature_names"]
            self.is_trained = True

            # Charger scalers depuis scalers.json
            scaler_path = "data/ml/scalers.json"
            if Path(scaler_path).exists():
                import json

                with Path(scaler_path).open() as f:
                    self.scaler_params = json.load(f).get("standard_scaler", None)

            logger.info(
                "[MLPredictor] Model loaded from %s (trained at: %s, features: %d, MAE test: %s)",
                self.model_path,
                model_data.get("trained_at", "unknown"),
                len(self.feature_names),
                model_data.get("metrics", {}).get("test", {}).get("mae", "N/A"),
            )
        except Exception as e:
            logger.error("[MLPredictor] Failed to load model: %s", e)
            self.is_trained = False


# Fonction helper pour faciliter l'utilisation
_global_predictor: DelayMLPredictor | None = None


def get_ml_predictor() -> DelayMLPredictor:
    """Récupère l'instance globale du prédicteur."""
    global _global_predictor  # noqa: PLW0603
    if _global_predictor is None:
        _global_predictor = DelayMLPredictor()
    return _global_predictor


def predict_with_feature_flag(
    booking: Any, driver: Any, current_time: datetime | None = None, request_id: str | None = None
) -> DelayPrediction:
    """Prédiction avec feature flag et logging exhaustif.

    Args:
        booking: Booking à prédire
        driver: Driver assigné
        current_time: Timestamp actuel (optionnel)
        request_id: ID de la requête pour tracking

    Returns:
        DelayPrediction (ML ou fallback selon feature flag)

    """
    from feature_flags import FeatureFlags

    booking_id = getattr(booking, "id", "unknown")
    driver_id = getattr(driver, "id", "unknown")

    # Vérifier feature flag
    use_ml = FeatureFlags.is_ml_enabled()

    try:
        if use_ml:
            # Utiliser ML
            predictor = get_ml_predictor()

            if not predictor.is_trained:
                logger.warning("[ML] Model not trained for booking %s, using fallback", booking_id)
                FeatureFlags.record_ml_failure()
                prediction = predictor.predict_delay(booking, driver, current_time)
            else:
                # ML prédiction
                import time

                start_time = time.time()

                prediction = predictor.predict_delay(booking, driver, current_time)

                elapsed_ms = (time.time() - start_time) * 1000

                # Logging exhaustif
                logger.info(
                    "[ML] Prediction for booking %s (driver %s): delay=%.2f, confidence=%.3f, risk=%s, time=%.1fms, request_id=%s",
                    booking_id,
                    driver_id,
                    prediction.predicted_delay_minutes,
                    prediction.confidence,
                    prediction.risk_level,
                    elapsed_ms,
                    request_id,
                )

                # Enregistrer succès
                FeatureFlags.record_ml_success()
        else:
            # Utiliser fallback
            logger.info("[ML] Using fallback for booking %s (ML disabled or outside traffic percentage)", booking_id)

            predictor = get_ml_predictor()
            prediction = predictor.predict_delay(booking, driver, current_time)

    except Exception as e:
        # En cas d'erreur, utiliser fallback si activé
        logger.exception("[ML] Prediction failed for booking %s: %s", booking_id, e)

        FeatureFlags.record_ml_failure()

        if FeatureFlags.should_fallback_on_error():
            logger.warning("[ML] Using fallback for booking %s after error", booking_id)

            # Fallback simple

            pickup_lat = float(getattr(booking, "pickup_lat", 0) or 0)
            pickup_lon = float(getattr(booking, "pickup_lon", 0) or 0)
            dropoff_lat = float(getattr(booking, "dropoff_lat", 0) or 0)
            dropoff_lon = float(getattr(booking, "dropoff_lon", 0) or 0)

            if all([pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]):
                from shared.geo_utils import haversine_distance

                distance_km = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
                predicted_delay = distance_km * 0.5
            else:
                predicted_delay = 5

            prediction = DelayPrediction(
                booking_id=int(booking_id) if isinstance(booking_id, (int, str)) and str(booking_id).isdigit() else 0,
                predicted_delay_minutes=predicted_delay,
                confidence=0.2,
                risk_level="medium",
                contributing_factors={"fallback_error": 1},
            )
        else:
            # Re-raise si fallback désactivé
            raise
    return prediction
