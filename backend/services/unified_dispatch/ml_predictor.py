# backend/services/unified_dispatch/ml_predictor.py
"""
Module de Machine Learning pour la prédiction avancée des retards.
Apprend des patterns historiques pour améliorer les prédictions futures.

Note: Nécessite scikit-learn pour l'entraînement du modèle.
Installation: pip install scikit-learn pandas
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Vérifier si scikit-learn est disponible
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "[MLPredictor] scikit-learn not available. "
        "Install with: pip install scikit-learn"
    )


@dataclass
class DelayPrediction:
    """Prédiction de retard par ML"""
    booking_id: int
    predicted_delay_minutes: float
    confidence: float  # 0.0 - 1.0
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


class DelayMLPredictor:
    """
    Prédicteur ML de retards basé sur l'historique.
    Utilise Random Forest pour la régression.
    """

    def __init__(self, model_path: str | None = None):
        """
        Args:
            model_path: Chemin vers le modèle sauvegardé (optionnel)
        """
        self.model_path = model_path or "backend/data/ml_models/delay_predictor.pkl"
        self.model: RandomForestRegressor | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: List[str] = []
        self.is_trained = False

        # Charger le modèle existant si disponible
        if os.path.exists(self.model_path):
            self.load_model()

    def extract_features(
        self,
        booking: Any,
        driver: Any,
        current_time: datetime | None = None
    ) -> Dict[str, float]:
        """
        Extrait les features pour la prédiction.
        
        Features utilisées:
        - time_of_day: Heure de la journée (0-23)
        - day_of_week: Jour de la semaine (0-6)
        - distance_km: Distance estimée
        - is_medical: Booking médical (0/1)
        - is_urgent: Booking urgent (0/1)
        - driver_punctuality_score: Score de ponctualité du chauffeur (0-1)
        - booking_priority: Priorité du booking (0-1)
        - weather_factor: Facteur météo (placeholder pour l'instant)
        - traffic_density: Densité du trafic estimée (0-1)
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
        is_medical = 1.0 if getattr(booking, "medical_facility", None) else 0.0
        is_urgent = 1.0 if getattr(booking, "is_urgent", False) else 0.0

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
        """Estime la distance en km (Haversine)"""
        try:
            import math

            lat1 = float(getattr(booking, "pickup_lat", 46.2044))
            lon1 = float(getattr(booking, "pickup_lon", 6.1432))
            lat2 = float(getattr(booking, "dropoff_lat", 46.2044))
            lon2 = float(getattr(booking, "dropoff_lon", 6.1432))

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
        except Exception:
            return 5.0  # Distance par défaut

    def _calculate_driver_punctuality(self, driver: Any) -> float:
        """
        Calcule un score de ponctualité du chauffeur (0-1) basé sur l'historique réel.
        
        Méthode :
        - Récupère les 50 dernières courses du chauffeur
        - Calcule le % de courses terminées à temps (delay <= 5 min)
        - Retourne 0.75 par défaut si < 10 courses (pas assez de données)
        
        Returns:
            Score entre 0.0 (toujours en retard) et 1.0 (toujours à l'heure)
        """
        try:
            from datetime import datetime, timedelta

            from sqlalchemy import and_

            from models import Booking, BookingStatus

            driver_id = getattr(driver, "id", None)
            if not driver_id:
                return 0.75  # Valeur par défaut si driver inconnu

            # Récupérer les 50 dernières courses terminées dans les 90 derniers jours
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            recent_bookings = Booking.query.filter(
                and_(
                    Booking.driver_id == driver_id,
                    Booking.status == BookingStatus.COMPLETED,
                    Booking.completed_at >= cutoff_date
                )
            ).order_by(Booking.completed_at.desc()).limit(50).all()

            # Minimum 10 courses pour avoir des statistiques significatives
            if len(recent_bookings) < 10:
                logger.debug(f"[MLPredictor] Driver #{driver_id} : seulement {len(recent_bookings)} courses, score par défaut 0.75")
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
                delay_minutes = delay_seconds / 60.0

                total_count += 1

                # Considérer "à l'heure" si retard <= 5 minutes
                if delay_minutes <= 5:
                    on_time_count += 1

            if total_count == 0:
                return 0.75

            # Score = % de courses à l'heure
            punctuality_score = on_time_count / total_count

            logger.debug(f"[MLPredictor] Driver #{driver_id} : {on_time_count}/{total_count} courses à l'heure = {punctuality_score:.2f}")

            return punctuality_score

        except Exception as e:
            # En cas d'erreur (DB non accessible, etc.), retourner la valeur par défaut
            logger.warning(f"[MLPredictor] Erreur calcul ponctualité : {e}")
            return 0.75

    def _estimate_traffic_density(self, hour: int, day_of_week: int) -> float:
        """
        Estime la densité du trafic (0-1) basée sur l'heure et le jour.
        """
        # Heures de pointe: 7-9h et 17-19h en semaine
        is_weekday = day_of_week < 5
        is_morning_rush = 7 <= hour <= 9
        is_evening_rush = 17 <= hour <= 19

        if is_weekday and (is_morning_rush or is_evening_rush):
            return 0.8  # Trafic dense
        elif is_weekday and 9 < hour < 17:
            return 0.5  # Trafic moyen
        else:
            return 0.3  # Trafic faible

    def train_on_historical_data(
        self,
        historical_data: List[Dict[str, Any]],
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Entraîne le modèle sur des données historiques.
        
        Args:
            historical_data: Liste de dicts avec features + actual_delay_minutes
            save_model: Sauvegarder le modèle après entraînement
        
        Returns:
            Métriques d'entraînement
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for training. Install with: pip install scikit-learn")

        if not historical_data:
            raise ValueError("No historical data provided for training")

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

            feature_vector = [features.get(f, 0.0) for f in self.feature_names]
            X.append(feature_vector)
            y.append(actual_delay)

        if not X or not y:
            raise ValueError("No valid training data after preprocessing")

        X_array = np.array(X)
        y_array = np.array(y)

        logger.info(f"[MLPredictor] Training on {len(X)} samples with {len(self.feature_names)} features")

        # Standardiser les features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_array)

        # Entraîner le modèle
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_scaled, y_array)
        self.is_trained = True

        # Métriques
        train_score = self.model.score(X_scaled, y_array)

        metrics = {
            "samples_count": len(X),
            "features_count": len(self.feature_names),
            "r2_score": float(train_score),
            "feature_importance": {
                name: float(importance)
                for name, importance in zip(self.feature_names, self.model.feature_importances_, strict=False)
            }
        }

        logger.info(f"[MLPredictor] Training complete. R² score: {train_score:.3f}")

        # Sauvegarder
        if save_model:
            self.save_model()

        return metrics

    def predict_delay(
        self,
        booking: Any,
        driver: Any,
        current_time: datetime | None = None
    ) -> DelayPrediction:
        """
        Prédit le retard pour une assignation.
        
        Returns:
            DelayPrediction avec la prédiction et la confiance
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train_on_historical_data() first.")

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for prediction")

        # Extraire features
        features = self.extract_features(booking, driver, current_time)

        # Préparer le vecteur de features dans le bon ordre
        feature_vector = np.array([[features.get(f, 0.0) for f in self.feature_names]])

        # Standardiser
        if self.scaler:
            feature_vector_scaled = self.scaler.transform(feature_vector)
        else:
            feature_vector_scaled = feature_vector

        # Prédire
        predicted_delay = float(self.model.predict(feature_vector_scaled)[0])

        # Calculer la confiance (basée sur la variance des arbres)
        # Plus les arbres sont d'accord, plus la confiance est élevée
        tree_predictions = [tree.predict(feature_vector_scaled)[0] for tree in self.model.estimators_]
        std = np.std(tree_predictions)
        confidence = max(0.0, min(1.0, 1.0 - (std / 30.0)))  # Normaliser

        # Déterminer le niveau de risque
        abs_delay = abs(predicted_delay)
        if abs_delay < 5:
            risk_level = "low"
        elif abs_delay < 10:
            risk_level = "medium"
        else:
            risk_level = "high"

        # Facteurs contributifs (feature importance pour cette prédiction)
        contributing_factors = {
            name: float(features.get(name, 0.0) * self.model.feature_importances_[i])
            for i, name in enumerate(self.feature_names)
        }

        return DelayPrediction(
            booking_id=getattr(booking, "id", 0),
            predicted_delay_minutes=predicted_delay,
            confidence=confidence,
            risk_level=risk_level,
            contributing_factors=contributing_factors
        )

    def save_model(self) -> None:
        """Sauvegarde le modèle sur disque"""
        if not self.model or not self.scaler:
            raise RuntimeError("No model to save")

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "trained_at": datetime.now().isoformat(),
        }

        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"[MLPredictor] Model saved to {self.model_path}")

    def load_model(self) -> None:
        """Charge le modèle depuis le disque"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.is_trained = model_data["is_trained"]

        logger.info(
            f"[MLPredictor] Model loaded from {self.model_path} "
            f"(trained at: {model_data.get('trained_at', 'unknown')})"
        )


# Fonction helper pour faciliter l'utilisation
_global_predictor: DelayMLPredictor | None = None


def get_ml_predictor(model_path: str | None = None) -> DelayMLPredictor:
    """Récupère ou crée un prédicteur ML global"""
    global _global_predictor

    if _global_predictor is None:
        _global_predictor = DelayMLPredictor(model_path)

    return _global_predictor

