"""Epic 4.1 - Modèle de prédiction de retard (ETA delta) avec gradient boosting.

Ce module implémente:
- Gradient boosting (XGBoost/LightGBM) pour prédiction de retard
- Features: heure, pluie, densité zone, historique chauffeur, distance OSRM, marge fenêtre
- Intégration RealtimeOptimizer: si P(retard)>p0 → Notify + Reassign candidates
- Critères: AUC > 0.75; réduction notifications tardives -25%
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import dynamique avec fallback
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb  # type: ignore[import-untyped]
    XGBOOST_AVAILABLE = True
    logger.info("[ETADelayModel] XGBoost disponible")
except ImportError:
    xgb = None  
    logger.warning(
        "[ETADelayModel] XGBoost non disponible. Installer avec: pip install xgboost"
    )

try:
    import lightgbm as lgb  # type: ignore[import-untyped]
    LIGHTGBM_AVAILABLE = True
    logger.info("[ETADelayModel] LightGBM disponible")
except ImportError:
    lgb = None  
    logger.warning(
        "[ETADelayModel] LightGBM non disponible. Installer avec: pip install lightgbm"
    )

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    mean_absolute_error = None  
    mean_squared_error = None
    roc_auc_score = None
    StandardScaler = None
    logger.error(
        "[ETADelayModel] scikit-learn requis. Installer avec: pip install scikit-learn"
    )

# Constantes
DELAY_THRESHOLD_MINUTES = 5
AUC_TARGET = 0.75
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Rush hours
MORNING_RUSH_START = 7
MORNING_RUSH_END = 9
EVENING_RUSH_START = 17
EVENING_RUSH_END = 19
WEEKEND_THRESHOLD = 5  # vendredi (index 4) = 0-4 = semaine
MIN_BOOKINGS_FOR_STATS = 10  # Minimum bookings pour statistiques significatives

# Zone density
ZONE_RADIUS_KM = 2  # Rayon de zone en km pour calcul densité
ZONE_DENSITY_MAX_BOOKINGS = 50  # Au-delà de ce nombre = max densité
ZONE_DENSITY_LOOKBACK_DAYS = 7  # Regarder 7 derniers jours

@dataclass
class ETADelayPrediction:
    """Prédiction de retard pour un booking."""
    
    booking_id: int
    predicted_delay_minutes: float
    probability_delay: float  # Probabilité P(retard > 5 min)
    confidence: float  # 0.0 - 1.0
    risk_level: str  # "low", "medium", "high"
    contributing_factors: Dict[str, float]
    model_type: str  # "xgboost", "lightgbm", "random_forest"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "booking_id": self.booking_id,
            "predicted_delay_minutes": round(self.predicted_delay_minutes, 2),
            "probability_delay": round(self.probability_delay, 3),
            "confidence": round(self.confidence, 3),
            "risk_level": self.risk_level,
            "contributing_factors": self.contributing_factors,
            "model_type": self.model_type,
        }


class ETADelayModel:
    """Modèle ML pour prédiction de retard ETA avec gradient boosting.
    
    Utilise XGBoost ou LightGBM pour prédire:
    - Le retard en minutes (régression)
    - La probabilité P(retard > seuil) (classification)
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",  # "xgboost" ou "lightgbm"
        model_path: str | None = None,
        delay_threshold: int = DELAY_THRESHOLD_MINUTES
    ):
        """Initialise le modèle.
        
        Args:
            model_type: Type de modèle ("xgboost" ou "lightgbm")
            model_path: Chemin vers modèle sauvegardé
            delay_threshold: Seuil de retard en minutes pour classification
        """
        super().__init__()
        self.model_type = model_type
        self.model_path = model_path or f"data/ml/models/eta_delay_{model_type}.pkl"
        self.delay_threshold = delay_threshold
        
        # Modèles
        self.regression_model: Any = None  # Pour prédire minutes de retard
        self.classification_model: Any = None  # Pour prédire P(retard > seuil)
        
        # Preprocessing
        self.scaler: Any | None = None
        self.scaler_params: Dict[str, Any] | None = None
        self.feature_names: List[str] = []
        
        self.is_trained = False
        
        # Charger si existe
        if Path(self.model_path).exists():
            self.load_model()
        else:
            logger.info("[ETADelayModel] Aucun modèle pré-entraîné trouvé")
    
    def extract_features(
        self,
        booking: Any,
        driver: Any | None = None,
        current_time: datetime | None = None
    ) -> Dict[str, float]:
        """Extrait les features pour prédiction.
        
        Features:
        - time_of_day, day_of_week, month, is_weekend
        - distance_km, duration_seconds
        - is_medical, is_urgent, booking_priority
        - driver_punctuality_score (historique)
        - traffic_density, weather_factor
        - zone_density (densité zone géographique)
        - window_margin (marge fenêtre horaire)
        
        Args:
            booking: Objet Booking
            driver: Objet Driver (optionnel)
            current_time: Timestamp actuel
            
        Returns:
            Dict de features
        """
        if current_time is None:
            current_time = datetime.now()
        
        scheduled_time = getattr(booking, "scheduled_time", current_time) or current_time
        
        # === Features temporelles ===
        time_of_day = float(scheduled_time.hour if scheduled_time else current_time.hour)
        day_of_week = float(scheduled_time.weekday() if scheduled_time else current_time.weekday())
        month = float(scheduled_time.month if scheduled_time else current_time.month)
        is_weekend = 1 if day_of_week >= WEEKEND_THRESHOLD else 0
        is_morning_rush = 1 if MORNING_RUSH_START <= time_of_day <= MORNING_RUSH_END else 0
        is_evening_rush = 1 if EVENING_RUSH_START <= time_of_day <= EVENING_RUSH_END else 0
        
        # Encodage cyclique
        hour_sin = np.sin(2 * np.pi * time_of_day / 24)
        hour_cos = np.cos(2 * np.pi * time_of_day / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # === Features géographiques ===
        try:
            from shared.geo_utils import haversine_distance
            
            pickup_lat = float(getattr(booking, "pickup_lat", 0) or 0)
            pickup_lon = float(getattr(booking, "pickup_lon", 0) or 0)
            dropoff_lat = float(getattr(booking, "dropoff_lat", 0) or 0)
            dropoff_lon = float(getattr(booking, "dropoff_lon", 0) or 0)
            
            if all([pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]):
                distance_km = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
            else:
                distance_km = float(getattr(booking, "distance_meters", 0) or 0) / 1000
        except Exception:
            distance_km = 0
        
        duration_seconds = float(getattr(booking, "duration_seconds", 0) or distance_km * 420)  # ~7 min/km
        
        # === Features booking ===
        is_medical = 1 if getattr(booking, "medical_facility", None) else 0
        is_urgent = 1 if getattr(booking, "is_urgent", False) else 0
        is_round_trip = 1 if getattr(booking, "is_round_trip", False) else 0
        booking_priority = 0.8 if (is_medical or is_urgent) else 0.5
        
        # === Features driver ===
        driver_punctuality_score = 0.75  # Default
        driver_experience_level = 1  # Default intermédiaire
        
        if driver:
            try:
                from sqlalchemy import and_

                from models import Booking, BookingStatus
                
                driver_id = getattr(driver, "id", None)
                if driver_id:
                    cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
                    recent_bookings = Booking.query.filter(
                        and_(
                            Booking.driver_id == driver_id,
                            Booking.status == BookingStatus.COMPLETED,  # type: ignore[arg-type]
                            Booking.completed_at >= cutoff_date
                        )
                    ).order_by(Booking.completed_at.desc()).limit(50).all()
                    
                    if len(recent_bookings) >= MIN_BOOKINGS_FOR_STATS:
                        on_time_count = 0
                        total_delays = []
                        
                        for b in recent_bookings:
                            scheduled = getattr(b, "scheduled_time", None)
                            actual_pickup = getattr(b, "actual_pickup_time", None) or getattr(b, "completed_at", None)
                            
                            if scheduled and actual_pickup:
                                delay_seconds = (actual_pickup - scheduled).total_seconds()
                                delay_minutes = delay_seconds / 60
                                total_delays.append(abs(delay_minutes))
                                
                                if abs(delay_minutes) <= DELAY_THRESHOLD_MINUTES:
                                    on_time_count += 1
                        
                        if total_delays:
                            driver_punctuality_score = on_time_count / len(recent_bookings)
                            avg_delay = np.mean(total_delays)
                            
                            if avg_delay < DELAY_THRESHOLD_MINUTES:
                                driver_experience_level = 2  # Expert
                            elif avg_delay < DELAY_THRESHOLD_MINUTES + 1:
                                driver_experience_level = 1  # Intermédiaire
                            else:
                                driver_experience_level = 0  # Novice
            except Exception as e:
                logger.debug("[ETADelayModel] Erreur calcul driver: %s", e)
        
        # === Features trafic et météo ===
        # Traffic density
        if MORNING_RUSH_START <= time_of_day <= MORNING_RUSH_END or EVENING_RUSH_START <= time_of_day <= EVENING_RUSH_END:
            traffic_density = 0.8
        elif time_of_day in [6, 10, 16, 20]:
            traffic_density = 0.6
        else:
            traffic_density = 0.3
        
        # Weather (intégrer API météo si disponible)
        try:
            from services.weather_service import get_weather_factor
            pickup_lat = float(getattr(booking, "pickup_lat", 0) or 0)
            pickup_lon = float(getattr(booking, "pickup_lon", 0) or 0)
            weather_factor = get_weather_factor(pickup_lat, pickup_lon) if pickup_lat and pickup_lon else 0.5
        except Exception:
            weather_factor = 0.5
        
        # === Zone density - Calculer depuis données historiques ===
        zone_density = self._calculate_zone_density(booking)
        
        # === Window margin (marge fenêtre horaire) ===
        try:
            window_start = getattr(booking, "window_start", None)
            window_end = getattr(booking, "window_end", None)
            window_margin = ((window_end - window_start).total_seconds() / 3600) if (window_start and window_end) else 0.5  # heures ou default 30 min
        except Exception:
            window_margin = 0.5
        
        return {
            # Temporelles
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "is_morning_rush": is_morning_rush,
            "is_evening_rush": is_evening_rush,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            
            # Géographiques
            "distance_km": distance_km,
            "duration_seconds": duration_seconds,
            
            # Booking
            "is_medical": is_medical,
            "is_urgent": is_urgent,
            "is_round_trip": is_round_trip,
            "booking_priority": booking_priority,
            
            # Driver
            "driver_punctuality_score": driver_punctuality_score,
            "driver_experience_level": driver_experience_level,
            
            # Environnement
            "traffic_density": traffic_density,
            "weather_factor": weather_factor,
            "zone_density": zone_density,
            "window_margin": window_margin,
            
            # Interactions
            "distance_x_traffic": distance_km * traffic_density,
            "distance_x_weather": distance_km * weather_factor,
            "medical_x_traffic": is_medical * traffic_density,
        }
    
    def predict(
        self,
        booking: Any,
        driver: Any | None = None,
        current_time: datetime | None = None
    ) -> ETADelayPrediction:
        """Prédit le retard pour un booking.
        
        Args:
            booking: Objet Booking
            driver: Objet Driver (optionnel)
            current_time: Timestamp actuel
            
        Returns:
            ETADelayPrediction
        """
        if not self.is_trained:
            logger.warning("[ETADelayModel] Modèle non entraîné, utilisation fallback")
            return self._fallback_prediction(booking, driver)
        
        try:
            # Extraire features
            features = self.extract_features(booking, driver, current_time)
            
            # Créer DataFrame
            if not self.feature_names:
                # Créer feature_names depuis features
                self.feature_names = sorted(features.keys())
            
            feature_df = pd.DataFrame([features])
            
            # Prédire retard (régression)
            predicted_delay = float(self.regression_model.predict(feature_df)[0])
            
            # Prédire probabilité de retard (classification)
            if self.classification_model:
                prob_delay = float(self.classification_model.predict_proba(feature_df)[0][1])
            else:
                # Fallback: probabilité basée sur retard prédit
                prob_delay = min(1.0, max(0.0, predicted_delay / self.delay_threshold))
            
            # Confiance (basée sur variance des prédictions)
            if self.model_type == "xgboost" and hasattr(self.regression_model, "get_booster"):
                try:
                    # XGBoost: contrib feature importance
                    _ = self.regression_model.get_booster().get_score(importance_type="total_gain")
                    # TODO: utiliser feature importance pour calculer confidence
                    confidence = 0.8  # Placeholder
                except Exception:
                    confidence = 0.7
            else:
                confidence = 0.7
            
            # Niveau de risque
            abs_delay = abs(predicted_delay)
            if abs_delay < DELAY_THRESHOLD_MINUTES - 2:
                risk_level = "low"
            elif abs_delay < DELAY_THRESHOLD_MINUTES + 2:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            # Facteurs contributifs (top 5)
            contributing_factors = self._calculate_contributing_factors(features)
            
            return ETADelayPrediction(
                booking_id=getattr(booking, "id", 0),
                predicted_delay_minutes=predicted_delay,
                probability_delay=prob_delay,
                confidence=confidence,
                risk_level=risk_level,
                contributing_factors=contributing_factors,
                model_type=self.model_type
            )
            
        except Exception as e:
            logger.error("[ETADelayModel] Erreur prédiction: %s", e)
            return self._fallback_prediction(booking, driver)
    
    def _fallback_prediction(
        self,
        booking: Any,
        _driver: Any | None
    ) -> ETADelayPrediction:
        """Prédiction de fallback en cas d'erreur."""
        # Estimation simple basée sur distance
        try:
            from shared.geo_utils import haversine_distance
            
            pickup_lat = float(getattr(booking, "pickup_lat", 0) or 0)
            pickup_lon = float(getattr(booking, "pickup_lon", 0) or 0)
            dropoff_lat = float(getattr(booking, "dropoff_lat", 0) or 0)
            dropoff_lon = float(getattr(booking, "dropoff_lon", 0) or 0)
            
            if all([pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]):
                distance_km = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
                predicted_delay = distance_km * 0.3  # Simplifié
            else:
                predicted_delay = 3
        except Exception:
            predicted_delay = 3
        
        return ETADelayPrediction(
            booking_id=getattr(booking, "id", 0),
            predicted_delay_minutes=predicted_delay,
            probability_delay=0.5,
            confidence=0.3,
            risk_level="medium",
            contributing_factors={"fallback": 1.0},
            model_type="fallback"
        )
    
    def _calculate_contributing_factors(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calcule les facteurs contributifs pour une prédiction."""
        # TODO: Implémenter analyse feature importance
        return {
            "traffic_density": features.get("traffic_density", 0),
            "distance_km": features.get("distance_km", 0),
            "weather_factor": features.get("weather_factor", 0),
        }
    
    def _calculate_zone_density(self, booking: Any) -> float:
        """Calcule la densité de bookings dans la zone géographique.
        
        Args:
            booking: Objet Booking avec pickup_lat/pickup_lon
            
        Returns:
            Densité normalisée entre 0.0 et 1.0
        """
        try:
            pickup_lat = float(getattr(booking, "pickup_lat", 0) or 0)
            pickup_lon = float(getattr(booking, "pickup_lon", 0) or 0)
            
            # Si pas de coordonnées, retourner densité neutre
            if not (pickup_lat and pickup_lon):
                return 0.5
            
            from sqlalchemy import and_

            from models import Booking, BookingStatus
            
            # Définir zone: ~2km radius (environ 0.018 degrés latitude)
            # 1 degré latitude ≈ 111 km
            zone_radius_deg = ZONE_RADIUS_KM / 111.0
            zone_radius_lat = zone_radius_deg  # km en latitude
            zone_radius_lon = zone_radius_deg / np.cos(np.radians(pickup_lat))  # Ajusté pour longitude
            
            lat_min = pickup_lat - zone_radius_lat
            lat_max = pickup_lat + zone_radius_lat
            lon_min = pickup_lon - zone_radius_lon
            lon_max = pickup_lon + zone_radius_lon
            
            # Compter bookings dans cette zone (7 derniers jours)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=ZONE_DENSITY_LOOKBACK_DAYS)
            
            bookings_in_zone = Booking.query.filter(
                and_(
                    Booking.pickup_lat.between(lat_min, lat_max),
                    Booking.pickup_lon.between(lon_min, lon_max),
                    Booking.status.in_([BookingStatus.COMPLETED, BookingStatus.ACCEPTED]),
                    Booking.scheduled_time >= cutoff_date
                )
            ).count()
            
            # Normaliser: 0 = vide, 1 = très dense
            normalized_density = min(1.0, bookings_in_zone / ZONE_DENSITY_MAX_BOOKINGS)
            
            logger.debug(
                "[ETADelayModel] Zone density: %d bookings in zone (lat:%.4f, lon:%.4f) → density=%.2f",
                bookings_in_zone, pickup_lat, pickup_lon, normalized_density
            )
            
            return normalized_density
            
        except Exception as e:
            logger.debug("[ETADelayModel] Erreur calcul zone density: %s", e)
            return 0.5  # Fallback: densité neutre
    
    def train(
        self,
        training_data: List[Dict[str, Any]],
        save_model: bool = True
    ) -> Dict[str, Any]:
        """Entraîne le modèle sur données historiques.
        
        Args:
            training_data: Liste de dicts avec 'features' et 'actual_delay_minutes'
            save_model: Sauvegarder le modèle
            
        Returns:
            Métriques d'entraînement
        """
        if not XGBOOST_AVAILABLE and not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "XGBoost ou LightGBM requis. Installer avec: pip install xgboost lightgbm"
            )
        
        if not training_data:
            raise ValueError("Pas de données d'entraînement")
        
        # Préparer données
        X = []
        y_regression = []  # retard en minutes
        y_classification = []  # retard > seuil (0 ou 1)
        
        for record in training_data:
            features = record.get("features", {})
            actual_delay = record.get("actual_delay_minutes", 0)
            
            if not features:
                continue
            
            if not self.feature_names:
                self.feature_names = sorted(features.keys())
            
            feature_vector = [features.get(f, 0) for f in self.feature_names]
            X.append(feature_vector)
            
            y_regression.append(actual_delay)
            y_classification.append(1 if actual_delay > self.delay_threshold else 0)
        
        if not X:
            raise ValueError("Pas de données valides après preprocessing")
        
        X_array = np.array(X)
        y_reg_array = np.array(y_regression)
        y_class_array = np.array(y_classification)
        
        logger.info(
            "[ETADelayModel] Entraînement sur %d échantillons avec %d features",
            len(X), len(self.feature_names)
        )
        
        # Normaliser
        if StandardScaler is None:
            raise ImportError("scikit-learn requis pour l'entraînement")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Entraîner modèle
        if self.model_type == "xgboost" and XGBOOST_AVAILABLE and xgb is not None:
            self.regression_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.regression_model.fit(X_scaled, y_reg_array)
            
            self.classification_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.classification_model.fit(X_scaled, y_class_array)
            
        elif self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE and lgb is not None:
            self.regression_model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.regression_model.fit(X_scaled, y_reg_array)
            
            self.classification_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.classification_model.fit(X_scaled, y_class_array)
        
        else:
            raise ValueError(f"Modèle {self.model_type} non disponible")
        
        # Métriques
        train_score_reg = self.regression_model.score(X_scaled, y_reg_array)
        
        train_pred_class = self.classification_model.predict_proba(X_scaled)[:, 1]
        if roc_auc_score is None or mean_absolute_error is None or mean_squared_error is None:
            raise ImportError("scikit-learn requis pour les métriques")
        train_auc = roc_auc_score(y_class_array, train_pred_class)
        
        mae = mean_absolute_error(y_reg_array, self.regression_model.predict(X_scaled))
        rmse = np.sqrt(mean_squared_error(y_reg_array, self.regression_model.predict(X_scaled)))
        
        metrics = {
            "samples_count": len(X),
            "features_count": len(self.feature_names),
            "mae": mae,
            "rmse": rmse,
            "r2_score": train_score_reg,
            "auc": train_auc,
            "model_type": self.model_type,
        }
        
        logger.info(
            "[ETADelayModel] Entraînement terminé: MAE=%.2f, RMSE=%.2f, R²=%.3f, AUC=%.3f",
            mae, rmse, train_score_reg, train_auc
        )
        
        self.is_trained = True
        
        # Sauvegarder
        if save_model:
            self.save_model()
        
        return metrics
    
    def save_model(self) -> None:
        """Sauvegarde le modèle."""
        if not self.regression_model:
            raise RuntimeError("Aucun modèle à sauvegarder")
        
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "regression_model": self.regression_model,
            "classification_model": self.classification_model,
            "feature_names": self.feature_names,
            "scaler_params": self.scaler_params,
            "is_trained": self.is_trained,
            "model_type": self.model_type,
            "delay_threshold": self.delay_threshold,
            "trained_at": datetime.now().isoformat(),
        }
        
        with Path(self.model_path).open("wb") as f:
            pickle.dump(model_data, f)
        
        logger.info("[ETADelayModel] Modèle sauvegardé: %s", self.model_path)
    
    def load_model(self) -> None:
        """Charge le modèle."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Modèle introuvable: {self.model_path}")
        
        try:
            with Path(self.model_path).open("rb") as f:
                model_data = pickle.load(f)
            
            self.regression_model = model_data["regression_model"]
            self.classification_model = model_data.get("classification_model", None)
            self.feature_names = model_data["feature_names"]
            self.scaler_params = model_data.get("scaler_params", None)
            self.is_trained = model_data.get("is_trained", False)
            self.model_type = model_data.get("model_type", "xgboost")
            self.delay_threshold = model_data.get("delay_threshold", DELAY_THRESHOLD_MINUTES)
            
            logger.info(
                "[ETADelayModel] Modèle chargé: %s (entraîné à: %s)",
                self.model_path,
                model_data.get("trained_at", "inconnu")
            )
        except Exception as e:
            logger.error("[ETADelayModel] Erreur chargement modèle: %s", e)
            self.is_trained = False


# Instance globale
_global_eta_model: ETADelayModel | None = None


def get_eta_delay_model() -> ETADelayModel:
    """Récupère l'instance globale du modèle ETA."""
    global _global_eta_model  # noqa: PLW0603
    
    if _global_eta_model is None:
        # Préférer XGBoost si disponible
        if XGBOOST_AVAILABLE:
            _global_eta_model = ETADelayModel(model_type="xgboost")
        elif LIGHTGBM_AVAILABLE:
            _global_eta_model = ETADelayModel(model_type="lightgbm")
        else:
            logger.error("[ETADelayModel] Aucun modèle GBM disponible")
            _global_eta_model = ETADelayModel(model_type="fallback")
    
    return _global_eta_model


