"""Service de monitoring ML pour tracking et analytics.

Fonctionnalités:
- Stockage prédictions ML
- Calcul métriques temps réel (MAE, R²)
- Détection anomalies
- Export rapports
"""
# pyright: reportAttributeAccessIssue=false
# datetime.utcnow() intentionnel, SQLAlchemy dynamic backref
import json
import logging
from datetime import timedelta
from typing import Any

import numpy as np

from db import db
from feature_flags import FeatureFlags
from models.ml_prediction import MLPrediction
from shared.time_utils import now_utc

logger = logging.getLogger(__name__)


class MLMonitoringService:
    """Service centralié pour monitoring ML."""

    @staticmethod
    def log_prediction(
        booking_id: int,
        driver_id: int | None,
        predicted_delay: float,
        confidence: float,
        risk_level: str,
        contributing_factors: dict[str, Any],
        prediction_time_ms: float,
        request_id: str | None = None,
        model_version: str | None = None,
    ) -> MLPrediction:
        """Enregistre une prédiction ML dans la base.

        Args:
            booking_id: ID du booking
            driver_id: ID du driver
            predicted_delay: Retard prédit (minutes)
            confidence: Score de confiance (0-1)
            risk_level: Niveau de risque (low/medium/high)
            contributing_factors: Facteurs contributifs (dict)
            prediction_time_ms: Temps de calcul (ms)
            request_id: ID de la requête
            model_version: Version du modèle

        Returns:
            MLPrediction créée

        """
        try:
            stats = FeatureFlags.get_stats()

            prediction = MLPrediction()
            prediction.booking_id = booking_id
            prediction.driver_id = driver_id
            prediction.request_id = request_id
            prediction.predicted_delay_minutes = predicted_delay
            prediction.confidence = confidence
            prediction.risk_level = risk_level
            prediction.contributing_factors = json.dumps(contributing_factors)
            prediction.model_version = model_version
            prediction.prediction_time_ms = prediction_time_ms
            prediction.feature_flag_enabled = stats["ml_enabled"]
            prediction.traffic_percentage = stats["ml_traffic_percentage"]

            db.session.add(prediction)
            db.session.commit()

            logger.debug(
                "[MLMonitoring] Logged prediction for booking %s: %s",
                booking_id, predicted_delay
            )

            return prediction

        except Exception as e:
            logger.error("[MLMonitoring] Failed to log prediction: %s", e)
            db.session.rollback()
            raise

    @staticmethod
    def update_actual_delay(booking_id: int, actual_delay: float) -> None:
        """Met à jour le retard réel pour une prédiction.

        Args:
            booking_id: ID du booking
            actual_delay: Retard réel (minutes)

        """
        try:
            # Récupérer la dernière prédiction pour ce booking
            prediction = (
                MLPrediction.query
                .filter_by(booking_id=booking_id)
                .order_by(MLPrediction.created_at.desc())
                .first()
            )

            if prediction:
                prediction.update_actual_delay(actual_delay)
                db.session.commit()

                logger.info(
                    "[MLMonitoring] Updated actual delay for booking %s: predicted=%s actual=%s error=%s",
                    booking_id, prediction.predicted_delay_minutes, actual_delay, prediction.prediction_error
                )
            else:
                logger.warning(
                    "[MLMonitoring] No prediction found for booking %s",
                    booking_id
                )

        except Exception as e:
            logger.error("[MLMonitoring] Failed to update actual delay: %s", e)
            db.session.rollback()

    @staticmethod
    def get_metrics(hours: int = 24) -> dict[str, Any]:
        """Calcule les métriques des dernières heures.

        Args:
            hours: Nombre d'heures à analyser

        Returns:
            Dict avec métriques MAE, R², count, etc.

        """
        try:
            cutoff = now_utc() - timedelta(hours=hours)

            # Récupérer prédictions avec résultats réels
            predictions = (
                MLPrediction.query
                .filter(MLPrediction.created_at >= cutoff)
                .filter(MLPrediction.actual_delay_minutes.isnot(None))
                .all()
            )

            if not predictions:
                return {
                    "period_hours": hours,
                    "count": 0,
                    "mae": None,
                    "rmse": None,
                    "r2": None,
                    "accuracy_rate": None,
                    "avg_confidence": None,
                    "avg_prediction_time_ms": None,
                }

            # Calculer métriques
            predicted = np.array([p.predicted_delay_minutes for p in predictions])
            actual = np.array([p.actual_delay_minutes for p in predictions])
            errors = np.abs(predicted - actual)

            mae = float(np.mean(errors))
            rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))

            # R² score
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            # Autres métriques
            accuracy_rate = sum(1 for p in predictions if p.is_accurate) / len(predictions)
            avg_confidence = float(np.mean([p.confidence for p in predictions]))
            avg_time_ms = float(np.mean([p.prediction_time_ms for p in predictions if p.prediction_time_ms]))

            return {
                "period_hours": hours,
                "count": len(predictions),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2": round(r2, 4),
                "accuracy_rate": round(accuracy_rate, 2),  # % predictions < 3min error
                "avg_confidence": round(avg_confidence, 2),
                "avg_prediction_time_ms": round(avg_time_ms, 1),
            }

        except Exception as e:
            logger.error("[MLMonitoring] Failed to calculate metrics: %s", e)
            return {"error": str(e)}

    @staticmethod
    def get_daily_metrics(days: int = 7) -> list[dict[str, Any]]:
        """Calcule les métriques par jour.

        Args:
            days: Nombre de jours à analyser

        Returns:
            Liste de dict avec métriques par jour

        """
        try:
            daily_metrics = []

            for day_offset in range(days):
                day_start = now_utc().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=day_offset)
                day_end = day_start + timedelta(days=1)

                # Prédictions de ce jour avec résultats réels
                predictions = (
                    MLPrediction.query
                    .filter(MLPrediction.created_at >= day_start)
                    .filter(MLPrediction.created_at < day_end)
                    .filter(MLPrediction.actual_delay_minutes.isnot(None))
                    .all()
                )

                if predictions:

                    predicted = np.array([p.predicted_delay_minutes for p in predictions])
                    actual = np.array([p.actual_delay_minutes for p in predictions])
                    errors = np.abs(predicted - actual)

                    mae = float(np.mean(errors))

                    ss_res = np.sum((actual - predicted) ** 2)
                    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

                    daily_metrics.append({
                        "date": day_start.date().isoformat(),
                        "count": len(predictions),
                        "mae": round(mae, 2),
                        "r2": round(r2, 4),
                        "accuracy_rate": sum(1 for p in predictions if p.is_accurate) / len(predictions),
                    })
                else:
                    daily_metrics.append({
                        "date": day_start.date().isoformat(),
                        "count": 0,
                        "mae": None,
                        "r2": None,
                        "accuracy_rate": None,
                    })

            return daily_metrics

        except Exception as e:
            logger.error("[MLMonitoring] Failed to calculate daily metrics: %s", e)
            return []

    @staticmethod
    def get_recent_predictions(limit: int = 100) -> list[dict[str, Any]]:
        """Récupère les prédictions récentes.

        Args:
            limit: Nombre max de prédictions

        Returns:
            Liste de prédictions

        """
        try:
            predictions = (
                MLPrediction.query
                .order_by(MLPrediction.created_at.desc())
                .limit(limit)
                .all()
            )

            return [p.to_dict() for p in predictions]

        except Exception as e:
            logger.error("[MLMonitoring] Failed to get recent predictions: %s", e)
            return []

    @staticmethod
    def detect_anomalies(threshold_mae: float = 5.0) -> list[dict[str, Any]]:
        """Détecte les anomalies (prédictions très imprécises).

        Args:
            threshold_mae: Seuil d'erreur pour anomalie (minutes)

        Returns:
            Liste des anomalies

        """
        try:
            anomalies = (
                MLPrediction.query
                .filter(MLPrediction.prediction_error > threshold_mae)
                .filter(MLPrediction.created_at >= now_utc() - timedelta(hours=24))
                .order_by(MLPrediction.prediction_error.desc())
                .limit(50)
                .all()
            )

            return [
                {
                    "booking_id": a.booking_id,
                    "predicted": a.predicted_delay_minutes,
                    "actual": a.actual_delay_minutes,
                    "error": a.prediction_error,
                    "confidence": a.confidence,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
                for a in anomalies
            ]

        except Exception as e:
            logger.error("[MLMonitoring] Failed to detect anomalies: %s", e)
            return []

    @staticmethod
    def get_summary() -> dict[str, Any]:
        """Résumé complet du système ML.

        Returns:
            Dict avec toutes les métriques importantes

        """
        try:
            # Métriques 24h
            metrics_24h = MLMonitoringService.get_metrics(hours=24)

            # Métriques 7 jours
            metrics_7d = MLMonitoringService.get_metrics(hours=24 * 7)

            # Feature flags
            ff_stats = FeatureFlags.get_stats()

            # Anomalies
            anomalies = MLMonitoringService.detect_anomalies()

            # Total prédictions
            total_predictions = MLPrediction.query.count()

            return {
                "total_predictions": total_predictions,
                "metrics_24h": metrics_24h,
                "metrics_7d": metrics_7d,
                "feature_flags": ff_stats,
                "anomalies_count": len(anomalies),
                "timestamp": now_utc().isoformat(),
            }

        except Exception as e:
            logger.error("[MLMonitoring] Failed to get summary: %s", e)
            return {"error": str(e)}

