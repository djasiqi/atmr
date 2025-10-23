# ruff: noqa: DTZ003, DTZ005, W293
# pyright: reportAttributeAccessIssue=false
"""
Service A/B Testing pour comparer ML vs Heuristique.
"""
import logging
import time
from datetime import datetime
from typing import Any

from shared.geo_utils import haversine_distance

logger = logging.getLogger(__name__)


class ABTestingService:
    """Service pour comparer ML vs Heuristique."""

    @staticmethod
    def run_ab_test(
        booking: Any,
        driver: Any,
        current_time: datetime | None = None
    ) -> dict[str, Any]:
        """
        Exécute test A/B : ML vs Heuristique.
        
        Args:
            booking: Booking à tester
            driver: Driver assigné
            current_time: Timestamp du test
            
        Returns:
            dict avec résultats comparatifs
        """
        if current_time is None:
            current_time = datetime.utcnow()

        logger.info(f"[AB Test] Running for booking {booking.id}, driver {driver.id}")

        # 1. Prédiction ML
        ml_start = time.time()
        ml_prediction = ABTestingService._run_ml_prediction(booking, driver, current_time)
        ml_time = (time.time() - ml_start) * 1000  # ms

        # 2. Prédiction Heuristique
        heuristic_start = time.time()
        heuristic_prediction = ABTestingService._run_heuristic_prediction(booking, driver)
        heuristic_time = (time.time() - heuristic_start) * 1000  # ms

        # 3. Comparaison
        result = {
            "booking_id": booking.id,
            "driver_id": driver.id,
            "test_timestamp": current_time.isoformat(),

            # ML
            "ml_delay_minutes": ml_prediction["delay_minutes"],
            "ml_confidence": ml_prediction.get("confidence", 0.0),
            "ml_risk_level": ml_prediction.get("risk_level", "unknown"),
            "ml_prediction_time_ms": ml_time,
            "ml_weather_factor": ml_prediction.get("weather_factor", 0.5),

            # Heuristique
            "heuristic_delay_minutes": heuristic_prediction["delay_minutes"],
            "heuristic_prediction_time_ms": heuristic_time,

            # Comparaison
            "difference_minutes": abs(ml_prediction["delay_minutes"] - heuristic_prediction["delay_minutes"]),
            "ml_faster": ml_time < heuristic_time,
            "speed_advantage_ms": heuristic_time - ml_time,
        }

        logger.info(
            f"[AB Test] Booking {booking.id}: "
            f"ML={ml_prediction['delay_minutes']:.2f}min ({ml_time:.1f}ms), "
            f"Heuristic={heuristic_prediction['delay_minutes']:.2f}min ({heuristic_time:.1f}ms)"
        )

        return result

    @staticmethod
    def _run_ml_prediction(
        booking: Any,
        driver: Any,
        current_time: datetime
    ) -> dict[str, Any]:
        """Exécute prédiction ML."""
        try:
            from services.unified_dispatch.ml_predictor import get_ml_predictor

            predictor = get_ml_predictor()
            prediction = predictor.predict_delay(booking, driver, current_time)

            return {
                "delay_minutes": prediction.predicted_delay_minutes,
                "confidence": prediction.confidence,
                "risk_level": prediction.risk_level,
                "weather_factor": prediction.contributing_factors.get("weather_factor", 0.5),
            }
        except Exception as e:
            logger.error(f"[AB Test] ML prediction failed: {e}", exc_info=True)
            return {
                "delay_minutes": 5.0,
                "confidence": 0.0,
                "risk_level": "error",
                "weather_factor": 0.5,
            }

    @staticmethod
    def _run_heuristic_prediction(booking: Any, driver: Any) -> dict[str, Any]:
        """
        Exécute prédiction heuristique simple.
        
        Basé sur distance Haversine : 0.5 min/km (30 km/h moyen).
        """
        try:
            pickup_lat = float(getattr(booking, "pickup_lat", 0) or 0)
            pickup_lon = float(getattr(booking, "pickup_lon", 0) or 0)
            dropoff_lat = float(getattr(booking, "dropoff_lat", 0) or 0)
            dropoff_lon = float(getattr(booking, "dropoff_lon", 0) or 0)

            if not all([pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]):
                return {"delay_minutes": 5.0}

            distance_km = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

            # Heuristique simple : 0.5 min/km + buffer 3 min
            delay_minutes = (distance_km * 0.5) + 3.0

            return {"delay_minutes": delay_minutes}
        except Exception as e:
            logger.error(f"[AB Test] Heuristic prediction failed: {e}", exc_info=True)
            return {"delay_minutes": 5.0}

    @staticmethod
    def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Calcule métriques agrégées des tests A/B.
        
        Args:
            results: Liste des résultats A/B
            
        Returns:
            dict avec métriques agrégées
        """
        if not results:
            return {}

        n = len(results)

        # Moyennes ML
        ml_delays = [r["ml_delay_minutes"] for r in results]
        ml_times = [r["ml_prediction_time_ms"] for r in results]
        ml_confidences = [r["ml_confidence"] for r in results]

        # Moyennes Heuristique
        heuristic_delays = [r["heuristic_delay_minutes"] for r in results]
        heuristic_times = [r["heuristic_prediction_time_ms"] for r in results]

        # Différences
        differences = [r["difference_minutes"] for r in results]
        ml_faster_count = sum(1 for r in results if r["ml_faster"])

        metrics = {
            "total_tests": n,

            # ML
            "ml_avg_delay": sum(ml_delays) / n,
            "ml_avg_time_ms": sum(ml_times) / n,
            "ml_avg_confidence": sum(ml_confidences) / n,

            # Heuristique
            "heuristic_avg_delay": sum(heuristic_delays) / n,
            "heuristic_avg_time_ms": sum(heuristic_times) / n,

            # Comparaison
            "avg_difference_minutes": sum(differences) / n,
            "ml_faster_percentage": (ml_faster_count / n) * 100,
            "avg_speed_advantage_ms": sum(r["speed_advantage_ms"] for r in results) / n,
        }

        logger.info(f"[AB Test] Metrics calculated for {n} tests")
        logger.info(f"  ML avg: {metrics['ml_avg_delay']:.2f} min ({metrics['ml_avg_time_ms']:.1f}ms)")
        logger.info(f"  Heuristic avg: {metrics['heuristic_avg_delay']:.2f} min ({metrics['heuristic_avg_time_ms']:.1f}ms)")
        logger.info(f"  ML faster: {metrics['ml_faster_percentage']:.1f}%")

        return metrics


def run_batch_ab_tests(booking_ids: list[int], driver_ids: list[int]) -> dict[str, Any]:
    """
    Exécute tests A/B en batch.
    
    Args:
        booking_ids: Liste IDs bookings
        driver_ids: Liste IDs drivers (appariés)
        
    Returns:
        dict avec résultats et métriques
    """
    from models.booking import Booking
    from models.driver import Driver

    if len(booking_ids) != len(driver_ids):
        raise ValueError("booking_ids et driver_ids doivent avoir même longueur")

    results = []
    current_time = datetime.utcnow()

    logger.info(f"[AB Test] Starting batch of {len(booking_ids)} tests")

    for booking_id, driver_id in zip(booking_ids, driver_ids, strict=False):
        try:
            booking = Booking.query.get(booking_id)
            driver = Driver.query.get(driver_id)

            if not booking or not driver:
                logger.warning(f"[AB Test] Skipping: booking {booking_id} or driver {driver_id} not found")
                continue

            result = ABTestingService.run_ab_test(booking, driver, current_time)
            results.append(result)
        except Exception as e:
            logger.error(f"[AB Test] Failed for booking {booking_id}, driver {driver_id}: {e}")

    metrics = ABTestingService.calculate_metrics(results)

    return {
        "results": results,
        "metrics": metrics,
        "timestamp": current_time.isoformat(),
        "total_tests": len(results),
    }

