# backend/services/analytics/metrics_collector.py
"""
Service de collecte de métriques pour le système de dispatch.
Collecte et sauvegarde les métriques après chaque dispatch run.
"""

import logging
from datetime import date

from ext import db
from models import Assignment, Booking, BookingStatus, DispatchMetrics, Driver

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collecte et sauvegarde les métriques de dispatch"""

    def collect_dispatch_metrics(
        self,
        dispatch_run_id: int,
        company_id: int,
        day: date
    ) -> DispatchMetrics | None:
        """
        Collecte les métriques après un dispatch run.
        Appelé automatiquement à la fin d'un dispatch.
        
        Args:
            dispatch_run_id: ID du dispatch run
            company_id: ID de l'entreprise
            day: Date du dispatch
            
        Returns:
            DispatchMetrics créé ou None en cas d'erreur
        """

        try:
            # Récupérer toutes les assignations du dispatch
            assignments = Assignment.query.filter_by(
                dispatch_run_id=dispatch_run_id
            ).all()

            if not assignments:
                logger.warning(
                    f"[MetricsCollector] No assignments found for dispatch run {dispatch_run_id}"
                )
                return None

            bookings = [a.booking for a in assignments if a.booking]

            if not bookings:
                logger.warning(
                    f"[MetricsCollector] No bookings found for dispatch run {dispatch_run_id}"
                )
                return None

            # Calculer les métriques de base
            total_bookings = len(bookings)
            on_time = sum(1 for b in bookings if self._is_on_time(b))
            delayed = sum(1 for b in bookings if self._is_delayed(b))
            cancelled = sum(1 for b in bookings
                           if b.status == BookingStatus.CANCELED)

            # Métriques de retard
            delays = [self._calculate_delay(b) for b in bookings]
            avg_delay = sum(delays) / len(delays) if delays else 0.0
            max_delay = max(delays) if delays else 0
            total_delay = sum(delays)

            # Métriques chauffeurs
            drivers_in_assignments = set(a.driver_id for a in assignments if a.driver_id)
            total_drivers = Driver.query.filter_by(
                company_id=company_id,
                is_active=True
            ).count()
            active_drivers = len(drivers_in_assignments)
            avg_per_driver = (
                total_bookings / active_drivers if active_drivers > 0 else 0.0
            )

            # Métriques de distance
            total_distance = sum(
                self._estimate_booking_distance(b) for b in bookings
            )
            avg_distance = (
                total_distance / total_bookings if total_bookings > 0 else 0.0
            )

            # Score de qualité (0-100)
            quality_score = self._calculate_quality_score(
                on_time, total_bookings, avg_delay, cancelled
            )

            # Créer l'enregistrement
            metrics = DispatchMetrics(
                company_id=company_id,
                dispatch_run_id=dispatch_run_id,
                date=day,
                total_bookings=total_bookings,
                on_time_bookings=on_time,
                delayed_bookings=delayed,
                cancelled_bookings=cancelled,
                average_delay_minutes=avg_delay,
                max_delay_minutes=max_delay,
                total_delay_minutes=total_delay,
                total_drivers=total_drivers,
                active_drivers=active_drivers,
                avg_bookings_per_driver=avg_per_driver,
                total_distance_km=total_distance,
                avg_distance_per_booking=avg_distance,
                suggestions_generated=0,  # Sera mis à jour par realtime_optimizer
            suggestions_applied=0,
            quality_score=quality_score,
            extra_data={
                "dispatch_mode": "auto",
                "optimization_engine": "heuristics",
                "bookings_ids": [b.id for b in bookings[:10]],  # Limité à 10 pour pas surcharger
            }
            )

            db.session.add(metrics)
            db.session.commit()

            logger.info(
                f"[MetricsCollector] Collected metrics for dispatch run {dispatch_run_id}: "
                f"Quality={quality_score:.1f}, On-time={on_time}/{total_bookings}, "
                f"Avg delay={avg_delay:.1f}min"
            )

            return metrics

        except Exception as e:
            logger.error(
                f"[MetricsCollector] Failed to collect metrics for dispatch run {dispatch_run_id}: {e}",
                exc_info=True
            )
            db.session.rollback()
            return None

    def _is_on_time(self, booking: Booking) -> bool:
        """Détermine si un booking est à l'heure (<= 5 min de retard)"""
        delay = self._calculate_delay(booking)
        return delay <= 5

    def _is_delayed(self, booking: Booking) -> bool:
        """Détermine si un booking est en retard (> 5 min)"""
        delay = self._calculate_delay(booking)
        return delay > 5

    def _calculate_delay(self, booking: Booking) -> int:
        """
        Calcule le retard en minutes.
        Pour le moment, retourne 0 car les retards réels sont calculés en temps réel.
        Cette métrique sera mise à jour par le realtime_optimizer.
        """
        # Les retards sont calculés en temps réel par le realtime_optimizer
        # Ici on ne calcule que les métriques de base (nombre de courses, etc.)
        return 0

    def _estimate_booking_distance(self, booking: Booking) -> float:
        """Estime la distance d'un booking en km (Haversine)"""
        try:
            import math

            if not all([
                booking.pickup_lat, booking.pickup_lon,
                booking.dropoff_lat, booking.dropoff_lon
            ]):
                return 0.0

            # Calcul Haversine direct
            R = 6371.0  # Rayon de la Terre en km
            lat1, lon1 = booking.pickup_lat, booking.pickup_lon
            lat2, lon2 = booking.dropoff_lat, booking.dropoff_lon

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

        except Exception as e:
            logger.warning(f"[MetricsCollector] Failed to calculate distance for booking {booking.id}: {e}")
            return 0.0

    def _calculate_quality_score(
        self,
        on_time: int,
        total: int,
        avg_delay: float,
        cancelled: int
    ) -> float:
        """
        Calcule un score de qualité global (0-100).
        
        Formule:
        - 50% basé sur le taux de ponctualité
        - 30% basé sur le retard moyen
        - 20% basé sur le taux d'annulation
        
        Args:
            on_time: Nombre de bookings à l'heure
            total: Nombre total de bookings
            avg_delay: Retard moyen en minutes
            cancelled: Nombre de bookings annulés
            
        Returns:
            Score entre 0 et 100
        """
        if total == 0:
            return 0.0

        # Taux de ponctualité (0-50 points)
        on_time_rate = on_time / total
        on_time_score = on_time_rate * 50

        # Retard moyen (0-30 points, décroissant)
        # 0 min = 30 pts, 15+ min = 0 pts
        delay_score = max(0, 30 - (avg_delay / 15 * 30))

        # Taux d'annulation (0-20 points)
        cancel_rate = cancelled / total
        cancel_score = max(0, 20 - (cancel_rate * 100))

        total_score = on_time_score + delay_score + cancel_score

        return min(100, max(0, total_score))

    def update_suggestions_count(
        self,
        dispatch_run_id: int,
        generated: int | None = None,
        applied: int | None = None
    ) -> bool:
        """
        Met à jour le nombre de suggestions générées/appliquées pour un dispatch run.
        Utilisé par le realtime_optimizer.
        
        Args:
            dispatch_run_id: ID du dispatch run
            generated: Nombre de suggestions générées (optionnel)
            applied: Nombre de suggestions appliquées (optionnel)
            
        Returns:
            True si succès, False sinon
        """
        try:
            metrics = DispatchMetrics.query.filter_by(
                dispatch_run_id=dispatch_run_id
            ).first()

            if not metrics:
                logger.warning(
                    f"[MetricsCollector] No metrics found for dispatch run {dispatch_run_id}"
                )
                return False

            if generated is not None:
                metrics.suggestions_generated = generated

            if applied is not None:
                metrics.suggestions_applied = applied

            db.session.commit()

            logger.debug(
                f"[MetricsCollector] Updated suggestions count for dispatch run {dispatch_run_id}: "
                f"generated={generated}, applied={applied}"
            )

            return True

        except Exception as e:
            logger.error(
                f"[MetricsCollector] Failed to update suggestions count: {e}",
                exc_info=True
            )
            db.session.rollback()
            return False


# Instance globale
_metrics_collector = MetricsCollector()


def collect_dispatch_metrics(
    dispatch_run_id: int,
    company_id: int,
    day: date
) -> DispatchMetrics | None:
    """
    Helper function pour collecter les métriques.
    
    Usage:
        metrics = collect_dispatch_metrics(run_id, company_id, date.today())
    """
    return _metrics_collector.collect_dispatch_metrics(dispatch_run_id, company_id, day)


def update_suggestions_count(
    dispatch_run_id: int,
    generated: int | None = None,
    applied: int | None = None
) -> bool:
    """
    Helper function pour mettre à jour le nombre de suggestions.
    
    Usage:
        update_suggestions_count(run_id, generated=5, applied=3)
    """
    return _metrics_collector.update_suggestions_count(dispatch_run_id, generated, applied)

