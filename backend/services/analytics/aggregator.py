# backend/services/analytics/aggregator.py
"""
Service d'agrégation de métriques.
Agrège les métriques quotidiennes et génère des statistiques par période.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict

from sqlalchemy import and_

from ext import db
from models import DailyStats, DispatchMetrics

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """Agrège les métriques pour différentes périodes"""

    def aggregate_daily_stats(self, company_id: int, day: date) -> DailyStats | None:
        """
        Agrège toutes les métriques d'un jour en statistiques journalières.
        À exécuter à la fin de chaque journée (tâche Celery).
        
        Args:
            company_id: ID de l'entreprise
            day: Date à agréger
            
        Returns:
            DailyStats créé/mis à jour ou None en cas d'erreur
        """

        try:
            # Récupérer toutes les métriques du jour
            metrics = DispatchMetrics.query.filter(
                and_(
                    DispatchMetrics.company_id == company_id,
                    DispatchMetrics.date == day
                )
            ).all()

            if not metrics:
                logger.warning(
                    f"[Aggregator] No metrics found for company {company_id} on {day}"
                )
                return None

            # Agréger les métriques
            total_bookings = sum(m.total_bookings for m in metrics)
            on_time_total = sum(m.on_time_bookings for m in metrics)

            # Calculs de moyennes pondérées
            delays = [m.average_delay_minutes for m in metrics if m.average_delay_minutes > 0]
            quality_scores = [m.quality_score for m in metrics if m.quality_score > 0]

            on_time_rate = (
                (on_time_total / total_bookings * 100) if total_bookings > 0 else 0.0
            )
            avg_delay = sum(delays) / len(delays) if delays else 0.0
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

            # Calculer les tendances (vs jour précédent)
            previous_day = day - timedelta(days=1)
            previous_stats = DailyStats.query.filter_by(
                company_id=company_id, date=previous_day
            ).first()

            if previous_stats:
                bookings_trend = (
                    ((total_bookings - previous_stats.total_bookings) /
                     previous_stats.total_bookings * 100)
                    if previous_stats.total_bookings > 0 else 0.0
                )
                delay_trend = (
                    ((avg_delay - previous_stats.avg_delay) /
                     previous_stats.avg_delay * 100)
                    if previous_stats.avg_delay > 0 else 0.0
                )
            else:
                bookings_trend = 0.0
                delay_trend = 0.0

            # Créer ou mettre à jour
            daily_stats = DailyStats.query.filter_by(
                company_id=company_id, date=day
            ).first()

            if daily_stats:
                # Mettre à jour
                daily_stats.total_bookings = total_bookings
                daily_stats.on_time_rate = on_time_rate
                daily_stats.avg_delay = avg_delay
                daily_stats.quality_score = avg_quality
                daily_stats.bookings_trend = bookings_trend
                daily_stats.delay_trend = delay_trend
                daily_stats.updated_at = datetime.utcnow()
            else:
                # Créer
                daily_stats = DailyStats(
                    company_id=company_id,
                    date=day,
                    total_bookings=total_bookings,
                    on_time_rate=on_time_rate,
                    avg_delay=avg_delay,
                    quality_score=avg_quality,
                    bookings_trend=bookings_trend,
                    delay_trend=delay_trend
                )
                db.session.add(daily_stats)

            db.session.commit()

            logger.info(
                f"[Aggregator] Daily stats aggregated for {day}: "
                f"Quality={avg_quality:.1f}, On-time={on_time_rate:.1f}%, "
                f"Avg delay={avg_delay:.1f}min"
            )

            return daily_stats

        except Exception as e:
            logger.error(
                f"[Aggregator] Failed to aggregate daily stats for {day}: {e}",
                exc_info=True
            )
            db.session.rollback()
            return None

    def get_period_analytics(
        self,
        company_id: int,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """
        Récupère les analytics pour une période donnée.
        Utilisé par l'API analytics pour générer les dashboards.
        
        Args:
            company_id: ID de l'entreprise
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            Dict contenant les analytics de la période
        """

        try:
            stats = DailyStats.query.filter(
                and_(
                    DailyStats.company_id == company_id,
                    DailyStats.date >= start_date,
                    DailyStats.date <= end_date
                )
            ).order_by(DailyStats.date).all()

            if not stats:
                logger.warning(
                    f"[Aggregator] No stats found for company {company_id} "
                    f"between {start_date} and {end_date}"
                )
                return {
                    "period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat(),
                        "days": 0
                    },
                    "summary": {
                        "total_bookings": 0,
                        "avg_on_time_rate": 0.0,
                        "avg_delay_minutes": 0.0,
                        "avg_quality_score": 0.0
                    },
                    "trends": [],
                    "daily_breakdown": []
                }

            # Résumé de la période
            total_bookings = sum(s.total_bookings for s in stats)
            avg_on_time_rate = sum(s.on_time_rate for s in stats) / len(stats)
            avg_delay = sum(s.avg_delay for s in stats) / len(stats)
            avg_quality = sum(s.quality_score for s in stats) / len(stats)

            # Tendances jour par jour
            trends = [
                {
                    "date": s.date.isoformat(),
                    "bookings": s.total_bookings,
                    "on_time_rate": round(s.on_time_rate, 2),
                    "avg_delay": round(s.avg_delay, 2),
                    "quality_score": round(s.quality_score, 2),
                    "bookings_trend": round(s.bookings_trend, 2),
                    "delay_trend": round(s.delay_trend, 2)
                }
                for s in stats
            ]

            return {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": len(stats)
                },
                "summary": {
                    "total_bookings": total_bookings,
                    "avg_on_time_rate": round(avg_on_time_rate, 2),
                    "avg_delay_minutes": round(avg_delay, 2),
                    "avg_quality_score": round(avg_quality, 2)
                },
                "trends": trends,
                "daily_breakdown": trends
            }

        except Exception as e:
            logger.error(
                f"[Aggregator] Failed to get period analytics: {e}",
                exc_info=True
            )
            return {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": 0
                },
                "summary": {},
                "trends": [],
                "error": str(e)
            }

    def get_weekly_summary(self, company_id: int, week_start: date) -> Dict[str, Any]:
        """
        Génère un résumé hebdomadaire (pour les rapports automatiques).
        
        Args:
            company_id: ID de l'entreprise
            week_start: Date de début de semaine (lundi)
            
        Returns:
            Dict contenant le résumé hebdomadaire
        """
        week_end = week_start + timedelta(days=6)
        analytics = self.get_period_analytics(company_id, week_start, week_end)

        # Enrichir avec des insights hebdomadaires
        if analytics["trends"]:
            best_day = max(analytics["trends"], key=lambda x: x["quality_score"])
            worst_day = min(analytics["trends"], key=lambda x: x["quality_score"])

            analytics["insights"] = {
                "best_day": {
                    "date": best_day["date"],
                    "quality_score": best_day["quality_score"]
                },
                "worst_day": {
                    "date": worst_day["date"],
                    "quality_score": worst_day["quality_score"]
                },
                "total_courses": analytics["summary"]["total_bookings"],
                "avg_punctuality": analytics["summary"]["avg_on_time_rate"]
            }

        return analytics


# Instance globale
_aggregator = MetricsAggregator()


def aggregate_daily_stats(company_id: int, day: date) -> DailyStats | None:
    """
    Helper function pour agréger les stats d'un jour.
    
    Usage:
        stats = aggregate_daily_stats(company_id, date.today())
    """
    return _aggregator.aggregate_daily_stats(company_id, day)


def get_period_analytics(
    company_id: int,
    start_date: date,
    end_date: date
) -> Dict[str, Any]:
    """
    Helper function pour récupérer les analytics d'une période.
    
    Usage:
        analytics = get_period_analytics(company_id, start, end)
    """
    return _aggregator.get_period_analytics(company_id, start_date, end_date)


def get_weekly_summary(company_id: int, week_start: date) -> Dict[str, Any]:
    """
    Helper function pour récupérer le résumé hebdomadaire.
    
    Usage:
        summary = get_weekly_summary(company_id, monday_date)
    """
    return _aggregator.get_weekly_summary(company_id, week_start)

