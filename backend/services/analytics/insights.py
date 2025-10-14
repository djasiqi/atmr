# backend/services/analytics/insights.py
"""
Service de génération d'insights intelligents.
Analyse les tendances et génère des recommandations.
"""

import logging
from datetime import date, timedelta
from typing import List, Dict, Any
from models import DailyStats
from sqlalchemy import and_

logger = logging.getLogger(__name__)


def generate_insights(company_id: int, analytics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Génère des insights intelligents basés sur les analytics.
    
    Args:
        company_id: ID de l'entreprise
        analytics: Dict contenant les analytics de période
        
    Returns:
        Liste d'insights avec type, message et priorité
    """
    
    insights = []
    
    if not analytics or not analytics.get("trends"):
        return insights
    
    summary = analytics.get("summary", {})
    trends = analytics["trends"]
    
    # Insight 1: Taux de ponctualité
    on_time_rate = summary.get("avg_on_time_rate", 0)
    if on_time_rate < 70:
        insights.append({
            "type": "warning",
            "category": "punctuality",
            "priority": "high",
            "title": "Taux de ponctualité faible",
            "message": f"Votre taux de ponctualité ({on_time_rate:.1f}%) est inférieur à 70%. "
                      "Recommandation : Analysez les causes de retards récurrents.",
            "action": "view_delays"
        })
    elif on_time_rate >= 90:
        insights.append({
            "type": "success",
            "category": "punctuality",
            "priority": "low",
            "title": "Excellente ponctualité",
            "message": f"Votre taux de ponctualité ({on_time_rate:.1f}%) est excellent ! Continuez ainsi.",
            "action": None
        })
    
    # Insight 2: Retard moyen
    avg_delay = summary.get("avg_delay_minutes", 0)
    if avg_delay > 15:
        insights.append({
            "type": "warning",
            "category": "delays",
            "priority": "high",
            "title": "Retard moyen élevé",
            "message": f"Le retard moyen est de {avg_delay:.1f} minutes. "
                      "Cela impacte la satisfaction client.",
            "action": "optimize_planning"
        })
    
    # Insight 3: Tendances (évolution sur la période)
    if len(trends) >= 7:  # Au moins une semaine de données
        recent_quality = [t["quality_score"] for t in trends[-7:]]
        previous_quality = [t["quality_score"] for t in trends[:7]]
        
        if recent_quality and previous_quality:
            recent_avg = sum(recent_quality) / len(recent_quality)
            previous_avg = sum(previous_quality) / len(previous_quality)
            evolution = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
            
            if evolution > 10:
                insights.append({
                    "type": "success",
                    "category": "trend",
                    "priority": "medium",
                    "title": "Amélioration continue",
                    "message": f"Votre score de qualité s'améliore (+{evolution:.1f}% sur la période). Excellente progression !",
                    "action": None
                })
            elif evolution < -10:
                insights.append({
                    "type": "warning",
                    "category": "trend",
                    "priority": "high",
                    "title": "Dégradation de la qualité",
                    "message": f"Votre score de qualité diminue ({evolution:.1f}% sur la période). Action requise.",
                    "action": "review_operations"
                })
    
    # Insight 4: Jours problématiques
    if trends:
        # Identifier les jours avec le plus de retards
        high_delay_days = [t for t in trends if t["avg_delay"] > 20]
        if len(high_delay_days) > len(trends) / 3:  # Plus d'1/3 des jours
            insights.append({
                "type": "info",
                "category": "pattern",
                "priority": "medium",
                "title": "Retards fréquents",
                "message": f"{len(high_delay_days)} jours sur {len(trends)} ont des retards >20 min. "
                          "Analysez les patterns (jour de la semaine, heure, etc.)",
                "action": "analyze_patterns"
            })
    
    # Insight 5: Volume de courses
    total_bookings = summary.get("total_bookings", 0)
    days_count = len(trends)
    if days_count > 0:
        avg_daily_bookings = total_bookings / days_count
        if avg_daily_bookings < 10:
            insights.append({
                "type": "info",
                "category": "volume",
                "priority": "low",
                "title": "Volume faible",
                "message": f"Moyenne de {avg_daily_bookings:.1f} courses/jour. "
                          "Opportunité de croissance.",
                "action": None
            })
        elif avg_daily_bookings > 50:
            insights.append({
                "type": "success",
                "category": "volume",
                "priority": "low",
                "title": "Volume élevé",
                "message": f"Moyenne de {avg_daily_bookings:.1f} courses/jour. Activité soutenue !",
                "action": None
            })
    
    # Insight 6: Score de qualité global
    quality_score = summary.get("avg_quality_score", 0)
    if quality_score >= 85:
        insights.append({
            "type": "success",
            "category": "quality",
            "priority": "low",
            "title": "Score de qualité excellent",
            "message": f"Score global de {quality_score:.1f}/100. Vous dépassez les standards de l'industrie !",
            "action": None
        })
    elif quality_score < 60:
        insights.append({
            "type": "warning",
            "category": "quality",
            "priority": "critical",
            "title": "Score de qualité faible",
            "message": f"Score global de {quality_score:.1f}/100. Des améliorations urgentes sont nécessaires.",
            "action": "improvement_plan"
        })
    
    logger.info(f"[Insights] Generated {len(insights)} insights for company {company_id}")
    
    return insights


def detect_patterns(company_id: int, lookback_days: int = 30) -> Dict[str, Any]:
    """
    Détecte des patterns récurrents dans les données (jour de la semaine, etc.).
    
    Args:
        company_id: ID de l'entreprise
        lookback_days: Nombre de jours à analyser
        
    Returns:
        Dict contenant les patterns détectés
    """
    
    
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    
    stats = DailyStats.query.filter(
        and_(
            DailyStats.company_id == company_id,
            DailyStats.date >= start_date,
            DailyStats.date <= end_date
        )
    ).all()
    
    if not stats:
        return {"patterns": [], "message": "Pas assez de données pour détecter des patterns"}
    
    # Grouper par jour de la semaine (0=lundi, 6=dimanche)
    by_weekday = {i: [] for i in range(7)}
    for s in stats:
        weekday = s.date.weekday()
        by_weekday[weekday].append({
            "delay": s.avg_delay,
            "on_time_rate": s.on_time_rate,
            "bookings": s.total_bookings
        })
    
    # Calculer les moyennes par jour
    weekday_names = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    weekday_analysis = []
    
    for weekday, data in by_weekday.items():
        if not data:
            continue
        
        avg_delay = sum(d["delay"] for d in data) / len(data)
        avg_on_time = sum(d["on_time_rate"] for d in data) / len(data)
        avg_bookings = sum(d["bookings"] for d in data) / len(data)
        
        weekday_analysis.append({
            "weekday": weekday,
            "weekday_name": weekday_names[weekday],
            "avg_delay": round(avg_delay, 2),
            "avg_on_time_rate": round(avg_on_time, 2),
            "avg_bookings": round(avg_bookings, 1),
            "sample_size": len(data)
        })
    
    # Identifier les patterns
    patterns = []
    
    # Jour avec le plus de retards
    if weekday_analysis:
        worst_day = max(weekday_analysis, key=lambda x: x["avg_delay"])
        if worst_day["avg_delay"] > 10:
            patterns.append({
                "type": "high_delay_day",
                "message": f"{worst_day['weekday_name']} a systématiquement plus de retards (moy: {worst_day['avg_delay']:.1f} min)",
                "recommendation": f"Ajoutez du temps buffer ou des chauffeurs supplémentaires le {worst_day['weekday_name']}"
            })
        
        # Jour avec le plus de courses
        busiest_day = max(weekday_analysis, key=lambda x: x["avg_bookings"])
        if busiest_day["avg_bookings"] > 0:
            patterns.append({
                "type": "busy_day",
                "message": f"{busiest_day['weekday_name']} est le jour le plus chargé (moy: {busiest_day['avg_bookings']:.0f} courses)",
                "recommendation": "Assurez-vous d'avoir assez de chauffeurs disponibles"
            })
    
    return {
        "patterns": patterns,
        "weekday_analysis": weekday_analysis
    }

