"""Génération de rapports quotidiens pour l'agent dispatch."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from flask import current_app

from models.autonomous_action import AutonomousAction

logger = logging.getLogger(__name__)


def generate_daily_report(company_id: int) -> Dict[str, Any]:
    """Génère un rapport quotidien pour l'agent.

    Args:
        company_id: ID de l'entreprise

    Returns:
        Rapport structuré avec KPIs, décisions clés, qualité, incidents

    """
    with current_app.app_context():
        # Date du jour
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Récupérer toutes les actions de la journée
        actions = (
            AutonomousAction.query.filter(
                AutonomousAction.company_id == company_id,
                AutonomousAction.created_at >= today_start,
                AutonomousAction.trigger_source == "agent_dispatch",
            )
            .order_by(AutonomousAction.created_at.desc())
            .all()
        )

        # KPIs
        LATE_THRESHOLD_MINUTES = -5
        P95_THRESHOLD = 20
        
        total_actions = len(actions)
        assigned = len([a for a in actions if a.action_type == "assign" and a.success])
        reassigned = len(
            [a for a in actions if a.action_type in ["assign", "reassign"] and a.success]
        )

        # Retards >5min (approximation depuis les actions)
        late_actions = [
            a
            for a in actions
            if a.expected_improvement_minutes
            and a.expected_improvement_minutes < LATE_THRESHOLD_MINUTES
        ]
        late_5min = len(late_actions)

        # Décisions clés (top 10 avec reasoning_brief)
        key_decisions = []
        for action in actions[:10]:
            if action.action_description:
                key_decisions.append(
                    {
                        "time": action.created_at.isoformat() if action.created_at else None,
                        "action": action.action_type,
                        "booking_id": action.booking_id,
                        "driver_id": action.driver_id,
                        "reasoning": action.action_description[:100],
                        "success": action.success,
                    }
                )

        # Qualité & SLA (approximation)
        # ETA p50/p95 : calculer depuis execution_time_ms si disponible
        execution_times = [
            a.execution_time_ms
            for a in actions
            if a.execution_time_ms and a.execution_time_ms > 0
        ]
        if execution_times:
            sorted_times = sorted(execution_times)
            p50 = sorted_times[len(sorted_times) // 2]
            p95 = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > P95_THRESHOLD else sorted_times[-1]
        else:
            p50 = 0
            p95 = 0

        # Équité : max gap (approximation)
        # Compter actions par driver
        driver_counts: Dict[int, int] = {}
        for action in actions:
            if action.driver_id and action.success:
                driver_counts[action.driver_id] = (
                    driver_counts.get(action.driver_id, 0) + 1
                )
        max_gap = (
            max(driver_counts.values()) - min(driver_counts.values())
            if driver_counts
            else 0
        )

        # Incidents & remédiations
        incidents = []
        for action in actions:
            if not action.success and action.error_message:
                incidents.append(
                    {
                        "time": action.created_at.isoformat() if action.created_at else None,
                        "action": action.action_type,
                        "error": action.error_message[:200],
                    }
                )

        # Prochaines actions (bullet points)
        next_actions = [
            "Continuer monitoring des urgences",
            "Optimiser selon santé OSRM",
            "Maintenir équité entre chauffeurs",
        ]

        return {
            "date": today_start.strftime("%Y-%m-%d"),
            "volume": {
                "jobs_total": total_actions,
                "assigned": assigned,
                "reassigned": reassigned,
                "late_5min": late_5min,
            },
            "decisions_cles": key_decisions,
            "qualite_sla": {
                "eta_p50_ms": p50,
                "eta_p95_ms": p95,
                "max_gap": max_gap,
                "cb_open_time_min": 0,  # TODO: Calculer depuis osrm_health logs
            },
            "incidents_remediations": incidents,
            "prochaines_actions": next_actions,
        }

