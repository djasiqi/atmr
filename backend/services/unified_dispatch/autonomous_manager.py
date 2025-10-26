"""Gestionnaire central pour le dispatch autonome.
Gère les 3 modes et orchestre les actions automatiques.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from models import Company, DispatchMode
from services.unified_dispatch.reactive_suggestions import Suggestion, apply_suggestion
from shared.time_utils import now_local

logger = logging.getLogger(__name__)


class AutonomousDispatchManager:
    """Gestionnaire central du dispatch autonome.
    Décide quelles actions peuvent être effectuées selon le mode de l'entreprise.
    Modes de fonctionnement :
    - MANUAL : Aucune automatisation, tout est manuel
    - SEMI_AUTO : Dispatch sur demande, suggestions non appliquées
    - FULLY_AUTO : Système 100% autonome avec application automatique.
    """

    def __init__(self, company_id: int):  # pyright: ignore[reportMissingSuperCall]
        """Initialise le gestionnaire pour une entreprise.

        Args:
            company_id: ID de l'entreprise
        Raises:
            ValueError: Si l'entreprise n'existe pas

        """
        self.company_id = company_id
        self.company = Company.query.get(company_id)
        if not self.company:
            msg = f"Company {company_id} not found"
            raise ValueError(msg)

        self.mode = self.company.dispatch_mode
        self.config = self.company.get_autonomous_config()

        logger.debug(
            "[AutonomousManager] Initialized for company %s with mode: %s",
            company_id, self.mode.value
        )

    def should_run_autorun(self) -> bool:
        """Détermine si le dispatch automatique périodique doit s'exécuter.

        Returns:
            True si le dispatch automatique doit tourner

        """
        if self.mode == DispatchMode.MANUAL:
            # En mode manuel : jamais d'autorun
            return False

        if self.mode == DispatchMode.SEMI_AUTO:
            # En semi-auto : seulement si explicitement activé dans la config
            return self.config["auto_dispatch"]["enabled"]

        # En fully auto : toujours actif
        return self.mode == DispatchMode.FULLY_AUTO

    def should_run_realtime_optimizer(self) -> bool:
        """Détermine si le RealtimeOptimizer doit tourner en continu.

        Returns:
            True si le monitoring temps réel doit être actif

        """
        if self.mode == DispatchMode.MANUAL:
            # Pas de monitoring en mode manuel
            return False

        # En semi-auto et fully-auto : selon la configuration
        return self.config["realtime_optimizer"]["enabled"]

    def can_auto_apply_suggestion(self, suggestion: Suggestion) -> bool:
        """Détermine si une suggestion peut être appliquée automatiquement.

        Args:
            suggestion: La suggestion à évaluer
        Returns:
            True si l'application automatique est autorisée

        """
        result = False
        
        # Seulement en mode fully auto
        if self.mode != DispatchMode.FULLY_AUTO:
            return result

        # La suggestion elle-même doit être marquée auto-applicable
        if not suggestion.auto_applicable:
            return result

        # Vérifier les règles d'application automatique selon le type d'action
        rules = self.config["auto_apply_rules"]

        if suggestion.action == "notify_customer":
            # Notifications clients : selon config
            result = rules.get("customer_notifications", True)
        elif suggestion.action == "adjust_time":
            # Ajustements de temps : uniquement si en dessous du seuil de sécurité
            delay = suggestion.additional_data.get(
                "delay_minutes", 0) if suggestion.additional_data else 0
            threshold = self.config["safety_limits"]["require_approval_delay_minutes"]

            # Si retard > seuil : validation manuelle requise
            if abs(delay) <= threshold:
                result = rules.get("minor_time_adjustments", False)
        elif suggestion.action == "reassign":
            # Réassignations : toujours avec précaution (désactivé par défaut)
            result = rules.get("reassignments", False)
        elif suggestion.action == "redistribute":
            # Redistribution de charges : jamais auto (trop critique)
            result = False
        else:
            # Par défaut : pas d'application automatique
            result = False

        return result

    def should_trigger_reoptimization(
        self,
        trigger_type: str,
        context: Dict[str, Any]
    ) -> bool:
        """Détermine si une ré-optimisation automatique doit être déclenchée.

        Args:
            trigger_type: Type de déclencheur ('delay', 'driver_unavailable', 'better_driver_available')
            context: Contexte avec les détails (delay_minutes, booking_id, etc.)

        Returns:
            True si la ré-optimisation doit être lancée

        """
        # Seulement en mode fully auto
        if self.mode != DispatchMode.FULLY_AUTO:
            return False

        triggers_config = self.config["re_optimize_triggers"]

        if trigger_type == "delay":
            # Retard détecté : comparer au seuil
            delay_minutes = context.get("delay_minutes", 0)
            threshold = triggers_config.get("delay_threshold_minutes", 15)
            return delay_minutes >= threshold

        if trigger_type == "driver_unavailable":
            # Chauffeur devient indisponible : selon config
            return triggers_config.get("driver_became_unavailable", True)

        if trigger_type == "better_driver_available":
            # Meilleur chauffeur disponible : vérifier le gain minimal
            gain_minutes = context.get("gain_minutes", 0)
            threshold = triggers_config.get(
                "better_driver_available_gain_minutes", 10)
            return gain_minutes >= threshold

        return False

    def check_safety_limits(self, action_type: str) -> tuple[bool, str]:
        """Vérifie que les limites de sécurité ne sont pas dépassées.
        Implémente un rate limiting à plusieurs niveaux :
        - Limite globale par heure (toutes actions confondues)
        - Limite globale par jour
        - Limites spécifiques par type d'action
        Args:
            action_type: Type d'action ('notify', 'reassign', 'adjust_time', etc.).

        Returns:
            Tuple (can_proceed, reason)
                - can_proceed: True si l'action peut être effectuée
                - reason: Explication si bloqué

        """
        from models.autonomous_action import AutonomousAction

        # Récupérer les limites de sécurité depuis la config
        limits = self.config["safety_limits"]

        # 1. Vérifier limite globale horaire
        max_per_hour = limits.get("max_auto_actions_per_hour", 50)
        current_hour_count = AutonomousAction.count_actions_last_hour(
            self.company_id
        )

        if current_hour_count >= max_per_hour:
            return False, (
                f"Limite horaire globale atteinte: {current_hour_count}/{max_per_hour} actions/h. "
                f"Validation manuelle requise."
            )

        # 2. Vérifier limite globale journalière
        max_per_day = limits.get("max_auto_actions_per_day", 500)
        current_day_count = AutonomousAction.count_actions_today(
            self.company_id
        )

        if current_day_count >= max_per_day:
            return False, (
                f"Limite journalière globale atteinte: {current_day_count}/{max_per_day} actions/jour. "
                f"Système basculé en mode manuel jusqu'à demain."
            )

        # 3. Vérifier limites spécifiques par type d'action
        action_limits = limits.get("action_type_limits", {})

        if action_type in action_limits:
            type_limit_hour = action_limits[action_type].get("per_hour")
            if type_limit_hour:
                type_count_hour = AutonomousAction.count_actions_last_hour(
                    self.company_id,
                    action_type
                )

                if type_count_hour >= type_limit_hour:
                    return False, (
                        f"Limite horaire pour '{action_type}' atteinte: "
                        f"{type_count_hour}/{type_limit_hour} actions/h."
                    )

            type_limit_day = action_limits[action_type].get("per_day")
            if type_limit_day:
                type_count_day = AutonomousAction.count_actions_today(
                    self.company_id,
                    action_type
                )

                if type_count_day >= type_limit_day:
                    return False, (
                        f"Limite journalière pour '{action_type}' atteinte: "
                        f"{type_count_day}/{type_limit_day} actions/jour."
                    )

        # Toutes les vérifications passées
        return True, "OK"

    def process_opportunities(
        self,
        opportunities: List[Any],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Traite une liste d'opportunités d'optimisation.
        Applique automatiquement celles qui sont autorisées selon le mode et la config
        Args:
            opportunities: Liste d'OptimizationOpportunity à traiter
            dry_run: Si True, simule sans appliquer (pour tests).

        Returns:
            Statistiques des actions effectuées

        """
        stats = {
            "total_opportunities": len(opportunities),
            "auto_applied": 0,
            "manual_required": 0,
            "blocked_by_limits": 0,
            "errors": 0,
            "actions": []
        }

        for opp in opportunities:
            for suggestion in opp.suggestions:
                # Vérifier si auto-applicable
                if not self.can_auto_apply_suggestion(suggestion):
                    stats["manual_required"] += 1
                    logger.info(
                        "[AutonomousManager] Suggestion requires manual approval: %s (company=%s, mode=%s)",
                        suggestion.action, self.company_id, self.mode.value
                    )
                    continue

                # Vérifier les limites de sécurité
                can_proceed, reason = self.check_safety_limits(
                    suggestion.action)
                if not can_proceed:
                    stats["blocked_by_limits"] += 1
                    logger.warning(
                        "[AutonomousManager] Action blocked by safety limit: %s (company=%s, reason=%s)",
                        suggestion.action, self.company_id, reason
                    )
                    continue

                # Appliquer la suggestion
                try:
                    if not dry_run:
                        import json
                        import time

                        from db import db as database

                        start_time = time.time()
                        result = apply_suggestion(
                            suggestion, self.company_id, dry_run=False)
                        execution_time_ms = (time.time() - start_time) * 1000

                        if result.get("success"):
                            stats["auto_applied"] += 1
                            stats["actions"].append({
                                "action": suggestion.action,
                                "booking_id": suggestion.booking_id,
                                "driver_id": suggestion.driver_id,
                                "applied_at": now_local().isoformat(),
                                "result": "success",
                                "message": suggestion.message
                            })
                            logger.info(
                                "[AutonomousManager] ✅ Auto-applied: %s for booking %s (company=%s)",
                                suggestion.action, suggestion.booking_id, self.company_id
                            )

                            # Logger l'action dans la table autonomous_action
                            # (audit trail)
                            from models.autonomous_action import AutonomousAction
                            action_record = AutonomousAction()
                            action_record.company_id = self.company_id
                            action_record.booking_id = suggestion.booking_id
                            action_record.driver_id = suggestion.driver_id
                            action_record.action_type = suggestion.action
                            action_record.action_description = suggestion.message
                            action_record.action_data = json.dumps({
                                "suggestion": suggestion.to_dict() if hasattr(suggestion, "to_dict") else str(suggestion),
                                "result": result
                            })
                            action_record.success = True
                            action_record.execution_time_ms = execution_time_ms
                            action_record.confidence_score = getattr(suggestion, "confidence", None)
                            action_record.expected_improvement_minutes = getattr(suggestion, "expected_gain", None)
                            action_record.trigger_source = "autonomous_manager"
                            database.session.add(action_record)
                            database.session.commit()

                        else:
                            stats["errors"] += 1
                            logger.error(
                                "[AutonomousManager] ❌ Failed to apply: %s (error=%s)",
                                suggestion.action, result.get("error")
                            )

                            # Logger l'échec aussi (pour monitoring)
                            from models.autonomous_action import AutonomousAction
                            action_record = AutonomousAction()
                            action_record.company_id = self.company_id
                            action_record.booking_id = suggestion.booking_id
                            action_record.driver_id = suggestion.driver_id
                            action_record.action_type = suggestion.action
                            action_record.action_description = suggestion.message
                            action_record.action_data = json.dumps({
                                "suggestion": suggestion.to_dict() if hasattr(suggestion, "to_dict") else str(suggestion),
                                "error": result.get("error")
                            })
                            action_record.success = False
                            action_record.error_message = result.get("error")
                            action_record.execution_time_ms = execution_time_ms
                            database.session.add(action_record)
                            database.session.commit()
                    else:
                        stats["auto_applied"] += 1
                        logger.info(
                            "[AutonomousManager] [DRY RUN] Would auto-apply: %s (company=%s)",
                            suggestion.action, self.company_id
                        )

                except Exception:
                    stats["errors"] += 1
                    logger.exception(
                        "[AutonomousManager] Exception while applying suggestion: %s (company=%s)",
                        suggestion.action, self.company_id
                    )

        logger.info(
            "[AutonomousManager] Processed %d opportunities for company %s: %d auto-applied, %d manual, %d blocked, %d errors",
            stats["total_opportunities"],
            self.company_id,
            stats["auto_applied"],
            stats["manual_required"],
            stats["blocked_by_limits"],
            stats["errors"]
        )

        return stats


def get_manager_for_company(company_id: int) -> AutonomousDispatchManager:
    """Factory pour créer un gestionnaire autonome pour une entreprise.

    Args:
        company_id: ID de l'entreprise
    Returns:
        Instance de AutonomousDispatchManager
    Raises:
        ValueError: Si l'entreprise n'existe pas

    """
    return AutonomousDispatchManager(company_id)
