"""Politique de sécurité et garde-fous pour l'agent.

Vérifie les limites de sécurité avant actions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

from flask import current_app

from models import Booking, Client
from models.autonomous_action import AutonomousAction

logger = logging.getLogger(__name__)


class SafetyPolicy:
    """Vérifie les limites de sécurité avant actions."""

    def __init__(self, company_id: int):
        """Initialise la politique de sécurité.

        Args:
            company_id: ID de l'entreprise

        """
        super().__init__()
        self.company_id = company_id

    def check_action(self, action_type: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Vérifie si une action peut être effectuée.

        Args:
            action_type: Type d'action ('assign', 'reassign', etc.)
            context: Contexte de l'action (booking_id, driver_id, etc.)

        Returns:
            (can_proceed, reason)

        """
        # Vérifier limites horaires/jour
        can_proceed, reason = self._check_rate_limits(action_type)
        if not can_proceed:
            return False, reason

        # Vérifier VIP clients
        booking_id = context.get("job_id") or context.get("booking_id")
        if booking_id:
            can_proceed, reason = self._check_vip_client(booking_id)
            if not can_proceed:
                return False, reason

        # Vérifier réassignations multiples
        if action_type in ["assign", "reassign"] and booking_id:
            driver_id = context.get("driver_id")
            if driver_id:
                can_proceed, reason = self._check_reassignment_limits(booking_id, int(driver_id))
                if not can_proceed:
                    return False, reason

        return True, "OK"

    def _check_rate_limits(self, action_type: str) -> Tuple[bool, str]:
        """Vérifie les limites de taux.

        Args:
            action_type: Type d'action

        Returns:
            (can_proceed, reason)

        """
        # Limites globales (augmentées pour mode fully_auto)
        max_per_hour = 200  # Augmenté de 50 à 200 pour permettre plus d'actions automatiques
        max_per_day = 2000  # Augmenté de 500 à 2000 pour mode fully_auto

        # Limites par type (augmentées pour mode fully_auto)
        action_limits = {
            "assign": {"per_hour": 150, "per_day": 1500},  # Augmenté pour fully_auto
            "reassign": {"per_hour": 50, "per_day": 500},  # Augmenté pour fully_auto
            "reoptimize": {"per_hour": 20, "per_day": 100},  # Augmenté pour fully_auto
        }

        # Vérifier limite globale horaire
        current_hour_count = AutonomousAction.count_actions_last_hour(self.company_id)
        if current_hour_count >= max_per_hour:
            return False, (
                f"Limite horaire globale atteinte: {current_hour_count}/{max_per_hour} actions/h. "
                "Validation manuelle requise."
            )

        # Vérifier limite globale journalière
        current_day_count = AutonomousAction.count_actions_today(self.company_id)
        if current_day_count >= max_per_day:
            return False, (
                f"Limite journalière globale atteinte: {current_day_count}/{max_per_day} actions/jour. "
                "Système basculé en mode manuel jusqu'à demain."
            )

        # Vérifier limites spécifiques par type
        if action_type in action_limits:
            limits = action_limits[action_type]
            type_limit_hour = limits.get("per_hour")
            type_limit_day = limits.get("per_day")

            if type_limit_hour:
                type_count_hour = AutonomousAction.count_actions_last_hour(self.company_id, action_type)
                if type_count_hour >= type_limit_hour:
                    return False, (
                        f"Limite horaire pour '{action_type}' atteinte: {type_count_hour}/{type_limit_hour} actions/h."
                    )

            if type_limit_day:
                type_count_day = AutonomousAction.count_actions_today(self.company_id, action_type)
                if type_count_day >= type_limit_day:
                    return False, (
                        f"Limite journalière pour '{action_type}' atteinte: "
                        f"{type_count_day}/{type_limit_day} actions/jour."
                    )

        return True, "OK"

    def _check_vip_client(self, booking_id: int) -> Tuple[bool, str]:
        """Vérifie si un client VIP nécessite approbation.

        Args:
            booking_id: ID du booking

        Returns:
            (can_proceed, reason)

        """
        with current_app.app_context():
            try:
                booking = Booking.query.get(booking_id)
                if not booking or not booking.client_id:
                    return True, "OK"

                client = Client.query.get(booking.client_id)
                if not client:
                    return True, "OK"

                # Vérifier si client VIP (marqué is_vip ou client hospitalier)
                # Note: À adapter selon votre modèle Client
                is_vip = getattr(client, "is_vip", False) or getattr(client, "is_hospital", False)

                if is_vip:
                    return False, (f"Client VIP (ID: {client.id}) - Approbation manuelle requise")

                return True, "OK"
            except Exception as e:
                logger.warning("[SafetyPolicy] Error checking VIP client: %s", e)
                # En cas d'erreur, autoriser (sécurité par défaut)
                return True, "OK"

    def _check_reassignment_limits(
        self,
        booking_id: int,
        driver_id: int,  # noqa: ARG002
    ) -> Tuple[bool, str]:
        """Vérifie les limites de réassignation pour un client.

        Args:
            booking_id: ID du booking
            driver_id: ID du nouveau driver

        Returns:
            (can_proceed, reason)

        """
        if not booking_id:
            return True, "OK"

        with current_app.app_context():
            try:
                booking = Booking.query.get(booking_id)
                if not booking or not booking.client_id:
                    return True, "OK"

                # Compter réassignations pour ce client dans les 30 dernières minutes
                thirty_min_ago = datetime.now(timezone.utc) - timedelta(minutes=30)

                recent_reassignments = AutonomousAction.query.filter(
                    AutonomousAction.company_id == self.company_id,
                    AutonomousAction.booking_id == booking_id,
                    AutonomousAction.action_type.in_(["assign", "reassign"]),
                    AutonomousAction.created_at >= thirty_min_ago,
                    AutonomousAction.success == True,  # noqa: E712
                ).count()

                max_reassignments_per_30min = 3
                if recent_reassignments >= max_reassignments_per_30min:
                    return False, (
                        f"Trop de réassignations pour ce booking ({recent_reassignments} "
                        f"en 30 min) - Limite: {max_reassignments_per_30min}"
                    )

                return True, "OK"
            except Exception as e:
                logger.warning("[SafetyPolicy] Error checking reassignment limits: %s", e)
                # En cas d'erreur, autoriser (sécurité par défaut)
                return True, "OK"
