#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false

"""Service d'alertes proactives pour prédiction de retards et explicabilité RL.

Ce service analyse les risques de retard et génère des alertes préventives
avec explications détaillées des décisions RL.

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List

import numpy as np

# Constantes pour éviter les valeurs magiques
PREDICTED_DELAY_MINUTES_ZERO = 0
PREDICTED_DELAY_MINUTES_THRESHOLD = 5
TIME_REMAINING_THRESHOLD = 15
DISTANCE_THRESHOLD = 20
DRIVER_LOAD_THRESHOLD = 3
PRIORITY_THRESHOLD = 4
PROBABILITY_ZERO = 0
PROBABILITY_HIGH_RISK = 0.7
PROBABILITY_MEDIUM_RISK = 0.5
PROBABILITY_CRITICAL = 0.8
PROBABILITY_WARNING = 0.6
PROBABILITY_LOW = 0.4
DRIVER_LOAD_WARNING = 2
CURRENT_DISTANCE_THRESHOLD = 10

# Imports conditionnels pour éviter les erreurs
try:
    from services.notification_service import NotificationService
except ImportError:
    NotificationService = None

try:
    from services.unified_dispatch.ml_predictor import MLPredictor
except ImportError:
    MLPredictor = None

logger = logging.getLogger(__name__)


class ProactiveAlertsService:
    """Service d'alertes proactives avec explicabilité RL.

    Features:
    - Prédiction de risque de retard via delay_predictor.pkl
    - Explicabilité des décisions RL (top-K features, règles métier)
    - Système de debounce anti-spam
    - Alertes temps réel via Socket.IO
    - Intégration avec notification_service
    """

    def __init__(self, notification_service=None, delay_predictor=None):
        """Initialise le service d'alertes proactives.

        Args:
            notification_service: Service de notification optionnel (pour injection de dépendances dans les tests)
            delay_predictor: Prédicteur de retard optionnel (pour injection de dépendances dans les tests)
        """
        super().__init__()
        self.notification_service = notification_service or (NotificationService() if NotificationService else None)
        self.ml_predictor = MLPredictor() if MLPredictor else None

        # Seuils configurables
        self.delay_risk_thresholds = {
            "low": 0.3,  # 30% - Alerte info
            "medium": 0.6,  # 60% - Alerte warning
            "high": 0.8,  # 80% - Alerte critical
        }

        # Système de debounce avancé
        self.alert_history: Dict[str, Dict[str, Any]] = {}
        self.debounce_minutes = 15  # 15 min entre alertes pour même booking
        self.max_alerts_per_hour = 10  # Limite d'alertes par heure par booking
        self.alert_frequency_tracker: Dict[str, List[datetime]] = {}

        # Modèle de prédiction de retard
        self.delay_predictor = delay_predictor
        if self.delay_predictor is None:
            self._load_delay_predictor()

        # Cache pour explicabilité
        self.explanation_cache: Dict[str, Dict[str, Any]] = {}

        logger.info("[ProactiveAlerts] Service initialisé avec seuils: %s", self.delay_risk_thresholds)

    def _load_delay_predictor(self) -> None:
        """Charge le modèle de prédiction de retard."""
        try:
            # Utiliser le DelayMLPredictor existant
            from services.unified_dispatch.ml_predictor import DelayMLPredictor

            self.delay_predictor = DelayMLPredictor()

            if self.delay_predictor.is_trained:
                logger.info("[ProactiveAlerts] ✅ Modèle delay_predictor chargé via DelayMLPredictor")
            else:
                logger.warning("[ProactiveAlerts] ⚠️ Modèle delay_predictor non entraîné")
                self.delay_predictor = None

        except Exception as e:
            logger.error("[ProactiveAlerts] ❌ Erreur chargement delay_predictor: %s", e)
            self.delay_predictor = None

    def check_delay_risk(
        self, booking: Dict[str, Any], driver: Dict[str, Any], current_time: datetime | None = None
    ) -> Dict[str, Any]:
        """Analyse le risque de retard pour une assignation.

        Args:
            booking: Données du booking
            driver: Données du chauffeur
            current_time: Temps actuel (optionnel)

        Returns:
            Dictionnaire avec probabilité de retard et explication

        """
        if current_time is None:
            current_time = datetime.now(UTC)

        try:
            # Calculer probabilité de retard
            delay_probability = self._calculate_delay_probability(booking, driver, current_time)

            # Déterminer niveau de risque
            risk_level = self._determine_risk_level(delay_probability)

            # Générer explication
            explanation = self._generate_explanation(booking, driver, delay_probability, risk_level)

            # Calculer métriques additionnelles
            metrics = self._calculate_additional_metrics(booking, driver, current_time)

            result = {
                "booking_id": booking.get("id"),
                "driver_id": driver.get("id"),
                "delay_probability": delay_probability,
                "risk_level": risk_level,
                "explanation": explanation,
                "metrics": metrics,
                "timestamp": current_time.isoformat(),
                "should_alert": risk_level in ["medium", "high"],
            }

            logger.debug(
                "[ProactiveAlerts] Risque retard Booking %s → Driver %s: %.2f%% (%s)",
                booking.get("id"),
                driver.get("id"),
                delay_probability * 100,
                risk_level,
            )

            return result

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur analyse risque retard: %s", e)
            return {
                "booking_id": booking.get("id"),
                "driver_id": driver.get("id"),
                "delay_probability": 0,
                "risk_level": "unknown",
                "explanation": {"error": str(e)},
                "metrics": {},
                "timestamp": current_time.isoformat(),
                "should_alert": False,
            }

    def _calculate_delay_probability(
        self, booking: Dict[str, Any], driver: Dict[str, Any], current_time: datetime
    ) -> float:
        """Calcule la probabilité de retard."""
        try:
            if self.delay_predictor is None or not self.delay_predictor.is_trained:
                # Fallback: calcul heuristique basique
                return self._heuristic_delay_probability(booking, driver, current_time)

            # Utiliser le DelayMLPredictor pour une prédiction précise
            prediction = self.delay_predictor.predict_delay(booking, driver, current_time)

            # Convertir la prédiction de retard en probabilité
            predicted_delay_minutes = prediction.predicted_delay_minutes
            confidence = prediction.confidence

            # Calculer probabilité basée sur le retard prédit
            if predicted_delay_minutes <= PREDICTED_DELAY_MINUTES_ZERO:
                probability = 0.1  # Très faible probabilité
            elif predicted_delay_minutes <= PREDICTED_DELAY_MINUTES_THRESHOLD:
                probability = 0.3  # Probabilité faible
            elif predicted_delay_minutes <= PREDICTED_DELAY_MINUTES_THRESHOLD:
                probability = 0.6  # Probabilité moyenne
            else:
                probability = 0.9  # Probabilité élevée

            # Ajuster selon la confiance du modèle
            probability = probability * confidence + (1 - confidence) * 0.5

            logger.debug(
                "[ProactiveAlerts] Prédiction ML - Retard: %.1f min, Confiance: %.2f, Prob: %.2f",
                predicted_delay_minutes,
                confidence,
                probability,
            )

            return min(0.95, max(0.5, probability))

        except Exception as e:
            logger.warning("[ProactiveAlerts] Erreur prédiction modèle, fallback heuristique: %s", e)
            return self._heuristic_delay_probability(booking, driver, current_time)

    def _heuristic_delay_probability(
        self, booking: Dict[str, Any], driver: Dict[str, Any], current_time: datetime
    ) -> float:
        """Calcul heuristique de probabilité de retard."""
        try:
            # Temps restant avant pickup
            pickup_time = booking.get("pickup_time")
            if isinstance(pickup_time, str):
                pickup_time = datetime.fromisoformat(pickup_time.replace("Z", "+00:00"))

            time_remaining = (pickup_time - current_time).total_seconds() / 60 if pickup_time else 30

            # Distance estimée
            distance = self._estimate_distance(driver, booking)

            # Facteurs de risque
            risk_factors = []

            # Temps insuffisant
            if time_remaining < TIME_REMAINING_THRESHOLD:
                risk_factors.append(0.8)
            elif time_remaining < TIME_REMAINING_THRESHOLD:
                risk_factors.append(0.4)
            else:
                risk_factors.append(0.1)

            # Distance importante
            if distance > DISTANCE_THRESHOLD:
                risk_factors.append(0.6)
            elif distance > DISTANCE_THRESHOLD:
                risk_factors.append(0.3)
            else:
                risk_factors.append(0.1)

            # Charge chauffeur
            driver_load = driver.get("current_bookings", 0)
            if driver_load >= DRIVER_LOAD_THRESHOLD:
                risk_factors.append(0.5)
            elif driver_load >= DRIVER_LOAD_THRESHOLD:
                risk_factors.append(0.2)
            else:
                risk_factors.append(0)

            # Calcul probabilité combinée
            base_prob = float(np.mean(risk_factors))

            # Ajustement selon priorité
            priority = booking.get("priority", 3)
            if priority >= PRIORITY_THRESHOLD:
                base_prob *= 0.7  # Réduction pour priorités élevées

            return min(0.95, max(0.5, base_prob))

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur calcul heuristique: %s", e)
            return 0.5  # Probabilité neutre en cas d'erreur

    def _prepare_features(self, booking: Dict[str, Any], driver: Dict[str, Any], current_time: datetime) -> List[float]:
        """Prépare les features pour le modèle de prédiction."""
        try:
            features = []

            # Features temporelles
            pickup_time = booking.get("pickup_time")
            if isinstance(pickup_time, str):
                pickup_time = datetime.fromisoformat(pickup_time.replace("Z", "+00:00"))

            time_remaining = (pickup_time - current_time).total_seconds() / 60 if pickup_time else 30
            features.extend(
                [
                    time_remaining,
                    current_time.hour,
                    current_time.weekday(),
                ]
            )

            # Features géographiques
            distance = self._estimate_distance(driver, booking)
            features.append(distance)

            # Features chauffeur
            features.extend(
                [
                    driver.get("current_bookings", 0),
                    driver.get("load", 0),
                    1 if driver.get("type") == "REGULAR" else 0,
                ]
            )

            # Features booking
            features.extend(
                [
                    booking.get("priority", 3),
                    1 if booking.get("is_outbound", True) else 0,
                    booking.get("estimated_duration", 30),
                ]
            )

            return features

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur préparation features: %s", e)
            return [0] * 10  # Features par défaut

    def _estimate_distance(self, driver: Dict[str, Any], booking: Dict[str, Any]) -> float:
        """Estime la distance entre chauffeur et pickup."""
        try:
            # Coordonnées chauffeur
            driver_lat = driver.get("lat", 46.2044)
            driver_lon = driver.get("lon", 6.1432)

            # Coordonnées pickup
            pickup_lat = booking.get("pickup_lat", 46.2044)
            pickup_lon = booking.get("pickup_lon", 6.1432)

            # Distance euclidienne simple (approximation)
            lat_diff = abs(driver_lat - pickup_lat)
            lon_diff = abs(driver_lon - pickup_lon)

            # Conversion approximative en km
            return ((lat_diff**2 + lon_diff**2) ** 0.5) * 111.32

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur calcul distance: %s", e)
            return 5  # Distance par défaut

    def _determine_risk_level(self, probability: float) -> str:
        """Détermine le niveau de risque basé sur la probabilité."""
        if probability >= self.delay_risk_thresholds["high"]:
            return "high"
        if probability >= self.delay_risk_thresholds["medium"]:
            return "medium"
        if probability >= self.delay_risk_thresholds["low"]:
            return "low"
        return "minimal"

    def _generate_explanation(
        self, booking: Dict[str, Any], driver: Dict[str, Any], probability: float, risk_level: str
    ) -> Dict[str, Any]:
        """Génère une explication détaillée du risque."""
        try:
            explanation = {
                "risk_level": risk_level,
                "probability_percent": round(probability * 100, 1),
                "primary_factors": [],
                "recommendations": [],
                "alternative_drivers": [],
                "business_impact": self._assess_business_impact(probability, booking),
            }

            # Analyser les facteurs principaux
            factors = self._analyze_risk_factors(booking, driver)
            explanation["primary_factors"] = factors

            # Générer recommandations
            recommendations = self._generate_recommendations(booking, driver, probability)
            explanation["recommendations"] = recommendations

            # Proposer alternatives
            alternatives = self._suggest_alternative_drivers(booking, driver)
            explanation["alternative_drivers"] = alternatives

            return explanation

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur génération explication: %s", e)
            return {
                "risk_level": risk_level,
                "probability_percent": round(probability * 100, 1),
                "primary_factors": [{"factor": "unknown", "impact": "unknown"}],
                "recommendations": ["Contactez le support technique"],
                "alternative_drivers": [],
                "business_impact": "unknown",
            }

    def _analyze_risk_factors(self, booking: Dict[str, Any], driver: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyse les facteurs de risque principaux."""
        factors = []

        # Facteur temps
        pickup_time = booking.get("pickup_time")
        if pickup_time:
            try:
                if isinstance(pickup_time, str):
                    pickup_time = datetime.fromisoformat(pickup_time.replace("Z", "+00:00"))

                time_remaining = (pickup_time - datetime.now(UTC)).total_seconds() / 60

                if time_remaining < TIME_REMAINING_THRESHOLD:
                    factors.append(
                        {
                            "factor": "temps_insuffisant",
                            "impact": "high",
                            "description": f"Seulement {time_remaining:.1f} min",
                            "value": time_remaining,
                        }
                    )
                elif time_remaining < TIME_REMAINING_THRESHOLD:
                    factors.append(
                        {
                            "factor": "temps_limite",
                            "impact": "medium",
                            "description": f"{time_remaining:.1f} min restantes",
                            "value": time_remaining,
                        }
                    )
            except Exception:
                pass

        # Facteur distance
        distance = self._estimate_distance(driver, booking)
        if distance > DISTANCE_THRESHOLD:
            factors.append(
                {
                    "factor": "distance_elevee",
                    "impact": "high",
                    "description": f"Distance {distance:.1f} km",
                    "value": distance,
                }
            )
        elif distance > DISTANCE_THRESHOLD:
            factors.append(
                {
                    "factor": "distance_moderee",
                    "impact": "medium",
                    "description": f"Distance {distance:.1f} km",
                    "value": distance,
                }
            )

        # Facteur charge chauffeur
        driver_load = driver.get("current_bookings", 0)
        if driver_load >= DRIVER_LOAD_THRESHOLD:
            factors.append(
                {
                    "factor": "charge_maximale",
                    "impact": "high",
                    "description": f"Chauffeur à {driver_load} courses",
                    "value": driver_load,
                }
            )
        elif driver_load >= DRIVER_LOAD_THRESHOLD:
            factors.append(
                {
                    "factor": "charge_elevee",
                    "impact": "medium",
                    "description": f"Chauffeur à {driver_load} courses",
                    "value": driver_load,
                }
            )

        return factors

    def _generate_recommendations(
        self,
        booking: Dict[str, Any],
        driver: Dict[str, Any],  # noqa: ARG002
        probability: float,
    ) -> List[str]:
        """Génère des recommandations basées sur l'analyse."""
        recommendations = []

        if probability > PROBABILITY_HIGH_RISK:
            recommendations.append("🚨 Risque élevé - Considérer un chauffeur plus proche")
            recommendations.append("📞 Prévenir le client du risque de retard")

        if probability > PROBABILITY_MEDIUM_RISK:
            recommendations.append("⚠️ Surveiller l'assignation en temps réel")
            recommendations.append("🔄 Préparer un plan de replanification")

        # Recommandations spécifiques
        pickup_time = booking.get("pickup_time")
        if pickup_time:
            try:
                if isinstance(pickup_time, str):
                    pickup_time = datetime.fromisoformat(pickup_time.replace("Z", "+00:00"))

                time_remaining = (pickup_time - datetime.now(UTC)).total_seconds() / 60

                if time_remaining < TIME_REMAINING_THRESHOLD:
                    recommendations.append("⏰ Temps critique - Accélérer le processus")

                if time_remaining < TIME_REMAINING_THRESHOLD:
                    recommendations.append("🚨 URGENCE - Contacter le chauffeur immédiatement")

            except Exception:
                pass

        return recommendations

    def _suggest_alternative_drivers(
        self, booking: Dict[str, Any], current_driver: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggère des chauffeurs alternatifs."""
        # Placeholder - dans une vraie implémentation,
        # on interrogerait la base de données pour trouver des alternatives
        alternatives = []

        # Simulation d'alternatives basée sur la distance
        current_distance = self._estimate_distance(current_driver, booking)

        if current_distance > CURRENT_DISTANCE_THRESHOLD:
            alternatives.append(
                {
                    "driver_id": "alt_001",
                    "estimated_distance": current_distance * 0.6,
                    "risk_reduction": 0.3,
                    "reason": "Chauffeur plus proche",
                }
            )

        if current_driver.get("current_bookings", 0) >= DRIVER_LOAD_WARNING:
            alternatives.append(
                {
                    "driver_id": "alt_002",
                    "estimated_distance": current_distance * 0.8,
                    "risk_reduction": 0.2,
                    "reason": "Chauffeur moins chargé",
                }
            )

        return alternatives

    def _assess_business_impact(self, probability: float, booking: Dict[str, Any]) -> str:
        """Évalue l'impact business du risque."""
        priority = booking.get("priority", 3)

        if probability > PROBABILITY_CRITICAL and priority >= PRIORITY_THRESHOLD:
            return "critical"
        if probability > PROBABILITY_WARNING and priority >= PRIORITY_THRESHOLD:
            return "high"
        if probability > PROBABILITY_LOW:
            return "medium"
        return "low"

    def _calculate_additional_metrics(
        self, booking: Dict[str, Any], driver: Dict[str, Any], current_time: datetime
    ) -> Dict[str, Any]:
        """Calcule des métriques additionnelles."""
        try:
            pickup_time = booking.get("pickup_time")
            if isinstance(pickup_time, str):
                pickup_time = datetime.fromisoformat(pickup_time.replace("Z", "+00:00"))

            time_remaining = (pickup_time - current_time).total_seconds() / 60 if pickup_time else 30
            distance = self._estimate_distance(driver, booking)

            return {
                "time_remaining_minutes": round(time_remaining, 1),
                "estimated_distance_km": round(distance, 1),
                "driver_load": driver.get("current_bookings", 0),
                "booking_priority": booking.get("priority", 3),
                "is_outbound": booking.get("is_outbound", True),
                "estimated_travel_time_minutes": round(distance * 2, 1),  # 30 km/h moyenne
                "buffer_time_minutes": round(time_remaining - (distance * 2), 1),
            }

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur calcul métriques: %s", e)
            return {}

    def send_proactive_alert(self, analysis_result: Dict[str, Any], company_id: str, force_send: bool = False) -> bool:
        """Envoie une alerte proactive si nécessaire avec système de debounce avancé.

        Args:
            analysis_result: Résultat de l'analyse de risque
            company_id: ID de l'entreprise
            force_send: Forcer l'envoi même si debounce

        Returns:
            True si alerte envoyée, False sinon

        """
        try:
            booking_id = analysis_result.get("booking_id")
            risk_level = analysis_result.get("risk_level", "unknown")
            current_time = datetime.now(UTC)

            # Vérifier debounce avancé
            if not force_send and booking_id:
                debounce_result = self._check_debounce_rules(booking_id, risk_level, current_time)
                if not debounce_result["should_send"]:
                    logger.debug(
                        "[ProactiveAlerts] Alerte debounced pour booking %s: %s", booking_id, debounce_result["reason"]
                    )
                    return False

            # Envoyer alerte si risque suffisant
            if analysis_result.get("should_alert", False):
                success = self._send_alert_notification(analysis_result, company_id)

                if success and booking_id:
                    self._update_alert_history(booking_id, risk_level, current_time)

                return success

            return False

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur envoi alerte: %s", e)
            return False

    def _send_alert_notification(self, analysis_result: Dict[str, Any], company_id: str) -> bool:
        """Envoie la notification d'alerte."""
        try:
            risk_level = analysis_result.get("risk_level")
            probability = analysis_result.get("delay_probability", 0)
            explanation = analysis_result.get("explanation", {})

            # Construire le message
            message = self._build_alert_message(analysis_result)

            # Envoyer via notification service
            notification_data = {
                "type": "delay_risk_alert",
                "level": risk_level,
                "probability": probability,
                "message": message,
                "explanation": explanation,
                "booking_id": analysis_result.get("booking_id"),
                "driver_id": analysis_result.get("driver_id"),
                "timestamp": analysis_result.get("timestamp"),
            }

            # Utiliser le service de notification existant
            if self.notification_service:
                success = self.notification_service.send_notification(
                    company_id=company_id, notification_type="delay_risk", data=notification_data
                )
            else:
                logger.warning("[ProactiveAlerts] NotificationService non disponible")
                success = False

            if success:
                logger.info(
                    "[ProactiveAlerts] ✅ Alerte envoyée - Booking %s, Risque %s (%.1f%%)",
                    analysis_result.get("booking_id"),
                    risk_level,
                    probability * 100,
                )
            else:
                logger.warning("[ProactiveAlerts] ⚠️ Échec envoi alerte - Booking %s", analysis_result.get("booking_id"))

            return success

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur construction/envoi notification: %s", e)
            return False

    def _build_alert_message(self, analysis_result: Dict[str, Any]) -> str:
        """Construit le message d'alerte."""
        risk_level = analysis_result.get("risk_level")
        probability = analysis_result.get("delay_probability", 0)
        explanation = analysis_result.get("explanation", {})

        # Emojis selon niveau de risque
        emoji_map = {"high": "🚨", "medium": "⚠️", "low": "ℹ️", "minimal": "✅"}

        emoji = emoji_map.get(risk_level or "unknown", "❓")

        # Message principal
        risk_level_str = (risk_level or "unknown").upper()
        message = f"{emoji} Risque de retard détecté\n\n"
        message += f"Probabilité: {probability * 100:.1f}%\n"
        message += f"Niveau: {risk_level_str}\n\n"

        # Facteurs principaux
        factors = explanation.get("primary_factors", [])
        if factors:
            message += "Facteurs de risque:\n"
            for factor in factors[:3]:  # Top 3
                message += f"• {factor.get('description', 'Facteur inconnu')}\n"
            message += "\n"

        # Recommandations
        recommendations = explanation.get("recommendations", [])
        if recommendations:
            message += "Recommandations:\n"
            for rec in recommendations[:3]:  # Top 3
                message += f"• {rec}\n"

        return message

    def get_explanation_for_decision(
        self, booking_id: str, driver_id: str, rl_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Génère une explication pour une décision RL.

        Args:
            booking_id: ID du booking
            driver_id: ID du chauffeur assigné
            rl_decision: Décision RL (Q-values, action choisie, etc.)

        Returns:
            Explication détaillée de la décision

        """
        try:
            explanation = {
                "decision_type": "rl_assignment",
                "booking_id": booking_id,
                "driver_id": driver_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "decision_factors": [],
                "q_values": rl_decision.get("q_values", {}),
                "confidence": rl_decision.get("confidence", 0),
                "alternative_options": [],
                "business_rules_applied": [],
            }

            # Analyser les facteurs de décision
            factors = self._analyze_rl_decision_factors(rl_decision)
            explanation["decision_factors"] = factors

            # Générer alternatives
            alternatives = self._generate_rl_alternatives(rl_decision)
            explanation["alternative_options"] = alternatives

            # Règles métier appliquées
            business_rules = self._identify_business_rules(rl_decision)
            explanation["business_rules_applied"] = business_rules

            return explanation

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur génération explication RL: %s", e)
            return {
                "decision_type": "rl_assignment",
                "booking_id": booking_id,
                "driver_id": driver_id,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def _analyze_rl_decision_factors(self, rl_decision: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyse les facteurs de la décision RL."""
        factors = []

        q_values = rl_decision.get("q_values", {})
        if q_values:
            # Top actions par Q-value
            sorted_actions = sorted(q_values.items(), key=lambda x: x[1], reverse=True)

            for i, (action, q_value) in enumerate(sorted_actions[:3]):
                factors.append(
                    {
                        "factor": f"q_value_rank_{i + 1}",
                        "action": action,
                        "q_value": q_value,
                        "description": f"Action {action} avec Q-value {q_value:.2f}",
                    }
                )

        # Facteurs de reward shaping
        reward_components = rl_decision.get("reward_components", {})
        for component, value in reward_components.items():
            factors.append(
                {
                    "factor": f"reward_{component}",
                    "value": value,
                    "description": f"Composant reward {component}: {value:.2f}",
                }
            )

        return factors

    def _generate_rl_alternatives(self, rl_decision: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Génère des alternatives à la décision RL."""
        alternatives = []

        q_values = rl_decision.get("q_values", {})
        if q_values:
            sorted_actions = sorted(q_values.items(), key=lambda x: x[1], reverse=True)

            # Top 3 alternatives
            for i, (action, q_value) in enumerate(sorted_actions[1:4]):
                alternatives.append(
                    {
                        "alternative_rank": i + 1,
                        "action": action,
                        "q_value": q_value,
                        "confidence": q_value / sorted_actions[0][1] if sorted_actions else 0,
                        "description": f"Alternative {i + 1}: Action {action}",
                    }
                )

        return alternatives

    def _identify_business_rules(self, rl_decision: Dict[str, Any]) -> List[str]:
        """Identifie les règles métier appliquées."""
        rules = []

        # Vérifier les contraintes appliquées
        constraints = rl_decision.get("constraints_applied", [])
        for constraint in constraints:
            rules.append(f"Contrainte appliquée: {constraint}")

        # Vérifier le reward shaping
        reward_profile = rl_decision.get("reward_profile", "DEFAULT")
        rules.append(f"Profil reward shaping: {reward_profile}")

        # Vérifier l'action masking
        if rl_decision.get("action_masked", False):
            rules.append("Action masking activé")

        return rules

    def _check_debounce_rules(self, booking_id: str, risk_level: str, current_time: datetime) -> Dict[str, Any]:
        """Vérifie les règles de debounce avancées.

        Args:
            booking_id: ID du booking
            risk_level: Niveau de risque actuel
            current_time: Temps actuel

        Returns:
            Dictionnaire avec should_send et reason

        """
        try:
            # Règle 1: Debounce temporel basique
            if booking_id in self.alert_history:
                last_alert_data = self.alert_history[booking_id]
                last_alert_time = last_alert_data.get("last_alert_time")

                if last_alert_time:
                    time_since_last = current_time - last_alert_time
                    if time_since_last.total_seconds() < (self.debounce_minutes * 60):
                        return {"should_send": False, "reason": f"Debounce temporel: {self.debounce_minutes} min"}

            # Règle 2: Limite de fréquence par heure
            if booking_id in self.alert_frequency_tracker:
                recent_alerts = self.alert_frequency_tracker[booking_id]
                one_hour_ago = current_time - timedelta(hours=1)

                # Filtrer les alertes de la dernière heure
                recent_count = len([t for t in recent_alerts if t > one_hour_ago])

                if recent_count >= self.max_alerts_per_hour:
                    return {
                        "should_send": False,
                        "reason": f"Limite fréquence: {recent_count}/{self.max_alerts_per_hour} par heure",
                    }

            # Règle 3: Escalade de risque (forcer si risque augmente)
            if booking_id in self.alert_history:
                last_risk_level = self.alert_history[booking_id].get("last_risk_level", "minimal")
                risk_escalation = self._get_risk_level_numeric(risk_level) > self._get_risk_level_numeric(
                    last_risk_level
                )

                if risk_escalation:
                    return {"should_send": True, "reason": f"Escalade de risque: {last_risk_level} → {risk_level}"}

            # Règle 4: Alerte critique toujours autorisée
            if risk_level == "high":
                return {"should_send": True, "reason": "Risque critique - toujours autorisé"}

            return {"should_send": True, "reason": "Règles de debounce respectées"}

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur vérification debounce: %s", e)
            return {"should_send": True, "reason": "Erreur debounce - autoriser par sécurité"}

    def _get_risk_level_numeric(self, risk_level: str) -> int:
        """Convertit le niveau de risque en valeur numérique."""
        risk_map = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        return risk_map.get(risk_level, 0)

    def _update_alert_history(self, booking_id: str, risk_level: str, current_time: datetime) -> None:
        """Met à jour l'historique des alertes."""
        try:
            # Mettre à jour l'historique principal
            self.alert_history[booking_id] = {
                "last_alert_time": current_time,
                "last_risk_level": risk_level,
                "total_alerts": self.alert_history.get(booking_id, {}).get("total_alerts", 0) + 1,
            }

            # Mettre à jour le tracker de fréquence
            if booking_id not in self.alert_frequency_tracker:
                self.alert_frequency_tracker[booking_id] = []

            self.alert_frequency_tracker[booking_id].append(current_time)

            # Nettoyer les anciennes entrées (plus de 24h)
            cutoff_time = current_time - timedelta(hours=24)
            self.alert_frequency_tracker[booking_id] = [
                t for t in self.alert_frequency_tracker[booking_id] if t > cutoff_time
            ]

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur mise à jour historique: %s", e)

    def clear_alert_history(self, booking_id: str | None = None) -> None:
        """Nettoie l'historique des alertes."""
        try:
            if booking_id:
                self.alert_history.pop(booking_id, None)
                self.alert_frequency_tracker.pop(booking_id, None)
                logger.info("[ProactiveAlerts] Historique nettoyé pour booking %s", booking_id)
            else:
                self.alert_history.clear()
                self.alert_frequency_tracker.clear()
                logger.info("[ProactiveAlerts] Historique complet nettoyé")

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur nettoyage historique: %s", e)

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques des alertes."""
        total_alerts = len(self.alert_history)

        # Analyser par niveau de risque (simulation)
        risk_levels = {"high": 0, "medium": 0, "low": 0, "minimal": 0}

        return {
            "total_alerts_sent": total_alerts,
            "active_debounce_count": len(self.alert_history),
            "risk_level_distribution": risk_levels,
            "debounce_minutes": self.debounce_minutes,
            "delay_predictor_loaded": self.delay_predictor is not None,
        }
