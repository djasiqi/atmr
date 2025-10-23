# ruff: noqa: T201, DTZ003, W293
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""
Shadow Mode Manager pour DQN Dispatch

Ce module permet de faire tourner le modèle DQN en parallèle du système
de dispatch existant, sans impacter les décisions réelles.

Il enregistre toutes les prédictions et permet de comparer les performances.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from models.booking import Booking
from models.driver import Driver
from services.rl.dqn_agent import DQNAgent

logger = logging.getLogger(__name__)


class ShadowModeManager:
    """
    Gestionnaire du mode Shadow pour le DQN Dispatch.
    
    Fait tourner le modèle DQN en parallèle du système actuel
    et enregistre toutes les décisions pour comparaison.
    """

    def __init__(
        self,
        model_path: str = "data/rl/models/dqn_best.pth",
        log_dir: str = "data/rl/shadow_mode",
        enable_logging: bool = True
    ):
        """
        Initialise le Shadow Mode Manager.
        
        Args:
            model_path: Chemin vers le modèle DQN à utiliser
            log_dir: Répertoire pour les logs de shadow mode
            enable_logging: Active/désactive le logging détaillé
        """
        self.model_path = model_path
        self.log_dir = Path(log_dir)
        self.enable_logging = enable_logging

        # Créer répertoire de logs
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Charger le modèle DQN
        self.agent: DQNAgent | None = None
        self._load_model()

        # Statistiques de session
        self.predictions_count = 0
        self.comparisons_count = 0
        self.agreements_count = 0

        logger.info(f"🔍 Shadow Mode Manager initialisé (model: {model_path})")

    def _load_model(self) -> None:
        """Charge le modèle DQN pour les prédictions shadow."""
        try:
            # TODO: Déterminer les dimensions du modèle à partir de la config
            # Pour l'instant, on utilise les valeurs du meilleur modèle
            state_dim = 82  # À ajuster selon la config réelle
            action_dim = 76  # À ajuster selon la config réelle

            self.agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                device="cpu"  # CPU pour production
            )

            self.agent.load(self.model_path)
            logger.info(f"✅ Modèle DQN chargé depuis {self.model_path}")

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle DQN: {e}")
            self.agent = None

    def predict_driver_assignment(
        self,
        booking: Booking,
        available_drivers: List[Driver],
        current_assignments: Dict[int, List[int]]
    ) -> Dict[str, Any] | None:
        """
        Fait une prédiction DQN pour l'assignation d'un booking.
        
        Cette fonction NE modifie PAS les assignations réelles.
        Elle enregistre seulement la prédiction du modèle.
        
        Args:
            booking: Le booking à assigner
            available_drivers: Liste des drivers disponibles
            current_assignments: Assignations actuelles {driver_id: [booking_ids]}
            
        Returns:
            Prédiction DQN avec score de confiance, ou None si erreur
        """
        if not self.agent:
            logger.warning("Agent DQN non disponible pour prédiction shadow")
            return None

        try:
            # Construire l'état pour le DQN
            state = self._build_state(booking, available_drivers, current_assignments)

            # Obtenir la prédiction DQN (mode exploitation)
            action = self.agent.select_action(state, training=False)
            q_values = self.agent.get_q_values(state)

            # Mapper l'action à un driver (ou "wait")
            if action < len(available_drivers):
                predicted_driver = available_drivers[action]
                predicted_driver_id = predicted_driver.id
                action_type = "assign"
            else:
                predicted_driver_id = None
                action_type = "wait"

            # Calculer la confiance (softmax des Q-values)
            confidence = self._compute_confidence(q_values, action)

            prediction = {
                "booking_id": booking.id,
                "predicted_driver_id": predicted_driver_id,
                "action_type": action_type,
                "confidence": confidence,
                "q_value": float(q_values[action]),
                "timestamp": datetime.utcnow().isoformat(),
                "available_drivers_count": len(available_drivers)
            }

            self.predictions_count += 1

            # Logger la prédiction si activé
            if self.enable_logging:
                self._log_prediction(prediction)

            return prediction

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction DQN shadow: {e}")
            return None

    def compare_with_actual_decision(
        self,
        prediction: Dict[str, Any],
        actual_driver_id: int | None,
        outcome_metrics: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Compare la prédiction DQN avec la décision réelle du système.
        
        Args:
            prediction: Prédiction DQN précédente
            actual_driver_id: Driver réellement assigné (None si wait)
            outcome_metrics: Métriques du résultat (distance, délai, etc.)
            
        Returns:
            Résultat de la comparaison
        """
        try:
            predicted_driver_id = prediction.get("predicted_driver_id")

            # Comparer les décisions
            agreement = predicted_driver_id == actual_driver_id

            comparison = {
                "booking_id": prediction["booking_id"],
                "predicted_driver_id": predicted_driver_id,
                "actual_driver_id": actual_driver_id,
                "agreement": agreement,
                "confidence": prediction.get("confidence", 0.0),
                "outcome_metrics": outcome_metrics or {},
                "timestamp": datetime.utcnow().isoformat()
            }

            self.comparisons_count += 1
            if agreement:
                self.agreements_count += 1

            # Logger la comparaison si activé
            if self.enable_logging:
                self._log_comparison(comparison)

            return comparison

        except Exception as e:
            logger.error(f"Erreur lors de la comparaison shadow: {e}")
            return {}

    def _build_state(
        self,
        booking: Booking,
        available_drivers: List[Driver],
        current_assignments: Dict[int, List[int]]
    ) -> np.ndarray:
        """
        Construit le vecteur d'état pour le DQN.
        
        Note: Doit correspondre exactement à l'environnement DispatchEnv.
        """
        # TODO: Implémenter la construction d'état réelle
        # Pour l'instant, retourne un état dummy
        # En production, utiliser la même logique que DispatchEnv

        state_dim = 82
        state = np.zeros(state_dim, dtype=np.float32)

        # Exemples de features à inclure:
        # - Infos booking: position, priorité, time window
        # - Infos drivers: positions, disponibilité, charge
        # - Context: heure, jour, météo, traffic

        return state

    def _compute_confidence(self, q_values: np.ndarray, action: int) -> float:
        """
        Calcule un score de confiance basé sur les Q-values.
        
        Utilise softmax pour normaliser les Q-values en probabilités.
        """
        # Softmax
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / exp_q.sum()

        confidence = float(probs[action])
        return confidence

    def _log_prediction(self, prediction: Dict[str, Any]) -> None:
        """Enregistre une prédiction dans les logs."""
        log_file = self.log_dir / f"predictions_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(prediction) + "\n")

    def _log_comparison(self, comparison: Dict[str, Any]) -> None:
        """Enregistre une comparaison dans les logs."""
        log_file = self.log_dir / f"comparisons_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(comparison) + "\n")

    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques actuelles du shadow mode.
        
        Returns:
            Dictionnaire avec les stats de session
        """
        agreement_rate = (
            self.agreements_count / self.comparisons_count
            if self.comparisons_count > 0
            else 0.0
        )

        return {
            "predictions_count": self.predictions_count,
            "comparisons_count": self.comparisons_count,
            "agreements_count": self.agreements_count,
            "agreement_rate": agreement_rate,
            "model_path": str(self.model_path),
            "log_dir": str(self.log_dir)
        }

    def generate_daily_report(self, date: str | None = None) -> Dict[str, Any]:
        """
        Génère un rapport quotidien des performances shadow mode.
        
        Args:
            date: Date au format YYYYMMDD (défaut: aujourd'hui)
            
        Returns:
            Rapport détaillé avec métriques de comparaison
        """
        if date is None:
            date = datetime.utcnow().strftime('%Y%m%d')

        predictions_file = self.log_dir / f"predictions_{date}.jsonl"
        comparisons_file = self.log_dir / f"comparisons_{date}.jsonl"

        predictions = []
        comparisons = []

        # Charger les prédictions
        if predictions_file.exists():
            with open(predictions_file, encoding="utf-8") as f:
                predictions = [json.loads(line) for line in f]

        # Charger les comparaisons
        if comparisons_file.exists():
            with open(comparisons_file, encoding="utf-8") as f:
                comparisons = [json.loads(line) for line in f]

        # Calculer les métriques
        total_predictions = len(predictions)
        total_comparisons = len(comparisons)
        agreements = sum(1 for c in comparisons if c.get("agreement", False))

        agreement_rate = agreements / total_comparisons if total_comparisons > 0 else 0.0

        # Métriques par type d'action
        dqn_assigns = sum(1 for p in predictions if p.get("action_type") == "assign")
        dqn_waits = sum(1 for p in predictions if p.get("action_type") == "wait")

        actual_assigns = sum(1 for c in comparisons if c.get("actual_driver_id") is not None)
        actual_waits = total_comparisons - actual_assigns

        # Confiance moyenne
        avg_confidence = (
            np.mean([p.get("confidence", 0) for p in predictions])
            if predictions else 0.0
        )

        report = {
            "date": date,
            "summary": {
                "total_predictions": total_predictions,
                "total_comparisons": total_comparisons,
                "agreements": agreements,
                "agreement_rate": agreement_rate,
                "avg_confidence": float(avg_confidence)
            },
            "action_distribution": {
                "dqn": {
                    "assigns": dqn_assigns,
                    "waits": dqn_waits,
                    "assign_rate": dqn_assigns / total_predictions if total_predictions > 0 else 0.0
                },
                "actual": {
                    "assigns": actual_assigns,
                    "waits": actual_waits,
                    "assign_rate": actual_assigns / total_comparisons if total_comparisons > 0 else 0.0
                }
            },
            "confidence_stats": {
                "mean": float(avg_confidence),
                "high_confidence_predictions": sum(
                    1 for p in predictions if p.get("confidence", 0) > 0.8
                ),
                "low_confidence_predictions": sum(
                    1 for p in predictions if p.get("confidence", 0) < 0.3
                )
            }
        }

        # Sauvegarder le rapport
        report_file = self.log_dir / f"daily_report_{date}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📊 Rapport quotidien généré: {report_file}")

        return report

