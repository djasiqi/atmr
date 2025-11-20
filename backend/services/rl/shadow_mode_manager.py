#!/usr/bin/env python3

# Constantes pour éviter les valeurs magiques
import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

X_ZERO = 0
AVG_ETA_IMPROVEMENT_THRESHOLD = 2
RL_ETA_IMPROVEMENT_THRESHOLD = -2
EXCELLENT_RL_ETA_IMPROVEMENT_THRESHOLD = -5
HIGH_AGREEMENT_RATE_THRESHOLD = 0.8
LOW_AGREEMENT_RATE_THRESHOLD = 0.4
INSIGHT_HIGH_AGREEMENT_THRESHOLD = 0.7
INSIGHT_LOW_AGREEMENT_THRESHOLD = 0.3
HIGH_RL_CONFIDENCE_THRESHOLD = 0.8
LOW_RL_CONFIDENCE_THRESHOLD = 0.5
HIGH_DRIVER_RATING_THRESHOLD = 4
MIN_VALUES_FOR_STD = 2
MIN_DAILY_REPORTS_FOR_TRENDS = 2
AGREEMENT_RATE_ZERO = 0
AVG_CONFIDENCE_ZERO = 0

"""Shadow Mode Manager enrichi avec KPIs détaillés.

Mesure les différences entre décisions humaines et RL,
génère des rapports quotidiens et pilote l'adoption.
"""


class ShadowModeManager:
    """Gestionnaire du mode shadow enrichi avec KPIs détaillés.

    Features:
        - Comparaison humain vs RL avec métriques détaillées
        - KPIs : delta ETA, delta retard, second best driver
        - Explicabilité des décisions RL
        - Rapports quotidiens par entreprise
        - Export CSV/JSON automatisé
    """

    def __init__(self, data_dir: str = "data/rl/shadow_mode"):  # pyright: ignore[reportMissingSuperCall]
        """Initialise le ShadowModeManager.

        Args:
            data_dir: Répertoire pour stocker les données shadow mode

        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Structure des KPIs
        self.kpi_metrics = {
            "eta_delta": [],  # Différence ETA humain vs RL
            "delay_delta": [],  # Différence retard humain vs RL
            "second_best_driver": [],  # Second meilleur driver suggéré
            "rl_confidence": [],  # Confiance RL dans la décision
            "human_confidence": [],  # Confiance humaine (si disponible)
            "decision_reasons": [],  # Raisons de la décision RL
            "constraint_violations": [],  # Violations de contraintes
            "performance_impact": [],  # Impact sur performance globale
        }

        # Métadonnées des décisions
        self.decision_metadata = {
            "timestamp": [],
            "company_id": [],
            "booking_id": [],
            "driver_id": [],
            "human_decision": [],
            "rl_decision": [],
            "context": [],
        }

        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Configure le logging."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def log_decision_comparison(
        self,
        company_id: str,
        booking_id: str,
        human_decision: Dict[str, Any],
        rl_decision: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enregistre une comparaison de décision humain vs RL.

        Args:
            company_id: ID de l'entreprise
            booking_id: ID de la réservation
            human_decision: Décision prise par l'humain
            rl_decision: Décision suggérée par RL
            context: Contexte de la décision

        Returns:
            Dictionnaire avec les KPIs calculés

        """
        timestamp = datetime.now(UTC)

        # Extraire les informations des décisions
        human_driver = human_decision.get("driver_id")
        rl_driver = rl_decision.get("driver_id")

        # Calculer les KPIs
        kpis = self._calculate_kpis(human_decision, rl_decision, context)

        # Enregistrer les métadonnées
        self.decision_metadata["timestamp"].append(timestamp)
        self.decision_metadata["company_id"].append(company_id)
        self.decision_metadata["booking_id"].append(booking_id)
        self.decision_metadata["driver_id"].append(human_driver)
        self.decision_metadata["human_decision"].append(human_decision)
        self.decision_metadata["rl_decision"].append(rl_decision)
        self.decision_metadata["context"].append(context)

        # Enregistrer les KPIs
        for metric, value in kpis.items():
            if metric in self.kpi_metrics:
                self.kpi_metrics[metric].append(value)

        # Log de la décision
        self.logger.info(
            "Decision logged: Company=%s, Booking=%s, Human=%s, RL=%s, ETA Delta=%s",
            company_id,
            booking_id,
            human_driver,
            rl_driver,
            kpis.get("eta_delta", 0),
        )

        return kpis

    def _calculate_kpis(
        self,
        human_decision: Dict[str, Any] | None = None,
        rl_decision: Dict[str, Any] | None = None,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Calcule les KPIs détaillés pour la comparaison.

        Si aucun paramètre n'est fourni, calcule des KPIs agrégés depuis les logs stockés.

        Args:
            human_decision: Décision humaine (optionnel)
            rl_decision: Décision RL (optionnel)
            context: Contexte (optionnel)

        Returns:
            Dictionnaire des KPIs calculés

        """
        # Si aucun paramètre n'est fourni, calculer des KPIs agrégés depuis les logs
        if human_decision is None and rl_decision is None and context is None:
            return self._calculate_aggregated_kpis()

        # Sinon, calculer les KPIs pour une seule décision
        if human_decision is None:
            human_decision = {}
        if rl_decision is None:
            rl_decision = {}
        if context is None:
            context = {}

        kpis = {}

        # 1. Delta ETA (minutes)
        human_eta = human_decision.get("eta_minutes", 0) or 0
        rl_eta = rl_decision.get("eta_minutes", 0) or 0
        kpis["eta_delta"] = rl_eta - human_eta

        # 2. Delta retard (minutes)
        human_delay = human_decision.get("delay_minutes", 0) or 0
        rl_delay = rl_decision.get("delay_minutes", 0) or 0
        kpis["delay_delta"] = rl_delay - human_delay

        # 3. Second best driver
        rl_alternatives = rl_decision.get("alternative_drivers", [])
        kpis["second_best_driver"] = rl_alternatives[1] if len(rl_alternatives) > 1 else None

        # 4. Confiance RL
        kpis["rl_confidence"] = rl_decision.get("confidence", 0)

        # 5. Confiance humaine (si disponible)
        kpis["human_confidence"] = human_decision.get("confidence")

        # 6. Raisons de la décision RL
        kpis["decision_reasons"] = self._extract_decision_reasons(rl_decision, context)

        # 7. Violations de contraintes
        kpis["constraint_violations"] = self._check_constraint_violations(rl_decision, context)

        # 8. Impact sur performance globale
        kpis["performance_impact"] = self._calculate_performance_impact(human_decision, rl_decision, context)

        return kpis

    def _calculate_aggregated_kpis(self) -> Dict[str, Any]:
        """Calcule les KPIs agrégés depuis les logs stockés.

        Returns:
            Dictionnaire des KPIs agrégés

        """
        # Vérifier si decision_logs existe (défini par les tests)
        if hasattr(self, "decision_logs") and self.decision_logs:
            logs = self.decision_logs
        # Sinon, construire depuis decision_metadata
        elif self.decision_metadata["human_decision"]:
            logs = []
            for i in range(len(self.decision_metadata["human_decision"])):
                human_dec = self.decision_metadata["human_decision"][i]
                rl_dec = self.decision_metadata["rl_decision"][i]
                logs.append(
                    {
                        "company_id": self.decision_metadata["company_id"][i],
                        "booking_id": self.decision_metadata["booking_id"][i],
                        "human_driver_id": human_dec.get("driver_id"),
                        "rl_driver_id": rl_dec.get("driver_id"),
                        "human_eta_minutes": human_dec.get("eta_minutes", 0) or 0,
                        "rl_eta_minutes": rl_dec.get("eta_minutes", 0) or 0,
                        "human_delay_minutes": human_dec.get("delay_minutes", 0) or 0,
                        "rl_delay_minutes": rl_dec.get("delay_minutes", 0) or 0,
                        "timestamp": self.decision_metadata["timestamp"][i],
                    }
                )
        else:
            # Aucun log disponible
            return {
                "total_decisions": 0,
                "rl_wins": 0,
                "human_wins": 0,
                "avg_human_eta": 0.0,
                "avg_rl_eta": 0.0,
                "avg_human_delay": 0.0,
                "avg_rl_delay": 0.0,
                "rl_win_rate": 0.0,
                "eta_improvement_rate": 0.0,
                "delay_reduction_rate": 0.0,
            }

        # Calculer les KPIs agrégés
        total_decisions = len(logs)
        if total_decisions == 0:
            return {
                "total_decisions": 0,
                "rl_wins": 0,
                "human_wins": 0,
                "avg_human_eta": 0.0,
                "avg_rl_eta": 0.0,
                "avg_human_delay": 0.0,
                "avg_rl_delay": 0.0,
                "rl_win_rate": 0.0,
                "eta_improvement_rate": 0.0,
                "delay_reduction_rate": 0.0,
            }

        rl_wins = 0
        human_wins = 0
        total_human_eta = 0.0
        total_rl_eta = 0.0
        total_human_delay = 0.0
        total_rl_delay = 0.0
        eta_improvements = 0
        delay_reductions = 0

        for log in logs:
            human_eta = float(log.get("human_eta_minutes", 0) or 0)
            rl_eta = float(log.get("rl_eta_minutes", 0) or 0)
            human_delay = float(log.get("human_delay_minutes", 0) or 0)
            rl_delay = float(log.get("rl_delay_minutes", 0) or 0)

            total_human_eta += human_eta
            total_rl_eta += rl_eta
            total_human_delay += human_delay
            total_rl_delay += rl_delay

            # Déterminer qui gagne (RL gagne si ETA ou delay est meilleur, ou si les deux sont égaux et RL a un meilleur delay)
            if rl_eta < human_eta or (rl_eta == human_eta and rl_delay < human_delay):
                rl_wins += 1
            elif human_eta < rl_eta or (human_eta == rl_eta and human_delay < rl_delay):
                human_wins += 1

            # Compter les améliorations
            if rl_eta < human_eta:
                eta_improvements += 1
            if rl_delay < human_delay:
                delay_reductions += 1

        avg_human_eta = total_human_eta / total_decisions if total_decisions > 0 else 0.0
        avg_rl_eta = total_rl_eta / total_decisions if total_decisions > 0 else 0.0
        avg_human_delay = total_human_delay / total_decisions if total_decisions > 0 else 0.0
        avg_rl_delay = total_rl_delay / total_decisions if total_decisions > 0 else 0.0
        rl_win_rate = rl_wins / total_decisions if total_decisions > 0 else 0.0
        eta_improvement_rate = eta_improvements / total_decisions if total_decisions > 0 else 0.0
        delay_reduction_rate = delay_reductions / total_decisions if total_decisions > 0 else 0.0

        return {
            "total_decisions": total_decisions,
            "rl_wins": rl_wins,
            "human_wins": human_wins,
            "avg_human_eta": avg_human_eta,
            "avg_rl_eta": avg_rl_eta,
            "avg_human_delay": avg_human_delay,
            "avg_rl_delay": avg_rl_delay,
            "rl_win_rate": rl_win_rate,
            "eta_improvement_rate": eta_improvement_rate,
            "delay_reduction_rate": delay_reduction_rate,
        }

    def _extract_decision_reasons(self, rl_decision: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Extrait les raisons de la décision RL."""
        reasons = []

        # Raisons basées sur les métriques
        if rl_decision.get("eta_minutes", 0) < context.get("avg_eta", 0):
            reasons.append("ETA inférieur à la moyenne")

        if rl_decision.get("distance_km", 0) < context.get("avg_distance", 0):
            reasons.append("Distance optimisée")

        if rl_decision.get("driver_load", 0) < context.get("avg_load", 0):
            reasons.append("Charge chauffeur équilibrée")

        # Raisons basées sur les contraintes
        if rl_decision.get("respects_time_window", True):
            reasons.append("Respecte la fenêtre horaire")

        if rl_decision.get("driver_available", True):
            reasons.append("Chauffeur disponible")

        # Raisons basées sur l'historique
        driver_id = rl_decision.get("driver_id")
        if (
            driver_id
            and context.get("driver_performance", {}).get(driver_id, {}).get("rating", 0) > HIGH_DRIVER_RATING_THRESHOLD
        ):
            reasons.append("Chauffeur bien noté")

        return reasons

    def _check_constraint_violations(self, rl_decision: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Vérifie les violations de contraintes."""
        violations = []

        # Vérifier fenêtre horaire
        if not rl_decision.get("respects_time_window", True):
            violations.append("Fenêtre horaire non respectée")

        # Vérifier disponibilité chauffeur
        if not rl_decision.get("driver_available", True):
            violations.append("Chauffeur non disponible")

        # Vérifier capacité véhicule
        if rl_decision.get("passenger_count", 0) > context.get("vehicle_capacity", 4):
            violations.append("Capacité véhicule dépassée")

        # Vérifier zone géographique
        if not rl_decision.get("in_service_area", True):
            violations.append("Hors zone de service")

        return violations

    def _calculate_performance_impact(
        self, human_decision: Dict[str, Any], rl_decision: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calcule l'impact sur la performance globale."""
        impact = {}

        # Impact sur ETA
        eta_improvement = human_decision.get("eta_minutes", 0) - rl_decision.get("eta_minutes", 0)
        impact["eta_improvement"] = eta_improvement

        # Impact sur distance
        distance_improvement = human_decision.get("distance_km", 0) - rl_decision.get("distance_km", 0)
        impact["distance_improvement"] = distance_improvement

        # Impact sur charge chauffeur
        load_balance = abs(rl_decision.get("driver_load", 0) - context.get("avg_load", 0)) - abs(
            human_decision.get("driver_load", 0) - context.get("avg_load", 0)
        )
        impact["load_balance"] = -load_balance  # Négatif = meilleur équilibre

        # Score global
        impact["global_score"] = eta_improvement * 0.4 + distance_improvement * 0.3 + load_balance * 0.3

        return impact

    def generate_daily_report(self, company_id: str, date: date | None = None) -> Dict[str, Any]:
        """Génère un rapport quotidien pour une entreprise.

        Args:
            company_id: ID de l'entreprise
            date: Date du rapport (par défaut aujourd'hui)

        Returns:
            Dictionnaire du rapport quotidien

        """
        if date is None:
            date = datetime.now(UTC).date()

        # Filtrer les données pour cette entreprise et cette date
        company_data = self._filter_data_by_company_and_date(company_id, date)

        if not company_data["decisions"]:
            return {
                "company_id": company_id,
                "date": date.isoformat(),
                "total_decisions": 0,
                "message": "Aucune décision enregistrée pour cette date",
            }

        # Calculer les statistiques
        stats = self._calculate_daily_statistics(company_data)

        # Générer le rapport
        report = {
            "company_id": company_id,
            "date": date.isoformat(),
            "total_decisions": len(company_data["decisions"]),
            "statistics": stats,
            "kpis_summary": self._generate_kpis_summary(company_data),
            "top_insights": self._generate_top_insights(company_data),
            "recommendations": self._generate_recommendations(company_data),
        }

        # Sauvegarder le rapport
        self._save_daily_report(report)

        return report

    def _filter_data_by_company_and_date(self, company_id: str, date: date) -> Dict[str, List[Any]]:
        """Filtre les données par entreprise et date."""
        filtered_data = {"decisions": [], "kpis": [], "metadata": []}

        for i, timestamp in enumerate(self.decision_metadata["timestamp"]):
            if timestamp.date() == date and self.decision_metadata["company_id"][i] == company_id:
                filtered_data["decisions"].append(
                    {
                        "timestamp": timestamp,
                        "booking_id": self.decision_metadata["booking_id"][i],
                        "driver_id": self.decision_metadata["driver_id"][i],
                        "human_decision": self.decision_metadata["human_decision"][i],
                        "rl_decision": self.decision_metadata["rl_decision"][i],
                        "context": self.decision_metadata["context"][i],
                    }
                )

                # Ajouter KPIs correspondants
                kpis = {}
                for metric, values in self.kpi_metrics.items():
                    if i < len(values):
                        kpis[metric] = values[i]
                filtered_data["kpis"].append(kpis)

                filtered_data["metadata"].append(
                    {
                        "timestamp": timestamp,
                        "company_id": company_id,
                        "booking_id": self.decision_metadata["booking_id"][i],
                    }
                )

        return filtered_data

    def _calculate_daily_statistics(self, company_data: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Calcule les statistiques quotidiennes."""
        stats = {}

        if not company_data["kpis"]:
            return stats

        # Statistiques ETA
        eta_deltas = [kpi.get("eta_delta", 0) for kpi in company_data["kpis"]]
        stats["eta_delta"] = {
            "mean": sum(eta_deltas) / len(eta_deltas),
            "median": sorted(eta_deltas)[len(eta_deltas) // 2],
            "min": min(eta_deltas),
            "max": max(eta_deltas),
            "std": self._calculate_std(eta_deltas),
        }

        # Statistiques retard
        delay_deltas = [kpi.get("delay_delta", 0) for kpi in company_data["kpis"]]
        stats["delay_delta"] = {
            "mean": sum(delay_deltas) / len(delay_deltas),
            "median": sorted(delay_deltas)[len(delay_deltas) // 2],
            "min": min(delay_deltas),
            "max": max(delay_deltas),
            "std": self._calculate_std(delay_deltas),
        }

        # Statistiques confiance RL
        rl_confidences = [kpi.get("rl_confidence", 0) for kpi in company_data["kpis"]]
        stats["rl_confidence"] = {
            "mean": sum(rl_confidences) / len(rl_confidences),
            "min": min(rl_confidences),
            "max": max(rl_confidences),
        }

        # Accord humain vs RL
        agreements = sum(
            1
            for decision in company_data["decisions"]
            if decision["human_decision"].get("driver_id") == decision["rl_decision"].get("driver_id")
        )
        stats["agreement_rate"] = agreements / len(company_data["decisions"])

        return stats

    def _calculate_std(self, values: List[float]) -> float:
        """Calcule l'écart-type."""
        if len(values) < MIN_VALUES_FOR_STD:
            return 0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def _generate_kpis_summary(self, company_data: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Génère un résumé des KPIs."""
        summary = {}

        if not company_data["kpis"]:
            return summary

        # Résumé des améliorations
        eta_improvements = [kpi.get("eta_delta", 0) for kpi in company_data["kpis"]]
        positive_eta = sum(1 for x in eta_improvements if x < X_ZERO)  # ETA RL < ETA humain
        summary["eta_improvement_rate"] = positive_eta / len(eta_improvements)

        # Résumé des violations
        violations = []
        for kpi in company_data["kpis"]:
            violations.extend(kpi.get("constraint_violations", []))
        summary["total_violations"] = len(violations)
        summary["violation_rate"] = len(violations) / len(company_data["kpis"])

        # Résumé des raisons
        all_reasons = []
        for kpi in company_data["kpis"]:
            all_reasons.extend(kpi.get("decision_reasons", []))

        reason_counts = {}
        for reason in all_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        summary["top_reasons"] = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return summary

    def _generate_top_insights(self, company_data: Dict[str, List[Any]]) -> List[str]:
        """Génère les insights principaux."""
        insights = []

        if not company_data["kpis"]:
            return insights

        # Insight 1: Performance ETA
        eta_deltas = [kpi.get("eta_delta", 0) for kpi in company_data["kpis"]]
        avg_eta_improvement = sum(eta_deltas) / len(eta_deltas)

        if avg_eta_improvement < RL_ETA_IMPROVEMENT_THRESHOLD:  # RL meilleur de plus de 2 minutes
            insights.append(f"RL améliore l'ETA de {abs(avg_eta_improvement)}")
        # Humain meilleur de plus de AVG_ETA_IMPROVEMENT_THRESHOLD minutes
        elif avg_eta_improvement > AVG_ETA_IMPROVEMENT_THRESHOLD:
            insights.append(f"L'humain améliore l'ETA de {avg_eta_improvement}")

        # Insight 2: Accord des décisions
        agreements = sum(
            1
            for decision in company_data["decisions"]
            if decision["human_decision"].get("driver_id") == decision["rl_decision"].get("driver_id")
        )
        agreement_rate = agreements / len(company_data["decisions"])

        if agreement_rate > INSIGHT_HIGH_AGREEMENT_THRESHOLD:
            insights.append(f"Taux d'accord élevé: {agreement_rate}")
        elif agreement_rate < INSIGHT_LOW_AGREEMENT_THRESHOLD:
            insights.append(f"Taux d'accord faible: {agreement_rate}")

        # Insight 3: Confiance RL
        rl_confidences = [kpi.get("rl_confidence", 0) for kpi in company_data["kpis"]]
        avg_confidence = sum(rl_confidences) / len(rl_confidences)

        if avg_confidence > HIGH_RL_CONFIDENCE_THRESHOLD:
            insights.append(f"Confiance RL élevée: {avg_confidence}")
        elif avg_confidence < LOW_RL_CONFIDENCE_THRESHOLD:
            insights.append(f"Confiance RL faible: {avg_confidence}")

        return insights

    def _generate_recommendations(self, company_data: Dict[str, List[Any]]) -> List[str]:
        """Génère des recommandations basées sur les données."""
        recommendations = []

        if not company_data["kpis"]:
            return recommendations

        # Recommandation 1: Basée sur l'accord
        agreements = sum(
            1
            for decision in company_data["decisions"]
            if decision["human_decision"].get("driver_id") == decision["rl_decision"].get("driver_id")
        )
        agreement_rate = agreements / len(company_data["decisions"])

        if agreement_rate > HIGH_AGREEMENT_RATE_THRESHOLD:
            recommendations.append("Taux d'accord élevé - Considérer l'activation du mode automatique")
        elif agreement_rate < LOW_AGREEMENT_RATE_THRESHOLD:
            recommendations.append("Taux d'accord faible - Analyser les différences de logique")

        # Recommandation 2: Basée sur les violations
        violations = []
        for kpi in company_data["kpis"]:
            violations.extend(kpi.get("constraint_violations", []))

        if len(violations) > len(company_data["kpis"]) * 0.1:  # Plus de 10% de violations
            recommendations.append("Taux de violations élevé - Revoir les contraintes RL")

        # Recommandation 3: Basée sur la performance
        eta_deltas = [kpi.get("eta_delta", 0) for kpi in company_data["kpis"]]
        avg_eta_improvement = sum(eta_deltas) / len(eta_deltas)

        if avg_eta_improvement < EXCELLENT_RL_ETA_IMPROVEMENT_THRESHOLD:  # RL meilleur de plus de 5 minutes
            recommendations.append("Performance RL excellente - Augmenter la confiance dans les suggestions")

        return recommendations

    def _save_daily_report(self, report: Dict[str, Any]) -> None:
        """Sauvegarde le rapport quotidien."""
        company_id = report["company_id"]
        date_str = report["date"]

        # Créer le répertoire de l'entreprise
        company_dir = self.data_dir / company_id
        company_dir.mkdir(exist_ok=True)

        # Sauvegarder en JSON
        json_path = company_dir / f"report_{date_str}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Sauvegarder en CSV (données tabulaires)
        csv_path = company_dir / f"data_{date_str}.csv"
        self._export_to_csv(report, csv_path)

        self.logger.info("Daily report saved: %s", json_path)

    def _export_to_csv(self, report: Dict[str, Any], csv_path: Path) -> None:
        """Exporte les données du rapport en CSV."""
        # Préparer les données pour le CSV
        csv_data = []

        company_id = report["company_id"]
        date = report["date"]

        # Ajouter les métadonnées de base
        base_data = {"company_id": company_id, "date": date, "total_decisions": report["total_decisions"]}

        # Ajouter les statistiques
        stats = report.get("statistics", {})
        for stat_name, stat_values in stats.items():
            if isinstance(stat_values, dict):
                for metric, value in stat_values.items():
                    base_data[f"{stat_name}_{metric}"] = value
            else:
                base_data[stat_name] = stat_values

        # Ajouter le résumé KPIs
        kpis_summary = report.get("kpis_summary", {})
        for kpi_name, kpi_value in kpis_summary.items():
            if isinstance(kpi_value, list):
                # Traiter les listes (comme top_reasons)
                for i, item in enumerate(kpi_value):
                    if isinstance(item, tuple):
                        base_data[f"{kpi_name}_{i}_reason"] = item[0]
                        base_data[f"{kpi_name}_{i}_count"] = item[1]
                    else:
                        base_data[f"{kpi_name}_{i}"] = item
            else:
                base_data[kpi_name] = kpi_value

        csv_data.append(base_data)

        # Créer le DataFrame et sauvegarder
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)

    def get_company_summary(self, company_id: str, days: int = 7) -> Dict[str, Any]:
        """Génère un résumé sur plusieurs jours pour une entreprise.

        Args:
            company_id: ID de l'entreprise
            days: Nombre de jours à analyser

        Returns:
            Résumé multi-jours

        """
        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=days - 1)

        daily_reports = []
        for i in range(days):
            date = start_date + timedelta(days=i)
            report = self.generate_daily_report(company_id, date)
            if report["total_decisions"] > 0:
                daily_reports.append(report)

        if not daily_reports:
            return {
                "company_id": company_id,
                "period_days": days,
                "total_decisions": 0,
                "message": "Aucune décision enregistrée sur cette période",
            }

        # Calculer les moyennes sur la période
        return {
            "company_id": company_id,
            "period_days": days,
            "total_decisions": sum(r["total_decisions"] for r in daily_reports),
            "avg_decisions_per_day": sum(r["total_decisions"] for r in daily_reports) / len(daily_reports),
            "avg_agreement_rate": self._calculate_avg_agreement_rate(daily_reports),
            "avg_eta_improvement": self._calculate_avg_eta_improvement(daily_reports),
            "trend_analysis": self._analyze_trends(daily_reports),
        }

    def _calculate_avg_agreement_rate(self, daily_reports: List[Dict[str, Any]]) -> float:
        """Calcule le taux d'accord moyen."""
        agreement_rates = []
        for report in daily_reports:
            stats = report.get("statistics", {})
            if "agreement_rate" in stats:
                agreement_rates.append(stats["agreement_rate"])

        return sum(agreement_rates) / len(agreement_rates) if agreement_rates else 0

    def _calculate_avg_eta_improvement(self, daily_reports: List[Dict[str, Any]]) -> float:
        """Calcule l'amélioration ETA moyenne."""
        eta_improvements = []
        for report in daily_reports:
            stats = report.get("statistics", {})
            eta_delta = stats.get("eta_delta", {})
            if "mean" in eta_delta:
                eta_improvements.append(eta_delta["mean"])

        return sum(eta_improvements) / len(eta_improvements) if eta_improvements else 0

    def _analyze_trends(self, daily_reports: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyse les tendances sur la période."""
        trends = {}

        if len(daily_reports) < MIN_DAILY_REPORTS_FOR_TRENDS:
            return trends

        # Tendance du taux d'accord
        agreement_rates = []
        for report in daily_reports:
            stats = report.get("statistics", {})
            if "agreement_rate" in stats:
                agreement_rates.append(stats["agreement_rate"])

        if len(agreement_rates) >= MIN_DAILY_REPORTS_FOR_TRENDS:
            if agreement_rates[-1] > agreement_rates[0]:
                trends["agreement_rate"] = "Amélioration"
            elif agreement_rates[-1] < agreement_rates[0]:
                trends["agreement_rate"] = "Dégradation"
            else:
                trends["agreement_rate"] = "Stable"

        # Tendance de l'amélioration ETA
        eta_improvements = []
        for report in daily_reports:
            stats = report.get("statistics", {})
            eta_delta = stats.get("eta_delta", {})
            if "mean" in eta_delta:
                eta_improvements.append(eta_delta["mean"])

        if len(eta_improvements) >= MIN_DAILY_REPORTS_FOR_TRENDS:
            if eta_improvements[-1] < eta_improvements[0]:  # ETA RL s'améliore
                trends["eta_improvement"] = "Amélioration"
            elif eta_improvements[-1] > eta_improvements[0]:
                trends["eta_improvement"] = "Dégradation"
            else:
                trends["eta_improvement"] = "Stable"

        return trends

    def clear_old_data(self, days_to_keep: int = 30) -> None:
        """Nettoie les anciennes données.

        Args:
            days_to_keep: Nombre de jours de données à conserver

        """
        cutoff_date = datetime.now(UTC) - timedelta(days=days_to_keep)

        # Filtrer les données récentes
        recent_indices = []
        for i, timestamp in enumerate(self.decision_metadata["timestamp"]):
            if timestamp >= cutoff_date:
                recent_indices.append(i)

        # Garder seulement les données récentes
        for key in self.decision_metadata:
            self.decision_metadata[key] = [self.decision_metadata[key][i] for i in recent_indices]

        for key in self.kpi_metrics:
            self.kpi_metrics[key] = [self.kpi_metrics[key][i] for i in recent_indices]

        self.logger.info("Cleaned old data, kept %s recent decisions", len(recent_indices))

    def generate_shadow_suggestions(
        self, bookings: List[Any], drivers: List[Any], current_assignments: Dict[int, int]
    ) -> List[Dict[str, Any]]:
        """Génère des suggestions RL en mode shadow (sans impact production).

        Args:
            dispatch_run_id: ID du dispatch run
            bookings: Liste des bookings à dispatcher
            drivers: Liste des drivers disponibles
            current_assignments: Dict {booking_id: driver_id} des assignations courantes

        Returns:
            Liste de suggestions RL formatées
        """
        suggestions = []

        try:
            from services.rl.suggestion_generator import get_suggestion_generator

            generator = get_suggestion_generator()

            for booking in bookings:
                # Simuler état actuel pour RL
                suggested_driver_id = current_assignments.get(booking.id)

                if not suggested_driver_id:
                    continue

                # Générer suggestion RL
                rl_suggestion = generator.generate_suggestion_for_booking(
                    booking=booking,
                    current_driver_id=suggested_driver_id,
                    available_drivers=drivers,
                    current_assignments=current_assignments,
                )

                if rl_suggestion:
                    suggestions.append(
                        {
                            "booking_id": booking.id,
                            "current_driver_id": suggested_driver_id,
                            "rl_suggested_driver_id": rl_suggestion.get("driver_id"),
                            "confidence": rl_suggestion.get("confidence", 0.0),
                            "score": rl_suggestion.get("score", 0.0),
                            "reason": rl_suggestion.get("reason", ""),
                        }
                    )

        except Exception as e:
            self.logger.error("[ShadowMode] Error generating suggestions: %s", e)

        return suggestions

    def store_shadow_suggestions(
        self, dispatch_run_id: int, suggestions: List[Dict[str, Any]], kpi_snapshot: Dict[str, Any] | None = None
    ) -> int:
        """Stocke les suggestions RL en mode shadow dans la DB.

        Args:
            dispatch_run_id: ID du dispatch run
            suggestions: Liste de suggestions à stocker
            kpi_snapshot: Snapshot des KPIs au moment de la suggestion

        Returns:
            Nombre de suggestions stockées
        """
        stored_count = 0

        try:
            from ext import db as ext_db
            from models import RLSuggestion

            for suggestion in suggestions:
                rl_sugg = RLSuggestion()
                rl_sugg.dispatch_run_id = dispatch_run_id
                rl_sugg.booking_id = suggestion["booking_id"]
                rl_sugg.driver_id = suggestion["rl_suggested_driver_id"]
                rl_sugg.score = suggestion.get("score", 0.0)
                rl_sugg.kpi_snapshot = kpi_snapshot
                ext_db.session.add(rl_sugg)
                stored_count += 1

            ext_db.session.commit()
            self.logger.info("[ShadowMode] Stored %s suggestions for dispatch_run %s", stored_count, dispatch_run_id)

        except Exception as e:
            from ext import db as ext_db

            ext_db.session.rollback()
            self.logger.error("[ShadowMode] Failed to store suggestions: %s", e)

        return stored_count

    def compare_shadow_with_actual(self, dispatch_run_id: int, actual_assignments: Dict[int, int]) -> Dict[str, Any]:
        """Compare les suggestions shadow avec les assignations réelles.

        Args:
            dispatch_run_id: ID du dispatch run
            actual_assignments: Dict {booking_id: driver_id} des assignations réelles

        Returns:
            Statistiques de comparaison
        """
        try:
            from models import RLSuggestion

            # Récupérer les suggestions shadow
            shadow_suggestions = RLSuggestion.query.filter_by(dispatch_run_id=dispatch_run_id).all()

            if not shadow_suggestions:
                return {"total": 0, "message": "No shadow suggestions found"}

            # Comparaison
            agreements = 0
            total = len(shadow_suggestions)

            for sugg in shadow_suggestions:
                actual_driver = actual_assignments.get(sugg.booking_id)
                if actual_driver == sugg.driver_id:
                    agreements += 1

            agreement_rate = agreements / total if total > 0 else 0

            return {
                "total_suggestions": total,
                "agreements": agreements,
                "disagreements": total - agreements,
                "agreement_rate": agreement_rate,
                "dispatch_run_id": dispatch_run_id,
            }

        except Exception as e:
            self.logger.error("[ShadowMode] Failed to compare: %s", e)
            return {"error": str(e)}
