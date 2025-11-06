"""Phase 5.1 - Front Pareto pour multi-objectif.

Permet de conserver plusieurs solutions non-dominées et choisir
selon le poids équité/efficacité du slider.

Critère: Capacité op d'ajuster sans redéploiement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ParetoSolution:
    """Solution avec métriques multi-objectif."""
    
    solution_id: int
    assignments: List[Dict[str, Any]]
    efficiency_score: float  # Distance totale minimisée
    fairness_score: float   # Distribution équitable
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "solution_id": self.solution_id,
            "assignments": self.assignments,
            "efficiency_score": self.efficiency_score,
            "fairness_score": self.fairness_score,
        }


class ParetoFront:
    """Gère le front Pareto pour solutions multi-objectif.
    
    Conserve uniquement les solutions non-dominées (Pareto-optimales).
    """
    
    def __init__(self, max_solutions: int = 3):
        super().__init__()
        """Initialise le front Pareto.
        
        Args:
            max_solutions: Nombre max de solutions à conserver
        """
        self.max_solutions = max_solutions
        self.solutions: List[ParetoSolution] = []
        self.next_id = 0
    
    def add_solution(
        self,
        assignments: List[Dict[str, Any]],
        efficiency_score: float,
        fairness_score: float
    ) -> bool:
        """Ajoute une solution au front Pareto si non-dominée.
        
        Args:
            assignments: Assignations de la solution
            efficiency_score: Score efficacité (à minimiser)
            fairness_score: Score équité (à maximiser)
            
        Returns:
            True si ajoutée, False si dominée
        """
        new_solution = ParetoSolution(
            solution_id=self.next_id,
            assignments=assignments,
            efficiency_score=efficiency_score,
            fairness_score=fairness_score
        )
        self.next_id += 1
        
        # Vérifier si cette solution est dominée
        dominated_by = self._find_dominating(new_solution)
        
        if dominated_by:
            logger.debug(
                "[ParetoFront] Solution %d dominée, ignorée",
                new_solution.solution_id
            )
            return False
        
        # Retirer les solutions dominées par la nouvelle
        self.solutions = [
            s for s in self.solutions
            if not self._is_dominated(s, new_solution)
        ]
        
        # Ajouter la nouvelle solution
        self.solutions.append(new_solution)
        
        # Limiter au max_solutions si nécessaire
        if len(self.solutions) > self.max_solutions:
            # Garder les plus diversifiées
            self.solutions = self._select_diverse(self.max_solutions)
        
        logger.info(
            "[ParetoFront] Solution ajoutée: efficiency=%.2f, fairness=%.2f (total=%d)",
            efficiency_score, fairness_score, len(self.solutions)
        )
        
        return True
    
    def select_best(
        self,
        fairness_weight: float
    ) -> ParetoSolution | None:
        """Sélectionne la meilleure solution selon le slider.
        
        Args:
            fairness_weight: Poids équité (0.0-1.0)
                0.0 = efficacité pure
                1.0 = équité pure
                
        Returns:
            ParetoSolution ou None
        """
        if not self.solutions:
            return None
        
        # Calculer score agrégé pour chaque solution
        best_solution = None
        best_score = float("-inf")
        
        for solution in self.solutions:
            # Score agrégé: trade-off équité/efficacité
            aggregated_score = (
                (1 - fairness_weight) * (1 / (1 + solution.efficiency_score))  # Maximiser efficacité
                + fairness_weight * solution.fairness_score  # Maximiser équité
            )
            
            if aggregated_score > best_score:
                best_score = aggregated_score
                best_solution = solution
        
        logger.info(
            "[ParetoFront] Solution sélectionnée: fairness_weight=%.2f, solution_id=%d, efficiency=%.2f, fairness=%.2f",
            fairness_weight,
            best_solution.solution_id if best_solution else -1,
            best_solution.efficiency_score if best_solution else 0,
            best_solution.fairness_score if best_solution else 0
        )
        
        return best_solution
    
    def _is_dominated(
        self,
        solution: ParetoSolution,
        other: ParetoSolution
    ) -> bool:
        """Vérifie si solution est dominée par other.
        
        Dominée = plus mauvaise sur TOUS les objectifs.
        
        Args:
            solution: Solution à vérifier
            other: Solution de référence
            
        Returns:
            True si solution est dominée
        """
        # Pour efficacité: plus petit = mieux (à minimiser)
        better_efficiency = other.efficiency_score <= solution.efficiency_score
        
        # Pour équité: plus grand = mieux (à maximiser)
        better_fairness = other.fairness_score >= solution.fairness_score
        
        # Dominance: better sur AU MOINS un objectif, et jamais pire
        return (
            (better_efficiency and other.fairness_score > solution.fairness_score) or
            (better_fairness and other.efficiency_score < solution.efficiency_score)
        ) and (
            other.efficiency_score <= solution.efficiency_score and
            other.fairness_score >= solution.fairness_score
        )
    
    def _find_dominating(
        self,
        solution: ParetoSolution
    ) -> ParetoSolution | None:
        """Trouve une solution dominante dans le front.
        
        Args:
            solution: Solution à vérifier
            
        Returns:
            Solution dominante ou None
        """
        for existing in self.solutions:
            if self._is_dominated(solution, existing):
                return existing
        return None
    
    def _select_diverse(self, n: int) -> List[ParetoSolution]:
        """Sélectionne les N solutions les plus diversifiées.
        
        Args:
            n: Nombre de solutions à garder
            
        Returns:
            Liste des solutions les plus diversifiées
        """
        if len(self.solutions) <= n:
            return self.solutions
        
        # Tri par diversité (distance euclidienne dans espace objectives)
        scored = []
        for i, sol in enumerate(self.solutions):
            # Distance à la solution moyenne
            avg_eff = sum(s.efficiency_score for s in self.solutions) / len(self.solutions)
            avg_fair = sum(s.fairness_score for s in self.solutions) / len(self.solutions)
            
            diversity = (
                (sol.efficiency_score - avg_eff) ** 2 +
                (sol.fairness_score - avg_fair) ** 2
            ) ** 0.5
            
            scored.append((diversity, i, sol))
        
        # Garder les plus diverses
        scored.sort(reverse=True, key=lambda x: x[0])
        
        return [sol for _, _, sol in scored[:n]]
    
    def get_all(self) -> List[ParetoSolution]:
        """Retourne toutes les solutions du front."""
        return self.solutions
    
    def clear(self) -> None:
        """Vide le front Pareto."""
        self.solutions = []
        self.next_id = 0


def calculate_efficiency_score(
    assignments: List[Dict[str, Any]],
    _problem: Dict[str, Any] | None = None
) -> float:
    """Calcule le score d'efficacité (distance totale).
    
    Args:
        assignments: Liste d'assignations
        problem: Problème avec time_matrix, etc.
        
    Returns:
        Score efficacité (à minimiser)
    """
    # Estimé basé sur nombre d'assignments (simplifié)
    # En production, calculer depuis time_matrix réelle
    if not assignments:
        return 999.0
    
    # Score inversement proportionnel au nombre d'assignments
    # Plus d'assignments = meilleur (couverture)
    efficiency = len(assignments)
    
    return 1.0 / (1.0 + efficiency)  # Normaliser pour minimisation


def calculate_fairness_score(
    assignments: List[Dict[str, Any]]
) -> float:
    """Calcule le score d'équité (distribution uniforme).
    
    Args:
        assignments: Liste d'assignations
        
    Returns:
        Score équité (0.0-1.0, à maximiser)
    """
    if not assignments:
        return 0.0
    
    from collections import Counter
    
    # Compter assignments par driver
    driver_counts = Counter(a.get("driver_id") for a in assignments)
    
    if not driver_counts:
        return 0.0
    
    counts = list(driver_counts.values())
    
    # Équité = 1 - écart-type normalisé
    mean_count = sum(counts) / len(counts)
    
    if mean_count == 0:
        return 0.0
    
    variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
    std_dev = variance ** 0.5
    
    # Normaliser (max std quand un driver a tout)
    max_std = mean_count if len(counts) > 1 else 0
    
    if max_std == 0:
        return 1.0
    
    fairness = 1.0 - (std_dev / max_std)
    
    return max(0.0, min(1.0, fairness))

