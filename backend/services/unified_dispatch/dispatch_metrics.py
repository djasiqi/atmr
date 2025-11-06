
# Constantes pour éviter les valeurs magiques
from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Tuple

from ext import db
from models import Assignment, Booking, BookingStatus, DispatchRun, Driver
from shared.time_utils import day_local_bounds, now_local

DELAY_MINUTES_THRESHOLD = 5
DELAYED_ZERO = 0
AVG_ZERO = 0
POOLING_WINDOW_SECONDS = 600  # 10 minutes en secondes
MIN_VALUES_FOR_GINI = 2  # ✅ C4: Minimum drivers pour calculer Gini

# ✅ B1: Version du calcul de quality score
QUALITY_FORMULA_VERSION = "v1.0"
QUALITY_WEIGHTS = {
    "assignment": 0.30,
    "on_time": 0.30,
    "pooling": 0.15,
    "fairness": 0.15,
    "delay": 0.10,
}
QUALITY_THRESHOLD = 70.0  # Seuil pour activer auto-apply RL

def get_quality_formula_hash() -> str:
    """Retourne le hash des poids pour traçabilité."""
    weights_str = str(sorted(QUALITY_WEIGHTS.items()))
    return hashlib.sha256(weights_str.encode()).hexdigest()[:8]

"""Système de métriques de qualité pour le dispatch.
Collecte, calcule et expose les indicateurs de performance.
"""


logger = logging.getLogger(__name__)


@dataclass
class DispatchQualityMetrics:
    """Métriques de qualité d'un dispatch."""

    # Identifiants
    dispatch_run_id: int | None
    company_id: int
    date: date
    calculated_at: datetime

    # Métriques d'assignation
    total_bookings: int
    assigned_bookings: int
    unassigned_bookings: int
    assignment_rate: float  # % assigné

    # Métriques de pooling
    pooled_bookings: int
    pooling_rate: float  # % regroupé

    # Métriques de retard
    on_time_bookings: int
    delayed_bookings: int
    average_delay_minutes: float
    max_delay_minutes: int

    # Métriques d'équité
    drivers_used: int
    avg_bookings_per_driver: float
    max_bookings_per_driver: int
    min_bookings_per_driver: int
    fairness_coefficient: float  # 0-1 (1 = parfaitement équitable)
    gini_index: float  # ✅ C4: Indice de Gini (0 = parfait, 1 = inéquitable)

    # Métriques de coût
    total_distance_km: float
    avg_distance_per_booking: float
    emergency_drivers_used: int
    emergency_bookings: int

    # Performance algorithmique
    solver_used: bool
    heuristic_used: bool
    fallback_used: bool
    execution_time_sec: float

    # Score global de qualité (0-100)
    quality_score: float
    quality_formula_version: str = QUALITY_FORMULA_VERSION
    quality_weights_hash: str = ""
    dominant_factors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON."""
        d = asdict(self)
        d["date"] = self.date.isoformat()
        d["calculated_at"] = self.calculated_at.isoformat()
        return d

    def to_summary(self) -> Dict[str, Any]:
        """Résumé compact pour dashboard."""
        return {
            "quality_score": round(self.quality_score, 1),
            "assignment_rate": round(self.assignment_rate, 1),
            "on_time_rate": round((self.on_time_bookings / max(1, self.total_bookings)) * 100, 1),
            "pooling_rate": round(self.pooling_rate, 1),
            "fairness": round(self.fairness_coefficient * 100, 1),
            "avg_delay": round(self.average_delay_minutes, 1),
        }


class DispatchMetricsCollector:
    """Collecteur de métriques pour un dispatch."""

    def __init__(self, company_id: int):  # pyright: ignore[reportMissingSuperCall]
        self.company_id = company_id

    def collect_for_run(self, dispatch_run_id: int) -> DispatchQualityMetrics:
        """Collecte les métriques pour un DispatchRun spécifique.

        Args:
            dispatch_run_id: ID du DispatchRun à analyser
        Returns:
            DispatchQualityMetrics avec toutes les métriques calculées

        """
        try:
            # Récupérer le DispatchRun
            run = db.session.get(DispatchRun, dispatch_run_id)
            if not run or run.company_id != self.company_id:
                msg = f"DispatchRun {dispatch_run_id} not found or access denied"
                raise ValueError(msg)

            run_date = run.day

            # Récupérer toutes les assignations de ce run
            assignments = Assignment.query.filter_by(
                dispatch_run_id=dispatch_run_id).all()

            # Récupérer tous les bookings du jour (pour calculer le taux
            # d'assignation)
            d0, d1 = day_local_bounds(run_date.strftime("%Y-%m-%d"))
            all_bookings = Booking.query.filter(
                Booking.company_id == self.company_id,
                Booking.scheduled_time >= d0,
                Booking.scheduled_time < d1,
                Booking.status.in_(
                    [BookingStatus.ACCEPTED, BookingStatus.ASSIGNED])
            ).all()

            # Calculer les métriques
            return self._calculate_metrics(
                dispatch_run_id=dispatch_run_id,
                run_date=run_date,
                assignments=assignments,
                all_bookings=all_bookings,
                run_metadata=run.metrics or {}
            )

        except Exception as e:
            logger.exception(
                "[DispatchMetrics] Failed to collect metrics for run %s: %s",
                dispatch_run_id,
                e)
            raise

    def collect_for_date(self, target_date: date |
                         str) -> DispatchQualityMetrics:
        """Collecte les métriques pour une date donnée (dernier run de la journée).

        Args:
            target_date: Date à analyser (date object ou string YYYY-MM-DD)

        Returns:
            DispatchQualityMetrics

        """
        if isinstance(target_date, str):
            target_date = date.fromisoformat(target_date)

        # Trouver le dernier DispatchRun de cette journée
        run = (
            DispatchRun.query
            .filter_by(company_id=self.company_id, day=target_date)
            .order_by(DispatchRun.completed_at.desc())
            .first()
        )

        if not run:
            msg = f"No DispatchRun found for company {self.company_id} on {target_date}"
            raise ValueError(msg)

        return self.collect_for_run(run.id)

    def _calculate_metrics(
        self,
        dispatch_run_id: int | None,
        run_date: date,
        assignments: List[Assignment],
        all_bookings: List[Booking],
        run_metadata: Dict[str, Any]
    ) -> DispatchQualityMetrics:
        """Calcule toutes les métriques à partir des données collectées."""
        # 1. Métriques d'assignation
        total_bookings = len(all_bookings)
        assigned_bookings = len(assignments)
        unassigned_bookings = total_bookings - assigned_bookings
        assignment_rate = (assigned_bookings / max(1, total_bookings)) * 100

        # 2. Métriques de pooling
        pooled_bookings = self._count_pooled_bookings(assignments)
        pooling_rate = (pooled_bookings / max(1, assigned_bookings)) * 100

        # 3. Métriques de retard
        on_time, delayed, avg_delay, max_delay = self._calculate_delay_metrics(
            assignments)

        # 4. Métriques d'équité
        driver_stats = self._calculate_driver_fairness(assignments)

        # 5. Métriques de coût
        distance_stats = self._calculate_distance_metrics(
            assignments, all_bookings)

        # 6. Performance algorithmique
        algo_stats = self._extract_algorithm_stats(run_metadata)

        # 7. Score global de qualité (0-100) avec version/hash
        quality_score, dominant_factors = self._calculate_quality_score(
            assignment_rate=assignment_rate,
            on_time_rate=(on_time / max(1, assigned_bookings)) * 100,
            pooling_rate=pooling_rate,
            fairness=driver_stats["fairness_coefficient"],
            avg_delay=avg_delay
        )

        return DispatchQualityMetrics(
            dispatch_run_id=dispatch_run_id,
            company_id=self.company_id,
            date=run_date,
            calculated_at=now_local(),

            total_bookings=total_bookings,
            assigned_bookings=assigned_bookings,
            unassigned_bookings=unassigned_bookings,
            assignment_rate=assignment_rate,

            pooled_bookings=pooled_bookings,
            pooling_rate=pooling_rate,

            on_time_bookings=on_time,
            delayed_bookings=delayed,
            average_delay_minutes=avg_delay,
            max_delay_minutes=max_delay,

            drivers_used=driver_stats["drivers_used"],
            avg_bookings_per_driver=driver_stats["avg_bookings_per_driver"],
            max_bookings_per_driver=driver_stats["max_bookings_per_driver"],
            min_bookings_per_driver=driver_stats["min_bookings_per_driver"],
            fairness_coefficient=driver_stats["fairness_coefficient"],
            gini_index=driver_stats["gini_index"],  # ✅ C4

            total_distance_km=distance_stats["total_distance_km"],
            avg_distance_per_booking=distance_stats["avg_distance_per_booking"],
            emergency_drivers_used=distance_stats["emergency_drivers_used"],
            emergency_bookings=distance_stats["emergency_bookings"],

            solver_used=algo_stats["solver_used"],
            heuristic_used=algo_stats["heuristic_used"],
            fallback_used=algo_stats["fallback_used"],
            execution_time_sec=algo_stats["execution_time_sec"],

            quality_score=quality_score,
            quality_formula_version=QUALITY_FORMULA_VERSION,
            quality_weights_hash=get_quality_formula_hash(),
            dominant_factors=dominant_factors
        )

    def _count_pooled_bookings(self, assignments: List[Assignment]) -> int:
        """Compte les bookings regroupés (même chauffeur, même heure ±10min)."""
        pooled = 0
        driver_times: Dict[int, List[datetime]] = {}

        for assignment in assignments:
            # Récupérer l'ID du chauffeur
            driver_id = assignment.driver_id

            booking = db.session.get(Booking, assignment.booking_id)
            if not booking:
                continue

            scheduled_time_raw = booking.scheduled_time
            if scheduled_time_raw is None:
                continue

            # Cast pour éviter Column[datetime]
            scheduled_time = scheduled_time_raw

            # Utiliser driver_id directement (SQLAlchemy gère les types)
  
            if driver_id not in driver_times:
                driver_times[driver_id] = []  # type: ignore

            # Vérifier si un autre booking du même chauffeur est à moins de
            # 10min
            for existing_time in driver_times[driver_id]:  # type: ignore
                if abs((scheduled_time - existing_time).total_seconds()
                       ) <= POOLING_WINDOW_SECONDS:  # 10min
                    pooled += 1
                    break

            driver_times[driver_id].append(scheduled_time)  # type: ignore

        return pooled

    def _calculate_delay_metrics(
            self, assignments: List[Assignment]) -> tuple[int, int, float, int]:
        """Calcule les métriques de retard."""
        on_time = 0
        delayed = 0
        total_delay = 0
        max_delay = 0

        # ✅ PERF: Charger tous les bookings en une seule query (évite N+1)
        booking_ids = [a.booking_id for a in assignments]
        bookings_map = {
            b.id: b for b in Booking.query.filter(Booking.id.in_(booking_ids)).all()
        } if booking_ids else {}

        for assignment in assignments:
            booking = bookings_map.get(assignment.booking_id)
            if not booking:
                continue

            scheduled_time = booking.scheduled_time
            if scheduled_time is None:
                continue

            # Calculer le retard
            eta_pickup = assignment.eta_pickup_at
            if eta_pickup is not None and scheduled_time is not None:
                delay_minutes = int(
                    (eta_pickup - scheduled_time).total_seconds() / 60)

                if delay_minutes <= DELAY_MINUTES_THRESHOLD:
                    on_time += 1
                else:
                    delayed += 1
                    total_delay += delay_minutes
                    max_delay = max(max_delay, delay_minutes)
            else:
                # Pas d'ETA → considéré comme à l'heure par défaut
                on_time += 1

        avg_delay = total_delay / \
            max(1, delayed) if delayed > DELAYED_ZERO else DELAYED_ZERO

        return on_time, delayed, avg_delay, max_delay

    def _calculate_driver_fairness(
            self, assignments: List[Assignment]) -> Dict[str, Any]:
        """Calcule les métriques d'équité entre chauffeurs."""
        driver_counts: Dict[int, int] = {}

        for assignment in assignments:
            driver_id = assignment.driver_id
            driver_counts[driver_id] = driver_counts.get(driver_id, 0) + 1  # type: ignore

        if not driver_counts:
            return {
                "drivers_used": 0,
                "avg_bookings_per_driver": 0,
                "max_bookings_per_driver": 0,
                "min_bookings_per_driver": 0,
                "fairness_coefficient": 1.0,
                "gini_index": 0.0  # ✅ C4: Parfaitement équitable
            }

        counts = list(driver_counts.values())
        avg = sum(counts) / len(counts)
        max_count = max(counts)
        min_count = min(counts)

        # Coefficient d'équité : 1 - (écart-type / moyenne)
        # 1.0 = parfaitement équitable, 0.0 = très inéquitable
        if avg > AVG_ZERO:
            variance = sum((c - avg) ** 2 for c in counts) / len(counts)
            std_dev = variance ** 0.5
            fairness = max(0, 1 - (std_dev / avg))
        else:
            fairness = 1.0

        # ✅ C4: Calcul de l'indice de Gini
        gini = self._calculate_gini_index(counts)

        return {
            "drivers_used": len(driver_counts),
            "avg_bookings_per_driver": avg,
            "max_bookings_per_driver": max_count,
            "min_bookings_per_driver": min_count,
            "fairness_coefficient": fairness,
            "gini_index": gini
        }
    
    def _calculate_gini_index(self, values: List[int]) -> float:
        """✅ C4: Calcule l'indice de Gini pour mesurer l'inégalité de répartition.
        
        L'indice de Gini mesure l'inégalité de distribution :
        - 0 = parfaitement équitable (tous les drivers ont le même nombre de courses)
        - 1 = parfaitement inéquitable (un seul driver a toutes les courses)
        
        Args:
            values: Liste des nombres de courses par driver
            
        Returns:
            Indice de Gini (0-1)
        """
        if not values or len(values) < MIN_VALUES_FOR_GINI:
            return 0.0
        
        n = len(values)
        values_sorted = sorted(values)
        
        # Formule de Gini : G = (2 * Σ(i * value_i)) / (n * Σ(value_i)) - (n+1)/n
        cumsum_values = sum(value * (i + 1) for i, value in enumerate(values_sorted))
        total = sum(values_sorted)
        
        if total == 0:
            return 0.0
        
        gini = (2 * cumsum_values) / (n * total) - (n + 1) / n
        
        return max(0.0, min(1.0, gini))

    def _calculate_distance_metrics(
            self, assignments: List[Assignment], all_bookings: List[Booking]) -> Dict[str, Any]:
        """Calcule les métriques de distance."""
        total_distance = 0
        emergency_drivers = set()
        emergency_bookings_count = 0

        # ✅ PERF: Utiliser all_bookings déjà fourni au lieu de db.session.get()
        bookings_map = {b.id: b for b in all_bookings}

        for assignment in assignments:
            booking = bookings_map.get(assignment.booking_id)  # type: ignore
            if booking:
                # Distance en km
                distance_m = getattr(booking, "distance_meters", 0) or 0
                total_distance += distance_m / 1000

                # Chauffeurs d'urgence
                driver_id = assignment.driver_id
                driver = db.session.get(Driver, driver_id)
                if driver and getattr(driver, "is_emergency", False):
                    emergency_drivers.add(driver_id)
                    emergency_bookings_count += 1

        avg_distance = total_distance / max(1, len(assignments))

        return {
            "total_distance_km": total_distance,
            "avg_distance_per_booking": avg_distance,
            "emergency_drivers_used": len(emergency_drivers),
            "emergency_bookings": emergency_bookings_count
        }

    def _extract_algorithm_stats(
            self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les stats algorithmiques du metadata."""
        return {
            "solver_used": metadata.get("used_solver", False),
            "heuristic_used": metadata.get("used_heuristic", True),
            "fallback_used": metadata.get("used_fallback", False),
            "execution_time_sec": metadata.get("execution_time", 0)
        }

    def _calculate_quality_score(
        self,
        assignment_rate: float,
        on_time_rate: float,
        pooling_rate: float,
        fairness: float,
        avg_delay: float
    ) -> Tuple[float, Dict[str, float]]:
        """Calcule un score global de qualité (0-100).

        Pondération :
        - 30% : Taux d'assignation
        - 30% : Taux de ponctualité
        - 15% : Taux de pooling
        - 15% : Équité chauffeurs
        - 10% : Retard moyen (pénalité)
        
        Returns:
            Tuple de (score, dominant_factors)
        """
        score = 0
        
        # Calculer chaque contribution
        assignment_contrib = (assignment_rate / 100) * 30
        on_time_contrib = (on_time_rate / 100) * 30
        pooling_contrib = (pooling_rate / 100) * 15
        fairness_contrib = fairness * 15
        delay_penalty = max(0, 10 - (avg_delay / 3))
        
        score = assignment_contrib + on_time_contrib + pooling_contrib + fairness_contrib + delay_penalty
        
        # Calculer les facteurs dominants (top 3)
        contributions = {
            "assignment": assignment_contrib,
            "on_time": on_time_contrib,
            "pooling": pooling_contrib,
            "fairness": fairness_contrib,
            "delay": delay_penalty
        }
        
        # Trier par contribution et garder top 3
        sorted_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        dominant_factors = dict(sorted_factors[:3])
        
        return round(min(100, max(0, score)), 2), dominant_factors


def collect_dispatch_metrics(
        dispatch_run_id: int, company_id: int, day: date) -> DispatchQualityMetrics:
    """Fonction helper pour collecter les métriques d'un dispatch.
    Utilisée par engine.py après chaque dispatch.

    Args:
        dispatch_run_id: ID du DispatchRun
        company_id: ID de l'entreprise
        day: Date du dispatch

    Returns:
        DispatchQualityMetrics calculées

    """
    collector = DispatchMetricsCollector(company_id)
    _ = day  # Supprimer l'avertissement du paramètre non utilisé
    return collector.collect_for_run(dispatch_run_id)
