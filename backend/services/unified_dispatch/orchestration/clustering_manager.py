# backend/services/unified_dispatch/orchestration/clustering_manager.py
"""Gestionnaire de clustering géographique.

Ce module gère le clustering géographique pour le dispatch. Il est responsable de :
- La décision d'utiliser le clustering (basée sur le nombre de bookings)
- La création de zones géographiques
- Le dispatch indépendant de chaque zone
- La fusion des résultats de toutes les zones

Le clustering est utilisé pour diviser un grand problème en sous-problèmes
plus petits et géographiquement cohérents, améliorant les performances et
la qualité des solutions.

Side-effects:
    - Accès DB (lecture bookings, drivers via clustering)
    - Appels récursifs au dispatch pour chaque zone
"""

from __future__ import annotations

import logging
from typing import Any

from models import Company
from services.unified_dispatch.data.clustering import GeographicClustering

logger = logging.getLogger(__name__)


class ClusteringManager:
    """Gestionnaire de clustering géographique pour le dispatch.

    Cette classe centralise la logique de clustering géographique :
    - Décision d'activation basée sur le nombre de bookings et les settings
    - Création de zones géographiques avec GeographicClustering
    - Dispatch indépendant de chaque zone
    - Fusion des résultats de toutes les zones

    Exemple:
        >>> manager = ClusteringManager()
        >>> if manager.should_use_clustering(problem, settings):
        ...     zones = manager.create_zones(problem, settings)
        ...     result = manager.dispatch_zones(
        ...         zones=zones,
        ...         company=company,
        ...         problem=problem,
        ...         mode="auto",
        ...         settings=settings
        ...     )
        ...     final_assignments = result["assignments"]
    """

    # Constante pour le seuil par défaut de clustering
    CLUSTERING_BOOKINGS_THRESHOLD = 100

    def should_use_clustering(self, problem: dict[str, Any], settings: Any) -> bool:
        """Détermine si le clustering géographique doit être utilisé.

        Le clustering est activé si :
        - Le feature flag est activé dans les settings
        - Le nombre de bookings dépasse le seuil configuré

        Args:
            problem: Dict contenant les données du problème avec clé "bookings"
            settings: Settings de dispatch avec configuration clustering

        Returns:
            True si le clustering doit être utilisé, False sinon
        """
        use_clustering = getattr(settings.features, "enable_clustering", False)
        if not use_clustering:
            return False

        n_bookings = len(problem.get("bookings", []))
        clustering_threshold = getattr(
            settings.clustering,
            "bookings_threshold",
            self.CLUSTERING_BOOKINGS_THRESHOLD,
        )

        return n_bookings > clustering_threshold

    def create_zones(self, problem: dict[str, Any], settings: Any) -> list[Any]:
        """Crée les zones géographiques pour le clustering.

        Utilise GeographicClustering pour diviser les bookings et drivers
        en zones géographiquement cohérentes.

        Args:
            problem: Dict contenant les données du problème (bookings, drivers)
            settings: Settings de dispatch avec configuration clustering

        Returns:
            List de zones (objets Zone créés par GeographicClustering)

        Raises:
            ValueError, TypeError, AttributeError: Si paramètres invalides
        """
        n_bookings = len(problem.get("bookings", []))
        n_drivers = len(problem.get("drivers", []))

        logger.info(
            "[ClusteringManager] Activating geographic clustering: %d bookings, %d drivers",
            n_bookings,
            n_drivers,
        )

        max_bookings = getattr(settings.clustering, "max_bookings_per_zone", 100)
        cross_tolerance = getattr(settings.clustering, "cross_zone_tolerance", 0.1)

        clustering = GeographicClustering(max_bookings_per_zone=max_bookings)
        bookings_list = problem.get("bookings", [])
        drivers_list = problem.get("drivers", [])

        return clustering.create_zones(
            bookings=bookings_list,
            drivers=drivers_list,
            cross_zone_tolerance=cross_tolerance,
        )

    def dispatch_zones(
        self,
        zones: list[Any],
        company: Company,
        problem: dict[str, Any],
        mode: str,
        settings: Any,
    ) -> dict[str, Any]:
        """Dispatch chaque zone indépendamment et fusionne les résultats.

        Dispatch chaque zone géographique de manière indépendante en utilisant
        GeographicClustering.dispatch_zones(). Fusionne ensuite tous les
        résultats en un seul résultat final.

        Args:
            zones: List de zones géographiques à dispatcher
            company: Objet Company
            problem: Dict contenant les données du problème original
            mode: Mode de dispatch
            settings: Settings de dispatch

        Returns:
            Dict avec clés :
            - assignments: List des assignations finales (toutes zones fusionnées)
            - unassigned: List des IDs de bookings non assignés
            - meta: Métadonnées sur le clustering (nombre de zones, etc.)

        Side-effects:
            - Appels récursifs au dispatch pour chaque zone
            - Accès DB (lecture/écriture via dispatch de chaque zone)
        """
        if len(zones) <= 1:
            return {
                "assignments": [],
                "unassigned": [],
                "meta": {"zones_count": len(zones)},
            }

        logger.info(
            "[ClusteringManager] Created %d zones, dispatching independently",
            len(zones),
        )

        max_bookings = getattr(settings.clustering, "max_bookings_per_zone", 100)
        clustering = GeographicClustering(max_bookings_per_zone=max_bookings)

        clustering_result = clustering.dispatch_zones(
            zones=zones,
            company=company,
            problem=problem,
            mode=mode,
            settings=settings,
        )

        clustering_final_assignments = clustering_result["assignments"]
        clustering_unassigned_ids = clustering_result["unassigned"]

        logger.info("[ClusteringManager] Using clustering results as final assignments")

        return {
            "assignments": clustering_final_assignments,
            "unassigned": clustering_unassigned_ids,
            "meta": {
                "zones_count": len(zones),
                "assignments_count": len(clustering_final_assignments),
                "unassigned_count": len(clustering_unassigned_ids),
            },
        }
