# backend/services/unified_dispatch/warm_start.py
"""Warm-start OR-Tools avec solution heuristique."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def create_routing_hints(
    heuristic_assignments: List[Any], problem: Dict[str, Any], manager: Any
) -> Dict[int, List[int]]:
    """Crée des hints de routage depuis les assignments heuristiques.

    Convertit les assignments heuristiques (booking_id, driver_id) en
    hints OR-Tools (véhicule index -> liste de node indices).

    Args:
        heuristic_assignments: Liste d'assignments heuristiques
        problem: Dict du problème VRPTW avec pickup_nodes, dropoff_nodes
        manager: RoutingIndexManager OR-Tools

    Returns:
        Dict {vehicle_index: [node1, node2, ...]} pour SetInitialSolutionFromRoutes
    """
    if not heuristic_assignments:
        return {}

    try:
        # Récupérer les mappings depuis le problème
        pickup_nodes = problem.get("pickup_nodes", [])
        dropoff_nodes = problem.get("dropoff_nodes", [])
        booking_id_to_p_node = problem.get("booking_id_to_p_node", {})

        # Créer mapping pickup -> dropoff
        p_to_d = {}
        for i, p_node in enumerate(pickup_nodes):
            if i < len(dropoff_nodes):
                p_to_d[p_node] = dropoff_nodes[i]

        # Grouper les assignments par driver (véhicule)
        routes: Dict[int, List[int]] = {}

        for assignment in heuristic_assignments:
            booking_id = getattr(assignment, "booking_id", None)
            driver_id = getattr(assignment, "driver_id", None)

            if booking_id is None or driver_id is None:
                continue

            # Trouver les nodes pour ce booking
            p_node = booking_id_to_p_node.get(int(booking_id))
            if p_node is None:
                continue

            d_node = p_to_d.get(p_node)
            if d_node is None:
                continue

            # Convertir en indices OR-Tools
            vehicle_index = int(driver_id)

            # Créer la route si elle n'existe pas
            if vehicle_index not in routes:
                routes[vehicle_index] = []

            # Ajouter pickup puis dropoff
            p_index = manager.NodeToIndex(p_node)
            d_index = manager.NodeToIndex(d_node)

            routes[vehicle_index].append(p_index)
            routes[vehicle_index].append(d_index)

        logger.info(
            "[WarmStart] Created hints for %d vehicles, %d total assignments", len(routes), len(heuristic_assignments)
        )

        return routes

    except Exception as e:
        logger.error("[WarmStart] Failed to create routing hints: %s", e)
        return {}


def apply_warm_start(routing: Any, heuristic_assignments: List[Any], problem: Dict[str, Any], manager: Any) -> bool:
    """Applique les hints de warm-start au solveur OR-Tools.

    Args:
        routing: RoutingModel OR-Tools
        heuristic_assignments: Liste d'assignments heuristiques
        problem: Dict du problème VRPTW
        manager: RoutingIndexManager OR-Tools

    Returns:
        True si les hints ont été appliqués avec succès
    """
    try:
        # Créer les hints
        routes = create_routing_hints(heuristic_assignments, problem, manager)

        if not routes:
            return False

        # Convertir en format OR-Tools: List[List[int]]
        routes_list: List[List[int]] = []

        # Obtenir le nombre de véhicules depuis le manager
        num_vehicles = routing.vehicles()

        for vehicle_index in range(num_vehicles):
            if vehicle_index in routes:
                routes_list.append(routes[vehicle_index])
            else:
                # Véhicule vide
                routes_list.append([])

        # Appliquer les hints
        # Note: SetInitialSolutionFromRoutes prend une List[List[int]]
        routing.SetInitialSolutionFromRoutes(routes_list)

        logger.info(
            "[WarmStart] Applied warm-start hints: %d routes with %d total tasks",
            len([r for r in routes_list if r]),
            sum(len(r) for r in routes_list) // 2,  # Puisque c'est pickup+dropoff
        )

        return True

    except Exception as e:
        logger.warning("[WarmStart] Failed to apply warm-start: %s", e)
        return False
