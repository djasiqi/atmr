# backend/services/unified_dispatch/solver.py
from __future__ import annotations

import contextlib
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from services.unified_dispatch.settings import Settings

if TYPE_CHECKING:
    from models import Booking, Driver

DEFAULT_SETTINGS = Settings()

# total nodes = drivers + 2*bookings
SAFE_MAX_NODES = int(os.getenv("UD_SOLVER_MAX_NODES", "800"))
SAFE_MAX_TASKS = int(os.getenv("UD_SOLVER_MAX_TASKS", "250"))  # bookings
SAFE_MAX_VEH = int(os.getenv("UD_SOLVER_MAX_VEHICLES", "120"))  # drivers

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Types de sortie
# -------------------------------------------------------------------


@dataclass
class SolverAssignment:
    booking_id: int
    driver_id: int
    reason: str = "solver"  # homogène avec heuristics
    route_index: int = 0  # position dans la tournée (pickup index)
    # note: on ne sort pas les timestamps exacts ici, OR-Tools peut les fournir si besoin
    # Estimations min depuis t0 (base_time du problème)
    estimated_pickup_min: int = 0
    estimated_dropoff_min: int = 0
    base_time: datetime | None = None
    dispatch_run_id: int | None = None  # Ensure this field is included

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation compatible API:
        - convertit les minutes relatives en datetimes ISO (local naïf) basés sur base_time.
        - inclut dispatch_run_id pour persistance.
        """
        bt = self.base_time or datetime.now()
        try:
            est_pickup_dt = bt + timedelta(minutes=int(self.estimated_pickup_min))
            est_drop_dt = bt + timedelta(minutes=int(self.estimated_dropoff_min))
        except Exception:
            est_pickup_dt = bt
            est_drop_dt = bt
        return {
            "booking_id": int(self.booking_id),
            "driver_id": int(self.driver_id),
            "status": "proposed",
            "estimated_pickup_arrival": est_pickup_dt,
            "estimated_dropoff_arrival": est_drop_dt,
            "reason": self.reason,
            "route_index": int(self.route_index),
            "dispatch_run_id": self.dispatch_run_id,  # Include dispatch_run_id in the dict
        }


@dataclass
class SolverResult:
    assignments: List[SolverAssignment]
    unassigned_booking_ids: List[int]
    debug: Dict[str, Any]


# -------------------------------------------------------------------
# Solveur OR-Tools
# -------------------------------------------------------------------


def solve(problem: Dict[str, Any], settings: Settings = DEFAULT_SETTINGS) -> SolverResult:
    """Solve VRPTW.
    time_matrix/service_times/time_windows/driver_windows en MINUTES, horizon en MINUTES.
    """
    if not problem or not problem.get("bookings") or not problem.get("drivers"):
        return SolverResult(assignments=[], unassigned_booking_ids=[], debug={"reason": "empty_problem"})

    bookings: List[Booking] = problem["bookings"]
    drivers: List[Driver] = problem["drivers"]
    time_matrix: List[List[int]] = problem["time_matrix"]
    starts: List[int] = problem["starts"]
    ends: List[int] = problem["ends"]
    num_vehicles: int = problem["num_vehicles"]
    tw: List[Tuple[int, int]] = problem["time_windows"]
    service_times: List[int] = problem["service_times"]
    driver_windows: List[Tuple[int, int]] = problem.get("driver_windows", [])
    pair_min_gaps: List[int] = problem.get("pair_min_gaps", [])
    # Capacités véhicules (optionnel) - défaut: 1 place / chauffeur
    vehicle_capacities: List[int] | None = problem.get("vehicle_capacities")
    # horizon minutes depuis settings, avec fallback robuste
    horizon: int = int(
        problem.get(
            "horizon",
            getattr(getattr(settings, "time", None), "horizon_minutes", 12 * 60),
        )
    )
    base_time: datetime | None = problem.get("base_time")
    dispatch_run_id: int | None = problem.get("dispatch_run_id")

    # -------- Sanity checks
    n_nodes = len(time_matrix)
    if n_nodes == 0:
        return SolverResult(
            assignments=[],
            unassigned_booking_ids=[int(getattr(b, "id", 0) or 0) for b in bookings],
            debug={"reason": "empty_matrix"},
        )

    # Matrice carrée
    for r in time_matrix:
        if len(r) != n_nodes:
            msg = f"time_matrix must be square, got {n_nodes}x{len(r)}"
            raise ValueError(msg)
    expected_nodes = num_vehicles + 2 * len(bookings)
    if expected_nodes != n_nodes:
        msg = f"time_matrix size mismatch: expected {expected_nodes}, got {n_nodes}"
        raise ValueError(msg)
    if len(tw) != 2 * len(bookings):
        msg = f"time_windows size mismatch: expected {2 * len(bookings)}, got {len(tw)}"
        raise ValueError(msg)
    if len(service_times) != 2 * len(bookings):
        msg = f"service_times size mismatch: expected {2 * len(bookings)}, got {len(service_times)}"
        raise ValueError(msg)
    if len(starts) != num_vehicles or len(ends) != num_vehicles:
        msg = "starts/ends must have length = num_vehicles"
        raise ValueError(msg)

    # -------- Indexation des tasks
    depot_count = num_vehicles
    task_nodes_start = depot_count
    pickup_nodes: List[int] = []
    dropoff_nodes: List[int] = []
    node_to_booking: Dict[int, int] = {}
    node_is_pickup: Dict[int, bool] = {}
    booking_id_to_p_node: Dict[int, int] = {}

    for i, b in enumerate(bookings):
        b_id = int(getattr(b, "id", 0) or 0)
        p_node = task_nodes_start + i * 2
        d_node = p_node + 1
        pickup_nodes.append(p_node)
        dropoff_nodes.append(d_node)
        node_to_booking[p_node] = b_id
        node_to_booking[d_node] = b_id
        node_is_pickup[p_node] = True
        node_is_pickup[d_node] = False
        booking_id_to_p_node[b_id] = p_node

    # --- Pré-traitement pour identifier les paires Aller/Retour ---
    booking_map: Dict[int, Any] = {int(getattr(b, "id", 0) or 0): b for b in bookings}
    return_to_outbound_map = {}
    for b in bookings:
        parent_id = getattr(b, "parent_booking_id", None)
        is_ret = bool(getattr(b, "is_return", False) or False)
        b_id = int(getattr(b, "id", 0) or 0)
        if parent_id is not None and is_ret:
            p_int = int(parent_id)
            if p_int in booking_map:
                return_to_outbound_map[b_id] = p_int

    # ---- Safety guard: cap problem size to avoid native crashes
    if len(bookings) > SAFE_MAX_TASKS or num_vehicles > SAFE_MAX_VEH or n_nodes > SAFE_MAX_NODES:
        logger.warning(
            "[Solver] Problem too large -> fallback (veh=%d, tasks=%d, nodes=%d; caps=%d/%d/%d)",
            num_vehicles,
            len(bookings),
            n_nodes,
            SAFE_MAX_VEH,
            SAFE_MAX_TASKS,
            SAFE_MAX_NODES,
        )
        # Ne pas tenter OR-Tools -> on rend tout "non assigné" => engine fera
        # le fallback heuristique
        return SolverResult(
            assignments=[],
            unassigned_booking_ids=[int(getattr(b, "id", 0) or 0) for b in bookings],
            debug={"status": "too_large", "veh": num_vehicles, "tasks": len(bookings), "nodes": n_nodes},
        )

    # -------- OR-Tools
    manager = pywrapcp.RoutingIndexManager(n_nodes, num_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    # --- Arc costs par véhicule pour pénaliser les urgences ---
    # Base: travel (min) + service(from_node) ; si véhicule d'urgence:
    #   travel *= emergency_distance_multiplier
    #   + emergency_per_stop_penalty (si to_node est une tâche)
    try:
        from models import DriverType  # si dispo
    except Exception:
        DriverType = None

    emergency_mult = float(getattr(getattr(settings, "emergency", None), "emergency_distance_multiplier", 1.0))
    emergency_per_stop = int(getattr(getattr(settings, "emergency", None), "emergency_per_stop_penalty", 0))

    veh_is_emergency: List[bool] = []
    for d in drivers:
        is_emg = False
        t = getattr(d, "driver_type", None)
        is_emg = t == DriverType.EMERGENCY if DriverType is not None else bool(t) and str(t).endswith("EMERGENCY")
        veh_is_emergency.append(is_emg)

    def _service_time_for_from_node(from_node: int) -> int:
        if from_node >= task_nodes_start:
            svc_idx = from_node - task_nodes_start
            if 0 <= svc_idx < len(service_times):
                return int(service_times[svc_idx])
        return 0

    # --- Callback TEMPS unique pour la dimension "Time"
    #     (utilise le temps réel: travel + service ; SANS pénalité/multiplicateur urgence)
    def _time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel = int(time_matrix[from_node][to_node])
        service = _service_time_for_from_node(from_node)
        return travel + service

    time_cb_index = routing.RegisterTransitCallback(_time_callback)

    transit_cb_per_vehicle: List[int] = []
    for vid in range(num_vehicles):

        def _make_cb(v_is_emg: bool):
            def _cb(from_index: int, to_index: int) -> int:
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                travel = int(time_matrix[from_node][to_node])
                if v_is_emg and emergency_mult > 1.0:
                    travel = math.ceil(travel * emergency_mult)
                service = _service_time_for_from_node(from_node)
                add_stop_pen = emergency_per_stop if (v_is_emg and to_node >= task_nodes_start) else 0
                return travel + service + add_stop_pen

            return _cb

        cb_idx = routing.RegisterTransitCallback(_make_cb(veh_is_emergency[vid]))
        transit_cb_per_vehicle.append(cb_idx)
        routing.SetArcCostEvaluatorOfVehicle(cb_idx, vid)

    for vid, d in enumerate(drivers):
        base_cost = int(getattr(getattr(settings, "solver", None), "vehicle_fixed_cost", 0))
        emg_fixed = int(getattr(getattr(settings, "emergency", None), "emergency_vehicle_fixed_cost", 0))
        if (DriverType is not None and getattr(d, "driver_type", None) == DriverType.EMERGENCY) or (
            DriverType is None and getattr(d, "driver_type", None) and str(d.driver_type).endswith("EMERGENCY")
        ):
            base_cost += emg_fixed
        if base_cost:
            with contextlib.suppress(Exception):
                routing.SetFixedCostOfVehicle(int(base_cost), vid)

    # Dimension temps (positionnel pour compat SWIG)
    routing.AddDimension(
        time_cb_index,  # transit
        horizon,  # slack_max (autorise l'attente)
        horizon,  # capacity (horizon max)
        False,  # fix_start_cumul_to_zero
        "Time",  # name
    )
    time_dim = routing.GetDimensionOrDie("Time")

    def _clamp_tw(a, b):
        a = max(0, int(a))
        b = min(horizon, int(b))
        if b <= a:
            b = a + 1
        return a, b

    # TW chauffeurs (départs/arrivées véhicules) si activé dans les settings
    if getattr(settings.solver, "add_driver_work_windows", True):
        strict_end = bool(getattr(settings.solver, "strict_driver_end_window", True))
        for v in range(num_vehicles):
            start_index = routing.Start(v)
            end_index = routing.End(v)
            if v < len(driver_windows):
                s, e = driver_windows[v]
            else:
                s, e = (0, horizon)
            s, e = _clamp_tw(s, e)
            time_dim.CumulVar(start_index).SetRange(s, e)
            if strict_end:
                time_dim.CumulVar(end_index).SetRange(s, e)
            else:
                # borne de fin souple : autorise un retour un peu plus tard
                time_dim.CumulVar(end_index).SetRange(0, e)
    else:
        # Comportement historique: aucune contrainte spécifique -> bornes
        # larges
        for v in range(num_vehicles):
            time_dim.CumulVar(routing.Start(v)).SetRange(0, horizon)
            time_dim.CumulVar(routing.End(v)).SetRange(0, horizon)

    # TW pickups/dropoffs
    for i, _ in enumerate(bookings):
        p_node = pickup_nodes[i]
        d_node = dropoff_nodes[i]
        p_index = manager.NodeToIndex(p_node)
        d_index = manager.NodeToIndex(d_node)

        p_tw = _clamp_tw(*tw[i * 2])
        d_tw = _clamp_tw(*tw[i * 2 + 1])

        time_dim.CumulVar(p_index).SetRange(*p_tw)
        time_dim.CumulVar(d_index).SetRange(*d_tw)

    # Pickup & Delivery (toujours même véhicule + ordre pickup <= dropoff)
    use_pairs = getattr(settings.solver, "use_pickup_dropoff_pairs", True)
    for i in range(len(bookings)):
        p_index = manager.NodeToIndex(pickup_nodes[i])
        d_index = manager.NodeToIndex(dropoff_nodes[i])
        if use_pairs:
            routing.AddPickupAndDelivery(p_index, d_index)
        # Même véhicule, ordre temporel et "actif ensemble"
        routing.solver().Add(routing.VehicleVar(p_index) == routing.VehicleVar(d_index))
        routing.solver().Add(time_dim.CumulVar(p_index) <= time_dim.CumulVar(d_index))
        routing.solver().Add(routing.ActiveVar(p_index) == routing.ActiveVar(d_index))

        # ✅ CONTRAINTE CLÉ : imposer un écart minimal entre pickup et dropoff
        #    (durée de trajet estimée + buffer post-course).
        if i < len(pair_min_gaps) and pair_min_gaps[i] > 0:
            min_gap = int(pair_min_gaps[i]) + 1
            routing.solver().Add(time_dim.CumulVar(d_index) >= time_dim.CumulVar(p_index) + min_gap)

    # -------------------------------------------------------------------
    # DIMENSION CAPACITÉ : 1 passager max par véhicule (empêche les chevauchements)
    # -------------------------------------------------------------------
    # Demande: +1 au pickup, -1 au dropoff
    demands = [0] * n_nodes
    for i in range(len(bookings)):
        demands[pickup_nodes[i]] = 1
        demands[dropoff_nodes[i]] = -1

    # Capacité véhicules - par défaut = 1 si non fourni
    if not vehicle_capacities or len(vehicle_capacities) != num_vehicles:
        vehicle_capacities = [1 for _ in range(num_vehicles)]

    def demand_callback(from_index: int) -> int:
        node = manager.IndexToNode(from_index)
        return int(demands[node])

    demand_cb_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_cb_index,
        0,  # slack (aucun backorder)
        vehicle_capacities,
        True,  # cumul fixé à zéro au départ
        "Capacity",
    )
    # Récupérable si besoin
    _ = routing.GetDimensionOrDie("Capacity")

    # Sécurité: pickup actif ⇒ cap >= 1 à ce point; dropoff possible ⇒ cap redescend
    # (OR-Tools gère déjà la borne 0..capacity via la dimension)

    # ----- Span cost (robuste Windows) -----
    try:
        disable_span = os.getenv("UD_SOLVER_DISABLE_SPAN_COST", "0") == "1"
        coeff = int(getattr(settings.solver, "global_span_cost", 0))
        if coeff > 0 and not disable_span:
            # Préfère par véhicule si disponible ; évite le chemin d'abort sous
            # Windows
            if hasattr(time_dim, "SetSpanCostCoefficientForVehicle"):
                for v in range(num_vehicles):
                    time_dim.SetSpanCostCoefficientForVehicle(coeff, v)
            # Global : seulement hors Windows
            elif os.name != "nt":
                time_dim.SetGlobalSpanCostCoefficient(coeff)
    except Exception:
        pass

    # Finalizers : Start par défaut (End désactivé pour contourner un autre
    # chemin de crash)
    for v in range(num_vehicles):
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
    if os.getenv("UD_SOLVER_FINALIZE_END", "0") == "1":
        for v in range(num_vehicles):
            routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    # Dimension "Visits" (cap de stops / véhicule)
    def visits_callback(_from_index: int, to_index: int) -> int:
        to_node = manager.IndexToNode(to_index)
        return 1 if to_node >= task_nodes_start else 0

    visits_cb_idx = routing.RegisterTransitCallback(visits_callback)
    routing.AddDimension(
        visits_cb_idx,  # transit
        0,  # slack_max
        # capacity (pickup+dropoff)
        int(2 * settings.solver.max_bookings_per_driver),
        True,  # fix_start_cumul_to_zero
        "Visits",
    )

    # Disjunctions: GROUPÉES par PAIRE (pickup+dropoff) avec grosse pénalité si non servi
    # NB: une seule disjonction par paire => on paie la pénalité au plus une
    # fois par course complète.
    base_penalty_raw = getattr(getattr(settings, "solver", None), "unassigned_penalty_base", 10000)
    try:
        base_penalty = int(base_penalty_raw)
    except Exception:
        base_penalty = 10000

    # On surestime par rapport aux coûts de trajet (en minutes) pour éviter le
    # non-service "facile"
    penalty = max(base_penalty, 20 * int(horizon))
    for i in range(len(bookings)):
        p_idx = manager.NodeToIndex(pickup_nodes[i])
        d_idx = manager.NodeToIndex(dropoff_nodes[i])
        # ✅ Disjonction groupée pickup+dropoff : on paie au plus une pénalité par course complète
        # (et on évite des états "pickup absent / dropoff présent" même si ActiveVar égalise déjà).
        routing.AddDisjunction([p_idx, d_idx], int(penalty))

    # --- NOUVEAU: Pénalité "soft" pour encourager le MÊME chauffeur sur les Aller/Retour ---
    # On ajoute une pénalité au coût total si les chauffeurs sont différents.
    # Cela incite le solveur à préférer l'attente sur place.
    solver = routing.solver()
    round_trip_penalty_cost = int(getattr(settings.solver, "round_trip_driver_penalty_min", 120))

    for return_id, outbound_id in return_to_outbound_map.items():
        outbound_p_node = booking_id_to_p_node.get(outbound_id)
        return_p_node = booking_id_to_p_node.get(return_id)

        if outbound_p_node is not None and return_p_node is not None:
            outbound_p_index = manager.NodeToIndex(outbound_p_node)
            return_p_index = manager.NodeToIndex(return_p_node)

            # Crée une variable qui vaut 1 si les chauffeurs sont différents, 0
            # sinon.
            are_drivers_different = solver.BoolVar(f"are_drivers_different_{return_id}")
            solver.Add(
                are_drivers_different == (routing.VehicleVar(outbound_p_index) != routing.VehicleVar(return_p_index))
            )
            # Ajoute la pénalité au coût global si la variable est à 1.
            routing.AddVariableMinimizedByFinalizer(are_drivers_different * round_trip_penalty_cost)

    # Paramètres de recherche
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.FromSeconds(int(settings.solver.time_limit_sec))
    # Bornes conservatrices pour maîtriser l'espace de recherche/mémoire:
    search_params.solution_limit = 1
    search_params.log_search = False

    # ✅ B5: Warm-start: Injecter les assignments heuristiques comme hint initial
    heuristic_assignments = problem.get("heuristic_assignments")
    if heuristic_assignments and getattr(settings.solver, "enable_warm_start", True):
        try:
            from services.unified_dispatch.warm_start import apply_warm_start

            # Enrichir le problème avec les informations nécessaires
            enriched_problem = {
                **problem,
                "pickup_nodes": pickup_nodes,
                "dropoff_nodes": dropoff_nodes,
                "booking_id_to_p_node": booking_id_to_p_node,
            }
            warm_start_applied = apply_warm_start(routing, heuristic_assignments, enriched_problem, manager)
            if warm_start_applied:
                logger.info("[Solver] Warm-start applied from heuristic assignments")
        except Exception as e:
            logger.warning("[Solver] Failed to apply warm-start: %s", e)

    # ✅ B5: Mesurer le gain warm-start (optionnel, pour taille 100-200)
    try:
        # Taille cible pour mesure gain
        from services.unified_dispatch.warm_start_gain_tracker import (
            TARGET_SIZE_MAX,
            TARGET_SIZE_MIN,
            measure_warm_start_gain,
        )

        size = len(bookings)
        if TARGET_SIZE_MIN <= size <= TARGET_SIZE_MAX and heuristic_assignments:
            gain_result = measure_warm_start_gain(problem, heuristic_assignments, solve)

            if not gain_result.get("skipped"):
                gain_pct = gain_result.get("gain_pct", 0)
                logger.info(
                    "[B5] Warm-start gain: %.1f%% (size=%d, without=%.0fms, with=%.0fms)",
                    gain_pct,
                    size,
                    gain_result.get("without_ms", 0),
                    gain_result.get("with_ms", 0),
                )
    except Exception as e:
        logger.debug("[B5] Warm-start gain tracking not available: %s", e)

    # Résolution
    with contextlib.suppress(Exception):
        search_params.number_of_threads = 1
    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        _limit_sec = int(getattr(getattr(settings, "solver", None), "time_limit_sec", 0))
        logger.warning(
            "[Solver] No solution (limit=%ss, vehicles=%d, tasks=%d, nodes=%d, penalty=%d)",
            _limit_sec,
            num_vehicles,
            len(bookings),
            n_nodes,
            int(penalty),
        )
        return SolverResult(
            assignments=[],
            unassigned_booking_ids=[int(getattr(b, "id", 0) or 0) for b in bookings],
            debug={
                "status": "no_solution",
                "veh": num_vehicles,
                "tasks": len(bookings),
                "nodes": n_nodes,
                "penalty": int(penalty),
            },
        )

    # Extraction solution
    assigned_booking_to_driver: Dict[int, int] = {}
    assigned_pickup_rank: Dict[int, int] = {}
    pickup_time_min: Dict[int, int] = {}
    dropoff_time_min: Dict[int, int] = {}

    # --- Stats urgences (stops & km estimés) ---
    emergency_stops = 0
    emergency_travel_min = 0

    def _travel_minutes(a: int, b: int) -> int:
        try:
            return int(time_matrix[a][b])
        except Exception:
            return 0

    for v in range(num_vehicles):
        index = routing.Start(v)
        order = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node >= task_nodes_start:
                b_id = int(node_to_booking[node])
                if node_is_pickup[node]:
                    assigned_booking_to_driver[b_id] = int(getattr(drivers[v], "id", 0) or 0)
                    assigned_pickup_rank[b_id] = order
                    # temps accumulé au pickup
                    pickup_time_min[b_id] = int(solution.Value(time_dim.CumulVar(index)))
                else:
                    # temps accumulé au dropoff
                    dropoff_time_min[b_id] = int(solution.Value(time_dim.CumulVar(index)))
                # comptage "stops" urgences (pickup + dropoff) : on compte le
                # to_node
                if veh_is_emergency[v]:
                    emergency_stops += 1
            # accumuler travel min par véhicule d'urgence
            nxt = solution.Value(routing.NextVar(index))
            if veh_is_emergency[v]:
                a = manager.IndexToNode(index)
                b = manager.IndexToNode(nxt)
                emergency_travel_min += _travel_minutes(a, b)
            index = solution.Value(routing.NextVar(index))
            order += 1

    unassigned_ids: List[int] = []
    for b in bookings:
        b_id = int(getattr(b, "id", 0) or 0)
        if b_id not in assigned_booking_to_driver:
            unassigned_ids.append(b_id)

    assignments: List[SolverAssignment] = []
    for b in bookings:
        b_id = int(getattr(b, "id", 0) or 0)
        if b_id in assigned_booking_to_driver:
            assignments.append(
                SolverAssignment(
                    booking_id=b_id,
                    driver_id=int(assigned_booking_to_driver[b_id]),
                    route_index=int(assigned_pickup_rank.get(b_id, 0)),
                    estimated_pickup_min=int(pickup_time_min.get(b_id, 0)),
                    estimated_dropoff_min=int(dropoff_time_min.get(b_id, pickup_time_min.get(b_id, 0))),
                    base_time=base_time,
                    dispatch_run_id=dispatch_run_id,
                )
            )

    debug = {
        "vehicles": num_vehicles,
        "tasks": len(bookings),
        "unassigned": len(unassigned_ids),
        "regular_stops": int(sum(1 for v in range(num_vehicles) for _ in [0] if not veh_is_emergency[v])),
        "emergency_stops": int(emergency_stops),
        "pair_min_gaps_snapshot": list(pair_min_gaps[: min(len(pair_min_gaps), 10)]),
        "matrix_provider": problem.get("matrix_provider"),
    }
    # km estimés pour les urgences (temps de trajet * vitesse moyenne / 60)
    try:
        avg_kmh = float(getattr(getattr(settings, "matrix", None), "avg_speed_kmh", 25.0))
        emergency_km_est = (emergency_travel_min * avg_kmh) / 60.0
        debug["emergency_km_est"] = round(emergency_km_est, 2)
    except Exception:
        pass

    return SolverResult(assignments=assignments, unassigned_booking_ids=unassigned_ids, debug=debug)
