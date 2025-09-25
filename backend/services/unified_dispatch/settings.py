# backend/services/unified_dispatch/settings.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple
from ast import literal_eval
from datetime import timedelta


# paramètres centralisés (capacité, buffers, pénalités…)
# ------------------------------------------------------------
# Groupes de paramètres
# ------------------------------------------------------------

@dataclass
class HeuristicWeights:
    proximity: float = 0.45              # distance/temps vers pickup
    driver_load_balance: float = 0.30    # équité (courses du jour)
    priority: float = 0.15               # priorité booking (médical, VIP…)
    return_urgency: float = 0.08         # retours déclenchés à la demande
    regular_driver_bonus: float = 0.02   # chauffeur habituel du client

    def normalized(self) -> "HeuristicWeights":
        total = self.proximity + self.driver_load_balance + self.priority + self.return_urgency + self.regular_driver_bonus
        if total == 0:
            return self
        return HeuristicWeights(
            proximity=self.proximity/total,
            driver_load_balance=self.driver_load_balance/total,
            priority=self.priority/total,
            return_urgency=self.return_urgency/total,
            regular_driver_bonus=self.regular_driver_bonus/total,
        )

@dataclass
class SolverParams:
    # OR-Tools
    time_limit_sec: int = 60                 # limite max (adaptative possible)
    global_span_cost: int = 100              # compaction des tournées
    vehicle_fixed_cost: int = 10             # coût fixe par véhicule utilisé
    unassigned_penalty_base: int = 10000     # pénalité non-assigné (par tâche)
    max_bookings_per_driver: int = 6         # LIMITE UNIQUE (heuristique = solveur)
    pickup_dropoff_slack_min: int = 5        # marge autour des TW
    use_pickup_dropoff_pairs: bool = True    # arc obligatoire pickup->dropoff
    add_driver_work_windows: bool = True     # fenêtres de travail véhicule
    round_trip_driver_penalty_min: int = 120
    strict_driver_end_window: bool = True    # borne de fin stricte
    regular_first_two_phase: bool = True     # passe 1 réguliers, passe 2 urgences si besoin

@dataclass
class TimeSettings:
    buffer_min: int = 8                  # marge avant pickup (sécurité)
    service_time_pickup_min: int = 4     # temps service MINIMUM au pickup
    service_time_dropoff_min: int = 3    # temps service au dropoff
    horizon_minutes: int = 48 * 60       # planification sur 48h (VRPTW)
    post_trip_buffer_min: int = 15  

@dataclass
class RealtimeSettings:
    debounce_ms: int = 800              # coalescing demandes (créations/annulations)
    coalesce_window_ms: int = 800
    lock_ttl_sec: int = 30              # TTL du verrou (par company_id)
    max_queue_backlog: int = 100        # protection contre rafales

@dataclass
class FairnessSettings:
    enable_equalization: bool = True
    target_per_driver_per_day: int = 9999  # objectif « souple » ; on égalise par pénalités

@dataclass
class EmergencyPolicy:
    return_urgent_threshold_min: int = 20     # si retour < 20 min → urgent
    allow_emergency_drivers: bool = True
    emergency_vehicle_fixed_cost: int = 40    # surcoût véhicule d’astreinte
    emergency_per_stop_penalty: int = 30      # coût par stop (min) sur urgences
    emergency_distance_multiplier: float = 1.25  # multiplicateur temps/distance urgences

@dataclass
class MatrixSettings:
    provider: str = "osrm"                    # "osrm" | "haversine"
    cache_ttl_sec: int = 300                  # TTL du cache (secondes)
    grid_rounding_meters: int = 50
    avg_speed_kmh: float = 25.0               # utilisé par Haversine uniquement
    coord_precision: int = 5                  # précision d’arrondi des coords (décimales)
    redis_url: Optional[str] = None           # si défini, data.py pourra l’utiliser
    # Options OSRM
    osrm_base_url: str = "http://localhost:5001"
    osrm_profile: str = "driving"
    osrm_timeout_sec: int = 5
    osrm_max_sources_per_call: int = 60
    osrm_rate_limit_per_sec: int = 8
    # (optionnel) robustesse appels
    osrm_max_retries: int = 2
    osrm_retry_backoff_ms: int = 250

    # Alias de compatibilité pour data.py (cache_ttl_s attendu)
    @property
    def cache_ttl_s(self) -> int:  # type: ignore[override]
        try:
            return int(self.cache_ttl_sec)
        except Exception:
            return 300

@dataclass
class LoggingSettings:
    level: str = "INFO"
    emit_metrics: bool = True               # compteur assignations, latence, etc.

@dataclass
class FeatureFlags:
    enable_solver: bool = True             # peut être désactivé en mode dégradé
    enable_heuristics: bool = True
    enable_events: bool = True             # SocketIO + notifications
    enable_db_bulk_ops: bool = True        # writes atomiques/bulk

# ------------------------------------------------------------
# Configuration globale
# ------------------------------------------------------------

@dataclass
class Settings:
    heuristic: HeuristicWeights = field(default_factory=HeuristicWeights)
    solver: SolverParams = field(default_factory=SolverParams)
    time: TimeSettings = field(default_factory=TimeSettings)
    realtime: RealtimeSettings = field(default_factory=RealtimeSettings)
    fairness: FairnessSettings = field(default_factory=FairnessSettings)
    emergency: EmergencyPolicy = field(default_factory=EmergencyPolicy)
    matrix: MatrixSettings = field(default_factory=MatrixSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    features: FeatureFlags = field(default_factory=FeatureFlags)

    # Divers
    default_timezone: str = "Europe/Zurich"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # normaliser les poids pour éviter toute dérive
        d["heuristic"] = asdict(self.heuristic.normalized())
        return d

# ------------------------------------------------------------
# Helpers: chargement ENV + overrides par entreprise
# ------------------------------------------------------------

def _getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default
    
def _getenv_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default
    
def from_env() -> Settings:
    """Construit Settings à partir des variables d'environnement (facultatif)."""
    s = Settings()

    # Heuristique (poids)
    s.heuristic.proximity = _getenv_float("UD_WEIGHT_PROXIMITY", s.heuristic.proximity)
    s.heuristic.driver_load_balance = _getenv_float("UD_WEIGHT_LOAD", s.heuristic.driver_load_balance)
    s.heuristic.priority = _getenv_float("UD_WEIGHT_PRIORITY", s.heuristic.priority)
    s.heuristic.return_urgency = _getenv_float("UD_WEIGHT_URGENCY", s.heuristic.return_urgency)
    s.heuristic.regular_driver_bonus = _getenv_float("UD_WEIGHT_REGULAR", s.heuristic.regular_driver_bonus)

    # Solveur
    s.solver.time_limit_sec = _getenv_int("UD_SOLVER_TIME_LIMIT", s.solver.time_limit_sec)
    s.solver.global_span_cost = _getenv_int("UD_SOLVER_GLOBAL_SPAN_COST", s.solver.global_span_cost)
    s.solver.vehicle_fixed_cost = _getenv_int("UD_SOLVER_VEHICLE_FIXED_COST", s.solver.vehicle_fixed_cost)
    s.solver.unassigned_penalty_base = _getenv_int("UD_SOLVER_UNASSIGNED_PENALTY", s.solver.unassigned_penalty_base)
    s.solver.max_bookings_per_driver = _getenv_int("UD_SOLVER_MAX_BOOKINGS_PER_DRIVER", s.solver.max_bookings_per_driver)
    s.solver.pickup_dropoff_slack_min = _getenv_int("UD_SOLVER_SLACK_MIN", s.solver.pickup_dropoff_slack_min)
    s.solver.use_pickup_dropoff_pairs = os.getenv("UD_SOLVER_USE_PAIRS", str(s.solver.use_pickup_dropoff_pairs)).lower() == "true"
    s.solver.add_driver_work_windows = os.getenv("UD_SOLVER_DRIVER_WINDOWS", str(s.solver.add_driver_work_windows)).lower() == "true"
    s.solver.strict_driver_end_window = os.getenv("UD_SOLVER_STRICT_DRIVER_END_WINDOW", str(s.solver.strict_driver_end_window)).lower() == "true"
    s.solver.regular_first_two_phase = os.getenv("UD_SOLVER_REGULAR_FIRST_TWO_PHASE", str(s.solver.regular_first_two_phase)).lower() == "true"


    # Temps
    s.time.buffer_min = _getenv_int("UD_TIME_BUFFER_MIN", s.time.buffer_min)
    s.time.service_time_pickup_min = _getenv_int("UD_TIME_SERVICE_PICKUP", s.time.service_time_pickup_min)
    s.time.service_time_dropoff_min = _getenv_int("UD_TIME_SERVICE_DROPOFF", s.time.service_time_dropoff_min)
    s.time.horizon_minutes = _getenv_int("UD_TIME_HORIZON_MIN", s.time.horizon_minutes)
    s.time.post_trip_buffer_min = _getenv_int("UD_TIME_POST_TRIP_BUFFER", s.time.post_trip_buffer_min)


    # Temps réel
    s.realtime.debounce_ms = _getenv_int(
        "UD_RT_DEBOUNCE_MS",
        _getenv_int("UD_RTC_DEBOUNCE_MS", s.realtime.debounce_ms),
    )
    s.realtime.coalesce_window_ms = _getenv_int(
        "UD_RT_COALESCE_MS",
        _getenv_int("UD_RTC_COALESCE_MS", s.realtime.coalesce_window_ms),
    )
    s.realtime.lock_ttl_sec = _getenv_int(
        "UD_RT_LOCK_TTL_SEC",
        _getenv_int("UD_RTC_LOCK_TTL_SEC", s.realtime.lock_ttl_sec),
    )
    s.realtime.max_queue_backlog = _getenv_int(
        "UD_RT_MAX_BACKLOG",
        _getenv_int("UD_RTC_MAX_QUEUE_BACKLOG", s.realtime.max_queue_backlog),
    )

    # Fairness
    s.fairness.enable_equalization = os.getenv("UD_FAIRNESS_ENABLE", str(s.fairness.enable_equalization)).lower() == "true"
    s.fairness.target_per_driver_per_day = _getenv_int("UD_FAIRNESS_TARGET_PER_DAY", s.fairness.target_per_driver_per_day)

    # Urgence
    s.emergency.return_urgent_threshold_min = _getenv_int("UD_EMERGENCY_RETURN_THRESHOLD_MIN", s.emergency.return_urgent_threshold_min)
    s.emergency.allow_emergency_drivers = os.getenv("UD_EMERGENCY_ALLOW", str(s.emergency.allow_emergency_drivers)).lower() == "true"
    s.emergency.emergency_vehicle_fixed_cost = _getenv_int("UD_EMERGENCY_VEHICLE_FIXED_COST", s.emergency.emergency_vehicle_fixed_cost)
    s.emergency.emergency_per_stop_penalty = _getenv_int("UD_EMERGENCY_PER_STOP_PENALTY_MIN", s.emergency.emergency_per_stop_penalty)
    s.emergency.emergency_distance_multiplier = _getenv_float("UD_EMERGENCY_DISTANCE_MULTIPLIER", s.emergency.emergency_distance_multiplier)

    # Matrice
    s.matrix.provider = os.getenv("UD_MATRIX_PROVIDER", s.matrix.provider)
    s.matrix.cache_ttl_sec = _getenv_int("UD_MATRIX_CACHE_TTL_SEC", s.matrix.cache_ttl_sec)
    s.matrix.grid_rounding_meters = _getenv_int("UD_MATRIX_GRID_ROUND", s.matrix.grid_rounding_meters)
    s.matrix.avg_speed_kmh = _getenv_float("UD_MATRIX_AVG_SPEED_KMH", s.matrix.avg_speed_kmh)
    s.matrix.coord_precision = _getenv_int("UD_MATRIX_COORD_PRECISION", s.matrix.coord_precision)
    s.matrix.redis_url = os.getenv("REDIS_URL", s.matrix.redis_url)  # fallback standard

    # OSRM
    s.matrix.osrm_base_url = os.getenv("UD_OSRM_BASE_URL", s.matrix.osrm_base_url)
    s.matrix.osrm_profile = os.getenv("UD_OSRM_PROFILE", s.matrix.osrm_profile)
    s.matrix.osrm_timeout_sec = _getenv_int("UD_OSRM_TIMEOUT_SEC", s.matrix.osrm_timeout_sec)
    s.matrix.osrm_max_sources_per_call = _getenv_int("UD_OSRM_MAX_SOURCES_PER_CALL", s.matrix.osrm_max_sources_per_call)
    s.matrix.osrm_rate_limit_per_sec = _getenv_int("UD_OSRM_RATE_LIMIT_PER_SEC", s.matrix.osrm_rate_limit_per_sec)
    s.matrix.osrm_max_retries = _getenv_int("UD_OSRM_MAX_RETRIES", s.matrix.osrm_max_retries)
    s.matrix.osrm_retry_backoff_ms = _getenv_int("UD_OSRM_RETRY_BACKOFF_MS", s.matrix.osrm_retry_backoff_ms)

    # Logs
    s.logging.level = os.getenv("UD_LOG_LEVEL", s.logging.level)
    s.logging.emit_metrics = os.getenv("UD_EMIT_METRICS", str(s.logging.emit_metrics)).lower() == "true"

    # Features
    s.features.enable_solver = os.getenv("UD_FEATURE_SOLVER", str(s.features.enable_solver)).lower() == "true"
    s.features.enable_heuristics = os.getenv("UD_FEATURE_HEURISTICS", str(s.features.enable_heuristics)).lower() == "true"
    s.features.enable_events = os.getenv("UD_FEATURE_EVENTS", str(s.features.enable_events)).lower() == "true"
    s.features.enable_db_bulk_ops = os.getenv("UD_FEATURE_DB_BULK", str(s.features.enable_db_bulk_ops)).lower() == "true"

    s.default_timezone = os.getenv("UD_DEFAULT_TZ", s.default_timezone)

    # Normalise les poids
    s.heuristic = s.heuristic.normalized()
    return s

def merge_overrides(base: Settings, overrides: Dict[str, Any]) -> Settings:
    """
    Fusionne un dict (p. ex. depuis la DB entreprise) sur la config.
    Clés supportées : "heuristic", "solver", "time", "realtime", "fairness",
                      "emergency", "matrix", "logging", "features", "default_timezone".
    """
    # Conversion récursive simple
    for section, payload in overrides.items():
        if not hasattr(base, section):
            continue
        section_val = getattr(base, section)
        if isinstance(payload, dict):
            for k, v in payload.items():
                if hasattr(section_val, k):
                    setattr(section_val, k, v)
        else:
            setattr(base, section, payload)
    # normalise les poids après merge
    base.heuristic = base.heuristic.normalized()
    return base

def for_company(company: Any) -> Settings:
    """
    Construit les Settings pour une entreprise donnée.
    - Charge d'abord from_env()
    - Applique un override s'il existe (company.dispatch_settings ou company.dispatch_config)
      au format JSON (dict).
    """
    cfg = from_env()
    # Lazy import pour éviter les cycles si nécessaire
    # from models import Company

    raw = None
    for attr in ("dispatch_settings", "dispatch_config", "dispatch_params"):
        if hasattr(company, attr):
            raw = getattr(company, attr)
            break

    if isinstance(raw, dict):
        return merge_overrides(cfg, raw)
    if isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
            return merge_overrides(cfg, data if isinstance(data, dict) else {})
        except Exception:
            return cfg
    return cfg

# ------------------------------------------------------------
# Utilitaires pour horaires chauffeur (mapping DriverWorkingConfig)
# ------------------------------------------------------------

def driver_work_window_from_config(dw: Any) -> Tuple[int, int, int]:
    """
    Convertit un DriverWorkingConfig vers une fenêtre de travail en minutes depuis minuit.
    Retourne (start_min, end_min, daily_limit_min).

    Attendus (d'après votre modèle) :
      - earliest_start: minutes depuis minuit (int)
      - latest_start: minutes depuis minuit (int)  -> sert ici de 'fin'
      - total_working_minutes: limite journalière
    """
    if dw is None:
        # Fenêtre « large » si pas de config
        return (6 * 60, 22 * 60, 8 * 60)  # 06:00–22:00, 8h max

    start = getattr(dw, "earliest_start", 6 * 60) or 6 * 60
    end = getattr(dw, "latest_start", 22 * 60) or 22 * 60
    limit = getattr(dw, "total_working_minutes", 8 * 60) or 8 * 60

    # Sanity checks
    start = max(0, min(24 * 60, int(start)))
    end = max(0, min(24 * 60, int(end)))
    limit = max(60, min(24 * 60, int(limit)))

    if end <= start:
        # Corrige une saisie incohérente
        end = min(24 * 60, start + limit)

    return (start, end, limit)

# ------------------------------------------------------------
# Constante process globale (optionnel)
# ------------------------------------------------------------

# Instance par défaut (peut être rechargée si besoin)
DEFAULT_SETTINGS = from_env()
