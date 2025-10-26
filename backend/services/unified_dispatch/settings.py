# backend/services/unified_dispatch/settings.py
from __future__ import annotations

import copy
import json
import os
from ast import literal_eval
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any, Dict

# paramètres centralisés (capacité, buffers, pénalités…)
# ------------------------------------------------------------
# Groupes de paramètres
# ------------------------------------------------------------


@dataclass
class HeuristicWeights:
    # distance/temps vers pickup (réduit encore)
    proximity: float = 0.20
    # équité (courses du jour) - AUGMENTÉ à 70% pour forcer répartition 3-3-3
    driver_load_balance: float = 0.70
    priority: float = 0.06               # priorité booking (médical, VIP…)
    return_urgency: float = 0.03         # retours déclenchés à la demande
    regular_driver_bonus: float = 0.01   # chauffeur habituel du client

    def normalized(self) -> HeuristicWeights:
        total = self.proximity + self.driver_load_balance + \
            self.priority + self.return_urgency + self.regular_driver_bonus
        if total == 0:
            return self
        return HeuristicWeights(
            proximity=self.proximity / total,
            driver_load_balance=self.driver_load_balance / total,
            priority=self.priority / total,
            return_urgency=self.return_urgency / total,
            regular_driver_bonus=self.regular_driver_bonus / total,
        )


@dataclass
class SolverParams:
    # OR-Tools
    time_limit_sec: int = 60                 # limite max (adaptative possible)
    global_span_cost: int = 100              # compaction des tournées
    vehicle_fixed_cost: int = 10             # coût fixe par véhicule utilisé
    unassigned_penalty_base: int = 10000     # pénalité non-assigné (par tâche)
    # LIMITE UNIQUE (heuristique = solveur)
    max_bookings_per_driver: int = 6
    pickup_dropoff_slack_min: int = 5        # marge autour des TW
    use_pickup_dropoff_pairs: bool = True    # arc obligatoire pickup->dropoff
    add_driver_work_windows: bool = True     # fenêtres de travail véhicule
    round_trip_driver_penalty_min: int = 120
    strict_driver_end_window: bool = True    # borne de fin stricte
    # passe 1 réguliers, passe 2 urgences si besoin
    regular_first_two_phase: bool = True


@dataclass
class ServiceTimesSettings:
    """Paramètres de temps de service pour les courses."""

    pickup_service_min: int = 5              # temps de pickup (minutes)
    dropoff_service_min: int = 10            # temps de dropoff (minutes)
    # marge minimale entre deux courses (minutes)
    min_transition_margin_min: int = 15


@dataclass
class PoolingSettings:
    """Paramètres de regroupement de courses (ride-pooling)."""

    enabled: bool = True                     # activer le regroupement de courses
    # tolérance temporelle pour le pickup (±10min)
    time_tolerance_min: int = 10
    # distance maximale entre pickups (mètres)
    pickup_distance_m: int = 500
    # détour maximal acceptable pour les dropoffs (minutes)
    max_detour_min: int = 15


@dataclass
class TimeSettings:
    # Buffers et marges (minutes)
    # marge avant pickup (±5min → fenêtre 17h55-18h05 pour course à 18h00)
    pickup_buffer_min: int = 5
    dropoff_buffer_min: int = 5              # marge avant dropoff
    pickup_window_min: int = 10              # fenêtre de pickup
    dropoff_window_min: int = 10             # fenêtre de dropoff
    horizon_min: int = 240                   # horizon de planification (4h)
    horizon_max: int = 1440                  # horizon max (24h)
    # Seuils (minutes)
    late_threshold_min: int = 5              # seuil de retard
    early_threshold_min: int = 5             # seuil d'avance
    # Divers
    # utiliser l'heure locale (Europe/Zurich)
    use_local_time: bool = True


@dataclass
class RealtimeSettings:
    # Seuils de rafraîchissement
    refresh_threshold_min: int = 5           # seuil de rafraîchissement
    refresh_interval_min: int = 5            # intervalle de rafraîchissement
    # Divers
    enable_realtime: bool = True             # activer le temps réel
    enable_eta: bool = True                  # activer les ETA


@dataclass
class FairnessSettings:
    # Équité entre chauffeurs
    enable_fairness: bool = True             # activer l'équité
    fairness_window_days: int = 7            # fenêtre d'équité (jours)
    fairness_weight: float = 0.3             # poids de l'équité


@dataclass
class EmergencyPolicy:
    # Gestion des urgences
    allow_emergency_drivers: bool = True     # autoriser les chauffeurs d'urgence
    emergency_threshold_min: int = 30        # seuil d'urgence (minutes)
    emergency_priority: float = 0.8          # priorité des urgences


@dataclass
class MatrixSettings:
    # Matrices de distance/temps
    provider: str = "osrm"                   # fournisseur de matrice
    cache_ttl_sec: int = 3600                # TTL du cache (1h)
    enable_cache: bool = True                # activer le cache
    osrm_url: str = "http://localhost:5000"  # URL du serveur OSRM
    osrm_profile: str = "car"                # profil OSRM


@dataclass
class LoggingSettings:
    # Journalisation
    level: str = "INFO"                      # niveau de log
    enable_file: bool = False                # activer les logs fichier
    file_path: str = "logs/dispatch.log"     # chemin du fichier de log
    enable_metrics: bool = True              # activer les métriques


@dataclass
class AutorunSettings:
    # Autorun settings
    autorun_enabled: bool = True             # Enable autorun by default
    autorun_interval_sec: int = 300          # Default interval: 5 minutes


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
    service_times: ServiceTimesSettings = field(
        default_factory=ServiceTimesSettings)
    pooling: PoolingSettings = field(default_factory=PoolingSettings)
    time: TimeSettings = field(default_factory=TimeSettings)
    realtime: RealtimeSettings = field(default_factory=RealtimeSettings)
    fairness: FairnessSettings = field(default_factory=FairnessSettings)
    emergency: EmergencyPolicy = field(default_factory=EmergencyPolicy)
    matrix: MatrixSettings = field(default_factory=MatrixSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    autorun: AutorunSettings = field(
        default_factory=AutorunSettings)  # Added autorun settings

    # Divers
    default_timezone: str = "Europe/Zurich"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # normaliser les poids pour éviter toute dérive
        d["heuristic"] = asdict(self.heuristic.normalized())
        return d

# ------------------------------------------------------------
# Fonctions utilitaires
# ------------------------------------------------------------


def _get_env_or_default(key: str, default: Any) -> Any:
    """Récupère une variable d'environnement ou une valeur par défaut."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _merge_dicts(base: Dict[str, Any],
                 override: Dict[str, Any]) -> Dict[str, Any]:
    """Fusionne deux dictionnaires de manière récursive."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def merge_overrides(base: Settings, overrides: Dict[str, Any]) -> Settings:
    """Applique des overrides dict sur une instance Settings (avec sous-dataclasses).
    - Ignore les clés inconnues (mode, run_async, ...)
    - Conserve les types des sous-objets (pas de dict qui remplace une dataclass).
    """

    def _merge_into(obj: Any, ov: Dict[str, Any]) -> Any:
        for k, v in ov.items():
            if not hasattr(obj, k):
                # clé inconnue → on ignore
                continue
            cur = getattr(obj, k)
            if is_dataclass(cur) and isinstance(v, dict):
                _merge_into(cur, v)
            else:
                try:
                    setattr(obj, k, v)
                except Exception:
                    # si assignation impossible, on ignore
                    continue
        return obj

    new_settings = copy.deepcopy(base)
    return _merge_into(new_settings, overrides)


def from_dict(d: Dict[str, Any]) -> Settings:
    """Crée une configuration à partir d'un dictionnaire."""
    return Settings(**d)


def from_json(json_str: str) -> Settings:
    """Crée une configuration à partir d'une chaîne JSON."""
    return from_dict(json.loads(json_str))


def for_company(company) -> Settings:
    """Crée une configuration pour une entreprise."""
    # Configuration de base
    s = Settings()

    # Surcharges spécifiques à l'entreprise
    if hasattr(company, "dispatch_settings") and company.dispatch_settings:
        try:
            overrides = json.loads(company.dispatch_settings)
            s = merge_overrides(s, overrides)
        except (json.JSONDecodeError, TypeError):
            pass

    # Surcharges globales depuis l'environnement
    s.matrix.osrm_url = _get_env_or_default("UD_OSRM_URL", s.matrix.osrm_url)
    s.matrix.cache_ttl_sec = _get_env_or_default(
        "UD_MATRIX_CACHE_TTL_SEC", s.matrix.cache_ttl_sec)
    s.solver.time_limit_sec = _get_env_or_default(
        "UD_SOLVER_TIME_LIMIT_SEC", s.solver.time_limit_sec)
    s.autorun.autorun_interval_sec = _get_env_or_default(
        "DISPATCH_AUTORUN_INTERVAL_SEC", s.autorun.autorun_interval_sec)
    s.autorun.autorun_enabled = _get_env_or_default(
        "DISPATCH_AUTORUN_ENABLED", s.autorun.autorun_enabled)

    return s


def driver_work_window_from_config(_driver_config):
    """Extract driver work window from configuration.
    Retourne (start, end) en naïf local pour la journée courante.
    """
    from shared.time_utils import coerce_local_day, day_local_bounds
    today_date = datetime.now().date()
    day_str = coerce_local_day(today_date)  # 'YYYY-MM-DD'
    return day_local_bounds(day_str)
