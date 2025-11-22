# backend/services/unified_dispatch/settings.py
from __future__ import annotations

import copy
import json
import os
from ast import literal_eval
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Tuple, overload

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
    priority: float = 0.06  # priorité booking (médical, VIP…)
    return_urgency: float = 0.03  # retours déclenchés à la demande
    regular_driver_bonus: float = 0.01  # chauffeur habituel du client

    def normalized(self) -> HeuristicWeights:
        total = (
            self.proximity
            + self.driver_load_balance
            + self.priority
            + self.return_urgency
            + self.regular_driver_bonus
        )
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
    time_limit_sec: int = 60  # limite max (adaptative possible)
    global_span_cost: int = 100  # compaction des tournées
    vehicle_fixed_cost: int = 10  # coût fixe par véhicule utilisé
    unassigned_penalty_base: int = 10000  # pénalité non-assigné (par tâche)
    # LIMITE UNIQUE (heuristique = solveur)
    max_bookings_per_driver: int = 6
    pickup_dropoff_slack_min: int = 5  # marge autour des TW
    use_pickup_dropoff_pairs: bool = True  # arc obligatoire pickup->dropoff
    add_driver_work_windows: bool = True  # fenêtres de travail véhicule
    round_trip_driver_penalty_min: int = 120
    strict_driver_end_window: bool = True  # borne de fin stricte
    # passe 1 réguliers, passe 2 urgences si besoin
    regular_first_two_phase: bool = True
    # Warm-start
    enable_warm_start: bool = True  # utiliser warm-start pour -30% temps solver


@dataclass
class ServiceTimesSettings:
    """Paramètres de temps de service pour les courses."""

    pickup_service_min: int = 5  # temps de pickup (minutes)
    dropoff_service_min: int = 10  # temps de dropoff (minutes)
    # marge minimale entre deux courses (minutes)
    min_transition_margin_min: int = 15


@dataclass
class PoolingSettings:
    """Paramètres de regroupement de courses (ride-pooling)."""

    enabled: bool = True  # activer le regroupement de courses
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
    dropoff_buffer_min: int = 5  # marge avant dropoff
    pickup_window_min: int = 10  # fenêtre de pickup
    dropoff_window_min: int = 10  # fenêtre de dropoff
    horizon_min: int = 240  # horizon de planification (4h)
    horizon_max: int = 1440  # horizon max (24h)
    # Seuils (minutes)
    late_threshold_min: int = 5  # seuil de retard
    early_threshold_min: int = 5  # seuil d'avance
    # Divers
    # utiliser l'heure locale (Europe/Zurich)
    use_local_time: bool = True


@dataclass
class RealtimeSettings:
    # Seuils de rafraîchissement
    refresh_threshold_min: int = 5  # seuil de rafraîchissement
    refresh_interval_min: int = 5  # intervalle de rafraîchissement
    # Divers
    enable_realtime: bool = True  # activer le temps réel
    enable_eta: bool = True  # activer les ETA


@dataclass
class FairnessSettings:
    # Équité entre chauffeurs
    enable_fairness: bool = True  # activer l'équité
    fairness_window_days: int = 7  # fenêtre d'équité (jours)
    fairness_weight: float = 0.3  # poids de l'équité
    reset_daily_load: bool = False  # remettre les compteurs à zéro pour un run manuel


@dataclass
class EmergencyPolicy:
    # Gestion des urgences
    allow_emergency_drivers: bool = True  # autoriser les chauffeurs d'urgence
    emergency_threshold_min: int = 30  # seuil d'urgence (minutes)
    emergency_priority: float = 0.8  # priorité des urgences
    emergency_penalty: float = 900.0  # pénalité d'utilisation (0-1000), plus élevé = utilisé en dernier recours seulement


@dataclass
class MatrixSettings:
    # Matrices de distance/temps
    provider: str = "osrm"  # fournisseur de matrice
    cache_ttl_sec: int = 3600  # TTL du cache (1h)
    enable_cache: bool = True  # activer le cache
    osrm_url: str = "http://osrm:5000"  # URL du serveur OSRM (nom du service Docker)
    osrm_profile: str = "car"  # profil OSRM
    osrm_timeout_sec: int = 5  # Timeout pour les requêtes OSRM (secondes)
    osrm_max_retries: int = 2  # Nombre maximum de tentatives en cas d'échec
    osrm_max_sources_per_call: int = 60  # Nombre maximum de sources par requête OSRM
    osrm_rate_limit_per_sec: int = 8  # Limite de débit pour les requêtes OSRM
    osrm_retry_backoff_ms: int = (
        250  # Délai d'attente entre les tentatives (millisecondes)
    )


@dataclass
class LoggingSettings:
    # Journalisation
    level: str = "INFO"  # niveau de log
    enable_file: bool = False  # activer les logs fichier
    file_path: str = "logs/dispatch.log"  # chemin du fichier de log
    enable_metrics: bool = True  # activer les métriques


@dataclass
class AutorunSettings:
    # Autorun settings
    autorun_enabled: bool = True  # Enable autorun by default
    autorun_interval_sec: int = 300  # Default interval: 5 minutes


@dataclass
class RLSettings:
    """Paramètres pour le Reinforcement Learning."""

    # Alpha pour fusion heuristique + RL: final_score = (1-alpha)*heur + alpha*rl
    alpha: float = 0.2  # Poids RL (0 = heuristique pure, 1 = RL pur)
    # Seuils de backout automatique
    min_quality_score: float = 70.0  # Seuil minimum quality_score
    min_on_time_rate: float = 85.0  # Seuil minimum on_time_rate (%)
    max_avg_delay_min: float = 5.0  # Seuil maximum average delay (minutes)
    consecutive_failures_threshold: int = 2  # Backout après N cycles consécutifs
    # Garde-fous temporels
    min_minutes_before_pickup: int = 10  # Ne pas réassigner si < X min avant pickup
    # Mode shadow
    enable_shadow_mode: bool = True  # Activer le mode shadow (par défaut)


@dataclass
class ClusteringSettings:
    """Paramètres pour le clustering géographique."""

    # Seuils d'activation
    bookings_threshold: int = 100  # Activer clustering si > N bookings
    # Configuration K-Means
    max_bookings_per_zone: int = 100  # Nombre max de courses par zone
    # Tolérance cross-zone
    cross_zone_tolerance: float = 0.1  # Tolérance pour passerelles entre zones (10%)
    # Distance maximale pour assigner un driver à une zone
    max_zone_radius_km: float = 50.0  # Rayon maximal en km


@dataclass
class MultiObjectiveSettings:
    """Phase 5.1 - Paramètres multi-objectif : Équité vs Efficacité."""

    # Slider équité/efficacité (0.0 = efficacité pure, 1.0 = équité pure)
    fairness_weight: float = 0.5  # Par défaut: 50/50
    # efficiency_weight est calculé automatiquement: 1 - fairness_weight

    # Multi-objectif: conserver N solutions frontières Pareto
    pareto_solutions_count: int = 3  # Garder 3 solutions non-dominées

    # Capacité opérationnelle: ajustement sans redéploiement
    enable_realtime_adjustment: bool = True  # Activer ajustement temps réel

    @property
    def efficiency_weight(self) -> float:
        """Calcule efficiency_weight depuis fairness_weight."""
        return 1.0 - self.fairness_weight


@dataclass
class SafetySettings:
    """Paramètres de sécurité et garde-fous temporels."""

    # Marge minimum entre deux courses pour éviter conflits temporels
    min_gap_minutes: int = 30  # Écart minimum entre deux courses (minutes)
    # Validation stricte des conflits temporels
    strict_time_conflict_check: bool = True  # Vérifier busy_until AVANT scoring
    # Buffer post-course pour transition
    post_trip_buffer_min: int = 15  # Buffer après dropoff avant prochaine course
    # Timeout dynamique pour gros problèmes
    dynamic_timeout_enabled: bool = True  # Timeout adapté à la taille du problème


@dataclass
class FeatureFlags:
    enable_solver: bool = True  # peut être désactivé en mode dégradé
    enable_heuristics: bool = True
    enable_events: bool = True  # SocketIO + notifications
    enable_db_bulk_ops: bool = True  # writes atomiques/bulk
    # Phase 0: Nouveaux flags pour sécurité
    enable_rl: bool = False  # Activer RL dans le pipeline
    enable_rl_apply: bool = False  # RL auto-apply (sinon suggest only)
    enable_clustering: bool = False  # Clustering géographique
    enable_parallel_heuristics: bool = False  # Parallélisation heuristiques
    # A1: Prévention des conflits temporels
    enable_strict_temporal_conflict_check: bool = (
        True  # Validation stricte des conflits temporels
    )


# ------------------------------------------------------------
# Configuration globale
# ------------------------------------------------------------


@dataclass
class Settings:
    heuristic: HeuristicWeights = field(default_factory=HeuristicWeights)
    solver: SolverParams = field(default_factory=SolverParams)
    service_times: ServiceTimesSettings = field(default_factory=ServiceTimesSettings)
    pooling: PoolingSettings = field(default_factory=PoolingSettings)
    time: TimeSettings = field(default_factory=TimeSettings)
    realtime: RealtimeSettings = field(default_factory=RealtimeSettings)
    fairness: FairnessSettings = field(default_factory=FairnessSettings)
    emergency: EmergencyPolicy = field(default_factory=EmergencyPolicy)
    matrix: MatrixSettings = field(default_factory=MatrixSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    autorun: AutorunSettings = field(
        default_factory=AutorunSettings
    )  # Added autorun settings
    rl: RLSettings = field(default_factory=RLSettings)  # Phase 2: RL settings
    clustering: ClusteringSettings = field(
        default_factory=ClusteringSettings
    )  # Phase 3: Clustering settings
    multi_objective: MultiObjectiveSettings = field(
        default_factory=MultiObjectiveSettings
    )  # Phase 5.1: Multi-objectif
    safety: SafetySettings = field(
        default_factory=SafetySettings
    )  # Phase A1: Safety & temporal conflict prevention

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


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Fusionne deux dictionnaires de manière récursive."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def _validate_merge_result(
    new_settings: Settings,
    overrides: "Dict[str, Any]",
    modified_keys: "List[Tuple[str, Any, Any]]",
) -> "Dict[str, Any]":
    """Valide que les paramètres critiques demandés ont bien été appliqués.

    Args:
        new_settings: Settings après merge (utilisé pour vérifier les valeurs finales)
        overrides: Paramètres demandés
        modified_keys: Liste des clés modifiées (path, old_value, new_value)

    Retourne un dict avec :
    - applied: liste des clés appliquées
    - ignored: liste des clés ignorées (inconnues ou non applicables)
    - errors: liste des erreurs de validation
    """
    import logging

    logger = logging.getLogger(__name__)

    validation_result: dict[str, list[str]] = {
        "applied": [],
        "ignored": [],
        "errors": [],
        "critical_errors": [],
    }

    # Clés critiques qui doivent être dans Settings
    critical_keys = {
        "fairness": {"fairness_weight", "enabled", "reset_daily_load"},
        "heuristic": {"driver_load_balance", "proximity", "priority"},
        "solver": {"time_limit_sec"},
        "features": {"enable_solver", "enable_heuristics", "enable_rl"},
    }

    # Vérifier les paramètres appliqués et valider les valeurs finales
    applied_paths = {path for path, _, _ in modified_keys}
    for section, keys in critical_keys.items():
        for key in keys:
            path = f"{section}.{key}"
            if path in applied_paths:
                # Vérifier que la valeur finale dans new_settings correspond à la valeur demandée
                if (
                    section in overrides
                    and isinstance(overrides[section], dict)
                    and key in overrides[section]
                ):
                    requested_value = overrides[section][key]
                    # Récupérer la valeur finale depuis new_settings
                    section_obj = getattr(new_settings, section, None)
                    if section_obj is not None:
                        final_value = getattr(section_obj, key, None)
                        if final_value is not None and final_value != requested_value:
                            # Valeur différente de celle demandée → erreur
                            error_msg = f"Paramètre {path} appliqué avec valeur différente: {requested_value} → {final_value}"
                            validation_result["errors"].append(error_msg)
                            validation_result["critical_errors"].append(path)
                            logger.warning("[Settings] %s", error_msg)
                        else:
                            validation_result["applied"].append(path)
                    else:
                        validation_result["applied"].append(path)
                else:
                    validation_result["applied"].append(path)
            elif (
                section in overrides
                and isinstance(overrides[section], dict)
                and key in overrides[section]
            ):
                # Paramètre demandé mais non appliqué → erreur
                error_msg = f"Paramètre critique non appliqué: {path}"
                validation_result["errors"].append(error_msg)
                validation_result["critical_errors"].append(path)
                logger.warning("[Settings] %s", error_msg)

    # Identifier les clés ignorées (inconnues mais demandées)
    def _collect_requested_keys(ov: "Dict[str, Any]", path: str = "") -> "List[str]":
        """Collecte toutes les clés demandées dans overrides."""
        requested = []
        for k, v in ov.items():
            current_path = f"{path}.{k}" if path else k
            if isinstance(v, dict):
                requested.extend(_collect_requested_keys(v, current_path))
            else:
                requested.append(current_path)
        return requested

    all_requested = _collect_requested_keys(overrides)
    for requested_path in all_requested:
        if requested_path not in applied_paths:
            # Vérifier si c'est une clé connue mais ignorée (ex: preferred_driver_id)
            parts = requested_path.split(".")
            if parts[0] in critical_keys or parts[0] in [
                "mode",
                "run_async",
                "preferred_driver_id",
                "reset_existing",
                "fast_mode",
            ]:
                # Clé connue mais ignorée (normal pour certains paramètres)
                validation_result["ignored"].append(requested_path)
                logger.debug(
                    "[Settings] Clé connue ignorée (non dans Settings): %s",
                    requested_path,
                )
            else:
                # Clé vraiment inconnue
                validation_result["ignored"].append(requested_path)
                logger.debug("[Settings] Clé inconnue ignorée: %s", requested_path)

    return validation_result


@overload
def merge_overrides(
    base: Settings,
    overrides: Dict[str, Any],
) -> Settings: ...


@overload
def merge_overrides(
    base: Settings,
    overrides: Dict[str, Any],
    *,
    return_validation: Literal[True],
) -> Tuple[Settings, Dict[str, Any]]: ...


def merge_overrides(
    base: Settings,
    overrides: Dict[str, Any],
    *,
    return_validation: bool = False,
) -> Settings | Tuple[Settings, Dict[str, Any]]:
    """Applique des overrides dict sur une instance Settings (avec sous-dataclasses).
    - Ignore les clés inconnues (mode, run_async, preferred_driver_id, ...)
    - Conserve les types des sous-objets (pas de dict qui remplace une dataclass).
    - Log chaque modification pour traçabilité.
    - Valide que les paramètres critiques ont bien été appliqués.
    """
    import logging

    logger = logging.getLogger(__name__)

    modified_keys = []

    def _merge_into(obj: Any, ov: Dict[str, Any], path: str = "") -> Any:
        for key, v in ov.items():
            current_path = f"{path}.{key}" if path else key

            # ✅ Mapping des noms de paramètres frontend → backend
            # Le frontend envoie "emergency_per_stop_penalty" mais le backend attend "emergency_penalty"
            final_key = key
            if path == "emergency" and key == "emergency_per_stop_penalty":
                final_key = "emergency_penalty"
                current_path = f"{path}.{final_key}" if path else final_key
                logger.debug(
                    "[Settings] Mapping frontend→backend: emergency_per_stop_penalty → emergency_penalty"
                )

            if not hasattr(obj, final_key):
                # clé inconnue → on ignore (c'est normal pour preferred_driver_id, mode, etc.)
                logger.debug(
                    "[Settings] Clé inconnue ignorée dans overrides: %s", current_path
                )
                continue
            cur = getattr(obj, final_key)
            if is_dataclass(cur) and isinstance(v, dict):
                # Merge récursif dans la sous-dataclass
                _merge_into(cur, v, current_path)
            else:
                try:
                    old_value = cur
                    setattr(obj, final_key, v)
                    modified_keys.append((current_path, old_value, v))
                    logger.debug(
                        "[Settings] Override appliqué: %s = %r (était: %r)",
                        current_path,
                        v,
                        old_value,
                    )
                except Exception as e:
                    logger.warning(
                        "[Settings] Échec assignation %s = %r: %s", current_path, v, e
                    )
                    continue
        return obj

    new_settings = copy.deepcopy(base)
    _merge_into(new_settings, overrides)

    # ✅ Validation post-merge
    validation_result = _validate_merge_result(new_settings, overrides, modified_keys)

    if modified_keys:
        logger.info(
            "[Settings] %d override(s) appliqué(s): %s (applied=%d, ignored=%d, errors=%d)",
            len(modified_keys),
            [k for k, _, _ in modified_keys],
            len(validation_result["applied"]),
            len(validation_result["ignored"]),
            len(validation_result["errors"]),
        )

    # ✅ Logger les paramètres appliqués vs demandés
    if validation_result["applied"]:
        logger.info("[Settings] Paramètres appliqués: %s", validation_result["applied"])
    if validation_result["ignored"]:
        logger.debug(
            "[Settings] Paramètres ignorés (normaux): %s", validation_result["ignored"]
        )
    if validation_result["errors"]:
        logger.warning(
            "[Settings] Erreurs de validation: %s", validation_result["errors"]
        )
        # Lever une exception si des paramètres critiques n'ont pas été appliqués
        # (optionnel, peut être désactivé via variable d'env)
        strict_validation = (
            os.getenv("UD_SETTINGS_STRICT_VALIDATION", "false").lower() == "true"
        )
        if strict_validation:
            raise ValueError(
                f"Paramètres critiques non appliqués: {validation_result['errors']}"
            )

    if return_validation:
        return new_settings, validation_result
    return new_settings


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

    # ✅ PRIORITÉ 1: Surcharges depuis autonomous_config.dispatch_overrides (nouveau système)
    if hasattr(company, "get_autonomous_config"):
        try:
            autonomous_config = company.get_autonomous_config()
            dispatch_overrides = autonomous_config.get("dispatch_overrides")
            if dispatch_overrides:
                s = merge_overrides(s, dispatch_overrides)
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.debug("[Settings] Erreur lecture autonomous_config: %s", e)

    # ✅ PRIORITÉ 2: Surcharges depuis dispatch_settings (ancien système, pour compatibilité)
    if hasattr(company, "dispatch_settings") and company.dispatch_settings:
        try:
            overrides = json.loads(company.dispatch_settings)
            s = merge_overrides(s, overrides)
        except (json.JSONDecodeError, TypeError):
            pass

    # Surcharges globales depuis l'environnement
    s.matrix.osrm_url = _get_env_or_default("UD_OSRM_URL", s.matrix.osrm_url)
    s.matrix.cache_ttl_sec = _get_env_or_default(
        "UD_MATRIX_CACHE_TTL_SEC", s.matrix.cache_ttl_sec
    )
    s.solver.time_limit_sec = _get_env_or_default(
        "UD_SOLVER_TIME_LIMIT_SEC", s.solver.time_limit_sec
    )
    s.autorun.autorun_interval_sec = _get_env_or_default(
        "DISPATCH_AUTORUN_INTERVAL_SEC", s.autorun.autorun_interval_sec
    )
    s.autorun.autorun_enabled = _get_env_or_default(
        "DISPATCH_AUTORUN_ENABLED", s.autorun.autorun_enabled
    )

    # Safety settings
    s.safety.min_gap_minutes = _get_env_or_default(
        "UD_SAFETY_MIN_GAP_MINUTES", s.safety.min_gap_minutes
    )

    return s


def driver_work_window_from_config(_driver_config):
    """Extract driver work window from configuration.
    Retourne (start, end) en naïf local pour la journée courante.
    """
    from shared.time_utils import coerce_local_day, day_local_bounds

    today_date = datetime.now().date()
    day_str = coerce_local_day(today_date)  # 'YYYY-MM-DD'
    return day_local_bounds(day_str)
