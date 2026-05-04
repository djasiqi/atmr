# backend/services/eta_service.py

"""Service centralisé pour calculs ETA avec intégration OSRM et ML.

Ce service unifie toute la logique de calcul d'ETA :
- Calcul ETA de base via OSRM
- Correction ML si disponible
- Cache Redis
- Logging précision pour analytics
- Fallback Haversine amélioré
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Tuple

from ext import db, redis_client
from services.geolocation.osrm import route_info as osrm_route_info
from services.infrastructure.feature_flags import FeatureFlags
from services.ml.models.eta_delay import ETADelayModel, ETADelayPrediction
from shared.geo_utils import haversine_distance

logger = logging.getLogger(__name__)

# Constantes
DEFAULT_OSRM_URL = "http://osrm:5000"
DEFAULT_OSRM_PROFILE = "driving"
DEFAULT_OSRM_TIMEOUT = 3  # secondes
DEFAULT_COORD_PRECISION = 5
DEFAULT_AVG_SPEED_KMH = 40.0

# Seuils pour vitesse adaptative (fallback Haversine)
DISTANCE_CITY_KM = 5.0  # < 5 km = centre-ville
DISTANCE_SUBURB_KM = 20.0  # 5-20 km = banlieue
SPEED_CITY_KMH = 20.0
SPEED_SUBURB_KMH = 40.0
SPEED_HIGHWAY_KMH = 60.0


@dataclass
class EtaContext:
    """Contexte pour calcul ETA avec ML."""

    booking_id: int | None = None
    assignment_id: int | None = None
    driver_id: int | None = None
    company_id: int | None = None
    scheduled_time: datetime | None = None
    booking: Any = None  # Objet Booking (optionnel)
    driver: Any = None  # Objet Driver (optionnel)


@dataclass
class EtaResult:
    """Résultat d'un calcul ETA."""

    duration_seconds: int
    distance_meters: int
    source: str  # "osrm", "osrm_ml", "haversine", "haversine_adaptive"
    confidence: float = 1.0  # Confiance ML (0.0-1.0)
    ml_correction_factor: float | None = None  # Facteur de correction ML
    ml_predicted_delay_minutes: float | None = None  # Retard prédit par ML


class EtaService:
    """Service centralisé pour calculs ETA avec intégration OSRM et ML."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        osrm_url: str = DEFAULT_OSRM_URL,
        osrm_profile: str = DEFAULT_OSRM_PROFILE,
        osrm_timeout: int = DEFAULT_OSRM_TIMEOUT,
        coord_precision: int = DEFAULT_COORD_PRECISION,
        avg_speed_kmh_fallback: float = DEFAULT_AVG_SPEED_KMH,
        ml_predictor: ETADelayModel | None = None,
        redis_client_instance: Any | None = None,
    ):
        """Initialise le service ETA.

        Args:
            osrm_url: URL du serveur OSRM
            osrm_profile: Profil OSRM (driving, walking, etc.)
            osrm_timeout: Timeout OSRM en secondes
            coord_precision: Précision coordonnées pour cache
            avg_speed_kmh_fallback: Vitesse moyenne pour fallback Haversine
            ml_predictor: Modèle ML pour correction ETA (optionnel)
            redis_client_instance: Client Redis pour cache (optionnel)
        """
        self.osrm_url = osrm_url
        self.osrm_profile = osrm_profile
        self.osrm_timeout = osrm_timeout
        self.coord_precision = coord_precision
        self.avg_speed_kmh_fallback = avg_speed_kmh_fallback
        self.ml_predictor = ml_predictor
        self.redis_client = redis_client_instance or redis_client

        # Logging asynchrone (optionnel, pour éviter blocage)
        self._enable_accuracy_logging = True

    def calculate_eta(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        *,
        context: EtaContext | None = None,
        use_ml: bool = True,
        use_osrm: bool = True,
    ) -> EtaResult:
        """Calcule ETA avec ML si disponible.

        Args:
            origin: Position d'origine (lat, lon)
            destination: Position de destination (lat, lon)
            context: Contexte pour ML (booking, driver, etc.)
            use_ml: Activer correction ML si disponible (sera vérifié avec FeatureFlags)
            use_osrm: Utiliser OSRM (sinon fallback Haversine)

        Returns:
            EtaResult avec durée, distance, source, confiance
        """
        # 1. ETA OSRM de base
        base_eta_seconds: int = 0
        base_distance_meters: int = 0
        source = "haversine_adaptive"

        if use_osrm:
            try:
                # Utiliser route_info pour obtenir distance + durée
                route_data = osrm_route_info(
                    origin=origin,
                    destination=destination,
                    base_url=self.osrm_url,
                    profile=self.osrm_profile,
                    timeout=self.osrm_timeout,
                    redis_client=self.redis_client,
                    coord_precision=self.coord_precision,
                )

                if route_data:
                    base_eta_seconds = int(route_data.get("duration", 0))
                    base_distance_meters = int(route_data.get("distance", 0))
                    source = "osrm"

                    if base_eta_seconds <= 0:
                        raise ValueError("OSRM returned invalid duration")
            except Exception as e:
                logger.warning("[EtaService] OSRM failed → fallback haversine: %s", e)
                use_osrm = False

        # 2. Fallback Haversine si OSRM échoué ou désactivé
        if not use_osrm or base_eta_seconds <= 0:
            base_eta_seconds, base_distance_meters = self._haversine_eta(
                origin, destination
            )
            source = "haversine_adaptive"

        # 3. Correction ML si disponible et activé
        corrected_eta_seconds = base_eta_seconds
        ml_correction_factor: float | None = None
        ml_predicted_delay: float | None = None
        confidence = 1.0

        # ✅ Vérifier feature flag ML global (avec pourcentage de trafic)
        # Le feature flag contrôle à la fois l'activation globale ET le
        # pourcentage de trafic
        ml_enabled_globally = FeatureFlags.is_ml_enabled()
        should_use_ml_flag = FeatureFlags.should_use_ml()  # Gère aussi le pourcentage
        should_use_ml = use_ml and ml_enabled_globally and should_use_ml_flag

        if should_use_ml and self.ml_predictor and context and context.booking:
            try:
                # Prédire retard avec ML
                ml_prediction: ETADelayPrediction = self.ml_predictor.predict(
                    context.booking, context.driver
                )

                ml_predicted_delay = ml_prediction.predicted_delay_minutes
                confidence = ml_prediction.confidence

                # Convertir retard prédit en facteur de correction ETA
                # Si ML prédit un retard de +5 min, on augmente l'ETA de base
                # Facteur = 1.0 + (retard_prédit_minutes / 60) /
                # (eta_base_secondes / 60)
                if base_eta_seconds > 0:
                    eta_base_minutes = base_eta_seconds / 60.0
                    # Correction : si retard prédit, augmenter ETA proportionnellement
                    correction_factor = 1.0 + (ml_predicted_delay / eta_base_minutes)
                    # Limiter facteur entre 0.5 et 2.0 pour éviter corrections extrêmes
                    correction_factor = max(0.5, min(2.0, correction_factor))
                    ml_correction_factor = correction_factor

                    corrected_eta_seconds = int(base_eta_seconds * correction_factor)
                    source = "osrm_ml" if use_osrm else "haversine_ml"
                    logger.debug(
                        (
                            "[EtaService] ML correction appliquée: facteur=%.2f, "
                            "retard_prédit=%.1f min, ETA_base=%d s → ETA_corrigé=%d s"
                        ),
                        correction_factor,
                        ml_predicted_delay,
                        base_eta_seconds,
                        corrected_eta_seconds,
                    )
                    # Fallback gracieux : utiliser ETA de base sans correction
                else:
                    msg = (
                        "[EtaService] Impossible d'appliquer correction ML: "
                        "base_eta_seconds <= 0"
                    )
                    logger.warning(msg)

            except Exception as e:
                logger.warning(
                    "[EtaService] Erreur correction ML (fallback gracieux): %s", e
                )
                # Fallback gracieux : utiliser ETA de base sans correction

        # 4. Log pour métriques précision (async, non-bloquant)
        if self._enable_accuracy_logging and context:
            self._log_eta_prediction(
                origin=origin,
                destination=destination,
                predicted_eta_seconds=corrected_eta_seconds,
                context=context,
                source=source,
                ml_confidence=confidence if should_use_ml else None,
            )

        return EtaResult(
            duration_seconds=max(1, corrected_eta_seconds),
            distance_meters=max(0, base_distance_meters),
            source=source,
            confidence=confidence,
            ml_correction_factor=ml_correction_factor,
            ml_predicted_delay_minutes=ml_predicted_delay,
        )

    def _haversine_eta(
        self, origin: Tuple[float, float], destination: Tuple[float, float]
    ) -> Tuple[int, int]:
        """Calcule ETA via Haversine avec vitesse adaptative.

        Args:
            origin: Position d'origine (lat, lon)
            destination: Position de destination (lat, lon)

        Returns:
            Tuple (duration_seconds, distance_meters)
        """
        # Calculer distance
        distance_km = haversine_distance(
            origin[0], origin[1], destination[0], destination[1]
        )
        distance_meters = int(distance_km * 1000)

        # Vitesse adaptative selon distance
        speed_kmh = self._get_adaptive_speed_kmh(distance_km)

        # Calculer durée
        time_hours = distance_km / speed_kmh
        duration_seconds = int(time_hours * 3600)

        return max(1, duration_seconds), distance_meters

    def _get_adaptive_speed_kmh(self, distance_km: float) -> float:
        """Estime vitesse moyenne selon zone (heuristique).

        Args:
            distance_km: Distance en kilomètres

        Returns:
            Vitesse moyenne estimée en km/h
        """
        if distance_km < DISTANCE_CITY_KM:
            # < 5 km → centre-ville → 20 km/h
            return SPEED_CITY_KMH
        if distance_km < DISTANCE_SUBURB_KM:
            # 5-20 km → banlieue → 40 km/h
            return SPEED_SUBURB_KMH
        # > 20 km → autoroute possible → 60 km/h
        return SPEED_HIGHWAY_KMH

    def _log_eta_prediction(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        predicted_eta_seconds: int,
        context: EtaContext,
        source: str,
        ml_confidence: float | None = None,
    ) -> None:
        """Log prédiction ETA pour analytics (non-bloquant).

        Args:
            origin: Position d'origine
            destination: Position de destination
            predicted_eta_seconds: ETA prédit en secondes
            context: Contexte (booking, driver, etc.)
            source: Source ETA (osrm, osrm_ml, haversine, etc.)
            ml_confidence: Confiance ML (optionnel)
        """
        try:
            # Import ici pour éviter dépendance circulaire
            from models.eta_accuracy_log import EtaAccuracyLog

            log_entry = EtaAccuracyLog()
            log_entry.booking_id = context.booking_id
            log_entry.assignment_id = context.assignment_id
            log_entry.predicted_eta_seconds = predicted_eta_seconds
            log_entry.origin_lat = origin[0]
            log_entry.origin_lon = origin[1]
            log_entry.dest_lat = destination[0]
            log_entry.dest_lon = destination[1]
            log_entry.source = source
            log_entry.ml_confidence = ml_confidence

            db.session.add(log_entry)
            # Ne pas commit ici (laisser l'appelant gérer la transaction)
            # db.session.commit() sera fait par l'appelant

        except ImportError:
            # Modèle pas encore créé (migration en cours)
            logger.debug(
                "[EtaService] EtaAccuracyLog non disponible (migration en cours)"
            )
        except Exception as e:
            # Logging non-bloquant : ne pas faire échouer le calcul ETA
            logger.warning("[EtaService] Erreur logging précision ETA: %s", e)


# Instance globale (singleton)
_eta_service_instance: EtaService | None = None


def get_eta_service() -> EtaService:
    """Retourne l'instance globale du service ETA (singleton).

    Returns:
        Instance EtaService
    """
    global _eta_service_instance

    if _eta_service_instance is None:
        # Charger modèle ML si disponible
        ml_predictor: ETADelayModel | None = None
        try:
            ml_predictor = ETADelayModel()
            if ml_predictor.is_trained:
                logger.info("[EtaService] Modèle ML chargé et prêt")
            else:
                logger.warning("[EtaService] Modèle ML non entraîné")
                ml_predictor = None
        except Exception as e:
            logger.warning("[EtaService] Impossible de charger modèle ML: %s", e)
            ml_predictor = None

        _eta_service_instance = EtaService(
            osrm_url=DEFAULT_OSRM_URL,
            osrm_profile=DEFAULT_OSRM_PROFILE,
            osrm_timeout=DEFAULT_OSRM_TIMEOUT,
            coord_precision=DEFAULT_COORD_PRECISION,
            avg_speed_kmh_fallback=DEFAULT_AVG_SPEED_KMH,
            ml_predictor=ml_predictor,
            redis_client_instance=redis_client,
        )

    return _eta_service_instance
