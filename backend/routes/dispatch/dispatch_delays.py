# backend/routes/dispatch/dispatch_delays.py
"""Endpoints pour la gestion des retards."""

# ruff: noqa: I001  # Imports organisés manuellement pour meilleure lisibilité
import logging
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, cast

from flask import request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
    get_jwt_identity,
    jwt_required,
)
from flask_restx import Resource  # pyright: ignore[reportMissingImports]
from http import HTTPStatus

from ext import role_required
from models.enums import BookingStatus, UserRole
from repositories.assignment_repository import AssignmentRepository
from repositories.company_repository import CompanyRepository
from repositories.user_repository import UserRepository
from routes.dispatch import dispatch_ns
from routes.dispatch.dispatch_helpers import (
    _booking_time_expr,
    _calculate_eta_for_assignment,
    _classify_delay_severity,
    _get_current_company,
    _get_driver_previous_booking,
    _parse_date,
)
from routes.dispatch.dispatch_schemas import delay_model

# ✅ DDD: AuthService remplacé par use-cases
from infrastructure.dispatch.reactive_suggestions_adapter import (
    generate_reactive_suggestions as generate_suggestions,
)
from shared.error_handlers import APIErrorHandler
from shared.time_utils import day_local_bounds, now_local

logger = logging.getLogger(__name__)

# Initialisation des repositories
assignment_repo = AssignmentRepository()
company_repo = CompanyRepository()
user_repo = UserRepository()

# Constantes pour les retards
MAX_DELAY_ZERO = 0
PICKUP_DELAY_ZERO = 0
DELAY_MINUTES_THRESHOLD = 5  # Ancien seuil (conservé pour compatibilité)
DELAY_MINUTES_REASONABLE_MAX = 5  # 1-5 min : raisonnable
DELAY_MINUTES_MODERATE_MAX = 10  # 5-10 min : modéré
TIME_DIFF_SECONDS_THRESHOLD = 0.300
DELAY_MINUTES_ZERO = 0
PREVIOUS_BOOKING_RELEVANCE_WINDOW_SECONDS = (
    7200  # 2h : fenêtre pour considérer une course précédente comme pertinente
)


@dispatch_ns.route("/delays")
class DelaysResource(Resource):
    """Retards courants pour la journée."""

    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"date": "YYYY-MM-DD"})
    @dispatch_ns.marshal_list_with(delay_model)
    def get(self):
        """Retards courants (ETA > horaire + 5 minutes) pour la journée."""
        # Validation de la date
        date_str = request.args.get("date")
        if not date_str:
            return APIErrorHandler.handle_validation_error(
                "Paramètre 'date' manquant",
                field="date",
                expected_format="YYYY-MM-DD",
                logger_instance=logger,
            )

        try:
            d = _parse_date(date_str)
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                f"Format de date invalide: {e}",
                field="date",
                provided_value=date_str,
                expected_format="YYYY-MM-DD",
                logger_instance=logger,
            )

        try:
            d0, d1 = day_local_bounds(d.strftime("%Y-%m-%d"))

            company = _get_current_company()
            # ✅ P1: Eager loading pour éviter N+1 queries
            assigns = assignment_repo.find_models_by_company_with_time_range_and_excluded_statuses_eager_loading(
                company_id=company.id,
                start_datetime=d0,
                end_datetime=d1,
                excluded_statuses=[],
            )

            # ✅ P1: Eager loading déjà fait via joinedload dans la requête précédente
            # Les bookings et drivers sont déjà chargés, pas besoin de requêtes supplémentaires
            # Calculer les retards
            delays = []
            for a in assigns:
                b = a.booking  # ✅ Déjà chargé via joinedload
                if not b:
                    continue

                # Temps prévus
                pickup_time = getattr(b, "pickup_time", None) or getattr(
                    b, "scheduled_time", None
                )
                dropoff_time = getattr(b, "dropoff_time", None)

                # Coerce strings -> datetime when needed
                def _to_dt(v):
                    if v is None:
                        return None
                    if isinstance(v, datetime):
                        return v
                    try:
                        # naive ISO string
                        return datetime.fromisoformat(str(v))
                    except Exception:
                        return None

                pickup_time = _to_dt(pickup_time)
                dropoff_time = _to_dt(dropoff_time)

                # ETAs (compat: plusieurs noms possibles)
                pickup_eta = (
                    getattr(a, "pickup_eta", None)
                    or getattr(a, "eta_pickup_at", None)
                    or getattr(a, "estimated_pickup_arrival", None)
                )
                dropoff_eta = (
                    getattr(a, "dropoff_eta", None)
                    or getattr(a, "eta_dropoff_at", None)
                    or getattr(a, "estimated_dropoff_arrival", None)
                )

                pickup_eta = _to_dt(pickup_eta)
                dropoff_eta = _to_dt(dropoff_eta)

                # Calcul des retards
                pickup_delay = MAX_DELAY_ZERO
                if pickup_time and pickup_eta:
                    try:
                        pickup_delay = max(
                            MAX_DELAY_ZERO,
                            int((pickup_eta - pickup_time).total_seconds() // 60),
                        )
                    except Exception:
                        pickup_delay = MAX_DELAY_ZERO

                dropoff_delay = MAX_DELAY_ZERO
                if dropoff_time and dropoff_eta:
                    try:
                        dropoff_delay = max(
                            MAX_DELAY_ZERO,
                            int((dropoff_eta - dropoff_time).total_seconds() // 60),
                        )
                    except Exception:
                        dropoff_delay = MAX_DELAY_ZERO

                # Toujours renvoyer si on a un ETA; le front pourra afficher "À l'heure" (0)
                if pickup_eta or dropoff_eta:
                    max_delay = max(pickup_delay, dropoff_delay)

                    # ✨ NOUVEAUTÉ: Générer des suggestions intelligentes
                    suggestions_list = []
                    try:
                        if (
                            max_delay != MAX_DELAY_ZERO
                        ):  # Générer suggestions si retard ou avance
                            company_id_int = int(cast("Any", company.id))
                            suggestions_list = generate_suggestions(
                                a,
                                delay_minutes=max_delay
                                if pickup_delay > PICKUP_DELAY_ZERO
                                else -abs(max_delay),
                                company_id=company_id_int,
                            )
                            suggestions_list = [s.to_dict() for s in suggestions_list]
                    except Exception as e:
                        logger.warning(
                            "[Delays] Failed to generate suggestions for assignment %s: %s",
                            a.id,
                            e,
                        )

                    delay = {
                        "id": a.id,
                        "booking_id": a.booking_id,
                        "driver_id": a.driver_id,
                        "assignment_id": a.id,
                        "pickup_time": pickup_time,
                        "dropoff_time": dropoff_time,
                        "pickup_eta": pickup_eta,
                        "dropoff_eta": dropoff_eta,
                        "pickup_delay_minutes": pickup_delay,
                        "dropoff_delay_minutes": dropoff_delay,
                        "delay_minutes": max_delay,
                        # infos utiles côté front pour affichage
                        "scheduled_time": getattr(b, "scheduled_time", None),
                        "estimated_arrival": pickup_eta or dropoff_eta,
                        "booking": b,
                        # ✅ P1: Driver déjà chargé via joinedload
                        "driver": a.driver if hasattr(a, "driver") else None,
                        # ✨ Suggestions intelligentes
                        "suggestions": suggestions_list,
                    }
                    delays.append(delay)

            return delays

        except Exception as e:
            logger.exception("Erreur récupération retards: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/delays/live")
class LiveDelaysResource(Resource):
    """Retards en temps réel avec recalcul des ETAs et suggestions intelligentes."""

    @jwt_required()
    @role_required(UserRole.company)  # Note: Admin access handled in code
    @dispatch_ns.doc(
        params={
            "date": "YYYY-MM-DD",
            "company_id": "ID entreprise (optionnel pour ADMIN)",
        }
    )
    def get(self):
        """Retards en temps réel avec recalcul des ETAs et suggestions intelligentes.
        Inclut les retards actuels ET prédits, avec suggestions de réassignation
        et impact sur les courses suivantes.
        ✅ OPTIMISÉ: Parallélise les calculs d'ETA pour améliorer les performances.
        ✅ OPTIMISÉ: Timeout global de 15s pour éviter les timeouts frontend.
        """
        endpoint_start_time = time.time()
        ENDPOINT_TIMEOUT_SECONDS = 15  # ✅ Timeout global réduit à 15s

        # Validation de la date
        date_str = request.args.get("date")
        if not date_str:
            return APIErrorHandler.handle_validation_error(
                "Paramètre 'date' manquant",
                field="date",
                expected_format="YYYY-MM-DD",
                logger_instance=logger,
            )

        try:
            d = _parse_date(date_str)
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                f"Format de date invalide: {e}",
                field="date",
                provided_value=date_str,
                expected_format="YYYY-MM-DD",
                logger_instance=logger,
            )

        try:
            d0, d1 = day_local_bounds(d.strftime("%Y-%m-%d"))

            # ✅ Gérer le cas admin : accès total, company_id optionnel
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, err, code = _get_current_company_via_use_case()

            # Si erreur et c'est un 404 "No company", vérifier si c'est un admin
            if err and code == HTTPStatus.NOT_FOUND:
                user_public_id = get_jwt_identity()
                user = user_repo.find_by_public_id_first(user_public_id)

                if user and user.role == UserRole.admin:
                    # Admin : accès total, company_id optionnel
                    company_id_param = request.args.get("company_id")
                    if company_id_param:
                        try:
                            company_id = int(company_id_param)
                        except (ValueError, TypeError):
                            return APIErrorHandler.handle_validation_error(
                                "company_id doit être un nombre entier",
                                field="company_id",
                                provided_value=company_id_param,
                            )

                        company_obj = company_repo.find_model_by_id(company_id)
                        if not company_obj:
                            return APIErrorHandler.handle_not_found(
                                "Entreprise", company_id, logger
                            )
                        company = company_obj
                    else:
                        # Admin sans company_id : utiliser la première entreprise trouvée
                        company_obj = company_repo.find_first_model()
                        if not company_obj:
                            return {
                                "error": "Aucune entreprise trouvée dans le système"
                            }, HTTPStatus.NOT_FOUND
                        company = company_obj
                else:
                    # Pas admin, retourner l'erreur originale
                    return err, code
            elif err:
                # Autre erreur, la retourner
                return err, code

            # Si on arrive ici, company est défini
            if company is None:
                return APIErrorHandler.handle_not_found("Entreprise", None, logger)

            time_expr = _booking_time_expr()

            # ✅ CRITIQUE: Filtrer les assignations par proximité temporelle
            # (1h avant à 1h après le pickup)
            now = now_local()
            TIME_WINDOW_BEFORE_MINUTES = 60  # Commencer à surveiller 1 heure avant
            TIME_WINDOW_AFTER_MINUTES = 60  # Arrêter de surveiller 1 heure après
            time_window_start = now - timedelta(minutes=TIME_WINDOW_BEFORE_MINUTES)
            time_window_end = now + timedelta(minutes=TIME_WINDOW_AFTER_MINUTES)

            # ✅ Récupérer uniquement les assignations dans la fenêtre temporelle
            excluded_statuses = [
                BookingStatus.COMPLETED,
                BookingStatus.RETURN_COMPLETED,
                BookingStatus.CANCELED,
            ]
            assigns = (
                assignment_repo.find_models_by_company_with_time_expr_and_time_window(
                    company_id=company.id,
                    time_expr=time_expr,
                    day_start=d0,
                    day_end=d1,
                    window_start=time_window_start,
                    window_end=time_window_end,
                    excluded_statuses=excluded_statuses,
                    limit=50,
                )
            )

            logger.info(
                "[LiveDelays] Found %d assignments in time window [%s, %s] for company %s",
                len(assigns),
                time_window_start.isoformat(),
                time_window_end.isoformat(),
                company.id,
            )

            # ✅ Si aucune assignation dans la fenêtre temporelle, retourner rapidement
            if not assigns:
                logger.debug(
                    "[LiveDelays] No assignments in time window, returning empty response"
                )
                return {
                    "delays": [],
                    "summary": {
                        "total": 0,
                        "late": 0,
                        "early": 0,
                        "on_time": 0,
                        "average_delay": 0,
                    },
                    "timestamp": now.isoformat(),
                }, HTTPStatus.OK

            # ✅ ÉTAPE 1: Préparer les données pour le calcul parallèle des ETAs
            assignment_data = []
            for a in assigns:
                b = a.booking
                if not b:
                    continue

                # ✅ Double vérification : skip les courses terminées et IN_PROGRESS
                if b.status in [
                    BookingStatus.COMPLETED,
                    BookingStatus.RETURN_COMPLETED,
                    BookingStatus.CANCELED,
                    BookingStatus.IN_PROGRESS,  # ✅ Client déjà à bord, pas de retard
                ]:
                    continue

                # Récupérer le chauffeur pour position temps réel
                driver = a.driver if hasattr(a, "driver") else None

                # Position actuelle du chauffeur
                if driver:
                    driver_pos = (
                        getattr(
                            driver, "current_lat", getattr(driver, "latitude", 46.2044)
                        ),
                        getattr(
                            driver, "current_lon", getattr(driver, "longitude", 6.1432)
                        ),
                    )
                else:
                    driver_pos = None

                # Position pickup
                pickup_lat = getattr(b, "pickup_lat", None)
                pickup_lon = getattr(b, "pickup_lon", None)
                pickup_pos = (
                    (pickup_lat, pickup_lon) if pickup_lat and pickup_lon else None
                )

                assignment_data.append(
                    {
                        "assignment": a,
                        "booking": b,
                        "driver": driver,
                        "driver_pos": driver_pos,
                        "pickup_pos": pickup_pos,
                    }
                )

            # ✅ ÉTAPE 2: Vérifier le circuit breaker OSRM AVANT de lancer les calculs
            use_haversine_only = False
            try:
                from services.geolocation.osrm import _osrm_circuit_breaker

                if _osrm_circuit_breaker.state == "OPEN":
                    logger.info(
                        "[LiveDelays] OSRM circuit breaker is OPEN, using Haversine only"
                    )
                    use_haversine_only = True
            except Exception as e:
                logger.warning(
                    "[LiveDelays] Could not check OSRM circuit breaker: %s, will try OSRM first",
                    e,
                )

            # ✅ ÉTAPE 3: Calculer tous les ETAs en parallèle avec timeout global
            eta_results = {}  # {assignment_id: eta_seconds}
            assignments_needing_eta = [
                (i, data_item)
                for i, data_item in enumerate(assignment_data)
                if data_item["driver_pos"] and data_item["pickup_pos"]
            ]

            if assignments_needing_eta:
                start_time = time.time()
                # ✅ Timeout global réduit : 3s si Haversine (rapide), 5s si OSRM
                GLOBAL_TIMEOUT_SECONDS = 3 if use_haversine_only else 5

                def _calculate_with_index(index_data):
                    _, data_item = index_data
                    eta_sec = _calculate_eta_for_assignment(
                        data_item["driver_pos"],
                        data_item["pickup_pos"],
                        use_haversine_only=use_haversine_only,
                    )
                    return (data_item["assignment"].id, eta_sec)

                # ✅ Utiliser ThreadPoolExecutor avec max_workers réduit
                max_workers = min(5, len(assignments_needing_eta))  # Max 5 workers
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    try:
                        futures = {
                            executor.submit(_calculate_with_index, item): item[1][
                                "assignment"
                            ].id
                            for item in assignments_needing_eta
                        }
                        completed = 0
                        for future in as_completed(
                            futures, timeout=GLOBAL_TIMEOUT_SECONDS
                        ):
                            try:
                                assignment_id, eta_sec = future.result()
                                if eta_sec is not None:
                                    eta_results[assignment_id] = eta_sec
                                completed += 1
                                # Si on dépasse le timeout global, arrêter d'attendre
                                if time.time() - start_time >= GLOBAL_TIMEOUT_SECONDS:
                                    logger.warning(
                                        "[LiveDelays] Global timeout (%ds) reached, stopping ETA calculations (%d/%d completed)",
                                        GLOBAL_TIMEOUT_SECONDS,
                                        completed,
                                        len(futures),
                                    )
                                    break
                            except Exception as e:
                                assignment_id_key = futures.get(future, "unknown")
                                logger.warning(
                                    "[LiveDelays] ETA calculation failed for assignment %s: %s",
                                    assignment_id_key,
                                    e,
                                )
                    except FutureTimeoutError:
                        logger.warning(
                            "[LiveDelays] Global timeout (%ds) reached for ETA calculations, using partial results (%d/%d completed)",
                            GLOBAL_TIMEOUT_SECONDS,
                            len(eta_results),
                            len(assignments_needing_eta),
                        )
                    except Exception as e:
                        logger.warning(
                            "[LiveDelays] Error in parallel ETA calculation: %s", e
                        )

                elapsed_time = time.time() - start_time
                logger.info(
                    "[LiveDelays] Calculated %d ETAs in parallel in %.2fs (timeout: %ds, total: %d)",
                    len(eta_results),
                    elapsed_time,
                    GLOBAL_TIMEOUT_SECONDS,
                    len(assignments_needing_eta),
                )

            # ✅ ÉTAPE 4: Construire les delays avec les ETAs calculés
            delays = []
            for data_item in assignment_data:
                # ✅ Vérifier le timeout global pour éviter les timeouts frontend
                if time.time() - endpoint_start_time >= ENDPOINT_TIMEOUT_SECONDS:
                    logger.warning(
                        "[LiveDelays] Endpoint timeout (%ds) reached, returning partial results (%d/%d delays)",
                        ENDPOINT_TIMEOUT_SECONDS,
                        len(delays),
                        len(assignment_data),
                    )
                    break
                a = data_item["assignment"]
                b = data_item["booking"]
                driver = data_item["driver"]
                driver_pos = data_item["driver_pos"]
                pickup_pos = data_item["pickup_pos"]

                # Temps prévus
                pickup_time = getattr(b, "pickup_time", None) or getattr(
                    b, "scheduled_time", None
                )
                dropoff_time = getattr(b, "dropoff_time", None)

                # Coerce strings -> datetime
                def _to_dt(v):
                    if v is None:
                        return None
                    if isinstance(v, datetime):
                        return v
                    try:
                        return datetime.fromisoformat(str(v))
                    except Exception:
                        return None

                pickup_time = _to_dt(pickup_time)
                dropoff_time = _to_dt(dropoff_time)

                # ✅ ÉTAPE 4.1: Prendre en compte les courses précédentes du chauffeur
                effective_driver_pos = (
                    driver_pos  # Position à utiliser pour le calcul ETA
                )

                if driver and driver.id and pickup_time:
                    try:
                        prev_b, prev_a = _get_driver_previous_booking(
                            driver.id, b, company.id, d0, d1
                        )
                        if prev_b and prev_a:
                            # Vérifier si la course précédente est en cours ou récente
                            prev_status = getattr(prev_b, "status", None)
                            prev_dropoff_time = getattr(
                                prev_b, "dropoff_time", None
                            ) or getattr(prev_b, "scheduled_time", None)

                            if prev_status in [
                                BookingStatus.ASSIGNED,
                                BookingStatus.EN_ROUTE,
                                BookingStatus.IN_PROGRESS,
                            ] or (
                                prev_dropoff_time
                                and pickup_time
                                and (pickup_time - prev_dropoff_time).total_seconds()
                                < PREVIOUS_BOOKING_RELEVANCE_WINDOW_SECONDS
                            ):
                                # ✅ Utiliser la destination de la course précédente
                                prev_dropoff_lat = getattr(prev_b, "dropoff_lat", None)
                                prev_dropoff_lon = getattr(prev_b, "dropoff_lon", None)

                                if prev_dropoff_lat and prev_dropoff_lon:
                                    effective_driver_pos = (
                                        prev_dropoff_lat,
                                        prev_dropoff_lon,
                                    )
                                    logger.debug(
                                        "[LiveDelays] Using previous booking dropoff as starting point for booking %d (previous: %d, status: %s)",
                                        b.id,
                                        prev_b.id,
                                        prev_status,
                                    )
                    except Exception as e:
                        logger.warning(
                            "[LiveDelays] Error processing previous booking for assignment %s: %s",
                            a.id,
                            e,
                        )

                # ✅ Utiliser l'ETA calculé en parallèle OU recalculer avec position effective
                current_eta = None
                if a.id in eta_results and pickup_time:
                    # ✅ ETA calculé en parallèle disponible
                    eta_seconds = eta_results[a.id]

                    # ✅ Si on a une course précédente pertinente, recalculer depuis sa destination
                    if effective_driver_pos != driver_pos and pickup_pos:
                        try:
                            recalc_eta_seconds = _calculate_eta_for_assignment(
                                effective_driver_pos, pickup_pos
                            )
                            if recalc_eta_seconds:
                                current_eta = now_local() + timedelta(
                                    seconds=recalc_eta_seconds
                                )
                                logger.debug(
                                    "[LiveDelays] Recalculated ETA from previous booking dropoff for assignment %d: %d min",
                                    a.id,
                                    recalc_eta_seconds // 60,
                                )
                            else:
                                # Fallback sur ETA parallèle
                                current_eta = now_local() + timedelta(
                                    seconds=eta_seconds
                                )
                        except Exception as e:
                            logger.warning(
                                "[LiveDelays] Failed to recalculate ETA from previous booking for assignment %s: %s",
                                a.id,
                                e,
                            )
                            # Fallback sur ETA parallèle
                            current_eta = now_local() + timedelta(seconds=eta_seconds)
                    else:
                        # Pas de course précédente pertinente, utiliser ETA parallèle
                        current_eta = now_local() + timedelta(seconds=eta_seconds)
                elif effective_driver_pos and pickup_pos and pickup_time:
                    # Fallback: calculer maintenant avec position effective
                    try:
                        eta_seconds = _calculate_eta_for_assignment(
                            effective_driver_pos, pickup_pos
                        )
                        if eta_seconds:
                            current_eta = now_local() + timedelta(seconds=eta_seconds)
                    except Exception as e:
                        logger.warning(
                            "[LiveDelays] Failed to calculate ETA for assignment %s: %s",
                            a.id,
                            e,
                        )

                # Utiliser ETA planifié en fallback
                if not current_eta:
                    current_eta = (
                        getattr(a, "pickup_eta", None)
                        or getattr(a, "eta_pickup_at", None)
                        or getattr(a, "estimated_pickup_arrival", None)
                    )
                    current_eta = _to_dt(current_eta)

                # ✅ LOGIQUE INTELLIGENTE DE DÉTECTION DE RETARD
                delay_minutes = 0
                status = "unknown"
                booking_status = getattr(b, "status", None)

                if pickup_time and current_eta:
                    try:
                        current_time = now_local()
                        time_remaining_until_pickup = (
                            pickup_time - current_time
                        ).total_seconds() / 60.0  # en minutes

                        # Calculer l'ETA en minutes depuis maintenant
                        eta_from_now_seconds = (
                            current_eta - current_time
                        ).total_seconds()
                        eta_from_now_minutes = eta_from_now_seconds / 60.0

                        if pickup_time > current_time:
                            # ✅ Course dans le futur : logique intelligente
                            if eta_from_now_minutes <= time_remaining_until_pickup:
                                # ✅ Le chauffeur arrivera à temps
                                status = "on_time"
                                delay_minutes = 0
                            else:
                                # ⚠️ Le chauffeur sera en retard
                                potential_delay_minutes = int(
                                    eta_from_now_minutes - time_remaining_until_pickup
                                )

                                is_en_route = booking_status == BookingStatus.EN_ROUTE

                                if is_en_route:
                                    # ✅ Chauffeur en mouvement
                                    delay_minutes = potential_delay_minutes
                                    severity = _classify_delay_severity(delay_minutes)
                                    status = (
                                        severity if severity != "early" else "on_time"
                                    )
                                    logger.debug(
                                        "[LiveDelays] Driver EN_ROUTE but will be %d min late (ETA: %.1f min, remaining: %.1f min)",
                                        delay_minutes,
                                        eta_from_now_minutes,
                                        time_remaining_until_pickup,
                                    )
                                else:
                                    # ❌ Chauffeur pas encore en route
                                    delay_minutes = potential_delay_minutes
                                    severity = _classify_delay_severity(delay_minutes)
                                    status = (
                                        severity if severity != "early" else "on_time"
                                    )
                                    logger.warning(
                                        "[LiveDelays] Driver should be EN_ROUTE but status is %s. Will be %d min late (ETA: %.1f min, remaining: %.1f min)",
                                        booking_status,
                                        delay_minutes,
                                        eta_from_now_minutes,
                                        time_remaining_until_pickup,
                                    )
                        else:
                            # ✅ Course passée ou en cours : calculer le retard normalement
                            delay_seconds = (current_eta - pickup_time).total_seconds()
                            delay_minutes = int(delay_seconds / 60)
                            status = _classify_delay_severity(delay_minutes)
                    except Exception as e:
                        logger.warning(
                            "[LiveDelays] Error calculating intelligent delay: %s", e
                        )
                elif pickup_time and not current_eta:
                    # ⭐ FALLBACK : Si pas d'ETA disponible, comparer heure actuelle vs heure prévue
                    try:
                        current_time = now_local()
                        time_diff_seconds = (current_time - pickup_time).total_seconds()

                        if time_diff_seconds > TIME_DIFF_SECONDS_THRESHOLD:
                            delay_minutes = int(time_diff_seconds / 60)
                            status = _classify_delay_severity(delay_minutes)
                        elif time_diff_seconds < -TIME_DIFF_SECONDS_THRESHOLD:
                            delay_minutes = int(time_diff_seconds / 60)
                            status = "early"
                        else:
                            status = "on_time"
                    except Exception as e:
                        logger.warning(
                            "[LiveDelays] Failed to calculate time-based delay: %s", e
                        )

                # ✅ OPTIMISATION CRITIQUE: Désactiver suggestions et cascade
                # pour améliorer les performances
                suggestions_list = []
                cascade_impact = []

                # Construire la réponse
                if current_eta or delay_minutes != DELAY_MINUTES_ZERO:
                    # ✅ Classifier la sévérité du retard
                    delay_severity = _classify_delay_severity(delay_minutes)

                    delay = {
                        "id": a.id,
                        "booking_id": a.booking_id,
                        "driver_id": a.driver_id,
                        "assignment_id": a.id,
                        "delay_minutes": delay_minutes,
                        "status": status,
                        "delay_severity": delay_severity,
                        "current_eta": current_eta.isoformat() if current_eta else None,
                        "scheduled_time": pickup_time.isoformat()
                        if pickup_time
                        else None,
                        "pickup_time": pickup_time.isoformat() if pickup_time else None,
                        "dropoff_time": dropoff_time.isoformat()
                        if dropoff_time
                        else None,
                        "suggestions": suggestions_list,
                        "impacts_next_bookings": cascade_impact,
                        "booking": {
                            "id": b.id,
                            "reference": getattr(b, "reference", None),
                            "customer_name": getattr(b, "customer_name", None),
                            "pickup_address": getattr(b, "pickup_address", None),
                            "dropoff_address": getattr(b, "dropoff_address", None),
                        },
                        "driver": {
                            "id": driver.id,
                            "name": f"{driver.user.first_name} {driver.user.last_name}"
                            if driver and driver.user
                            else None,
                            "current_position": {
                                "lat": driver_pos[0] if driver_pos else None,
                                "lon": driver_pos[1] if driver_pos else None,
                            }
                            if driver_pos
                            else None,
                        }
                        if driver
                        else None,
                    }
                    delays.append(delay)

            # Statistiques globales
            total = len(delays)
            late = len(
                [
                    d
                    for d in delays
                    if d.get("status") in ["late", "reasonable", "moderate", "critical"]
                    or d.get("delay_severity") in ["reasonable", "moderate", "critical"]
                ]
            )
            early = len([d for d in delays if d["status"] == "early"])
            on_time = len([d for d in delays if d["status"] == "on_time"])

            # ✅ Statistiques détaillées par sévérité
            reasonable_count = len(
                [d for d in delays if d.get("delay_severity") == "reasonable"]
            )
            moderate_count = len(
                [d for d in delays if d.get("delay_severity") == "moderate"]
            )
            critical_count = len(
                [d for d in delays if d.get("delay_severity") == "critical"]
            )

            avg_delay = 0
            if delays:
                delay_values = [d["delay_minutes"] for d in delays]
                avg_delay = sum(delay_values) / len(delay_values) if delay_values else 0

            return {
                "delays": delays,
                "summary": {
                    "total": total,
                    "late": late,
                    "early": early,
                    "on_time": on_time,
                    "average_delay": round(avg_delay, 2),
                    "reasonable": reasonable_count,
                    "moderate": moderate_count,
                    "critical": critical_count,
                },
                "timestamp": now_local().isoformat(),
            }, HTTPStatus.OK

        except Exception as e:
            logger.exception("Erreur récupération retards live: %s", e)
            return APIErrorHandler.handle_exception(e, logger)
