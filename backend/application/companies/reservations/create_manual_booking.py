"""Use-case: création manuelle d'une réservation (aller simple, A/R ou récurrente).

Source de vérité unique pour la création de réservation depuis web et mobile.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from ext import db
from models import Booking
from models.enums import BillingSource, BookingStatus
from shared.constants import ErrorCodes
from shared.time_utils import parse_local_naive

logger = logging.getLogger(__name__)

# Constantes (alignées avec companies.py)
WEEKEND_START_INDEX = 5
SCHEDULED_HOUR_THRESHOLD = 9
PREFERENTIAL_RATE_ZERO = 0
MORNING_RUSH_START = 7
EVENING_RUSH_START = 17
LUNCH_START = 12


class CreateManualBookingError(Exception):
    """Erreur lors de la création manuelle d'une réservation."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        *,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}


@dataclass
class CreateManualBookingResult:
    """Résultat du use-case CreateManualBookingUseCase."""

    created_outbounds: list[Booking]
    created_returns: list[Booking]


def _geocode_with_nominatim(address: str) -> tuple[float, float] | None:
    """Géocode une adresse via Nominatim (gratuit, pas de clé API)."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params: dict[str, str | int] = {
            "q": address,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
        }
        headers = {"User-Agent": "ATMR-Transport/1"}
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        data = resp.json()
        if data and len(data) > 0:
            return (float(data[0]["lat"]), float(data[0]["lon"]))
        return None
    except Exception as e:
        logger.warning("Nominatim geocoding failed for '%s': %s", address, e)
        return None


class CreateManualBookingUseCase:
    """Use-case Application: création manuelle d'une réservation.

    Gère la résolution billing, récurrence, OSRM, pricing et création des Booking.
    Source de vérité unique pour web et mobile.
    """

    def execute(
        self,
        *,
        company_id: int,
        validated_data: dict[str, Any],
        client: Any,
        user: Any,
    ) -> CreateManualBookingResult:
        """Exécute la création des réservations.

        Args:
            company_id: ID de l'entreprise
            validated_data: Données validées par ManualBookingCreateSchema
            client: Client résolu (modèle Client)
            user: Utilisateur du client (modèle User)

        Returns:
            CreateManualBookingResult avec les Booking créés

        Raises:
            CreateManualBookingError: En cas d'erreur métier (validation, config, etc.)
        """
        cid = company_id
        client_id = validated_data["client_id"]

        def _norm_str(x: Any, default: str | None = None) -> str | None:
            if isinstance(x, str):
                return x.strip()
            return default

        # ---------- 0) Résolution du payeur ----------
        from services.billing.client_stay_resolver import (
            find_active_stay_for_client,
            get_clinic_address_for_stay,
        )

        scheduled_dt_for_stay = None
        with suppress(Exception):
            scheduled_dt_for_stay = parse_local_naive(validated_data["scheduled_time"])

        active_stay = None
        if scheduled_dt_for_stay:
            active_stay = find_active_stay_for_client(
                client_id=client_id, reference_date=scheduled_dt_for_stay
            )

        bill_to_patient_override = validated_data.get("bill_to_patient", False)
        billed_to_company_id = None

        if active_stay:
            if bill_to_patient_override:
                billed_to_type = "patient"
                billed_to_company_id = None
            else:
                clinic_info = get_clinic_address_for_stay(active_stay)
                if clinic_info and clinic_info.get("clinic_id"):
                    billed_to_type = "clinic"
                    billed_to_company_id = clinic_info.get("clinic_id")
                else:
                    _bt_raw = _norm_str(
                        getattr(client, "default_billed_to_type", "patient"), "patient"
                    )
                    billed_to_type = (_bt_raw or "patient").lower()
        else:
            _bt_raw = _norm_str(
                getattr(client, "default_billed_to_type", "patient"), "patient"
            )
            billed_to_type = _bt_raw or "patient"
            billed_to_type = billed_to_type.lower()

        if not billed_to_company_id:
            billed_to_company_id = validated_data.get("billed_to_company_id") or getattr(
                client, "default_billed_to_company_id", None
            )
        billed_to_contact = _norm_str(
            validated_data.get("billed_to_contact")
            or getattr(client, "default_billed_to_contact", None),
            None,
        )

        if billed_to_type not in ("patient", "clinic", "insurance"):
            raise CreateManualBookingError(
                "billed_to_type invalide (valeurs possibles: patient | clinic | insurance)",
                status_code=400,
            )

        if billed_to_type in ("clinic", "insurance"):
            if billed_to_company_id in (None, ""):
                raise CreateManualBookingError(
                    "billed_to_company_id est requis quand billed_to_type != 'patient'.",
                    status_code=400,
                )
            try:
                billed_to_company_id = int(billed_to_company_id)
            except (TypeError, ValueError) as e:
                raise CreateManualBookingError(
                    "billed_to_company_id doit être un entier.",
                    status_code=400,
                ) from e
            from repositories.company_repository import CompanyRepository

            company_repo = CompanyRepository()
            payer = company_repo.find_model_by_id(company_id=billed_to_company_id)
            if not payer:
                raise CreateManualBookingError(
                    f"Société payeuse {billed_to_company_id} introuvable",
                    status_code=404,
                )

        # ---------- 1) Parse des dates + Récurrence ----------
        try:
            scheduled = parse_local_naive(validated_data["scheduled_time"])
        except Exception as e:
            raise CreateManualBookingError(
                f"scheduled_time invalide: {e}",
                status_code=400,
            ) from e

        is_rt = bool(validated_data.get("is_round_trip", False))
        return_dt = None
        return_time_confirmed = True
        return_date_str = validated_data.get("return_date")
        return_time_str = validated_data.get("return_time")

        if is_rt and return_date_str:
            try:
                if return_time_str:
                    time_only = return_time_str
                    if "T" in return_time_str:
                        time_parts = return_time_str.split("T")
                        if len(time_parts) > 1:
                            time_only = time_parts[-1].split(":")[:2]
                            time_only = ":".join(time_only)
                    combined = f"{return_date_str}T{time_only}"
                    _HOUR_MIN_PARTS = 2
                    if len(combined.split("T")[1].split(":")) == _HOUR_MIN_PARTS:
                        combined = f"{combined}:00"
                    return_dt = parse_local_naive(combined)
                    return_time_confirmed = True
                else:
                    combined = f"{return_date_str}T00:00:00"
                    return_dt = parse_local_naive(combined)
                    return_time_confirmed = False
            except Exception as e:
                raise CreateManualBookingError(
                    f"return_date/return_time invalide: {e}",
                    status_code=400,
                ) from e

        is_recurring = bool(validated_data.get("is_recurring", False))
        recurrence_dates = [scheduled]

        if is_recurring:
            recurrence_type = validated_data.get("recurrence_type", "weekly")
            occurrences = int(validated_data.get("occurrences", 1))
            recurrence_days = validated_data.get("recurrence_days", [])
            recurrence_end_date_str = validated_data.get("recurrence_end_date")
            base_date = scheduled

            if recurrence_type == "daily" and base_date:
                for i in range(1, occurrences):
                    next_date = base_date + timedelta(days=i)
                    if recurrence_end_date_str:
                        try:
                            end_date = parse_local_naive(recurrence_end_date_str)
                            if end_date and next_date > end_date:
                                break
                        except Exception:
                            pass
                    recurrence_dates.append(next_date)
            elif recurrence_type == "weekly" and base_date:
                for i in range(1, occurrences):
                    next_date = base_date + timedelta(weeks=i)
                    if recurrence_end_date_str:
                        try:
                            end_date = parse_local_naive(recurrence_end_date_str)
                            if end_date and next_date > end_date:
                                break
                        except Exception:
                            pass
                    recurrence_dates.append(next_date)
            elif recurrence_type == "custom" and recurrence_days and base_date:
                for target_weekday in recurrence_days:
                    current_date = base_date
                    count = 0
                    max_iterations = occurrences * 10
                    iteration = 0
                    while count < occurrences and iteration < max_iterations:
                        iteration += 1
                        if (
                            current_date
                            and current_date.weekday() == target_weekday
                        ):
                            if recurrence_end_date_str:
                                try:
                                    end_date = parse_local_naive(
                                        recurrence_end_date_str
                                    )
                                    if end_date and current_date > end_date:
                                        break
                                except Exception:
                                    pass
                            if (
                                current_date != base_date
                                or (
                                    base_date
                                    and target_weekday == base_date.weekday()
                                )
                            ):
                                if current_date not in recurrence_dates:
                                    recurrence_dates.append(current_date)
                                count += 1
                        if current_date:
                            current_date += timedelta(days=1)

            recurrence_dates = [d for d in recurrence_dates if d is not None]
            recurrence_dates.sort()

        # ---------- 2) Estimation distance/durée avec OSRM ----------
        dur_s, dist_m = None, None
        final_pickup_coords = None
        final_dropoff_coords = None

        pickup_coords = None
        dropoff_coords = None
        need_pickup_geo = not validated_data.get("pickup_lat") or not validated_data.get(
            "pickup_lon"
        )
        need_dropoff_geo = not validated_data.get("dropoff_lat") or not validated_data.get(
            "dropoff_lon"
        )
        # Nominatim en parallèle (sinon jusqu'à ~10 s séquentiel pour aller + retour sans coords).
        if need_pickup_geo or need_dropoff_geo:
            tasks: list[tuple[str, str]] = []
            if need_pickup_geo:
                tasks.append(("pickup", validated_data["pickup_location"]))
            if need_dropoff_geo:
                tasks.append(("dropoff", validated_data["dropoff_location"]))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    name: pool.submit(_geocode_with_nominatim, addr)
                    for name, addr in tasks
                }
                for name, fut in futures.items():
                    try:
                        res = fut.result(timeout=6)
                    except Exception as e:
                        logger.warning("Geocode %s failed: %s", name, e)
                        res = None
                    if name == "pickup":
                        pickup_coords = res
                    else:
                        dropoff_coords = res

        if validated_data.get("pickup_lat") and validated_data.get("pickup_lon"):
            final_pickup_coords = (
                float(validated_data["pickup_lat"]),
                float(validated_data["pickup_lon"]),
            )
        elif pickup_coords:
            final_pickup_coords = pickup_coords

        if validated_data.get("dropoff_lat") and validated_data.get("dropoff_lon"):
            final_dropoff_coords = (
                float(validated_data["dropoff_lat"]),
                float(validated_data["dropoff_lon"]),
            )
        elif dropoff_coords:
            final_dropoff_coords = dropoff_coords

        if final_pickup_coords and final_dropoff_coords:
            from config import Config

            osrm_url = getattr(Config, "UD_OSRM_URL", "http://osrm:5000")
            try:
                from services.geolocation.osrm import _route

                route_data = _route(
                    base_url=osrm_url,
                    profile="driving",
                    origin=final_pickup_coords,
                    destination=final_dropoff_coords,
                    timeout=2,
                    overview="false",
                    geometries="geojson",
                    steps=False,
                    annotations=False,
                )
                if route_data.get("code") == "Ok" and route_data.get("routes"):
                    r0 = route_data["routes"][0]
                    base_dur_s = int(r0.get("duration", 0))
                    dist_m = int(r0.get("distance", 0))
                else:
                    raise ValueError(
                        "OSRM bad response: "
                        + f"{route_data.get('message', 'Unknown error')}"
                    )
            except Exception as osrm_error:
                base_dur_s = None
                dist_m = None
                logger.warning("OSRM timeout/error: %s", osrm_error)

            if base_dur_s is not None:
                scheduled_hour = (
                    scheduled.hour if scheduled else datetime.now(UTC).hour
                )
                rush_hour_factor = 1
                if (
                    MORNING_RUSH_START <= scheduled_hour < SCHEDULED_HOUR_THRESHOLD
                    or EVENING_RUSH_START <= scheduled_hour < SCHEDULED_HOUR_THRESHOLD
                ):
                    rush_hour_factor = 1.3
                elif LUNCH_START <= scheduled_hour < SCHEDULED_HOUR_THRESHOLD:
                    rush_hour_factor = 1.15
                dur_s = int(base_dur_s * rush_hour_factor)

        # ---------- 3) Montant et création des réservations ----------
        full_name = (
            f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}"
        ).strip()
        if bool(client.is_institution) and client.institution_name:
            display_name = client.institution_name
        else:
            display_name = full_name or (getattr(user, "username", "") or "Client")

        mission_type = (
            (validated_data.get("mission_type") or "patient_transport")
            .strip()
            .lower()
        )
        raw_desc = (validated_data.get("delivery_description") or "").strip()
        delivery_description = (
            " ".join(raw_desc.split()) if raw_desc else None
        )

        if mission_type == "material_delivery":
            from models import CompanyBillingSettings

            billing_settings = CompanyBillingSettings.query.filter_by(
                company_id=cid
            ).first()
            price_fixed = (
                billing_settings.material_delivery_price_fixed
                if billing_settings
                else None
            )
            if (
                not billing_settings
                or price_fixed is None
                or float(price_fixed) <= 0
            ):
                raise CreateManualBookingError(
                    "Configurez le prix fixe livraison dans Paramètres > Facturation avant de créer une livraison.",
                    status_code=400,
                    error_code=ErrorCodes.MATERIAL_DELIVERY_PRICE_NOT_CONFIGURED,
                    details={
                        "field": "material_delivery_price_fixed",
                        "booking_id": None,
                    },
                )
            amount_to_use = float(price_fixed)
            amount_source_used = "material_delivery_fixed"
            pricing_profile_id = None
            pricing_profile_version_id = None
            price_amount = None
            price_breakdown_json = None
        else:
            from models import PricingProfile
            from services.geo.geo_resolver import resolve_pickup_admin
            from services.pricing.pricing_engine import compute_price

            amount_source_requested = (
                str(validated_data.get("amount_source") or "").strip().lower()
            )
            provided_amount = validated_data.get("amount")
            has_provided_amount = (
                provided_amount is not None
                and float(provided_amount) > PREFERENTIAL_RATE_ZERO
            )
            amount_to_use = float(provided_amount or 0.0)
            pricing_profile_id = None
            pricing_profile_version_id = None
            price_amount = None
            price_breakdown_json = None

            clinic_preferential = None
            if active_stay and not bill_to_patient_override:
                clinic_info = get_clinic_address_for_stay(active_stay)
                clinic_preferential = (
                    clinic_info.get("preferential_rate") if clinic_info else None
                )
            client_preferential = client.preferential_rate

            if (
                clinic_preferential
                and float(clinic_preferential) > PREFERENTIAL_RATE_ZERO
            ):
                amount_to_use = float(clinic_preferential)
                amount_source_used = "preferential"
                price_breakdown_json = {
                    "overridden_by_preferential": True,
                    "preferential_source": "clinic",
                    "preferential_amount": f"{amount_to_use:.2f}",
                }
            elif (
                client_preferential
                and float(client_preferential) > PREFERENTIAL_RATE_ZERO
            ):
                amount_to_use = float(client_preferential)
                amount_source_used = "preferential"
                price_breakdown_json = {
                    "overridden_by_preferential": True,
                    "preferential_source": "client",
                    "preferential_amount": f"{amount_to_use:.2f}",
                }
            elif amount_source_requested == "manual" and has_provided_amount:
                amount_source_used = "manual"
            else:
                profile = (
                    PricingProfile.query.filter_by(
                        company_id=cid, is_active=True
                    )
                    .order_by(PricingProfile.created_at.desc())
                    .first()
                )
                version = None
                if profile:
                    pricing_profile_id = profile.id
                    version = profile.current_version
                    if not version and profile.versions:
                        version = sorted(
                            profile.versions,
                            key=lambda item: int(item.version),
                            reverse=True,
                        )[0]
                if version:
                    pricing_profile_version_id = version.id
                    scheduled_ref = scheduled or datetime.now()
                    pickup_zip_match = re.search(
                        r"\b(\d{4})\b",
                        validated_data.get("pickup_location") or "",
                    )
                    dropoff_zip_match = re.search(
                        r"\b(\d{4})\b",
                        validated_data.get("dropoff_location") or "",
                    )
                    pickup_admin = resolve_pickup_admin(
                        lat=final_pickup_coords[0] if final_pickup_coords else None,
                        lng=final_pickup_coords[1] if final_pickup_coords else None,
                        pickup_zip=(
                            pickup_zip_match.group(1) if pickup_zip_match else None
                        ),
                        pickup_text=validated_data.get("pickup_location"),
                    )
                    dropoff_admin = resolve_pickup_admin(
                        lat=final_dropoff_coords[0] if final_dropoff_coords else None,
                        lng=final_dropoff_coords[1] if final_dropoff_coords else None,
                        pickup_zip=(
                            dropoff_zip_match.group(1) if dropoff_zip_match else None
                        ),
                        pickup_text=validated_data.get("dropoff_location"),
                    )
                    now_ref = (
                        datetime.now(scheduled_ref.tzinfo)
                        if scheduled_ref.tzinfo
                        else datetime.now()
                    )
                    minutes_until = max(
                        0,
                        int((scheduled_ref - now_ref).total_seconds() // 60),
                    )
                    context = {
                        "is_weekend": scheduled_ref.weekday() >= WEEKEND_START_INDEX,
                        "is_round_trip": bool(is_rt),
                        "pickup_local_time": scheduled_ref.strftime("%H:%M"),
                        "minutes_until_pickup": minutes_until,
                        "distance_km": max(float(dist_m or 0) / 1000.0, 0.0),
                        "pickup_admin_token": pickup_admin.get("token"),
                        "dropoff_admin_token": dropoff_admin.get("token"),
                        "pickup_lat": (
                            final_pickup_coords[0] if final_pickup_coords else None
                        ),
                        "pickup_lng": (
                            final_pickup_coords[1] if final_pickup_coords else None
                        ),
                        "dropoff_lat": (
                            final_dropoff_coords[0] if final_dropoff_coords else None
                        ),
                        "dropoff_lng": (
                            final_dropoff_coords[1] if final_dropoff_coords else None
                        ),
                        "zones_count": (
                            1
                            if pickup_admin.get("token") == dropoff_admin.get("token")
                            else 2
                        ),
                        "requires_waiting": bool(
                            validated_data.get("requires_waiting")
                        ),
                    }
                    try:
                        computed_amount, computed_breakdown = compute_price(
                            validated_data, version, context
                        )
                        amount_to_use = float(computed_amount)
                        price_amount = amount_to_use
                        price_breakdown_json = dict(computed_breakdown or {})
                        amount_source_used = "simulated"
                    except Exception:
                        logger.exception(
                            "Pricing compute failed during manual booking create"
                        )
                        amount_source_used = (
                            "manual" if has_provided_amount else "fallback_zero"
                        )
                else:
                    amount_source_used = (
                        "manual" if has_provided_amount else "fallback_zero"
                    )

        created_outbounds: list[Booking] = []
        created_returns: list[Booking] = []

        for occurrence_date in recurrence_dates:
            occurrence_return_dt = None
            if is_rt:
                if return_dt and scheduled and occurrence_date:
                    time_diff = return_dt - scheduled
                    if time_diff.total_seconds() < 0:
                        occurrence_return_dt = occurrence_date.replace(
                            hour=return_dt.hour,
                            minute=return_dt.minute,
                            second=return_dt.second,
                        )
                    else:
                        occurrence_return_dt = occurrence_date + time_diff
                else:
                    occurrence_return_dt = None

            outbound = Booking()
            outbound.customer_name = display_name
            outbound.client_id = client.id
            outbound.scheduled_time = occurrence_date
            outbound.is_round_trip = is_rt
            outbound.pickup_location = validated_data["pickup_location"]
            outbound.dropoff_location = validated_data["dropoff_location"]
            outbound.amount = amount_to_use
            outbound.status = BookingStatus.ACCEPTED
            outbound.company_id = cid
            outbound.booking_type = "manual"
            outbound.user_id = user.id if user else None
            outbound.is_return = False
            outbound.duration_seconds = dur_s
            outbound.distance_meters = dist_m
            outbound.pricing_profile_id = pricing_profile_id
            outbound.pricing_profile_version_id = pricing_profile_version_id
            outbound.price_amount = (
                price_amount if price_amount is not None else amount_to_use
            )
            outbound_breakdown = dict(price_breakdown_json or {})
            outbound_breakdown["amount_source"] = amount_source_used
            outbound.price_breakdown_json = outbound_breakdown

            outbound.pickup_lat = (
                final_pickup_coords[0] if final_pickup_coords else None
            )
            outbound.pickup_lon = (
                final_pickup_coords[1] if final_pickup_coords else None
            )
            outbound.dropoff_lat = (
                final_dropoff_coords[0] if final_dropoff_coords else None
            )
            outbound.dropoff_lon = (
                final_dropoff_coords[1] if final_dropoff_coords else None
            )

            outbound.billed_to_type = billed_to_type
            outbound.billed_to_company_id = billed_to_company_id
            outbound.billed_to_contact = billed_to_contact

            if active_stay:
                if bill_to_patient_override:
                    outbound.billing_source = BillingSource.MANUAL_OVERRIDE
                    outbound.billing_source_ref = "manual_booking_patient_override"
                else:
                    outbound.billing_source = BillingSource.CLIENT_STAY
                    outbound.billing_source_ref = (
                        f"stay#{active_stay.id}" if active_stay else None
                    )
            else:
                outbound.billing_source = BillingSource.DEFAULT_CLIENT
                outbound.billing_source_ref = None

            outbound.medical_facility = validated_data.get("medical_facility")
            outbound.doctor_name = validated_data.get("doctor_name")
            outbound.hospital_service = validated_data.get("hospital_service")
            outbound.notes_medical = validated_data.get("notes_medical")
            outbound.pickup_access_notes = validated_data.get("pickup_access_notes")
            outbound.dropoff_access_notes = validated_data.get(
                "dropoff_access_notes"
            )
            outbound.wheelchair_client_has = validated_data.get(
                "wheelchair_client_has", False
            )
            outbound.wheelchair_need = validated_data.get("wheelchair_need", False)
            outbound.mission_type = mission_type
            outbound.delivery_description = delivery_description

            db.session.add(outbound)
            db.session.flush()
            created_outbounds.append(outbound)

            if is_rt:
                return_booking = Booking()
                return_booking.parent_booking_id = outbound.id
                return_booking.is_return = True
                # Avant scheduled_time : le validateur scheduled_time exige is_return=True si heure absente.
                return_booking.time_confirmed = (
                    bool(return_time_confirmed) if occurrence_return_dt is not None else False
                )
                return_booking.customer_name = outbound.customer_name
                return_booking.client_id = client.id
                return_booking.scheduled_time = occurrence_return_dt
                return_booking.status = BookingStatus.ACCEPTED
                return_booking.pickup_location = outbound.dropoff_location
                return_booking.dropoff_location = outbound.pickup_location
                return_booking.amount = amount_to_use
                return_booking.company_id = cid
                return_booking.booking_type = "manual"
                return_booking.mission_type = mission_type
                return_booking.delivery_description = delivery_description
                return_booking.user_id = user.id if user else None
                return_booking.is_round_trip = False
                return_booking.duration_seconds = dur_s
                return_booking.distance_meters = dist_m
                return_booking.pricing_profile_id = pricing_profile_id
                return_booking.pricing_profile_version_id = pricing_profile_version_id
                return_booking.price_amount = (
                    price_amount if price_amount is not None else amount_to_use
                )
                return_breakdown = dict(price_breakdown_json or {})
                return_breakdown["amount_source"] = amount_source_used
                return_booking.price_breakdown_json = return_breakdown

                return_booking.pickup_lat = outbound.dropoff_lat
                return_booking.pickup_lon = outbound.dropoff_lon
                return_booking.dropoff_lat = outbound.pickup_lat
                return_booking.dropoff_lon = outbound.pickup_lon

                return_booking.billed_to_type = billed_to_type
                return_booking.billed_to_company_id = billed_to_company_id
                return_booking.billed_to_contact = billed_to_contact
                return_booking.billing_source = outbound.billing_source
                return_booking.billing_source_ref = outbound.billing_source_ref

                return_booking.medical_facility = validated_data.get(
                    "medical_facility"
                )
                return_booking.doctor_name = validated_data.get("doctor_name")
                return_booking.hospital_service = validated_data.get(
                    "hospital_service"
                )
                return_booking.notes_medical = validated_data.get("notes_medical")
                return_booking.pickup_access_notes = validated_data.get(
                    "pickup_access_notes"
                )
                return_booking.dropoff_access_notes = validated_data.get(
                    "dropoff_access_notes"
                )
                return_booking.wheelchair_client_has = validated_data.get(
                    "wheelchair_client_has", False
                )
                return_booking.wheelchair_need = validated_data.get(
                    "wheelchair_need", False
                )

                db.session.add(return_booking)
                created_returns.append(return_booking)

        db.session.commit()

        return CreateManualBookingResult(
            created_outbounds=created_outbounds,
            created_returns=created_returns,
        )
