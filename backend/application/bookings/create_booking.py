"""Use-case: création de réservation (booking).

Migration progressive vers Clean Architecture:
- BookingService devient une façade (compat routes)
- La logique métier est portée par ce module Application
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from contextlib import nullcontext, suppress
from datetime import datetime, timezone
from typing import Any, Protocol, cast

from flask import has_app_context

from application.events.event_bus import publish_event
from domain.bookings.commands import CreateBookingCommand
from domain.client_dto import ClientDTO
from domain.events.events import BookingCreatedEvent
from ext import db
from models import PricingProfile
from schemas.booking_schemas import BookingCreateSchema
from services.geo.geo_resolver import (
    geo_unit_id_from_pickup_admin_token,
    resolve_pickup_admin,
)
from services.pricing.pricing_engine import compute_price
from shared.booking_company_resolution import (
    resolve_booking_owner_company_id_for_create,
)
from shared.client_portal_notes import compose_client_portal_notes_medical
from shared.geo_utils import GeoValidator
from shared.time_utils import api_scheduled_iso_to_naive_geneva

logger = logging.getLogger(__name__)
WEEKEND_START_INDEX = 5
BOOKING_FREEZE_FLAG = "FF_BOOKING_PRICE_FREEZE_ON_CREATE"


def _is_valid_positive_amount(value: Any) -> bool:
    try:
        return float(value) > 0
    except Exception:
        return False


def _to_positive_amount(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


class ClientRepoPort(Protocol):
    def find_by_id(self, client_id: int) -> ClientDTO | None: ...


class CompanyLookupPort(Protocol):
    def find_model_by_id(self, company_id: int) -> Any | None: ...


class GeocodingPort(Protocol):
    def geocode_address(
        self, address: str, *, country: str | None = None, language: str = "fr"
    ) -> dict[str, float] | None: ...


class BookingLike(Protocol):
    id: int
    company_id: int | None


class BookingWriterPort(Protocol):
    def create_and_commit(
        self,
        *,
        user_id: int,
        client_id: int,
        company_id: int | None,
        customer_name: str,
        pickup_location: str,
        dropoff_location: str,
        scheduled_time: Any,
        amount: float,
        medical_facility: str,
        doctor_name: str,
        hospital_service: str,
        duration_seconds: int,
        distance_meters: int,
        pickup_lat: float,
        pickup_lon: float,
        dropoff_lat: float,
        dropoff_lon: float,
        is_round_trip: bool,
        pickup_admin_token: str | None,
        pickup_canton_code: str | None,
        pickup_admin_source: str | None,
        pickup_admin_confidence: str | None,
        pickup_admin_label: str | None,
        pickup_admin_resolved_at: datetime | None,
        dropoff_admin_token: str | None,
        dropoff_canton_code: str | None,
        dropoff_admin_source: str | None,
        dropoff_admin_confidence: str | None,
        dropoff_admin_label: str | None,
        dropoff_admin_resolved_at: datetime | None,
        pickup_geo_unit_id: int | None,
        dropoff_geo_unit_id: int | None,
        pricing_profile_id: int | None,
        pricing_profile_version_id: int | None,
        price_amount: float | None,
        price_breakdown_json: dict[str, Any] | None,
        notes_medical: str | None = None,
        return_scheduled_time: Any | None = None,
        return_time_exact: bool = False,
    ) -> BookingLike: ...


class CreateBookingUseCase:
    """Use-case Application: créer une réservation.

    Ce use-case centralise la logique métier de création (validation, calcul
    distance/durée, géocodage optimiste + tâche async, persistance).

    Notes:
        - Publie `BookingCreatedEvent` après commit.
        - Retourne un modèle SQLAlchemy `Booking` (migration progressive vers DTO).

    Exemple:
        >>> uc = CreateBookingUseCase(
        ...     client_repo=ClientRepository(),
        ...     company_repo=CompanyRepository(),
        ...     geocoding_service=get_geocoding_service(),  # doctest: +SKIP
        ...     distance_duration_fn=get_distance_duration,  # doctest: +SKIP
        ... )
        >>> booking = uc.execute(
        ...     CreateBookingCommand(user_id=1, client_id=2, data={...})
        ... )  # doctest: +SKIP
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        *,
        client_repo: ClientRepoPort,
        company_lookup: CompanyLookupPort | None,
        booking_writer: BookingWriterPort,
        geocoding_service: GeocodingPort,
        distance_duration_fn: Callable[[str, str], tuple[int, int]],
        fallback_coords_fn: Callable[[Any | None], tuple[float, float]] | None = None,
        trigger_async_geocoding_fn: Callable[[int, str, str], None] | None = None,
    ) -> None:
        """Initialise le use-case.

        Args:
            client_repo: Repository client.
            company_repo: Repository company.
            geocoding_service: Service de géocodage (interface).
            distance_duration_fn: Fonction (pickup_address, dropoff_address)
                -> (duration_s, distance_m).
            fallback_coords_fn: Fonction fallback (company -> (lat, lon))
                utilisée quand le géocodage manque.
            trigger_async_geocoding_fn: Hook optionnel pour déclencher
                un géocodage async (tests).
        """
        self.client_repo = client_repo
        self.company_lookup = company_lookup
        self.booking_writer = booking_writer
        self.geocoding_service = geocoding_service
        self.distance_duration_fn = distance_duration_fn
        self.fallback_coords_fn = fallback_coords_fn
        self.trigger_async_geocoding_fn = trigger_async_geocoding_fn

    def execute(self, cmd: CreateBookingCommand) -> BookingLike:
        """Exécute la création d'une réservation.

        Args:
            cmd: Commande de création.

        Returns:
            Booking créé (type concret côté Infrastructure).

        Raises:
            ValueError: données invalides / client introuvable.
            RuntimeError: erreurs de géocodage ou calcul distance/durée.
        """
        # Validation des données (défense en profondeur)
        validated_data = cast(dict[str, Any], BookingCreateSchema().load(cmd.data))

        # Parser la date
        try:
            scheduled_time = api_scheduled_iso_to_naive_geneva(
                validated_data["scheduled_time"]
            )
        except Exception as date_error:
            logger.error("Erreur de conversion scheduled_time: %s", date_error)
            raise ValueError("Invalid scheduled_time format") from date_error
        if scheduled_time is None:
            raise ValueError("Invalid scheduled_time format")

        return_scheduled_time = None
        return_time_exact = False
        rt_raw = validated_data.get("return_time")
        rd_raw = validated_data.get("return_date")
        if rt_raw:
            try:
                return_scheduled_time = api_scheduled_iso_to_naive_geneva(rt_raw)
            except Exception as date_error:
                logger.error("Erreur de conversion return_time: %s", date_error)
                raise ValueError("Invalid return_time format") from date_error
            return_time_exact = True
        elif bool(validated_data.get("is_round_trip")) and rd_raw:
            rd_str = str(rd_raw).strip() if isinstance(rd_raw, str) else ""
            if rd_str:
                return_scheduled_time = None
                return_time_exact = False

        notes_medical = compose_client_portal_notes_medical(validated_data)

        client_dto = self.client_repo.find_by_id(cmd.client_id)
        if not client_dto:
            raise ValueError("Client non trouvé")
        company_id = resolve_booking_owner_company_id_for_create(client_dto)

        # ✅ Détecter si le client est hospitalisé et utiliser l'adresse de la clinique
        from services.billing.client_stay_resolver import (
            find_active_stay_for_client,
            get_clinic_address_for_stay,
        )

        active_stay = find_active_stay_for_client(
            client_id=cmd.client_id, reference_date=scheduled_time
        )
        manual_amount = validated_data.get("amount")
        amount_source = str(cmd.data.get("amount_source") or "").strip().lower()
        prefer_clinic = None
        prefer_client = getattr(client_dto, "preferential_rate", None)

        if active_stay:
            clinic_info = get_clinic_address_for_stay(active_stay)
            if clinic_info and clinic_info.get("address"):
                # Remplacer l'adresse de départ par celle de la clinique
                validated_data["pickup_location"] = clinic_info["address"]
                logger.info(
                    "🏥 Client hospitalisé: utilisation adresse clinique %s (id=%s) pour départ",
                    clinic_info.get("clinic_name"),
                    clinic_info.get("clinic_id"),
                )

                prefer_clinic = clinic_info.get("preferential_rate")

        if cmd.data.get("bill_to_patient"):
            prefer_clinic = None

        preferential_amount = None
        preferential_source = None
        clinic_amount = _to_positive_amount(prefer_clinic)
        client_amount = _to_positive_amount(prefer_client)
        manual_amount_value = _to_positive_amount(manual_amount)
        if clinic_amount is not None:
            preferential_amount = clinic_amount
            preferential_source = "clinic"
        elif client_amount is not None:
            preferential_amount = client_amount
            preferential_source = "client"

        # Calcul distance/duration
        try:
            duration_seconds, distance_meters = self.distance_duration_fn(
                validated_data["pickup_location"],
                validated_data["dropoff_location"],
            )
        except OSError as e:
            msg = (
                "❌ Erreur configuration géocodage pour booking (pickup=%s, "
                "dropoff=%s): %s"
            )
            logger.error(
                msg,
                validated_data["pickup_location"],
                validated_data["dropoff_location"],
                e,
            )
            raise RuntimeError(
                "Impossible de calculer la distance entre les adresses. "
                + "Vérifiez que les adresses sont valides et que le service de "
                + "géocodage est configuré."
            ) from e
        except RuntimeError as e:
            error_msg = str(e)
            logger.warning(
                "⚠️ Géocodage échoué pour booking (pickup=%s, dropoff=%s): %s",
                validated_data["pickup_location"],
                validated_data["dropoff_location"],
                error_msg,
            )
            if "ZERO_RESULTS" in error_msg or "NOT_FOUND" in error_msg:
                user_message = (
                    "Une ou plusieurs adresses n'ont pas pu être trouvées. "
                    "Vérifiez que les adresses sont complètes et valides."
                )
            elif "OVER_QUERY_LIMIT" in error_msg:
                user_message = (
                    "Service de géocodage temporairement indisponible. "
                    "Veuillez réessayer dans quelques instants."
                )
            else:
                user_message = (
                    "Impossible de calculer la distance entre les adresses. "
                    "Vérifiez que les adresses sont valides."
                )
            raise RuntimeError(user_message) from e
        except Exception as e:
            logger.exception(
                "❌ Erreur inattendue lors du géocodage (pickup=%s, dropoff=%s)",
                validated_data["pickup_location"],
                validated_data["dropoff_location"],
            )
            raise RuntimeError(
                "Erreur lors du calcul de la distance. Le service de géocodage "
                + "est temporairement indisponible. Veuillez réessayer dans "
                + "quelques instants."
            ) from e

        pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, geocode_miss = (
            self._geocode_booking_addresses(validated_data, company_id)
        )
        pickup_zip = self._extract_pickup_zip(validated_data.get("pickup_location"))
        pickup_admin = resolve_pickup_admin(
            lat=pickup_lat,
            lng=pickup_lon,
            pickup_zip=pickup_zip,
            pickup_text=validated_data.get("pickup_location"),
        )
        dropoff_zip = self._extract_dropoff_zip(validated_data.get("dropoff_location"))
        dropoff_admin = resolve_pickup_admin(
            lat=dropoff_lat,
            lng=dropoff_lon,
            pickup_zip=dropoff_zip,
            pickup_text=validated_data.get("dropoff_location"),
        )

        pickup_geo_unit_id = geo_unit_id_from_pickup_admin_token(
            str(pickup_admin.get("token") or "")
        )
        dropoff_geo_unit_id = geo_unit_id_from_pickup_admin_token(
            str(dropoff_admin.get("token") or "")
        )
        (
            pricing_profile_id,
            pricing_profile_version_id,
            price_amount,
            price_breakdown_json,
        ) = self._compute_pricing_freeze(
            company_id=company_id,
            booking_payload=validated_data,
            pickup_admin_token=pickup_admin.get("token"),
            dropoff_admin_token=dropoff_admin.get("token"),
            pickup_geo_unit_id=pickup_geo_unit_id,
            dropoff_geo_unit_id=dropoff_geo_unit_id,
            distance_meters=distance_meters,
            scheduled_time=scheduled_time,
            is_round_trip=bool(validated_data.get("is_round_trip", False)),
            pickup_lat=pickup_lat,
            pickup_lon=pickup_lon,
            dropoff_lat=dropoff_lat,
            dropoff_lon=dropoff_lon,
        )
        # Paiement client = montant reçu, sauf tarifs préf. / "manual" (plus bas) :
        # - PORTAL : company_id is None → pas de gel côté entreprise
        #   (le sous-traitant n'est pas encore connu) ; l'`amount` du body reprend
        #   la prévisualisation calculée côté plateforme
        #   (route preview + profil de référence PORTAL_CLIENT_PREVIEW_COMPANY_ID),
        #   cohérent avec Saferpay (booking.amount).
        # - TRANSPORT : `price_amount` peut remplacer le body si le moteur de
        #   price freeze renvoie un montant pour l'entreprise rattachée.

        if preferential_amount is not None:
            validated_data["amount"] = preferential_amount
            if isinstance(price_breakdown_json, dict):
                price_breakdown_json["overridden_by_preferential"] = True
                price_breakdown_json["preferential_source"] = preferential_source
                price_breakdown_json["preferential_amount"] = (
                    f"{preferential_amount:.2f}"
                )
            logger.info(
                "💰 Tarif préférentiel %s appliqué: %.2f CHF",
                preferential_source,
                preferential_amount,
            )
        elif amount_source == "manual" and manual_amount_value is not None:
            validated_data["amount"] = manual_amount_value
            if isinstance(price_breakdown_json, dict):
                price_breakdown_json["manual_amount_applied"] = True
        elif price_amount is not None and _is_valid_positive_amount(price_amount):
            validated_data["amount"] = float(price_amount)
            if isinstance(price_breakdown_json, dict):
                price_breakdown_json["pricing_amount_applied"] = True

        if has_app_context():
            get_transaction = getattr(db.session, "get_transaction", None)
            has_transaction = bool(
                get_transaction() if callable(get_transaction) else False
            )
            tx_context = nullcontext() if has_transaction else db.session.begin()
        else:
            tx_context = nullcontext()
        with tx_context:
            new_booking = self.booking_writer.create_and_commit(
                user_id=cmd.user_id,
                client_id=cmd.client_id,
                company_id=company_id,
                customer_name=validated_data["customer_name"],
                pickup_location=validated_data["pickup_location"],
                dropoff_location=validated_data["dropoff_location"],
                scheduled_time=scheduled_time,
                amount=float(validated_data["amount"]),
                medical_facility=validated_data.get("medical_facility", ""),
                doctor_name=validated_data.get("doctor_name", ""),
                hospital_service=validated_data.get("hospital_service", ""),
                duration_seconds=duration_seconds,
                distance_meters=distance_meters,
                pickup_lat=pickup_lat,
                pickup_lon=pickup_lon,
                dropoff_lat=dropoff_lat,
                dropoff_lon=dropoff_lon,
                is_round_trip=bool(validated_data.get("is_round_trip", False)),
                pickup_admin_token=pickup_admin.get("token"),
                pickup_canton_code=pickup_admin.get("canton_code"),
                pickup_admin_source=pickup_admin.get("source"),
                pickup_admin_confidence=pickup_admin.get("confidence"),
                pickup_admin_label=pickup_admin.get("label"),
                pickup_admin_resolved_at=datetime.now(timezone.utc),  # noqa: UP017
                dropoff_admin_token=dropoff_admin.get("token"),
                dropoff_canton_code=dropoff_admin.get("canton_code"),
                dropoff_admin_source=dropoff_admin.get("source"),
                dropoff_admin_confidence=dropoff_admin.get("confidence"),
                dropoff_admin_label=dropoff_admin.get("label"),
                dropoff_admin_resolved_at=datetime.now(timezone.utc),  # noqa: UP017
                pickup_geo_unit_id=pickup_geo_unit_id,
                dropoff_geo_unit_id=dropoff_geo_unit_id,
                pricing_profile_id=pricing_profile_id,
                pricing_profile_version_id=pricing_profile_version_id,
                price_amount=price_amount,
                price_breakdown_json=price_breakdown_json,
                notes_medical=notes_medical,
                return_scheduled_time=return_scheduled_time,
                return_time_exact=return_time_exact,
            )

        if geocode_miss:
            self._trigger_async_geocoding(
                int(getattr(new_booking, "id", 0) or 0),
                validated_data["pickup_location"],
                validated_data["dropoff_location"],
            )

        publish_event(
            BookingCreatedEvent(
                booking_id=int(getattr(new_booking, "id", 0) or 0),
                company_id=getattr(new_booking, "company_id", None),
            )
        )

        try:
            from middleware.trace_id import get_trace_id
            from security.audit_log import AuditLogger

            st = getattr(new_booking, "status", None)
            new_status = str(getattr(st, "value", st) or "")
            AuditLogger.log_action(
                action_type="booking_created",
                action_category="booking",
                user_id=cmd.user_id,
                user_type="client",
                company_id=getattr(new_booking, "company_id", None),
                booking_id=int(getattr(new_booking, "id", 0) or 0),
                correlation_id=get_trace_id(),
                action_details={
                    "source": "application.bookings.create_booking",
                    "previous_status": None,
                    "new_status": new_status,
                    "trigger": "user",
                },
            )
        except Exception as audit_exc:
            logger.critical(
                "[booking] audit booking_created failed booking_id=%s: %s",
                getattr(new_booking, "id", None),
                audit_exc,
                exc_info=True,
            )
            from services.monitoring.prometheus import inc_booking_audit_write_failed

            with suppress(Exception):
                inc_booking_audit_write_failed(action_type="booking_created")

        return new_booking

    def _compute_pricing_freeze(
        self,
        *,
        company_id: int | None,
        booking_payload: dict[str, Any],
        pickup_admin_token: str | None,
        dropoff_admin_token: str | None,
        pickup_geo_unit_id: int | None,
        dropoff_geo_unit_id: int | None,
        distance_meters: int,
        scheduled_time: datetime,
        is_round_trip: bool,
        pickup_lat: float | None,
        pickup_lon: float | None,
        dropoff_lat: float | None,
        dropoff_lon: float | None,
    ) -> tuple[int | None, int | None, float | None, dict[str, Any] | None]:
        if os.getenv(BOOKING_FREEZE_FLAG, "true").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return None, None, None, None
        if not has_app_context():
            return None, None, None, None
        if company_id is None or company_id <= 0:
            return None, None, None, None
        profile = (
            PricingProfile.query.filter_by(company_id=company_id, is_active=True)
            .order_by(PricingProfile.created_at.desc())
            .first()
        )
        if not profile:
            return None, None, None, None
        version = profile.current_version
        if not version and profile.versions:
            version = sorted(
                profile.versions, key=lambda item: int(item.version), reverse=True
            )[0]
        if not version:
            return profile.id, None, None, None

        now_ref = (
            datetime.now(scheduled_time.tzinfo)
            if scheduled_time.tzinfo
            else datetime.now()
        )
        minutes_until = max(0, int((scheduled_time - now_ref).total_seconds() // 60))
        context = {
            "is_weekend": scheduled_time.weekday() >= WEEKEND_START_INDEX,
            "is_round_trip": bool(is_round_trip),
            "pickup_local_time": scheduled_time.strftime("%H:%M"),
            "minutes_until_pickup": minutes_until,
            "distance_km": max(float(distance_meters or 0) / 1000.0, 0.0),
            "pickup_admin_token": pickup_admin_token,
            "dropoff_admin_token": dropoff_admin_token,
            "pickup_lat": pickup_lat,
            "pickup_lng": pickup_lon,
            "dropoff_lat": dropoff_lat,
            "dropoff_lng": dropoff_lon,
            "pickup_geo_unit_id": pickup_geo_unit_id,
            "dropoff_geo_unit_id": dropoff_geo_unit_id,
            "zones_count": 1 if pickup_admin_token == dropoff_admin_token else 2,
            "requires_waiting": bool(booking_payload.get("requires_waiting")),
        }
        try:
            amount, breakdown = compute_price(booking_payload, version, context)
            return profile.id, version.id, float(amount), breakdown
        except Exception:
            logger.exception(
                "Pricing freeze computation failed for company=%s", company_id
            )
            return profile.id, version.id, None, None

    def _geocode_booking_addresses(
        self, validated_data: dict[str, Any], company_id: int | None
    ) -> tuple[float, float, float, float, bool]:
        try:
            company_ctx = None
            if company_id and self.company_lookup is not None:
                company_ctx = self.company_lookup.find_model_by_id(int(company_id))

            pickup_coords = self.geocoding_service.geocode_address(
                validated_data["pickup_location"], country="CH"
            )
            pickup_lat, pickup_lon, pickup_geocoded = self._process_geocoding_result(
                pickup_coords,
                validated_data["pickup_location"],
                company_ctx,
                "pickup",
            )

            dropoff_coords = self.geocoding_service.geocode_address(
                validated_data["dropoff_location"], country="CH"
            )
            dropoff_lat, dropoff_lon, dropoff_geocoded = self._process_geocoding_result(
                dropoff_coords,
                validated_data["dropoff_location"],
                company_ctx,
                "dropoff",
            )
            geocode_miss = (not pickup_geocoded) or (not dropoff_geocoded)
            return pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, geocode_miss
        except ValueError as e:
            logger.error("❌ Erreur validation adresse lors géocodage: %s", e)
            raise ValueError(
                "L'adresse fournie est invalide ou vide. "
                + "Veuillez fournir une adresse complète et valide."
            ) from e
        except Exception as e:
            logger.exception("❌ Erreur inattendue lors du géocodage des adresses")
            raise RuntimeError(
                "Erreur lors du géocodage des adresses. Le service de géocodage "
                + "est temporairement indisponible. Veuillez réessayer dans "
                + "quelques instants."
            ) from e

    def _process_geocoding_result(
        self,
        coords: dict[str, float] | None,
        address: str,
        company: Any | None,
        address_type: str,
    ) -> tuple[float, float, bool]:
        if coords and "lat" in coords and "lon" in coords:
            lat_val = coords.get("lat")
            lon_val = coords.get("lon")
            if (
                lat_val is not None
                and lon_val is not None
                and GeoValidator.is_valid(lat_val, lon_val)
            ):
                logger.info(
                    "✅ %s géocodé: %s -> (%.6f, %.6f)",
                    address_type.capitalize(),
                    address,
                    lat_val,
                    lon_val,
                )
                return float(lat_val), float(lon_val), True

        fallback_fn = self.fallback_coords_fn
        if fallback_fn is None:
            msg = (
                f"Géocodage échoué pour {address_type} '{address}' et "
                "aucune fonction fallback configurée."
            )
            raise RuntimeError(msg)

        lat, lon = fallback_fn(company)
        logger.warning(
            "⚠️ %s géocodage échoué pour '%s', fallback: (%.6f, %.6f)",
            address_type.capitalize(),
            address,
            lat,
            lon,
        )
        return lat, lon, False

    def _trigger_async_geocoding(
        self, booking_id: int, pickup_address: str, dropoff_address: str
    ) -> None:
        if self.trigger_async_geocoding_fn is not None:
            self.trigger_async_geocoding_fn(booking_id, pickup_address, dropoff_address)
            return
        logger.info("ℹ️ Géocodage async non configuré (booking_id=%s).", booking_id)

    @staticmethod
    def _extract_pickup_zip(pickup_location: str | None) -> str | None:
        if not pickup_location:
            return None
        match = re.search(r"\b(\d{4})\b", pickup_location)
        return match.group(1) if match else None

    @staticmethod
    def _extract_dropoff_zip(dropoff_location: str | None) -> str | None:
        if not dropoff_location:
            return None
        match = re.search(r"\b(\d{4})\b", dropoff_location)
        return match.group(1) if match else None
