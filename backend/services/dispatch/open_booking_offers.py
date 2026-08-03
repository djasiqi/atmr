"""Offres dispatch pour réservations créées sans entreprise propriétaire (marché ouvert)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import not_

from ext import db
from models import Booking
from models.enums import BookingStatus, DispatchOfferStatus
from models.service_area_pricing import DispatchOffer
from services.dispatch.scoring_engine import (
    compute_candidates,
    compute_urgency_override_candidates,
    persist_offers_for_threshold,
    persist_urgency_offers,
)

logger = logging.getLogger(__name__)


def _canton_token_from_address_keywords(address: str | None) -> str | None:
    """Si geocodage inverse echoue, deduit ``canton:XX`` depuis des mots-cles canton dans l'adresse (ex. Geneve -> GE)."""
    from services.geo.geo_resolver import LEGACY_CANTON_MAP, normalize_text

    n = normalize_text(address)
    if not n:
        return None
    for keyword, code in LEGACY_CANTON_MAP.items():
        if keyword in n:
            return f"canton:{code}"
    return None


def _token_for_geo_unit_lookup(admin_payload: dict[str, Any]) -> str | None:
    """Construit un jeton utilisable par ``geo_unit_id_from_pickup_admin_token`` (Photon peut renvoyer ``canton_code`` sans ``token``)."""
    token = (admin_payload.get("token") or "").strip()
    if token:
        return token
    cc = (admin_payload.get("canton_code") or "").strip().upper()
    if cc:
        return f"canton:{cc}"
    return None


def ensure_booking_dispatch_geo_units(booking: Booking) -> bool:
    """Renseigne ``pickup_geo_unit_id`` / ``dropoff_geo_unit_id`` si absents mais coords connues.

    Permet au scoring dispatch de trouver des transporteurs après géocodage async ou pour
    d'anciennes lignes sans unités — sans cette étape, ``compute_candidates`` reçoit des
    chaînes vides et aucune offre PROPOSED n'est créée (dashboard entreprise vide).
    """
    from services.geo.geo_resolver import (
        geo_unit_id_from_pickup_admin_token,
        resolve_pickup_admin,
    )

    changed = False
    if getattr(booking, "pickup_geo_unit_id", None) is None:
        pu_lat = getattr(booking, "pickup_lat", None)
        pu_lon = getattr(booking, "pickup_lon", None)
        if pu_lat is not None and pu_lon is not None:
            pa = resolve_pickup_admin(
                lat=float(pu_lat),
                lng=float(pu_lon),
                pickup_zip=None,
                pickup_text=getattr(booking, "pickup_location", None),
            )
            token = _token_for_geo_unit_lookup(
                pa
            ) or _canton_token_from_address_keywords(
                getattr(booking, "pickup_location", None)
            )
            gid = geo_unit_id_from_pickup_admin_token(token)
            if gid:
                booking.pickup_geo_unit_id = int(gid)
                changed = True
    if getattr(booking, "dropoff_geo_unit_id", None) is None:
        do_lat = getattr(booking, "dropoff_lat", None)
        do_lon = getattr(booking, "dropoff_lon", None)
        if do_lat is not None and do_lon is not None:
            da = resolve_pickup_admin(
                lat=float(do_lat),
                lng=float(do_lon),
                pickup_zip=None,
                pickup_text=getattr(booking, "dropoff_location", None),
            )
            token = _token_for_geo_unit_lookup(
                da
            ) or _canton_token_from_address_keywords(
                getattr(booking, "dropoff_location", None)
            )
            gid = geo_unit_id_from_pickup_admin_token(token)
            if gid:
                booking.dropoff_geo_unit_id = int(gid)
                changed = True
    if changed:
        db.session.flush()
        db.session.expire(booking, ["pickup_geo_unit", "dropoff_geo_unit"])
    return changed


def _notify_companies_new_dispatch_offers(booking_id: int) -> None:
    """Socket : notifie chaque entreprise ayant une offre PROPOSED sur ce booking (room ``new_reservation``)."""
    try:
        from services.realtime.socketio import emit_company_event

        rows = (
            db.session.query(DispatchOffer.company_id)
            .filter(
                DispatchOffer.booking_id == booking_id,
                DispatchOffer.status == DispatchOfferStatus.PROPOSED,
            )
            .distinct()
            .all()
        )
        for (cid,) in rows:
            if cid is None:
                continue
            emit_company_event(
                int(cid),
                "new_reservation",
                {"booking_id": int(booking_id)},
            )
    except Exception:
        logger.exception(
            "[open_booking_offers] emit new_reservation echoue booking_id=%s",
            booking_id,
        )


def booking_open_market_has_dispatch_candidates(booking: Booking) -> bool:
    """Vrai si au moins une entreprise avec dispatch activé est candidate pour cette course.

    Réservé au **marché ouvert** (``company_id`` absent) : avant paiement en ligne, évite
    d'encaisser si aucun transporteur ne peut être sollicité par le moteur de dispatch.
    """
    if getattr(booking, "company_id", None) is not None:
        return True
    ensure_booking_dispatch_geo_units(booking)
    db.session.refresh(booking)
    pickup_gu = getattr(booking, "pickup_geo_unit", None)
    drop_gu = getattr(booking, "dropoff_geo_unit", None)
    candidates = compute_candidates(
        pickup_geo_unit=pickup_gu,
        drop_geo_unit=drop_gu,
    )
    return len(candidates) > 0


def seed_dispatch_offers_for_unassigned_booking(booking_id: int) -> int:
    """Crée des ``DispatchOffer`` PROPOSED pour les transporteurs éligibles.

    Réutilise la même logique que ``POST .../scoring/dispatch/<booking_id>`` mais
    sans exiger que le booking soit déjà rattaché à une entreprise.

    Returns:
        Nombre d'offres créées (0 si booking déjà assigné ou introuvable).
    """
    booking = db.session.get(Booking, booking_id)
    if booking is None:
        return 0
    assigned_company_id: int | None = getattr(booking, "company_id", None)
    if assigned_company_id is not None:
        return 0

    ensure_booking_dispatch_geo_units(booking)

    candidates = compute_candidates(
        pickup_geo_unit=booking.pickup_geo_unit,
        drop_geo_unit=booking.dropoff_geo_unit,
    )
    from services.platform_billing.capabilities import (
        BillingCapability,
        is_billing_capability_allowed,
    )

    candidates = [
        c
        for c in candidates
        if is_billing_capability_allowed(
            int(c.company_id),
            BillingCapability.RECEIVE_MARKETPLACE_OFFERS,
        )
    ]
    created_total: list[Any] = []
    for threshold in (100, 70, 50, 10):
        created = persist_offers_for_threshold(
            booking_id=booking.id, candidates=candidates, threshold=threshold
        )
        created_total.extend(created)
        if created:
            break

    pickup_at = booking.scheduled_time
    should_run_urgency = False
    if pickup_at:
        pickup_at_aware = (
            pickup_at.replace(tzinfo=UTC) if pickup_at.tzinfo is None else pickup_at
        )
        should_run_urgency = (
            pickup_at_aware - datetime.now(UTC)
        ).total_seconds() <= 15 * 60

    if not created_total and should_run_urgency:
        pickup_lat_raw = getattr(booking, "pickup_lat", None)
        pickup_lon_raw = getattr(booking, "pickup_lon", None)
        urgent_candidates = compute_urgency_override_candidates(
            pickup_lat=float(pickup_lat_raw) if pickup_lat_raw is not None else None,
            pickup_lon=float(pickup_lon_raw) if pickup_lon_raw is not None else None,
        )
        created_total = persist_urgency_offers(booking.id, urgent_candidates)

    if not created_total:
        logger.info(
            "[open_booking_offers] Aucune offre générée pour booking_id=%s (candidates=%s)",
            booking_id,
            len(candidates),
        )

    n_created = len(created_total)
    db.session.commit()
    if n_created > 0:
        _notify_companies_new_dispatch_offers(int(booking_id))
    return n_created


def refresh_dispatch_offers_after_online_payment(booking_id: int) -> None:
    """Réémet les offres marché ouvert après paiement si la course est encore sans transporteur.

    Utile lorsque le premier seed à la création n'a rien produit (géo incomplète) ou après
    passage AWAITING_CLIENT_PAYMENT -> PENDING.
    """
    try:
        booking = db.session.get(Booking, booking_id)
        if booking is None:
            return
        if getattr(booking, "company_id", None) is not None:
            return
        n = seed_dispatch_offers_for_unassigned_booking(int(booking_id))
        if n > 0:
            logger.info(
                "[open_booking_offers] Apres paiement en ligne: seed booking_id=%s offres=%s",
                booking_id,
                n,
            )
    except Exception:
        logger.exception(
            "[open_booking_offers] refresh apres paiement echoue booking_id=%s",
            booking_id,
        )


def seed_offers_for_pending_without_proposed_offers(
    *, limit: int = 100
) -> dict[str, Any]:
    """Retente ``seed_dispatch_offers_for_unassigned_booking`` pour les PENDING sans offre PROPOSED.

    Permet de « rattraper » les demandes orphelines (visibles client mais invisibles entreprises
    car aucune ligne ``DispatchOffer`` PROPOSED).
    """
    proposed_exists = (
        db.session.query(DispatchOffer.id)
        .filter(
            DispatchOffer.booking_id == Booking.id,
            DispatchOffer.status == DispatchOfferStatus.PROPOSED,
        )
        .exists()
    )
    rows = (
        db.session.query(Booking.id)
        .filter(
            Booking.company_id.is_(None),
            Booking.status == BookingStatus.PENDING,
            not_(proposed_exists),
        )
        .order_by(Booking.updated_at.desc().nullslast(), Booking.id.desc())
        .limit(max(1, min(limit, 500)))
        .all()
    )
    booking_ids = [int(r[0]) for r in rows]
    total_offers = 0
    errors = 0
    for bid in booking_ids:
        try:
            total_offers += seed_dispatch_offers_for_unassigned_booking(bid)
        except Exception:
            errors += 1
            logger.exception(
                "[open_booking_offers] repair seed echoue booking_id=%s",
                bid,
            )
    out: dict[str, Any] = {
        "booking_ids": booking_ids,
        "offers_created_total": total_offers,
        "errors": errors,
    }
    if booking_ids:
        logger.warning(
            "[open_booking_offers] repair PENDING sans PROPOSED: bookings=%s offres=%s erreurs=%s",
            len(booking_ids),
            total_offers,
            errors,
        )
    return out
