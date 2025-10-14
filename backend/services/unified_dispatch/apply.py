# backend/services/unified_dispatch/apply.py
from __future__ import annotations

import os
import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, cast


from sqlalchemy.orm import joinedload

from ext import db
from models import Booking, BookingStatus, Driver, Assignment, AssignmentStatus
from shared.time_utils import now_utc  # UTC centralisé

logger = logging.getLogger(__name__)

_Assignment = Any

def apply_assignments(
    company_id: int,
    assignments: List[_Assignment],
    *,
    dispatch_run_id: Optional[int] = None,
    allow_reassign: bool = True,
    respect_existing: bool = True,
    enforce_driver_checks: bool = True,
    return_pairs: bool = False,
) -> Dict[str, Any]:
    if not assignments:
        return {"applied": [], "skipped": {}, "conflicts": [], "driver_load": {}}

    # ✅ ROLLBACK DÉFENSIF AU DÉBUT
    try:
        db.session.rollback()
    except Exception:
        pass

    # Log pour tracer la propagation du dispatch_run_id
    if dispatch_run_id:
        logger.info("[Apply] Using dispatch_run_id=%s for assignments", dispatch_run_id)
    
    # Helper: attr ou clé dict 
    def _aget(obj: Any, name: str, default: Any = None) -> Any:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                pass
        if isinstance(obj, dict):
            return obj.get(name, default)
        return default

    # 1) Déduplication par booking_id
    chosen_by_booking: Dict[int, _Assignment] = {}
    for a in assignments:
        b_id = int(_aget(a, "booking_id"))
        if b_id not in chosen_by_booking:
            chosen_by_booking[b_id] = a
        else:
            prev = chosen_by_booking[b_id]
            a_score = _aget(a, "score", None)
            p_score = _aget(prev, "score", None)
            if a_score is not None and p_score is not None:
                if a_score > p_score:
                    chosen_by_booking[b_id] = a
            else:
                chosen_by_booking[b_id] = a

    booking_ids = list(chosen_by_booking.keys())
    # Utiliser le helper _aget pour supporter objets ET dicts
    driver_ids = sorted({
        int(_aget(chosen_by_booking[b], "driver_id"))
        for b in booking_ids
        if _aget(chosen_by_booking[b], "driver_id") is not None
    })

    # 2) Chargements + (optionnel) verrouillage
    bookings_q = (
        Booking.query.options(joinedload(Booking.driver))
        .filter(Booking.company_id == company_id, Booking.id.in_(booking_ids))
    )
    drivers_q = (
        Driver.query
        .filter(Driver.company_id == company_id, Driver.id.in_(driver_ids))
    )

    # Appliquer FOR UPDATE uniquement si supporté (SQLite: non)
    dialect_name = db.session.bind.dialect.name if db.session.bind else ""
    supports_for_update = dialect_name not in ("sqlite",)

    if supports_for_update:
        # Optionnel: SKIP LOCKED (Postgres) pour éviter le blocage si autre transaction tient un lock
        use_skip_locked = os.getenv("UD_APPLY_SKIP_LOCKED", "false").lower() == "true"
        bookings_q = bookings_q.with_for_update(nowait=False, of=Booking, skip_locked=use_skip_locked)
        drivers_q = drivers_q.with_for_update(nowait=False, of=Driver,   skip_locked=use_skip_locked)

    bookings = bookings_q.all()
    drivers = drivers_q.all()

    booking_map: Dict[int, Booking] = {b.id: b for b in bookings}
    driver_map: Dict[int, Driver] = {d.id: d for d in drivers}

    # 3) Prépare updates
    applied_ids: List[int] = []
    skipped: Dict[int, str] = {}
    conflicts: List[int] = []
    driver_load: Dict[int, int] = defaultdict(int)

    now = now_utc()  # ⟵ centralisé

    updates: List[Dict[str, Any]] = []
    applied_pairs: List[Tuple[int, int]] = []  # (booking_id, driver_id) - utile si besoin
    # Candidats à l'upsert dans Assignment (même si Booking inchangé)
    desired_assignments: Dict[int, Dict[str, Any]] = {}

    for b_id, a in chosen_by_booking.items():
        b = booking_map.get(b_id)
        if b is None:
            skipped[b_id] = "booking_not_found_or_wrong_company"
            continue

        if b.status not in (BookingStatus.PENDING, BookingStatus.ACCEPTED, BookingStatus.ASSIGNED):
            skipped[b_id] = f"status_is_{b.status}"
            continue

        d_id = int(_aget(a, "driver_id"))
        d = driver_map.get(d_id)
        if d is None:
            skipped[b_id] = "driver_not_found_or_wrong_company"
            continue
        d_any = cast(Any, d)
        is_active = bool(getattr(d_any, "is_active", False))
        is_available = bool(getattr(d_any, "is_available", False))
        if enforce_driver_checks and (not is_active or not is_available):
            skipped[b_id] = "driver_not_available"
            continue

        # Enregistrer la cible d'Assignment (ETA incluse si fournie)
        desired_assignments[b_id] = {
            "booking_id": b_id,
            "driver_id": d_id,
            "status": AssignmentStatus.SCHEDULED,
            "estimated_pickup_arrival": _aget(a, "estimated_pickup_arrival"),
            "estimated_dropoff_arrival": _aget(a, "estimated_dropoff_arrival"),
            "dispatch_run_id": dispatch_run_id if dispatch_run_id is not None else _aget(a, "dispatch_run_id"),  # Priorité au dispatch_run_id passé en param
        }

        b_any = cast(Any, b)
        b_status: BookingStatus = cast(BookingStatus, getattr(b_any, "status", None))

        cur_driver_id_raw = getattr(b_any, "driver_id", None)
        try:
            cur_driver_id: Optional[int] = int(cur_driver_id_raw) if cur_driver_id_raw is not None else None
        except Exception:
            cur_driver_id = None

        is_assigned = (b_status == BookingStatus.ASSIGNED)
        same_driver = (cur_driver_id == d_id)

        if respect_existing and is_assigned and same_driver:
            skipped[b_id] = "already_assigned_same_driver"
            continue

        if is_assigned and (cur_driver_id is not None) and (cur_driver_id != d_id) and (not allow_reassign):
            conflicts.append(b_id)
            skipped[b_id] = "reassign_blocked"
            continue


        payload = {
            "id": b.id,
            "driver_id": d_id,
            "status": BookingStatus.ASSIGNED,
        }
        # timestamps optionnels suivant le modèle
        if hasattr(b, "assigned_at"):
            payload["assigned_at"] = now
        if hasattr(b, "updated_at"):          # ⟵ ajoute updated_at uniquement si présent
            payload["updated_at"] = now

        updates.append(payload)
        applied_ids.append(b_id)
        applied_pairs.append((b_id, d_id))
        driver_load[d_id] += 1

    # 4) Write back Bookings + upsert Assignments (une seule transaction si possible)
    try:
        with db.session.begin_nested():  # ✅ Utilisation de begin_nested pour savepoint
            if updates:
                db.session.bulk_update_mappings(cast(Any, Booking), updates)

            # Upsert côté Assignment (y compris ETA si fournies)
            if desired_assignments:
                target_bids = list(desired_assignments.keys())
                existing = (
                    Assignment.query
                    .filter(Assignment.booking_id.in_(target_bids))
                    .all()
                )
                by_booking: Dict[int, Assignment] = {}
                for a0 in existing:
                    cur = by_booking.get(a0.booking_id)
                    if cur is None or (hasattr(a0, "created_at") and hasattr(cur, "created_at") and a0.created_at > cur.created_at):
                        by_booking[a0.booking_id] = a0

                for b_id, payload in desired_assignments.items():
                    cur = by_booking.get(b_id)
                    if cur is None:
                        # création
                        new = Assignment()
                        a_any = cast(Any, new)
                        a_any.booking_id = int(payload["booking_id"])
                        a_any.driver_id = payload["driver_id"]
                        a_any.status = payload.get("status", AssignmentStatus.SCHEDULED)
                        eta_pu = payload.get("estimated_pickup_arrival") or payload.get("eta_pickup_at")
                        eta_do = payload.get("estimated_dropoff_arrival") or payload.get("eta_dropoff_at")
                        if eta_pu is not None:
                            a_any.eta_pickup_at = eta_pu
                        if eta_do is not None:
                            a_any.eta_dropoff_at = eta_do
                        drid = payload.get("dispatch_run_id") or dispatch_run_id
                        if drid is not None:
                            a_any.dispatch_run_id = drid  # ✅ Assignation du dispatch_run_id
                        if hasattr(new, "created_at"):
                            a_any.created_at = now
                        if hasattr(new, "updated_at"):
                            a_any.updated_at = now
                        db.session.add(new)
                    else:
                        # mise à jour
                        c_any = cast(Any, cur)
                        c_any.driver_id = payload["driver_id"]
                        c_any.status = payload.get("status", AssignmentStatus.SCHEDULED)
                        eta_pu = payload.get("estimated_pickup_arrival") or payload.get("eta_pickup_at")
                        eta_do = payload.get("estimated_dropoff_arrival") or payload.get("eta_dropoff_at")
                        if eta_pu is not None:
                            c_any.eta_pickup_at = eta_pu
                        if eta_do is not None:
                            c_any.eta_dropoff_at = eta_do
                        drid = payload.get("dispatch_run_id")
                        if drid is not None:  # Only update if we have a valid dispatch_run_id
                            c_any.dispatch_run_id = payload["dispatch_run_id"]  # ✅ Mise à jour du dispatch_run_id
                        if hasattr(cur, "updated_at"):
                            c_any.updated_at = now
            else:
                logger.info("[Apply] No desired assignments to upsert (company_id=%s)", company_id)
        
        db.session.commit()  # ✅ Commit après le savepoint

    except Exception as e:
        logger.exception("[Apply] DB error while applying assignments (company_id=%s)", company_id)
        db.session.rollback()  # ✅ Rollback sur erreur
        return {
            "applied": [], "skipped": {b_id: "db_error" for b_id in applied_ids},
            "conflicts": [], "driver_load": {}, "error": str(e),
        }
    if dispatch_run_id:
        logger.info("[Apply] Linked %d assignments to dispatch_run_id=%s", len(desired_assignments), dispatch_run_id)


    if not updates:
        logger.info("[Apply] No booking updates (company_id=%s) — assignments/ETA refreshed only.", company_id)


    result = {
        "applied": applied_ids,
        "skipped": skipped,
        "conflicts": conflicts,
        "driver_load": dict(driver_load),
    }
    # Optionnel : retourner les paires (booking_id, driver_id) si demandé
    if return_pairs:
        result["applied_pairs"] = applied_pairs

    logger.info(
        "[Apply] company=%s applied=%d skipped=%d conflicts=%d (reasons=%s)",
        company_id, len(applied_ids), len(skipped), len(conflicts), dict(skipped)    )

    # 🔔 Notifications Socket.IO vers les chauffeurs pour MAJ en temps réel (mobile)
    try:
        if applied_pairs:
            # Charger les bookings et notifier chaque driver ciblé
            for (b_id, d_id) in applied_pairs:
                try:
                    b = Booking.query.get(b_id)
                    if b is None:
                        continue
                    from services.notification_service import notify_driver_new_booking
                    notify_driver_new_booking(int(d_id), b)
                except Exception:
                    logger.exception("[Apply] notify_driver_new_booking failed booking_id=%s driver_id=%s", b_id, d_id)
    except Exception:
        logger.exception("[Apply] driver notifications failed (company_id=%s)", company_id)
    return result