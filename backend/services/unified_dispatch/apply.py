# backend/services/unified_dispatch/apply.py v1.0.0
from __future__ import annotations

import contextlib
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple, cast

from sqlalchemy.orm import joinedload, scoped_session, sessionmaker

from ext import db
from models import Assignment, AssignmentStatus, Booking, BookingStatus, Driver
from services.unified_dispatch.transaction_helpers import _begin_tx, _in_tx
from shared.time_utils import now_utc  # UTC centralisé

logger = logging.getLogger(__name__)


def _get_scoped_session(db_instance):
    """
    Crée une scoped session compatible avec toutes les versions de Flask-SQLAlchemy.

    Args:
        db_instance: Instance SQLAlchemy de Flask-SQLAlchemy

    Returns:
        Scoped session pour requêtes indépendantes
    """
    try:
        # Essayer d'abord create_scoped_session si disponible (anciennes versions)
        if hasattr(db_instance, "create_scoped_session"):
            return db_instance.create_scoped_session()
    except AttributeError:
        pass

    # Fallback : créer une scoped_session manuellement
    try:
        # Obtenir l'engine de différentes manières selon la version
        engine = getattr(db_instance, "engine", None)
        if engine is None and hasattr(db_instance, "get_engine"):
            engine = db_instance.get_engine()  # Flask-SQLAlchemy v3+
        elif engine is None and hasattr(db_instance, "session"):
            # Flask-SQLAlchemy v3+ : utiliser l'engine de la session
            engine = db_instance.session.get_bind()

        if engine is None:
            logger.warning(
                "[Apply] Impossible de créer scoped_session, utilisation de db.session"
            )
            return db_instance.session

        return scoped_session(sessionmaker(bind=engine))
    except Exception as e:
        logger.warning(
            "[Apply] Erreur lors de la création de scoped_session: %s, utilisation de db.session",
            e,
        )
        # Dernier recours : utiliser la session principale
        return db_instance.session


_Assignment = Any


# ✅ A2: Compteur thread-safe pour conflits DB (contraintes uniques)
class DBConflictCounter:
    """Compteur thread-safe pour les violations de contraintes uniques."""

    _instance: "DBConflictCounter | None" = None

    def __init__(self):
        super().__init__()
        self._counter = 0

    @classmethod
    def get_instance(cls) -> "DBConflictCounter":
        """Retourne l'instance singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self) -> None:
        """Réinitialise le compteur."""
        self._counter = 0

    def increment(self) -> None:
        """Incrémente le compteur."""
        self._counter += 1

    def get_count(self) -> int:
        """Retourne le nombre total de conflits."""
        return self._counter


def reset_db_conflict_counter() -> None:
    """Réinitialise le compteur de conflits DB."""
    DBConflictCounter.get_instance().reset()


def get_db_conflict_count() -> int:
    """Retourne le nombre de conflits DB depuis le dernier reset."""
    return DBConflictCounter.get_instance().get_count()


def increment_db_conflict_counter() -> None:
    """Incrémente le compteur de conflits DB."""
    DBConflictCounter.get_instance().increment()


def apply_assignments(
    company_id: int,
    assignments: List[_Assignment],
    *,
    dispatch_run_id: int | None = None,
    allow_reassign: bool = True,
    respect_existing: bool = True,
    enforce_driver_checks: bool = True,
    return_pairs: bool = False,
) -> Dict[str, Any]:
    """Applique les assignations en base de données avec transaction atomique.

    Toutes les modifications (Booking, Assignment) sont effectuées dans une seule
    transaction pour garantir l'atomicité. En cas d'erreur, rollback complet.
    """
    if not assignments:
        return {"applied": [], "skipped": {}, "conflicts": [], "driver_load": {}}

    # ✅ ROLLBACK DÉFENSIF AU DÉBUT
    with contextlib.suppress(Exception):
        db.session.rollback()

    # Log pour tracer la propagation du dispatch_run_id
    if dispatch_run_id:
        logger.info("[Apply] Using dispatch_run_id=%s for assignments", dispatch_run_id)

    # ✅ Transaction globale pour garantir atomicité complète
    # Utilise _begin_tx() qui détecte si une transaction existe déjà (savepoint)
    # ou en crée une nouvelle si nécessaire
    # Vérifier si une transaction existe déjà avant d'appeler _begin_tx()
    had_existing_tx = _in_tx()
    try:
        with _begin_tx():
            result = _apply_assignments_inner(
                company_id=company_id,
                assignments=assignments,
                dispatch_run_id=dispatch_run_id,
                allow_reassign=allow_reassign,
                respect_existing=respect_existing,
                enforce_driver_checks=enforce_driver_checks,
                return_pairs=return_pairs,
            )
        # ✅ Commit la transaction principale après succès du bloc
        # Si aucune transaction n'existait avant, on doit commit explicitement
        # Sinon, le commit sera fait par le code appelant (engine.run())
        if not had_existing_tx:
            db.session.commit()
        return result
    except Exception as e:
        logger.exception(
            "[Apply] Transaction failed for company_id=%s: %s", company_id, e
        )
        # Rollback automatique en cas d'erreur
        db.session.rollback()
        # ✅ FIX RC2: Expirer tous les objets après rollback pour forcer le rechargement
        db.session.expire_all()
        # ✅ FIX RC2: S'assurer que tous les objets modifiés sont bien restaurés
        # En expirant tous les objets, SQLAlchemy les rechargera depuis la DB au prochain accès
        return {
            "applied": [],
            "skipped": {},
            "conflicts": [],
            "driver_load": {},
            "error": str(e),
        }


def _apply_assignments_inner(
    company_id: int,
    assignments: List[_Assignment],
    *,
    dispatch_run_id: int | None = None,
    allow_reassign: bool = True,
    respect_existing: bool = True,
    enforce_driver_checks: bool = True,
    return_pairs: bool = False,
) -> Dict[str, Any]:
    """Logique interne d'application des assignations (exécutée dans une transaction)."""

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
    driver_ids = sorted(
        {
            int(_aget(chosen_by_booking[b], "driver_id"))
            for b in booking_ids
            if _aget(chosen_by_booking[b], "driver_id") is not None
        }
    )

    # 2) Chargements + (optionnel) verrouillage
    # ✅ FIX RC4: Flush la session pour s'assurer que les objets en attente sont visibles
    db.session.flush()

    bookings_q = Booking.query.options(joinedload(Booking.driver)).filter(
        Booking.company_id == company_id, Booking.id.in_(booking_ids)
    )
    drivers_q = Driver.query.filter(
        Driver.company_id == company_id, Driver.id.in_(driver_ids)
    )

    # ✅ A2: Lock doux en lecture (read=True pour lock non-bloquant)
    dialect_name = db.session.bind.dialect.name if db.session.bind else ""
    supports_for_update = dialect_name not in ("sqlite",)

    if supports_for_update:
        # Optionnel: SKIP LOCKED (Postgres) pour éviter le blocage si autre
        # transaction tient un lock
        use_skip_locked = os.getenv("UD_APPLY_SKIP_LOCKED", "false").lower() == "true"
        # ✅ A2: Lock doux en lecture (lock partagé pour idempotence)
        # Note: avec_for_update(read=True) est un lock partagé PostgreSQL
        bookings_q = bookings_q.with_for_update(
            nowait=False, of=Booking, skip_locked=use_skip_locked
        )
        drivers_q = drivers_q.with_for_update(
            nowait=False, of=Driver, skip_locked=use_skip_locked
        )

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
    # (booking_id, driver_id) - utile si besoin
    applied_pairs: List[Tuple[int, int]] = []
    # Candidats à l'upsert dans Assignment (même si Booking inchangé)
    desired_assignments: Dict[int, Dict[str, Any]] = {}

    for b_id, a in chosen_by_booking.items():
        b = booking_map.get(b_id)
        # ✅ FIX RC2/RC4: Recharger le booking depuis la DB pour éviter problèmes de session
        if b is None:
            # Essayer de flush la session pour voir les objets en attente
            db.session.flush()
            # ✅ FIX: Utiliser filter au lieu de filter_by pour plus de flexibilité
            b = (
                db.session.query(Booking)
                .filter(Booking.id == b_id, Booking.company_id == company_id)
                .first()
            )
        if b is None:
            # ✅ FIX RC4: Logger plus d'infos pour debug
            logger.warning(
                "[Apply] Booking id=%s company_id=%s not found in booking_map (size=%d) or DB query",
                b_id,
                company_id,
                len(booking_map),
            )
            skipped[b_id] = "booking_not_found_or_wrong_company"
            continue

        if b.status not in (
            BookingStatus.PENDING,
            BookingStatus.ACCEPTED,
            BookingStatus.ASSIGNED,
        ):
            skipped[b_id] = f"status_is_{b.status}"
            continue

        d_id = int(_aget(a, "driver_id"))
        d = driver_map.get(d_id)
        if d is None:
            skipped[b_id] = "driver_not_found_or_wrong_company"
            continue
        d_any = cast("Any", d)
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
            # Priorité au dispatch_run_id passé en param
            "dispatch_run_id": dispatch_run_id
            if dispatch_run_id is not None
            else _aget(a, "dispatch_run_id"),
        }

        b_any = cast("Any", b)
        b_status: BookingStatus = cast("BookingStatus", getattr(b_any, "status", None))

        cur_driver_id_raw = getattr(b_any, "driver_id", None)
        try:
            cur_driver_id: int | None = (
                int(cur_driver_id_raw) if cur_driver_id_raw is not None else None
            )
        except Exception:
            cur_driver_id = None

        is_assigned = b_status == BookingStatus.ASSIGNED
        same_driver = cur_driver_id == d_id

        if respect_existing and is_assigned and same_driver:
            skipped[b_id] = "already_assigned_same_driver"
            continue

        if (
            is_assigned
            and (cur_driver_id is not None)
            and (cur_driver_id != d_id)
            and (not allow_reassign)
        ):
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
        if hasattr(b, "updated_at"):  # ⟵ ajoute updated_at uniquement si présent
            payload["updated_at"] = now

        updates.append(payload)
        applied_ids.append(b_id)
        applied_pairs.append((b_id, d_id))
        driver_load[d_id] += 1

    # 4) Write back Bookings + upsert Assignments
    # ✅ Déjà dans une transaction globale (_begin_tx), donc begin_nested créerait
    # un savepoint supplémentaire (optionnel mais peut être utile pour rollback partiel)
    # On garde begin_nested pour compatibilité et rollback partiel si nécessaire
    try:
        with db.session.begin_nested():  # Savepoint pour rollback partiel si nécessaire
            if updates:
                db.session.bulk_update_mappings(cast("Any", Booking), updates)

            # Upsert côté Assignment (y compris ETA si fournies)
            if desired_assignments:
                target_bids = list(desired_assignments.keys())
                existing = Assignment.query.filter(
                    Assignment.booking_id.in_(target_bids)
                ).all()
                by_booking: Dict[int, Assignment] = {}
                for a0 in existing:
                    cur = by_booking.get(a0.booking_id)
                    if cur is None or (
                        hasattr(a0, "created_at")
                        and hasattr(cur, "created_at")
                        and a0.created_at > cur.created_at
                    ):
                        by_booking[a0.booking_id] = a0

                # ✅ PERF: Séparer nouveaux vs existants pour bulk operations
                new_assignments: List[Dict[str, Any]] = []
                update_assignments: List[Dict[str, Any]] = []

                for b_id, payload in desired_assignments.items():
                    cur = by_booking.get(b_id)
                    if cur is None:
                        # ✅ PERF: Préparer pour bulk_insert_mappings
                        new_assignment = {
                            "booking_id": int(payload["booking_id"]),
                            "driver_id": payload["driver_id"],
                            "status": payload.get("status", AssignmentStatus.SCHEDULED),
                            "created_at": now,
                            "updated_at": now,
                        }

                        # ETA optionnels
                        eta_pu = payload.get("estimated_pickup_arrival") or payload.get(
                            "eta_pickup_at"
                        )
                        eta_do = payload.get(
                            "estimated_dropoff_arrival"
                        ) or payload.get("eta_dropoff_at")
                        if eta_pu is not None:
                            new_assignment["eta_pickup_at"] = eta_pu
                        if eta_do is not None:
                            new_assignment["eta_dropoff_at"] = eta_do

                        # dispatch_run_id
                        drid = payload.get("dispatch_run_id") or dispatch_run_id
                        if drid is not None:
                            new_assignment["dispatch_run_id"] = drid

                        new_assignments.append(new_assignment)
                    else:
                        # ✅ PERF: Préparer pour bulk_update_mappings
                        update_assignment = {
                            "id": cur.id,
                            "driver_id": payload["driver_id"],
                            "status": payload.get("status", AssignmentStatus.SCHEDULED),
                            "updated_at": now,
                        }

                        # ETA optionnels
                        eta_pu = payload.get("estimated_pickup_arrival") or payload.get(
                            "eta_pickup_at"
                        )
                        eta_do = payload.get(
                            "estimated_dropoff_arrival"
                        ) or payload.get("eta_dropoff_at")
                        if eta_pu is not None:
                            update_assignment["eta_pickup_at"] = eta_pu
                        if eta_do is not None:
                            update_assignment["eta_dropoff_at"] = eta_do

                        # dispatch_run_id
                        drid = payload.get("dispatch_run_id")
                        if drid is not None:
                            update_assignment["dispatch_run_id"] = drid

                        update_assignments.append(update_assignment)

                # ✅ A2: Idempotence avec UPSERT ON CONFLICT DO NOTHING
                if new_assignments:
                    # Utiliser PostgreSQL insert avec ON CONFLICT
                    from sqlalchemy.dialects.postgresql import insert

                    try:
                        # Pour chaque nouveau assignment, faire un upsert
                        conflicts_count = 0
                        for assignment in new_assignments:
                            try:
                                stmt = (
                                    insert(Assignment)
                                    .values(**assignment)
                                    .on_conflict_do_nothing(
                                        constraint="uq_assignment_run_booking"
                                    )
                                )
                                db.session.execute(stmt)
                            except Exception as conflict_err:
                                # ✅ A2: Compter les conflits de contrainte unique
                                if "unique" in str(
                                    conflict_err
                                ).lower() or "uq_assignment" in str(conflict_err):
                                    conflicts_count += 1
                                    increment_db_conflict_counter()
                                    logger.debug(
                                        "[Apply] Conflit unique ignoré (idempotence): %s",
                                        conflict_err,
                                    )
                                else:
                                    raise

                        if conflicts_count > 0:
                            logger.info(
                                "[Apply] UPSERT: %d insertions, %d conflits ignorés (idempotent)",
                                len(new_assignments) - conflicts_count,
                                conflicts_count,
                            )
                        else:
                            logger.info(
                                "[Apply] UPSERT inserted %d new assignments",
                                len(new_assignments),
                            )
                    except Exception as upsert_err:
                        # Fallback sur bulk_insert si ON CONFLICT non supporté
                        logger.warning(
                            "[Apply] ON CONFLICT not supported, falling back to bulk_insert: %s",
                            upsert_err,
                        )
                        db.session.bulk_insert_mappings(
                            cast("Any", Assignment), new_assignments
                        )

                if update_assignments:
                    db.session.bulk_update_mappings(
                        cast("Any", Assignment), update_assignments
                    )
                    logger.info(
                        "[Apply] Bulk updated %d existing assignments",
                        len(update_assignments),
                    )
            else:
                logger.info(
                    "[Apply] No desired assignments to upsert (company_id=%s)",
                    company_id,
                )

        # ✅ Commit le savepoint interne (begin_nested)
        # La transaction principale sera commitée par apply_assignments()
        db.session.commit()

    except Exception:
        logger.exception(
            "[Apply] DB error while applying assignments (company_id=%s)", company_id
        )
        # Rollback du savepoint en cas d'erreur
        # La transaction principale sera rollbackée par apply_assignments()
        db.session.rollback()
        # ✅ FIX RC2: Expirer tous les objets après rollback pour forcer le rechargement
        db.session.expire_all()
        raise  # Propager l'erreur pour que apply_assignments() gère le rollback global
    if dispatch_run_id:
        logger.info(
            "[Apply] Linked %d assignments to dispatch_run_id=%s",
            len(desired_assignments),
            dispatch_run_id,
        )

    if not updates:
        logger.info(
            "[Apply] No booking updates (company_id=%s) - assignments/ETA refreshed only.",
            company_id,
        )

    result = {
        "applied": applied_ids,
        "skipped": skipped,
        "conflicts": conflicts,
        "driver_load": dict(driver_load),
    }

    if skipped:
        for skipped_id, reason in skipped.items():
            booking_obj = booking_map.get(skipped_id)
            scheduled_time = (
                getattr(booking_obj, "scheduled_time", None) if booking_obj else None
            )
            time_confirmed = (
                getattr(booking_obj, "time_confirmed", None) if booking_obj else None
            )
            is_return = getattr(booking_obj, "is_return", None) if booking_obj else None
            logger.warning(
                "[Apply] Skipped booking_id=%s reason=%s scheduled_time=%s time_confirmed=%s is_return=%s",
                skipped_id,
                reason,
                scheduled_time,
                time_confirmed,
                is_return,
            )

    # Optionnel : retourner les paires (booking_id, driver_id) si demandé
    if return_pairs:
        result["applied_pairs"] = applied_pairs

    logger.info(
        "[Apply] company=%s applied=%d skipped=%d conflicts=%d (reasons=%s)",
        company_id,
        len(applied_ids),
        len(skipped),
        len(conflicts),
        dict(skipped),
    )

    # 🔔 Notifications Socket.IO vers les chauffeurs pour MAJ en temps réel (mobile)
    try:
        if applied_pairs:
            notif_booking_ids = [b_id for b_id, _ in applied_pairs]

            # ⚠️ Utiliser une session indépendante pour éviter les transactions closes
            # Compatibilité Flask-SQLAlchemy : utiliser get_scoped_session helper
            session = _get_scoped_session(db)
            try:
                notif_bookings = {
                    b.id: b
                    for b in session.query(Booking)
                    .filter(Booking.id.in_(notif_booking_ids))
                    .all()
                }

                from services.notification_service import notify_driver_new_booking

                for b_id, d_id in applied_pairs:
                    try:
                        booking_obj = notif_bookings.get(b_id)
                        if booking_obj is None:
                            continue
                        notify_driver_new_booking(int(d_id), booking_obj)
                    except Exception:
                        logger.exception(
                            "[Apply] notify_driver_new_booking failed booking_id=%s driver_id=%s",
                            b_id,
                            d_id,
                        )
            finally:
                session.close()
    except Exception:
        logger.exception(
            "[Apply] driver notifications failed (company_id=%s)", company_id
        )
    return result
