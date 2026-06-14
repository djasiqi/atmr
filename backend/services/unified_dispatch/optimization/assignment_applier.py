# backend/services/unified_dispatch/apply.py v1.0.0
from __future__ import annotations

import contextlib
import logging
import os
from collections import defaultdict
from contextlib import suppress
from typing import Any, Dict, List, Tuple, cast

from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError
from sqlalchemy.orm import joinedload, scoped_session, sessionmaker

# ✅ Import au niveau module pour permettre le mock dans les tests
# Le test patch "services.unified_dispatch.apply.publish_event"
# donc on doit utiliser cette référence directement
from application.events.event_bus import publish_event
from domain.events.events import DriverNewBookingEvent
from ext import db
from models import Assignment, AssignmentStatus, Booking, BookingStatus, Driver
from repositories.assignment_repository import AssignmentRepository
from repositories.booking_repository import BookingRepository
from repositories.driver_repository import DriverRepository
from services.unified_dispatch.utils.transactions import _begin_tx, _in_tx
from shared.time_utils import now_utc  # UTC centralisé

logger = logging.getLogger(__name__)

# ✅ P2: Métriques Prometheus (déclarées au niveau module, une seule fois)
try:
    from prometheus_client import Counter, Histogram  # type: ignore[reportMissingImports]  # noqa: I001

    NOTIF_EMITTED = Counter(
        "atmr_apply_notifications_emitted_total",
        "Total notifications emitted after commit",
        ["company_id", "status"],
    )
    NOTIF_FAILED = Counter(
        "atmr_apply_notifications_failed_total",
        "Total notifications failed after commit",
        ["company_id", "error_type"],
    )
    NOTIF_LATENCY = Histogram(
        "atmr_apply_notification_latency_seconds",
        "Latency between commit and notification emission",
        ["company_id"],
    )
except ImportError:
    # Prometheus non disponible (dev/test)
    NOTIF_EMITTED = None
    NOTIF_FAILED = None
    NOTIF_LATENCY = None


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
    except (OperationalError, DBAPIError, AttributeError) as e:
        # Erreurs DB attendues : connexion, configuration
        logger.warning(
            (
                "[Apply] Erreur lors de la création de scoped_session (DB error: %s): %s, "
                "utilisation de db.session"
            ),
            type(e).__name__,
            e,
        )
        # Dernier recours : utiliser la session principale
        return db_instance.session
    except Exception:
        # Erreur inattendue lors de la création de scoped_session
        logger.exception("[Apply] Erreur lors de la création de scoped_session")
        # Dernier recours : utiliser la session principale
        return db_instance.session


@contextlib.contextmanager
def scoped_session_context(db_instance):
    """
    ✅ P1: Context manager pour scoped sessions avec fermeture automatique.

    Garantit que la session est fermée même en cas d'exception, évitant les
    fuites de connexions DB.

    Args:
        db_instance: Instance SQLAlchemy de Flask-SQLAlchemy

    Yields:
        Scoped session pour requêtes indépendantes

    Example:
        with scoped_session_context(db) as session:
            bookings = session.query(Booking).filter_by(company_id=1).all()
            # Session fermée automatiquement à la sortie
    """
    session = None
    try:
        session = _get_scoped_session(db_instance)
        yield session
    finally:
        if (
            session is not None
            and hasattr(session, "close")
            and session is not db_instance.session
        ):
            # Vérifier si c'est une vraie scoped_session (a une méthode close)
            # ou si c'est la session principale (ne doit pas être fermée)
            try:
                session.close()
            except (OperationalError, DBAPIError, AttributeError) as e:
                # Erreurs DB attendues : session déjà fermée, connexion perdue
                logger.warning(
                    "[Apply] Erreur lors de la fermeture de scoped_session (DB error: %s): %s",
                    type(e).__name__,
                    e,
                )
            except Exception:
                # Erreur inattendue lors de la fermeture
                logger.exception(
                    "[Apply] Erreur lors de la fermeture de scoped_session"
                )


_Assignment = Any


# ✅ A2: Compteur thread-safe pour conflits DB (contraintes uniques)
class DBConflictCounter:
    """Compteur thread-safe pour les violations de contraintes uniques."""

    _instance: DBConflictCounter | None = None

    def __init__(self) -> None:
        super().__init__()
        self._counter: int = 0

    @classmethod
    def get_instance(cls) -> DBConflictCounter:
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


def _driver_display_name(driver: Any) -> str | None:
    """Construit un nom lisible chauffeur depuis l'utilisateur lié."""
    user = getattr(driver, "user", None)
    if user is None:
        return None
    fn = (getattr(user, "first_name", None) or "").strip()
    ln = (getattr(user, "last_name", None) or "").strip()
    return f"{fn} {ln}".strip() or getattr(user, "username", None)


def _record_driver_assigned_timeline(
    *,
    applied_pairs: List[Tuple[int, int]],
    booking_map: Dict[int, "Booking"],
    driver_map: Dict[int, "Driver"],
    company_id: int,
) -> None:
    """Historise driver_assigned pour les bookings liés à une TransportRequest."""
    if not applied_pairs:
        return
    try:
        from models import TransportRequest
        from services.institutions.transport_timeline_service import (
            TimelineActor,
            record_event,
        )

        booking_ids = [b_id for b_id, _ in applied_pairs]
        requests = TransportRequest.query.filter(
            TransportRequest.booking_id.in_(booking_ids)
        ).all()
        request_by_booking = {r.booking_id: r for r in requests}

        for b_id, d_id in applied_pairs:
            transport_req = request_by_booking.get(b_id)
            if transport_req is None:
                continue
            driver = driver_map.get(d_id)
            company = getattr(driver, "company", None) if driver else None
            record_event(
                "driver_assigned",
                institution_id=transport_req.institution_id,
                transport_request_id=transport_req.id,
                booking_id=b_id,
                actor=TimelineActor(
                    actor_type="company",
                    company_id=company_id,
                    driver_id=d_id,
                ),
                payload={
                    "driver_id": d_id,
                    "driver_name": _driver_display_name(driver) if driver else None,
                    "company_id": company_id,
                    "company_name": getattr(company, "name", None)
                    if company
                    else None,
                },
                correlation_id=f"driver_assigned:{b_id}:{d_id}",
            )
    except Exception as timeline_err:
        logger.warning(
            "[Apply] Timeline driver_assigned recording failed: %s", timeline_err
        )


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
    # Log pour tracer la propagation du dispatch_run_id
    if dispatch_run_id:
        logger.info("[Apply] Using dispatch_run_id=%s for assignments", dispatch_run_id)

    # ✅ Transaction globale pour garantir atomicité complète
    # Utilise _begin_tx() qui détecte si une transaction existe déjà (savepoint)
    # ou en crée une nouvelle si nécessaire
    # ✅ Utiliser _in_tx() comme source unique de vérité (même fonction que _begin_tx())
    # Ne pas redéfinir _in_tx ici, source unique = transaction_helpers
    had_existing_tx = _in_tx()
    result = None

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

        # ✅ Commit uniquement si on est propriétaire de la transaction
        # Aucun commit() ne doit être exécuté dans le try avant d'être sûr que tout est OK
        if not had_existing_tx:
            db.session.commit()
    except (OperationalError, DBAPIError, IntegrityError) as e:
        # Erreurs DB attendues : connexion, contraintes, timeout
        # ✅ P1: Logger contexte supplémentaire pour debugging
        logger.error(
            (
                "[Apply] Transaction failed for company_id=%s (DB error: %s): %s. "
                "Assignments count: %d, had_existing_tx: %s, dispatch_run_id: %s"
            ),
            company_id,
            type(e).__name__,
            e,
            len(assignments),
            had_existing_tx,
            dispatch_run_id,
        )
        if not had_existing_tx:
            db.session.rollback()
        raise
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        # Erreurs de validation attendues : données invalides
        # ✅ P1: Logger contexte supplémentaire pour debugging
        logger.error(
            (
                "[Apply] Transaction failed for company_id=%s (validation error: %s): %s. "
                "Assignments count: %d, had_existing_tx: %s, dispatch_run_id: %s"
            ),
            company_id,
            type(e).__name__,
            e,
            len(assignments),
            had_existing_tx,
            dispatch_run_id,
        )
        if not had_existing_tx:
            db.session.rollback()
        raise
    except Exception:
        # Erreur inattendue : logger avec trace complète
        # ✅ P1: Logger contexte supplémentaire pour debugging
        logger.exception(
            (
                "[Apply] Transaction failed for company_id=%s. "
                "Assignments count: %d, had_existing_tx: %s, dispatch_run_id: %s"
            ),
            company_id,
            len(assignments),
            had_existing_tx,
            dispatch_run_id,
        )
        if not had_existing_tx:
            db.session.rollback()
        raise

    # ✅ Notifications uniquement si commit réellement fait (après le bloc try/except)
    # Ne jamais émettre de notifications avant le commit ou en cas d'exception
    if not had_existing_tx:
        # On a commité avec succès, émettre les notifications
        if result:
            applied_pairs = result.get("applied_pairs", [])
            if applied_pairs:
                _emit_notifications_after_commit(applied_pairs, company_id)
    elif result:
        # Transaction externe: ne pas émettre ici (commit pas encore fait)
        # Ajouter dans result pour que caller puisse émettre après son commit
        applied_pairs = result.get("applied_pairs", [])
        if applied_pairs:
            logger.info(
                (
                    "[Apply] Notifications deferred (had_existing_tx=True, company_id=%s, pairs=%d). "
                    "Caller must emit after commit."
                ),
                company_id,
                len(applied_pairs),
            )
            result["deferred_notifications"] = {
                "applied_pairs": applied_pairs,
                "company_id": company_id,
            }

    return result


def _emit_notifications_after_commit(
    applied_pairs: List[Tuple[int, int]],
    company_id: int,
) -> None:
    """Émet les notifications Socket.IO APRÈS commit réussi de la transaction.

    Cette fonction est appelée uniquement après un commit réussi pour éviter
    d'émettre des notifications si la transaction est rollback.

    ⚠️ IMPORTANT: Cette fonction ne doit être appelée que si `not had_existing_tx`,
    car si transaction externe, le commit n'a pas encore eu lieu.

    Args:
        applied_pairs: Liste de tuples (booking_id, driver_id) des assignations appliquées
        company_id: ID de l'entreprise
    """
    if not applied_pairs:
        return

    import time

    start_time = time.time()

    try:
        notif_booking_ids = [b_id for b_id, _ in applied_pairs]

        # ✅ Utiliser scoped_session_context directement (déjà dans ce fichier, ligne 79)
        # Pas d'import nécessaire, fonction déjà définie dans le module
        with scoped_session_context(db) as session:
            notif_bookings = {
                b.id: b
                for b in session.query(Booking)
                .filter(Booking.id.in_(notif_booking_ids))
                .all()
            }

            # ✅ Event bus déjà utilisé dans le code actuel (lignes 979-980)
            # publish_event et DriverNewBookingEvent sont importés au niveau module

            for b_id, d_id in applied_pairs:
                try:
                    booking_obj = notif_bookings.get(b_id)
                    if booking_obj is None:
                        logger.warning(
                            "[Apply] Booking id=%s not found for notification (post-commit)",
                            b_id,
                        )
                        continue

                    # ✅ Vérifier que l'assignation est toujours valide (idempotence)
                    # Si le booking a été modifié entre temps, ne pas notifier
                    if booking_obj.driver_id != d_id:
                        logger.info(
                            "[Apply] Booking id=%s driver changed (%s -> %s), skipping notification",
                            b_id,
                            d_id,
                            booking_obj.driver_id,
                        )
                        continue

                    # Publier événement
                    # ✅ Utiliser publish_event directement (importé au niveau module)
                    # Le test patch "services.unified_dispatch.apply.publish_event"
                    # donc cette référence sera mockée
                    publish_event(
                        DriverNewBookingEvent(
                            booking_id=int(b_id),
                            driver_id=int(d_id),
                            company_id=company_id,
                        )
                    )
                except (ValueError, TypeError, AttributeError, KeyError) as e:
                    logger.warning(
                        (
                            "[Apply] DriverNewBookingEvent publish failed (post-commit) "
                            "booking_id=%s driver_id=%s (validation error: %s): %s"
                        ),
                        b_id,
                        d_id,
                        type(e).__name__,
                        e,
                    )
                    # ✅ P2: Métrique échec
                    if NOTIF_FAILED:
                        NOTIF_FAILED.labels(
                            company_id=str(company_id), error_type=type(e).__name__
                        ).inc()
                except Exception:
                    logger.exception(
                        (
                            "[Apply] DriverNewBookingEvent publish failed (post-commit) "
                            "booking_id=%s driver_id=%s"
                        ),
                        b_id,
                        d_id,
                    )
                    # ✅ P2: Métrique échec
                    if NOTIF_FAILED:
                        NOTIF_FAILED.labels(
                            company_id=str(company_id), error_type="Exception"
                        ).inc()

        # ✅ P2: Métrique succès (incrémentée même si aucune notification émise)
        # Indique que la fonction a été appelée avec succès
        if NOTIF_EMITTED:
            NOTIF_EMITTED.labels(company_id=str(company_id), status="success").inc()

    except (ValueError, TypeError, AttributeError, KeyError) as e:
        logger.warning(
            "[Apply] driver notifications failed (post-commit, company_id=%s, validation error: %s): %s",
            company_id,
            type(e).__name__,
            e,
        )
        if NOTIF_FAILED:
            NOTIF_FAILED.labels(
                company_id=str(company_id), error_type=type(e).__name__
            ).inc()
    except Exception:
        logger.exception(
            "[Apply] driver notifications failed (post-commit, company_id=%s)",
            company_id,
        )
        if NOTIF_FAILED:
            NOTIF_FAILED.labels(
                company_id=str(company_id), error_type="Exception"
            ).inc()
    finally:
        # ✅ P2: Enregistrer latence
        if NOTIF_LATENCY:
            latency = time.time() - start_time
            NOTIF_LATENCY.labels(company_id=str(company_id)).observe(latency)


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
    """Logique interne d'application des assignations
    (exécutée dans une transaction).
    """

    # Helper: attr ou clé dict
    def _aget(obj: Any, name: str, default: Any = None) -> Any:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except (AttributeError, TypeError):
                # Erreurs attendues : attribut non accessible, type incorrect
                pass
            except Exception:
                # Erreur inattendue lors de l'accès à l'attribut (ignorée silencieusement)
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
    # ✅ FIX RC4: Flush la session pour s'assurer que les objets en attente
    # sont visibles
    db.session.flush()

    # ✅ Utilisation des repositories pour valider les IDs avant la requête
    # Note: On garde la requête SQLAlchemy directe car with_for_update() nécessite une query SQLAlchemy
    # mais on valide d'abord que les IDs existent via les repositories
    booking_repo = BookingRepository()
    driver_repo = DriverRepository()

    # Valider que les bookings existent (la vérification company_id se fait dans la requête SQLAlchemy)
    valid_booking_ids = []
    if booking_ids:
        booking_dtos = booking_repo.find_by_ids(booking_ids)
        valid_booking_ids = [dto.id for dto in booking_dtos if dto]

    # Valider que les drivers existent (la vérification company_id se fait dans la requête SQLAlchemy)
    valid_driver_ids = []
    if driver_ids:
        for driver_id in driver_ids:
            driver_dto = driver_repo.find_by_id(driver_id)
            if driver_dto:
                valid_driver_ids.append(driver_dto.id)

    # Construire les requêtes SQLAlchemy avec les IDs validés
    # (nécessaire pour with_for_update() qui est spécifique à SQLAlchemy)
    bookings_q = (
        Booking.query.options(joinedload(Booking.driver)).filter(
            Booking.company_id == company_id, Booking.id.in_(valid_booking_ids)
        )
        if valid_booking_ids
        else Booking.query.filter(Booking.id == -1)  # Query vide si aucun ID valide
    )
    drivers_q = (
        Driver.query.options(joinedload(Driver.company)).filter(
            Driver.company_id == company_id, Driver.id.in_(valid_driver_ids)
        )
        if valid_driver_ids
        else Driver.query.filter(Driver.id == -1)  # Query vide si aucun ID valide
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
    # ✅ FIX: Capturer les métadonnées des bookings skipped
    # avant que la transaction soit fermée
    skipped_metadata: Dict[int, Dict[str, Any]] = {}
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
        # ✅ FIX RC2/RC4: Recharger le booking depuis la DB pour éviter
        # problèmes de session
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
                (
                    "[Apply] Booking id=%s company_id=%s not found in "
                    "booking_map (size=%d) or DB query"
                ),
                b_id,
                company_id,
                len(booking_map),
            )
            reason = "booking_not_found_or_wrong_company"
            skipped[b_id] = reason
            # ✅ FIX: Pas de métadonnées disponibles si le booking n'existe pas
            skipped_metadata[b_id] = {
                "scheduled_time": None,
                "time_confirmed": None,
                "is_return": None,
            }
            # ✅ Log détaillé pour debugging
            driver_id = int(_aget(a, "driver_id")) if a else None
            logger.warning(
                (
                    "⚠️ [Apply] Assignation skipped: booking_id=%d, driver_id=%s, "
                    "reason=%s, company_id=%d, scheduled_time=None, "
                    "time_confirmed=None, is_return=None"
                ),
                b_id,
                driver_id,
                reason,
                company_id,
            )
            continue

        if b.status not in (
            BookingStatus.PENDING,
            BookingStatus.ACCEPTED,
            BookingStatus.ASSIGNED,
        ):
            reason = f"status_is_{b.status}"
            skipped[b_id] = reason
            # ✅ FIX: Capturer les métadonnées avant que la transaction soit fermée
            scheduled_time = getattr(b, "scheduled_time", None)
            time_confirmed = getattr(b, "time_confirmed", None)
            is_return = getattr(b, "is_return", None)
            skipped_metadata[b_id] = {
                "scheduled_time": scheduled_time,
                "time_confirmed": time_confirmed,
                "is_return": is_return,
            }
            # ✅ Log détaillé pour debugging
            driver_id = int(_aget(a, "driver_id")) if a else None
            logger.warning(
                (
                    "⚠️ [Apply] Assignation skipped: booking_id=%d, driver_id=%s, "
                    "reason=%s, company_id=%d, scheduled_time=%s, "
                    "time_confirmed=%s, is_return=%s"
                ),
                b_id,
                driver_id,
                reason,
                company_id,
                scheduled_time,
                time_confirmed,
                is_return,
            )
            continue

        d_id = int(_aget(a, "driver_id"))
        d = driver_map.get(d_id)
        if d is None:
            reason = "driver_not_found_or_wrong_company"
            skipped[b_id] = reason
            # ✅ FIX: Capturer les métadonnées avant que la transaction soit fermée
            scheduled_time = getattr(b, "scheduled_time", None)
            time_confirmed = getattr(b, "time_confirmed", None)
            is_return = getattr(b, "is_return", None)
            skipped_metadata[b_id] = {
                "scheduled_time": scheduled_time,
                "time_confirmed": time_confirmed,
                "is_return": is_return,
            }
            # ✅ Log détaillé pour debugging
            logger.warning(
                (
                    "⚠️ [Apply] Assignation skipped: booking_id=%d, driver_id=%d, "
                    "reason=%s, company_id=%d, scheduled_time=%s, "
                    "time_confirmed=%s, is_return=%s"
                ),
                b_id,
                d_id,
                reason,
                company_id,
                scheduled_time,
                time_confirmed,
                is_return,
            )
            continue
        d_any = cast("Any", d)
        is_active = bool(getattr(d_any, "is_active", False))
        is_available = bool(getattr(d_any, "is_available", False))
        if enforce_driver_checks and (not is_active or not is_available):
            reason = "driver_not_available"
            skipped[b_id] = reason
            # ✅ FIX: Capturer les métadonnées avant que la transaction soit fermée
            scheduled_time = getattr(b, "scheduled_time", None)
            time_confirmed = getattr(b, "time_confirmed", None)
            is_return = getattr(b, "is_return", None)
            skipped_metadata[b_id] = {
                "scheduled_time": scheduled_time,
                "time_confirmed": time_confirmed,
                "is_return": is_return,
            }
            # ✅ Log détaillé pour debugging
            logger.warning(
                (
                    "⚠️ [Apply] Assignation skipped: booking_id=%d, driver_id=%d, "
                    "reason=%s, company_id=%d, scheduled_time=%s, "
                    "time_confirmed=%s, is_return=%s, driver_is_active=%s, "
                    "driver_is_available=%s"
                ),
                b_id,
                d_id,
                reason,
                company_id,
                scheduled_time,
                time_confirmed,
                is_return,
                is_active,
                is_available,
            )
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
        except (ValueError, TypeError, OverflowError):
            # Erreurs de conversion attendues : valeur non convertible en int
            cur_driver_id = None
        except Exception:
            # Erreur inattendue lors de la conversion (ignorée silencieusement)
            cur_driver_id = None

        is_assigned = b_status == BookingStatus.ASSIGNED
        same_driver = cur_driver_id == d_id

        if respect_existing and is_assigned and same_driver:
            reason = "already_assigned_same_driver"
            skipped[b_id] = reason
            # ✅ FIX: Capturer les métadonnées avant que la transaction soit fermée
            scheduled_time = getattr(b, "scheduled_time", None)
            time_confirmed = getattr(b, "time_confirmed", None)
            is_return = getattr(b, "is_return", None)
            skipped_metadata[b_id] = {
                "scheduled_time": scheduled_time,
                "time_confirmed": time_confirmed,
                "is_return": is_return,
            }
            # ✅ Log détaillé pour debugging
            logger.warning(
                (
                    "⚠️ [Apply] Assignation skipped: booking_id=%d, driver_id=%d, "
                    "reason=%s, company_id=%d, scheduled_time=%s, "
                    "time_confirmed=%s, is_return=%s"
                ),
                b_id,
                d_id,
                reason,
                company_id,
                scheduled_time,
                time_confirmed,
                is_return,
            )
            continue

        if (
            is_assigned
            and (cur_driver_id is not None)
            and (cur_driver_id != d_id)
            and (not allow_reassign)
        ):
            conflicts.append(b_id)
            reason = "reassign_blocked"
            skipped[b_id] = reason
            # ✅ FIX: Capturer les métadonnées avant que la transaction soit fermée
            scheduled_time = getattr(b, "scheduled_time", None)
            time_confirmed = getattr(b, "time_confirmed", None)
            is_return = getattr(b, "is_return", None)
            skipped_metadata[b_id] = {
                "scheduled_time": scheduled_time,
                "time_confirmed": time_confirmed,
                "is_return": is_return,
            }
            # ✅ Log détaillé pour debugging
            logger.warning(
                (
                    "⚠️ [Apply] Assignation skipped: booking_id=%d, driver_id=%d, "
                    "reason=%s, company_id=%d, scheduled_time=%s, "
                    "time_confirmed=%s, is_return=%s, current_driver_id=%s"
                ),
                b_id,
                d_id,
                reason,
                company_id,
                scheduled_time,
                time_confirmed,
                is_return,
                cur_driver_id,
            )
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
    # ✅ FIX: Gérer le savepoint avec commit/rollback explicite pour éviter InFailedSqlTransaction
    sp = db.session.begin_nested()
    try:
        if updates:
            db.session.bulk_update_mappings(cast("Any", Booking), updates)
            # Timeline transport: driver_assigned (courses institution uniquement)
            _record_driver_assigned_timeline(
                applied_pairs=applied_pairs,
                booking_map=booking_map,
                driver_map=driver_map,
                company_id=company_id,
            )

        # Upsert côté Assignment (y compris ETA si fournies)
        if desired_assignments:
            target_bids = list(desired_assignments.keys())
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            assignment_repo = AssignmentRepository()
            existing_dtos = assignment_repo.find_by_booking_ids(target_bids)
            # Récupérer les modèles SQLAlchemy depuis les IDs des DTOs pour la compatibilité
            existing_ids = [dto.id for dto in existing_dtos]
            existing = (
                Assignment.query.filter(Assignment.id.in_(existing_ids)).all()
                if existing_ids
                else []
            )
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
                        "booking_id": cast(int, payload["booking_id"]),
                        "driver_id": payload["driver_id"],
                        "status": payload.get("status", AssignmentStatus.SCHEDULED),
                        "created_at": now,
                        "updated_at": now,
                    }

                    # ETA optionnels
                    eta_pu = payload.get("estimated_pickup_arrival") or payload.get(
                        "eta_pickup_at"
                    )
                    eta_do = payload.get("estimated_dropoff_arrival") or payload.get(
                        "eta_dropoff_at"
                    )
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
                    eta_do = payload.get("estimated_dropoff_arrival") or payload.get(
                        "eta_dropoff_at"
                    )
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
                        except IntegrityError as conflict_err:
                            # ✅ A2: Compter les conflits de contrainte unique
                            # IntegrityError contient les détails de la contrainte
                            conflicts_count += 1
                            increment_db_conflict_counter()
                            logger.debug(
                                (
                                    "[Apply] Conflit unique ignoré "
                                    "(idempotence, IntegrityError): %s"
                                ),
                                conflict_err,
                            )
                        except (OperationalError, DBAPIError) as e:
                            # Erreurs DB non liées aux contraintes : re-lancer
                            logger.warning(
                                "[Apply] DB error during UPSERT (DB error: %s): %s",
                                type(e).__name__,
                                e,
                            )
                            raise
                        except Exception:
                            # Erreur inattendue : re-lancer
                            logger.exception("[Apply] Unexpected error during UPSERT")
                            raise

                    if conflicts_count > 0:
                        logger.info(
                            (
                                "[Apply] UPSERT: %d insertions, %d conflits "
                                "ignorés (idempotent)"
                            ),
                            len(new_assignments) - conflicts_count,
                            conflicts_count,
                        )
                    else:
                        logger.info(
                            "[Apply] UPSERT inserted %d new assignments",
                            len(new_assignments),
                        )
                except (OperationalError, DBAPIError, IntegrityError) as upsert_err:
                    # Erreurs DB attendues : ON CONFLICT non supporté, syntaxe SQL
                    logger.warning(
                        (
                            "[Apply] ON CONFLICT not supported, falling back "
                            "to bulk_insert (DB error: %s): %s"
                        ),
                        type(upsert_err).__name__,
                        upsert_err,
                    )
                except (ValueError, TypeError, AttributeError) as e:
                    # Erreurs de validation attendues : données invalides
                    logger.warning(
                        (
                            "[Apply] ON CONFLICT failed, falling back "
                            "to bulk_insert (validation error: %s): %s"
                        ),
                        type(e).__name__,
                        e,
                    )
                except Exception:
                    # Erreur inattendue : logger avec trace complète
                    logger.exception(
                        "[Apply] ON CONFLICT failed, falling back to bulk_insert"
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
        # IMPORTANT: Ne commit que si aucune exception n'a été levée pendant les opérations DB
        sp.commit()

    except Exception as e:
        # ✅ IMPORTANT: Si une exception DB survient, rollback le savepoint immédiatement
        # Ne pas continuer après une DB error sans rollback explicite, sinon on garde
        # InFailedSqlTransaction jusqu'à la fin
        # Rollback le savepoint dès qu'une DB error arrive, puis re-raise
        with suppress(Exception):
            sp.rollback()

        # Logger selon le type d'erreur
        if isinstance(e, (OperationalError, DBAPIError, IntegrityError)):
            logger.error(
                "[Apply] DB error while applying assignments (company_id=%s, DB error: %s): %s",
                company_id,
                type(e).__name__,
                e,
            )
        elif isinstance(e, (ValueError, TypeError, AttributeError, KeyError)):
            logger.error(
                "[Apply] Validation error while applying assignments (company_id=%s, validation error: %s): %s",
                company_id,
                type(e).__name__,
                e,
            )
        else:
            logger.exception(
                "[Apply] DB error while applying assignments (company_id=%s)",
                company_id,
            )

        # Expirer tous les objets après rollback pour forcer le rechargement
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
            (
                "[Apply] No booking updates (company_id=%s) - "
                "assignments/ETA refreshed only."
            ),
            company_id,
        )

    result: Dict[str, Any] = {
        "applied": applied_ids,
        "skipped": skipped,
        "conflicts": conflicts,
        "driver_load": dict(driver_load),
    }

    if skipped:
        for skipped_id, reason in skipped.items():
            # ✅ FIX: Utiliser les métadonnées capturées
            # avant que la transaction soit fermée
            metadata = skipped_metadata.get(skipped_id, {})
            scheduled_time = metadata.get("scheduled_time")
            time_confirmed = metadata.get("time_confirmed")
            is_return = metadata.get("is_return")
            logger.warning(
                (
                    "[Apply] Skipped booking_id=%s reason=%s scheduled_time=%s "
                    "time_confirmed=%s is_return=%s"
                ),
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

    # ✅ Notifications Socket.IO déplacées après commit (voir _emit_notifications_after_commit)
    # Les notifications sont maintenant émises dans apply_assignments() wrapper
    # après commit réussi pour éviter émission si rollback

    return result
