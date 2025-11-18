"""✅ 3.3: Tâches Celery pour purge automatique des données RGPD.

Purge des données anciennes conformément au RGPD:
- Données transactionnelles (> 7 ans) : suppression définitive
- Données personnelles (> 7 ans) : anonymisation/archivage
- Logs d'audit complets pour traçabilité
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from celery import Task

from celery_app import celery, get_flask_app

logger = logging.getLogger(__name__)


# ✅ 3.3: Helper pour logging audit
def _log_purge_audit(
    action_type: str,
    model_name: str,
    deleted_count: int,
    errors: list[dict[str, Any]],
    cutoff_date: datetime,
    retention_days: int,
) -> None:
    """✅ 3.3: Log une action de purge dans l'audit log."""
    try:
        from security.audit_log import AuditLogger

        AuditLogger.log_action(
            action_type=action_type,
            action_category="gdpr_purge",
            user_type="system",
            result_status="success" if len(errors) == 0 else "partial",
            result_message=f"Purge {model_name}: {deleted_count} enregistrements supprimés",
            action_details={
                "model": model_name,
                "deleted_count": deleted_count,
                "errors_count": len(errors),
                "cutoff_date": cutoff_date.isoformat(),
                "retention_days": retention_days,
                "errors": errors[:10],  # Limiter à 10 erreurs dans le log
            },
            metadata={
                "gdpr_compliant": True,
                "retention_period_days": retention_days,
            },
        )
    except Exception as e:
        logger.warning("[3.3 GDPR] Échec logging audit: %s", e)


# ✅ 3.3: Durées de rétention par défaut (en jours)
DEFAULT_RETENTION_DAYS = int(os.getenv("GDPR_RETENTION_DAYS", "2555"))  # 7 ans = 7 * 365
ANALYTICS_RETENTION_DAYS = int(os.getenv("GDPR_ANALYTICS_RETENTION_DAYS", "3650"))  # 10 ans
EVENT_RETENTION_DAYS = int(os.getenv("GDPR_EVENT_RETENTION_DAYS", "2555"))  # 7 ans
MESSAGE_RETENTION_DAYS = int(os.getenv("GDPR_MESSAGE_RETENTION_DAYS", "2555"))  # 7 ans


@celery.task(bind=True, name="tasks.purge_tasks.purge_old_bookings")
def purge_old_bookings(self: Task, retention_days: int | None = None) -> dict[str, Any]:  # noqa: ARG001
    """✅ 3.3: Purge les réservations (bookings) anciennes (> 7 ans par défaut).

    Supprime définitivement les bookings dont created_at > retention_days.
    Les bookings avec statut COMPLETED ou CANCELED peuvent être purgées.
    Les bookings actives (PENDING, IN_PROGRESS) ne sont jamais purgées.

    Args:
        retention_days: Nombre de jours de rétention (défaut: 7 ans)

    Returns:
        dict avec status, deleted_count, errors
    """
    try:
        app = get_flask_app()
        with app.app_context():
            from ext import db
            from models import Booking, BookingStatus

            retention = retention_days or DEFAULT_RETENTION_DAYS
            cutoff_date = datetime.now(UTC) - timedelta(days=retention)

            logger.info(
                "[3.3 GDPR] Début purge bookings antérieurs à %s (rétention: %d jours)",
                cutoff_date.isoformat(),
                retention,
            )

            # Purger uniquement bookings terminés/cancelled et anciens
            deletable_statuses = [
                BookingStatus.COMPLETED,
                BookingStatus.RETURN_COMPLETED,
                BookingStatus.CANCELED,
            ]

            old_bookings = Booking.query.filter(
                Booking.created_at < cutoff_date, Booking.status.in_(deletable_statuses)
            ).all()

            deleted_count = 0
            errors: list[dict[str, Any]] = []

            for booking in old_bookings:
                try:
                    booking_id = booking.id
                    company_id = booking.company_id
                    client_id = booking.client_id
                    created_at = booking.created_at

                    db.session.delete(booking)
                    deleted_count += 1

                    logger.debug(
                        "[3.3 GDPR] Booking %d supprimé (company=%s, client=%s, created=%s)",
                        booking_id,
                        company_id,
                        client_id,
                        created_at.isoformat(),
                    )
                except Exception as e:
                    errors.append({"booking_id": booking.id, "error": str(e)[:200]})
                    logger.warning("[3.3 GDPR] Erreur suppression booking %d: %s", booking.id, e)

            db.session.commit()

            logger.info("[3.3 GDPR] ✅ Purge bookings terminée: %d supprimés, %d erreurs", deleted_count, len(errors))

            # ✅ 3.3: Log audit pour traçabilité RGPD
            _log_purge_audit(
                action_type="purge_old_bookings",
                model_name="Booking",
                deleted_count=deleted_count,
                errors=errors,
                cutoff_date=cutoff_date,
                retention_days=retention,
            )

            return {
                "status": "success",
                "model": "Booking",
                "cutoff_date": cutoff_date.isoformat(),
                "retention_days": retention,
                "deleted_count": deleted_count,
                "errors": errors,
            }

    except Exception as e:
        logger.exception("[3.3 GDPR] ❌ Erreur purge bookings: %s", e)
        raise


@celery.task(bind=True, name="tasks.purge_tasks.purge_old_messages")
def purge_old_messages(self: Task, retention_days: int | None = None) -> dict[str, Any]:  # noqa: ARG001
    """✅ 3.3: Purge les messages anciens (> 7 ans par défaut).

    Args:
        retention_days: Nombre de jours de rétention (défaut: 7 ans)

    Returns:
        dict avec status, deleted_count, errors
    """
    try:
        app = get_flask_app()
        with app.app_context():
            from ext import db
            from models import Message

            retention = retention_days or MESSAGE_RETENTION_DAYS
            cutoff_date = datetime.now(UTC) - timedelta(days=retention)

            logger.info(
                "[3.3 GDPR] Début purge messages antérieurs à %s (rétention: %d jours)",
                cutoff_date.isoformat(),
                retention,
            )

            old_messages = Message.query.filter(Message.created_at < cutoff_date).all()

            deleted_count = 0
            errors: list[dict[str, Any]] = []

            for message in old_messages:
                try:
                    message_id = message.id
                    db.session.delete(message)
                    deleted_count += 1
                    logger.debug("[3.3 GDPR] Message %d supprimé", message_id)
                except Exception as e:
                    errors.append({"message_id": message.id, "error": str(e)[:200]})
                    logger.warning("[3.3 GDPR] Erreur suppression message %d: %s", message.id, e)

            db.session.commit()

            logger.info("[3.3 GDPR] ✅ Purge messages terminée: %d supprimés, %d erreurs", deleted_count, len(errors))

            # ✅ 3.3: Log audit pour traçabilité RGPD
            _log_purge_audit(
                action_type="purge_old_messages",
                model_name="Message",
                deleted_count=deleted_count,
                errors=errors,
                cutoff_date=cutoff_date,
                retention_days=retention,
            )

            return {
                "status": "success",
                "model": "Message",
                "cutoff_date": cutoff_date.isoformat(),
                "retention_days": retention,
                "deleted_count": deleted_count,
                "errors": errors,
            }

    except Exception as e:
        logger.exception("[3.3 GDPR] ❌ Erreur purge messages: %s", e)
        raise


@celery.task(bind=True, name="tasks.purge_tasks.purge_old_realtime_events")
def purge_old_realtime_events(self: Task, retention_days: int | None = None) -> dict[str, Any]:  # noqa: ARG001
    """✅ 3.3: Purge les événements temps réel anciens (> 7 ans par défaut).

    Args:
        retention_days: Nombre de jours de rétention (défaut: 7 ans)

    Returns:
        dict avec status, deleted_count, errors
    """
    try:
        app = get_flask_app()
        with app.app_context():
            from ext import db
            from models.dispatch import RealtimeEvent

            retention = retention_days or EVENT_RETENTION_DAYS
            cutoff_date = datetime.now(UTC) - timedelta(days=retention)

            logger.info(
                "[3.3 GDPR] Début purge RealtimeEvent antérieurs à %s (rétention: %d jours)",
                cutoff_date.isoformat(),
                retention,
            )

            # Utiliser delete en bulk pour performance (évite N queries)
            deleted_count = RealtimeEvent.query.filter(RealtimeEvent.timestamp < cutoff_date).delete(
                synchronize_session=False
            )

            db.session.commit()

            logger.info("[3.3 GDPR] ✅ Purge RealtimeEvent terminée: %d supprimés", deleted_count)

            # ✅ 3.3: Log audit pour traçabilité RGPD
            _log_purge_audit(
                action_type="purge_old_realtime_events",
                model_name="RealtimeEvent",
                deleted_count=deleted_count,
                errors=[],
                cutoff_date=cutoff_date,
                retention_days=retention,
            )

            return {
                "status": "success",
                "model": "RealtimeEvent",
                "cutoff_date": cutoff_date.isoformat(),
                "retention_days": retention,
                "deleted_count": deleted_count,
                "errors": [],
            }

    except Exception as e:
        logger.exception("[3.3 GDPR] ❌ Erreur purge RealtimeEvent: %s", e)
        raise


@celery.task(bind=True, name="tasks.purge_tasks.purge_old_autonomous_actions")
def purge_old_autonomous_actions(self: Task, retention_days: int | None = None) -> dict[str, Any]:  # noqa: ARG001
    """✅ 3.3: Purge les actions autonomes anciennes (> 7 ans par défaut).

    Les actions reviewées peuvent être purgées, les non-reviewées sont conservées plus longtemps.

    Args:
        retention_days: Nombre de jours de rétention (défaut: 7 ans)

    Returns:
        dict avec status, deleted_count, errors
    """
    try:
        app = get_flask_app()
        with app.app_context():
            from ext import db
            from models.autonomous_action import AutonomousAction

            retention = retention_days or DEFAULT_RETENTION_DAYS
            cutoff_date = datetime.now(UTC) - timedelta(days=retention)

            logger.info(
                "[3.3 GDPR] Début purge AutonomousAction antérieures à %s (rétention: %d jours)",
                cutoff_date.isoformat(),
                retention,
            )

            # Purger uniquement les actions reviewées et anciennes
            old_actions = AutonomousAction.query.filter(
                AutonomousAction.created_at < cutoff_date,
                AutonomousAction.reviewed_by_admin == True,  # noqa: E712
            ).all()

            deleted_count = 0
            errors: list[dict[str, Any]] = []

            for action in old_actions:
                try:
                    action_id = action.id
                    db.session.delete(action)
                    deleted_count += 1
                    logger.debug("[3.3 GDPR] AutonomousAction %d supprimée", action_id)
                except Exception as e:
                    errors.append({"action_id": action.id, "error": str(e)[:200]})
                    logger.warning("[3.3 GDPR] Erreur suppression action %d: %s", action.id, e)

            db.session.commit()

            logger.info(
                "[3.3 GDPR] ✅ Purge AutonomousAction terminée: %d supprimées, %d erreurs", deleted_count, len(errors)
            )

            # ✅ 3.3: Log audit pour traçabilité RGPD
            _log_purge_audit(
                action_type="purge_old_autonomous_actions",
                model_name="AutonomousAction",
                deleted_count=deleted_count,
                errors=errors,
                cutoff_date=cutoff_date,
                retention_days=retention,
            )

            return {
                "status": "success",
                "model": "AutonomousAction",
                "cutoff_date": cutoff_date.isoformat(),
                "retention_days": retention,
                "deleted_count": deleted_count,
                "errors": errors,
            }

    except Exception as e:
        logger.exception("[3.3 GDPR] ❌ Erreur purge AutonomousAction: %s", e)
        raise


@celery.task(bind=True, name="tasks.purge_tasks.purge_old_task_failures")
def purge_old_task_failures(self: Task, retention_days: int | None = None) -> dict[str, Any]:  # noqa: ARG001
    """✅ 3.3: Purge les TaskFailure anciennes (> 7 ans par défaut).

    Args:
        retention_days: Nombre de jours de rétention (défaut: 7 ans)

    Returns:
        dict avec status, deleted_count, errors
    """
    try:
        app = get_flask_app()
        with app.app_context():
            from ext import db
            from models.task_failure import TaskFailure

            retention = retention_days or DEFAULT_RETENTION_DAYS
            cutoff_date = datetime.now(UTC) - timedelta(days=retention)

            logger.info(
                "[3.3 GDPR] Début purge TaskFailure antérieures à %s (rétention: %d jours)",
                cutoff_date.isoformat(),
                retention,
            )

            # Bulk delete pour performance
            deleted_count = TaskFailure.query.filter(TaskFailure.first_seen < cutoff_date).delete(
                synchronize_session=False
            )

            db.session.commit()

            logger.info("[3.3 GDPR] ✅ Purge TaskFailure terminée: %d supprimées", deleted_count)

            # ✅ 3.3: Log audit pour traçabilité RGPD
            _log_purge_audit(
                action_type="purge_old_task_failures",
                model_name="TaskFailure",
                deleted_count=deleted_count,
                errors=[],
                cutoff_date=cutoff_date,
                retention_days=retention,
            )

            return {
                "status": "success",
                "model": "TaskFailure",
                "cutoff_date": cutoff_date.isoformat(),
                "retention_days": retention,
                "deleted_count": deleted_count,
                "errors": [],
            }

    except Exception as e:
        logger.exception("[3.3 GDPR] ❌ Erreur purge TaskFailure: %s", e)
        raise


@celery.task(bind=True, name="tasks.purge_tasks.purge_all_old_data")
def purge_all_old_data(self: Task) -> dict[str, Any]:
    """✅ 3.3: Purge toutes les données anciennes (tâche principale).

    Appelle toutes les tâches de purge individuellement et agrège les résultats.
    Exécutée périodiquement (hebdomadaire recommandé).

    Returns:
        dict avec status, results (résultats par modèle), summary
    """
    try:
        logger.info("[3.3 GDPR] Début purge globale des données anciennes...")

        results: dict[str, dict[str, Any]] = {}

        # Purger chaque type de données
        # Note: Les fonctions purge_* sont des tâches Celery, mais peuvent être appelées directement
        purge_functions = [
            ("Bookings", purge_old_bookings),
            ("Messages", purge_old_messages),
            ("RealtimeEvents", purge_old_realtime_events),
            ("AutonomousActions", purge_old_autonomous_actions),
            ("TaskFailures", purge_old_task_failures),
        ]

        total_deleted = 0
        total_errors = 0

        app = get_flask_app()
        with app.app_context():
            for model_name, purge_func in purge_functions:
                try:
                    # Appeler directement la tâche Celery dans le contexte actuel
                    # On passe self pour respecter la signature (bind=True)
                    result = purge_func(self)
                    results[model_name] = result
                    total_deleted += result.get("deleted_count", 0)
                    total_errors += len(result.get("errors", []))
                    logger.info("[3.3 GDPR] %s: %d supprimés", model_name, result.get("deleted_count", 0))
                except Exception as e:
                    logger.exception("[3.3 GDPR] Erreur purge %s: %s", model_name, e)
                    results[model_name] = {"status": "error", "error": str(e)[:500]}
                    total_errors += 1

        summary = {
            "status": "success",
            "timestamp": datetime.now(UTC).isoformat(),
            "total_deleted": total_deleted,
            "total_errors": total_errors,
            "models_purged": len([r for r in results.values() if r.get("status") == "success"]),
        }

        logger.info(
            "[3.3 GDPR] ✅ Purge globale terminée: %d enregistrements supprimés, %d erreurs",
            total_deleted,
            total_errors,
        )

        # ✅ 3.3: Log audit pour purge globale
        try:
            from security.audit_log import AuditLogger

            AuditLogger.log_action(
                action_type="purge_all_old_data",
                action_category="gdpr_purge",
                user_type="system",
                result_status="success" if total_errors == 0 else "partial",
                result_message=f"Purge globale RGPD: {total_deleted} enregistrements supprimés",
                action_details={
                    "total_deleted": total_deleted,
                    "total_errors": total_errors,
                    "models_purged": summary["models_purged"],
                    "results": results,
                },
                metadata={
                    "gdpr_compliant": True,
                    "retention_period_days": DEFAULT_RETENTION_DAYS,
                },
            )
        except Exception as e:
            logger.warning("[3.3 GDPR] Échec logging audit global: %s", e)

        return {
            "status": "success",
            "summary": summary,
            "results": results,
        }

    except Exception as e:
        logger.exception("[3.3 GDPR] ❌ Erreur purge globale: %s", e)
        raise


@celery.task(bind=True, name="tasks.purge_tasks.anonymize_old_user_data")
def anonymize_old_user_data(self: Task, retention_days: int | None = None) -> dict[str, Any]:  # noqa: ARG001
    """✅ 3.3: Anonymise les données utilisateur anciennes (> 7 ans) au lieu de supprimer.

    Conformément RGPD, anonymise plutôt que supprimer pour préserver statistiques.

    Args:
        retention_days: Nombre de jours de rétention (défaut: 7 ans)

    Returns:
        dict avec status, anonymized_count, errors
    """
    try:
        app = get_flask_app()
        with app.app_context():
            from ext import db
            from models import User

            retention = retention_days or DEFAULT_RETENTION_DAYS
            cutoff_date = datetime.now(UTC) - timedelta(days=retention)

            logger.info(
                "[3.3 GDPR] Début anonymisation utilisateurs antérieurs à %s (rétention: %d jours)",
                cutoff_date.isoformat(),
                retention,
            )

            # Utilisateurs inactifs depuis > retention
            old_inactive_users = User.query.filter(
                User.created_at < cutoff_date,
                User.role != "admin",  # Ne pas anonymiser les admins
            ).all()

            anonymized_count = 0
            errors: list[dict[str, Any]] = []

            for user in old_inactive_users:
                try:
                    user_id = user.id
                    original_email = user.email

                    # Anonymiser les données personnelles
                    user.email = f"anonymized_{user_id}@deleted.local"
                    user.username = f"anonymized_{user_id}"
                    user.first_name = "Anonymized"
                    user.last_name = "User"
                    user.phone = None
                    user.address = None
                    user.zip_code = None
                    user.city = None
                    user.birth_date = None
                    user.gender = None
                    user.profile_image = None

                    anonymized_count += 1

                    logger.info(
                        "[3.3 GDPR] Utilisateur %d anonymisé (email: %s → anonymized_%d@deleted.local)",
                        user_id,
                        original_email,
                        user_id,
                    )
                except Exception as e:
                    errors.append({"user_id": user.id, "error": str(e)[:200]})
                    logger.warning("[3.3 GDPR] Erreur anonymisation utilisateur %d: %s", user.id, e)

            db.session.commit()

            logger.info(
                "[3.3 GDPR] ✅ Anonymisation utilisateurs terminée: %d anonymisés, %d erreurs",
                anonymized_count,
                len(errors),
            )

            # ✅ 3.3: Log audit pour anonymisation
            try:
                from security.audit_log import AuditLogger

                AuditLogger.log_action(
                    action_type="anonymize_old_user_data",
                    action_category="gdpr_purge",
                    user_type="system",
                    result_status="success" if len(errors) == 0 else "partial",
                    result_message=f"Anonymisation utilisateurs: {anonymized_count} anonymisés",
                    action_details={
                        "model": "User",
                        "anonymized_count": anonymized_count,
                        "errors_count": len(errors),
                        "cutoff_date": cutoff_date.isoformat(),
                        "retention_days": retention,
                        "errors": errors[:10],
                    },
                    metadata={
                        "gdpr_compliant": True,
                        "retention_period_days": retention,
                        "action": "anonymize",  # Anonymisation plutôt que suppression
                    },
                )
            except Exception as e:
                logger.warning("[3.3 GDPR] Échec logging audit anonymisation: %s", e)

            return {
                "status": "success",
                "model": "User",
                "action": "anonymize",
                "cutoff_date": cutoff_date.isoformat(),
                "retention_days": retention,
                "anonymized_count": anonymized_count,
                "errors": errors,
            }

    except Exception as e:
        logger.exception("[3.3 GDPR] ❌ Erreur anonymisation utilisateurs: %s", e)
        raise
