# backend/celery_app.py
# pyright: reportImportCycles=false
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus

from celery import Celery
from celery.schedules import crontab

if TYPE_CHECKING:
    from flask import Flask

from services.demo.environment_guard import (
    build_demo_environment_snapshot,
    enforce_demo_environment_or_raise,
)

logger = logging.getLogger(__name__)

# Default configuration
# (⚠️ par défaut on pointe vers le service Docker 'redis', pas localhost)

# Construire les URLs Redis avec échappement du mot de passe si nécessaire
_redis_password = os.getenv("REDIS_PASSWORD", "")
_redis_host = os.getenv("REDIS_HOST", "redis")
_redis_port = os.getenv("REDIS_PORT", "6379")
_redis_db = os.getenv("REDIS_DB", "0")

# Si REDIS_URL est fourni, l'utiliser tel quel
# Sinon, construire depuis REDIS_PASSWORD avec échappement
if os.getenv("REDIS_URL"):
    REDIS_URL = os.getenv("REDIS_URL")
elif _redis_password:
    # Échapper le mot de passe pour l'URL
    _redis_password_escaped = quote_plus(_redis_password)
    REDIS_URL = (
        f"redis://:{_redis_password_escaped}@{_redis_host}:{_redis_port}/{_redis_db}"
    )
else:
    REDIS_URL = f"redis://{_redis_host}:{_redis_port}/{_redis_db}"

# ⚠️ IMPORTANT: Construire les URLs Redis avec échappement correct du mot de passe
# Si REDIS_PASSWORD est disponible, toujours construire les URLs avec échappement
# même si CELERY_BROKER_URL/CELERY_RESULT_BACKEND sont définis
# (pour éviter les URLs non échappées)
_broker_url_env = os.getenv("CELERY_BROKER_URL")
_result_backend_env = os.getenv("CELERY_RESULT_BACKEND")

# Si REDIS_PASSWORD est disponible et qu'une des URLs n'est pas définie
# ou contient des caractères non échappés,
# reconstruire les URLs avec échappement correct
if _redis_password:
    # Construire l'URL avec échappement depuis REDIS_PASSWORD
    _redis_password_escaped = quote_plus(_redis_password)
    _redis_url_escaped = (
        f"redis://:{_redis_password_escaped}@{_redis_host}:{_redis_port}/{_redis_db}"
    )

    # Utiliser les URLs échappées si les URLs de l'environnement ne sont pas définies
    # ou si elles contiennent le mot de passe non échappé (détection simple)
    if not _broker_url_env or _redis_password in _broker_url_env:
        CELERY_BROKER_URL = _redis_url_escaped
    else:
        CELERY_BROKER_URL = _broker_url_env

    if not _result_backend_env or _redis_password in _result_backend_env:
        CELERY_RESULT_BACKEND = _redis_url_escaped
    else:
        CELERY_RESULT_BACKEND = _result_backend_env
else:
    # Pas de mot de passe, utiliser les URLs fournies ou REDIS_URL par défaut
    CELERY_BROKER_URL = _broker_url_env if _broker_url_env else REDIS_URL
    CELERY_RESULT_BACKEND = _result_backend_env if _result_backend_env else REDIS_URL
CELERY_TIMEZONE = os.getenv("CELERY_TIMEZONE", "Europe/Zurich")
DISPATCH_AUTORUN_INTERVAL_SEC = int(os.getenv("DISPATCH_AUTORUN_INTERVAL_SEC", "300"))
CELERY_TASK_TIME_LIMIT = int(os.getenv("CELERY_TASK_TIME_LIMIT", "900"))
CELERY_TASK_SOFT_TIME_LIMIT = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "840"))
if CELERY_TASK_SOFT_TIME_LIMIT >= CELERY_TASK_TIME_LIMIT:
    CELERY_TASK_SOFT_TIME_LIMIT = max(1, CELERY_TASK_TIME_LIMIT - 60)

# Guard fail-fast pour worker/beat demo (pas de fallback vers prod)
enforce_demo_environment_or_raise(build_demo_environment_snapshot())

# Create Celery instance
# ✅ Forcer explicitement le transport Redis pour éviter confusion avec AMQP
celery: Celery = Celery(
    "atmr",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    broker_transport="redis",  # ✅ Forcer transport Redis explicitement
    include=[
        "tasks.dispatch_tasks",
        "tasks.planning_tasks",
        "tasks.rl_tasks",
        "tasks.secret_rotation_tasks",  # ✅ 2.5: Rotation secrets
        "tasks.vault_rotation_tasks",  # ✅ 4.1: Rotation secrets via Vault
        "tasks.purge_tasks",  # ✅ 3.3: Purge données RGPD
        "tasks.profiling_tasks",  # ✅ 3.4: Profiling CPU/mémoire automatique
        "tasks.dlq_cleanup_task",  # ✅ DLQ: Cleanup automatique DLQ
        "tasks.archive_tasks",  # ✅ 3.5.2: Archivage positions automatique
        "tasks.notification_tasks",  # ✅ Notifications push avec fallback SMS/Email
        "tasks.institution_invitation_tasks",  # Emails invitation institution
        "tasks.event_tasks",  # ✅ CRITIQUE: Gestion des événements domain (DriverNewBookingEvent, etc.)
        "tasks.geocoding_tasks",  # ✅ P1: Géocodage asynchrone des adresses de bookings
        "tasks.analytics_tasks",  # ✅ Analytics et rapports automatiques
        "tasks.osrm_precompute_tasks",  # ✅ P1: Pré-calcul matrices OSRM pour zones fréquentes
        "tasks.request_offer_tasks",  # ✅ ÉTAPE 4: Expiration/escalade offres institution
        "tasks.change_request_tasks",  # ✅ PR2: Expiration demandes de validation modification
        "tasks.patient_sync_tasks",  # ✅ Curatelle: sync patient cross-plateforme
        "tasks.security_tasks",  # ✅ Security Tab V2: purge audit logs
        "tasks.demo_access_tasks",  # ✅ Demo 24h: expiration automatique des acces demo
        "tasks.platform_billing_tasks",  # Facturation plateforme LIRIE V1
        "tasks.saferpay_reconciliation_tasks",  # Paiements Saferpay PENDING (beat)
        "tasks.invoice_pdf_tasks",  # V2 : PDF facture transport async (file d'attente)
        "tasks.transport_invoicing_automation",  # V3 : rappel mensuel / batch (stubs)
        "tasks.health_tasks",  # Smoke test Celery broker/worker (runbook incident)
        "tasks.tracking_health_tasks",  # Tracking BG: silent wake stale + purge device health
    ],
)

# Calculer worker_max_tasks_per_child avant l'appel à celery.conf.update
# ⚠️ IMPORTANT: 0 n'est plus accepté par billiard, on le convertit en None
_max_tasks_env = os.getenv("CELERY_WORKER_MAX_TASKS_PER_CHILD")
if _max_tasks_env is None:
    _max_tasks = 200  # Default: 200 tasks par worker
else:
    _max_tasks_int = int(_max_tasks_env)
    _max_tasks = None if _max_tasks_int == 0 else _max_tasks_int

# Plafond RSS par processus enfant (prefork) — unité : kilooctets (Ko), Billiard/Celery.
# 0 ou absent : pas de limite mémoire par enfant (recyclage uniquement via max_tasks_per_child).
_max_mem_env = os.getenv("CELERY_WORKER_MAX_MEMORY_PER_CHILD")
if _max_mem_env is None:
    _max_mem: int | None = None
else:
    _max_mem_int = int(_max_mem_env)
    _max_mem = None if _max_mem_int == 0 else _max_mem_int

# Configure Celery
celery.conf.update(
    broker_transport="redis",  # ✅ Forcer explicitement le transport Redis
    broker_connection_retry=True,  # Retry automatique en cas d'erreur de connexion
    broker_connection_retry_on_startup=True,  # important avec Docker
    broker_connection_max_retries=10,  # Nombre max de tentatives
    timezone=CELERY_TIMEZONE,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    task_time_limit=CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=CELERY_TASK_SOFT_TIME_LIMIT,
    worker_max_tasks_per_child=_max_tasks,  # None ou int > 0 (0 converti en None)
    worker_max_memory_per_child=_max_mem,  # None ou Ko > 0 (0 = désactivé)
    worker_prefetch_multiplier=1,  # One task at a time
    task_acks_late=True,  # Acknowledge task after execution
    task_reject_on_worker_lost=True,  # Requeue task if worker dies
    # ✅ Sprint 2 H5: Routage explicite par domaines pour isolation des workloads
    task_routes={
        "tasks.dispatch_tasks.*": {"queue": "dispatch"},
        "tasks.dispatch_tasks.realtime_monitoring_tick": {"queue": "realtime"},
        "tasks.dispatch_tasks.autorun_tick": {"queue": "dispatch"},
        "tasks.planning_tasks.*": {"queue": "dispatch"},
        "tasks.geocoding_tasks.*": {"queue": "geocoding"},
        "tasks.rl_tasks.*": {"queue": "rl"},
        "tasks.rl_*": {"queue": "rl"},
        "tasks.billing_tasks.*": {"queue": "billing"},
        "tasks.platform_billing_tasks.*": {"queue": "billing"},
        "tasks.notification_tasks.*": {
            "queue": "notifications"
        },  # Queue dédiée pour notifications
    },
    task_default_queue="default",
    task_create_missing_queues=True,
    task_default_exchange="default",
    task_default_exchange_type="direct",
    # ✅ DLQ: Configuration pour envoi en DLQ après échecs répétés
    task_default_rate_limit=None,
    task_ignore_result=False,  # Garder les résultats pour debugging
    # ✅ DLQ: Configuration queue DLQ explicite
    task_routes_dlq={
        # Toutes les tâches échouées après max_retries iront en DLQ
        "*": {"queue": "dlq"},
    },
    # Routage automatique vers DLQ après échecs
    task_acks_on_failure_or_timeout=False,  # Ne pas ack si échec (permet DLQ)
    # ✅ Forcer explicitement les URLs de broker pour Redis
    broker_write_url=CELERY_BROKER_URL,  # Forcer l'URL de broker pour écriture
    broker_read_url=CELERY_BROKER_URL,  # Forcer l'URL de broker pour lecture
)

# Configure Beat schedule
# ✅ 2.6: Jitter ajouté à tous les jobs pour éviter thundering herd
celery.conf.beat_schedule = {
    # Dispatch automatique périodique (toutes les 5 min par défaut)
    "dispatch-autorun": {
        "task": "tasks.dispatch_tasks.autorun_tick",
        "schedule": DISPATCH_AUTORUN_INTERVAL_SEC,
        "options": {
            "expires": DISPATCH_AUTORUN_INTERVAL_SEC * 2,
            "jitter": 30,  # ✅ 2.6: Jitter jusqu'à 30 secondes
        },
    },
    # 🆕 Monitoring temps réel pour dispatch autonome (toutes les 2 min)
    "realtime-monitoring": {
        "task": "tasks.dispatch_tasks.realtime_monitoring_tick",
        "schedule": 120.0,  # 2 minutes
        "options": {
            "expires": 240,  # Expire après 4 min
            "jitter": 15,  # ✅ 2.6: Jitter jusqu'à 15 secondes (tasks fréquentes)
        },
    },
    # Planning scans (daily at ~04:15, seconds granularity acceptable in dev)
    "planning-compliance-scan": {
        "task": "planning.compliance_scan_all",
        "schedule": 24 * 3600,
        "options": {
            "expires": 6 * 3600,
            "jitter": 300,  # ✅ 2.6: Jitter jusqu'à 5 minutes (tasks quotidiennes)
        },
    },
    # 🆕 RL: Ré-entraînement DQN hebdomadaire (dimanche à 3h)
    "rl-retrain-weekly": {
        "task": "tasks.rl_retrain_model",
        "schedule": 7 * 24 * 3600,  # 1 semaine
        "options": {
            "expires": 12 * 3600,  # Expire après 12h
            "jitter": 1800,  # ✅ 2.6: Jitter jusqu'à 30 minutes
            # (tasks hebdomadaires)
        },
    },
    # 🤖 Agent Dispatch: Démarrer automatiquement tous les agents
    # en mode FULLY_AUTO (toutes les 5 min)
    "ensure-agents-running": {
        "task": "tasks.dispatch_tasks.ensure_agents_running",
        "schedule": 300.0,  # 5 minutes
        "options": {
            "expires": 600,  # Expire après 10 min
            "jitter": 30,  # ✅ 2.6: Jitter jusqu'à 30 secondes
        },
    },
    # ✅ 3.5.2: Archivage automatique positions (hebdomadaire, dimanche à 2h)
    "archive-old-positions": {
        "task": "tasks.archive_tasks.archive_old_positions_task",
        "schedule": 7 * 24 * 3600,  # 1 semaine (604800 secondes)
        "options": {
            "expires": 12 * 3600,  # Expire après 12h
            "jitter": 1800,  # ✅ 2.6: Jitter jusqu'à 30 minutes (tasks hebdomadaires)
        },
    },
    # 🆕 RL: Nettoyage feedbacks mensuels (1er du mois à 4h)
    "rl-cleanup-monthly": {
        "task": "tasks.rl_cleanup_old_feedbacks",
        "schedule": 30 * 24 * 3600,  # ~1 mois
        "options": {
            "expires": 24 * 3600,
            "jitter": 3600,  # ✅ 2.6: Jitter jusqu'à 1 heure (tasks mensuelles)
        },
    },
    # 🆕 RL: Rapport hebdomadaire (lundi à 8h)
    "rl-weekly-report": {
        "task": "tasks.rl_generate_weekly_report",
        "schedule": 7 * 24 * 3600,  # 1 semaine
        "options": {
            "expires": 6 * 3600,
            "jitter": 1800,  # ✅ 2.6: Jitter jusqu'à 30 minutes
            # (tasks hebdomadaires)
        },
    },
    # ✅ P1: Pré-calcul matrices OSRM pour zones fréquentes (quotidien à 3h)
    "osrm-precompute-matrices": {
        "task": "tasks.osrm_precompute_matrices",
        "schedule": 24 * 3600,  # 1 jour (86400 secondes)
        "options": {
            "expires": 6 * 3600,  # Expire après 6h
            "jitter": 1800,  # ✅ 2.6: Jitter jusqu'à 30 minutes (tasks quotidiennes)
        },
    },
    # ✅ S3: Vérification et rotation automatique des secrets
    # (tous les 7 jours pour détecter si rotation due et l'effectuer)
    "secret-rotation-check": {
        "task": "tasks.secret_rotation_tasks.check_and_rotate_all",
        "schedule": 7 * 24 * 3600,  # 1 semaine
        "options": {
            "expires": 12 * 3600,
            "jitter": 3600,  # ✅ 2.6: Jitter jusqu'à 1 heure (tasks hebdomadaires)
        },
    },
    # ✅ ÉTAPE 4: Expiration et escalade des offres de transport institutionnelles
    # (toutes les minutes pour traiter les timeouts rapidement)
    "process-expired-offers": {
        "task": "tasks.request_offer_tasks.process_expired_offers",
        "schedule": 60.0,  # 1 minute
        "options": {
            "expires": 120,  # Expire après 2 min
            "jitter": 10,  # ✅ Jitter jusqu'à 10 secondes
        },
    },
    # ✅ PR2: Expiration des demandes de validation de modification (toutes les minutes)
    "expire-pending-change-requests": {
        "task": "tasks.change_request_tasks.expire_pending_change_requests",
        "schedule": 60.0,  # 1 minute
        "options": {
            "expires": 120,  # Expire après 2 min
            "jitter": 10,
        },
    },
    # ✅ GO-LIVE: Métriques métier institution (toutes les heures)
    "log-institution-metrics": {
        "task": "tasks.request_offer_tasks.log_institution_metrics",
        "schedule": 3600.0,  # 1 heure
        "args": [24],  # période = 24h
        "options": {
            "expires": 3600,  # Expire après 1h
            "jitter": 300,  # Jitter jusqu'à 5 min
        },
    },
    # Saferpay: PENDING avec session non finalisée depuis 30+ min (logs / alerting)
    "saferpay-stale-pending-scan": {
        "task": "saferpay.reconcile_stale_pending",
        "schedule": 3600.0,
        "options": {
            "expires": 1800,
            "jitter": 120,
        },
    },
    # Push: nettoyage tokens zombies (quotidien ~04:30 UTC)
    "deactivate-stale-push-tokens": {
        "task": "notifications.deactivate_stale_device_tokens",
        "schedule": 24 * 3600,
        "options": {
            "expires": 6 * 3600,
            "jitter": 300,
        },
    },
    # Push: refresh gauges couverture (horaire)
    "refresh-push-coverage-gauges": {
        "task": "notifications.refresh_push_coverage_gauges",
        "schedule": 3600.0,
        "options": {
            "expires": 1800,
            "jitter": 120,
        },
    },
    # Tracking BG: silent push wake pour drivers en mission avec fix stale
    "tracking-stale-wake": {
        "task": "tasks.tracking_health_tasks.stale_tracking_wake_tick",
        "schedule": 60.0,
        "options": {
            "expires": 120,
            "jitter": 5,
        },
    },
    # Purge historique device-health (>30 jours)
    "purge-device-health-events": {
        "task": "tasks.tracking_health_tasks.purge_device_health_events",
        "schedule": 24 * 3600,
        "options": {
            "expires": 6 * 3600,
            "jitter": 600,
        },
    },
    # Marché ouvert : PENDING sans entreprise et sans offre PROPOSED (rattrapage centralisé)
    "dispatch-repair-open-offers": {
        "task": "tasks.dispatch_tasks.repair_open_booking_offers_tick",
        "schedule": 300.0,
        "options": {
            "expires": 240,
            "jitter": 45,
        },
    },
    # ✅ 3.3: Purge automatique données RGPD (hebdomadaire)
    "gdpr-purge-all": {
        "task": "tasks.purge_tasks.purge_all_old_data",
        "schedule": 7 * 24 * 3600,  # 1 semaine
        "options": {
            "expires": 24 * 3600,  # Expire après 24h
            "jitter": 3600,  # ✅ 2.6: Jitter jusqu'à 1 heure (tasks hebdomadaires)
        },
    },
    # ✅ 3.3: Anonymisation données utilisateur (mensuelle)
    "gdpr-anonymize-users": {
        "task": "tasks.purge_tasks.anonymize_old_user_data",
        "schedule": 30 * 24 * 3600,  # 1 mois
        "options": {
            "expires": 48 * 3600,  # Expire après 48h
            "jitter": 7200,  # ✅ 2.6: Jitter jusqu'à 2 heures (tasks mensuelles)
        },
    },
    # ✅ 3.4: Profiling automatique hebdomadaire
    "profiling-weekly": {
        "task": "tasks.profiling_tasks.run_weekly_profiling",
        "schedule": 7 * 24 * 3600,  # 1 semaine
        "options": {
            "expires": 24 * 3600,  # Expire après 24h
            "jitter": 3600,  # ✅ 2.6: Jitter jusqu'à 1 heure (tasks hebdomadaires)
        },
    },
    # ✅ 4.1: Rotation automatique JWT secret via Vault (tous les 30 jours)
    "vault-rotate-jwt": {
        "task": "tasks.vault_rotation_tasks.rotate_jwt_secret",
        "schedule": 30 * 24 * 3600,  # 30 jours
        "options": {
            "expires": 48 * 3600,  # Expire après 48h
            "jitter": 7200,  # ✅ 2.6: Jitter jusqu'à 2 heures (tasks mensuelles)
        },
    },
    # ✅ 4.1: Rotation automatique SECRET_KEY Flask via Vault (tous les 90 jours)
    "vault-rotate-flask-secret": {
        "task": "tasks.vault_rotation_tasks.rotate_flask_secret_key",
        "schedule": 90 * 24 * 3600,  # 90 jours
        "options": {
            "expires": 48 * 3600,  # Expire après 48h
            "jitter": 7200,  # ✅ 2.6: Jitter jusqu'à 2 heures (tasks trimestrielles)
        },
    },
    # ✅ 4.1: Rotation automatique encryption key via Vault (tous les 90 jours)
    "vault-rotate-encryption": {
        "task": "tasks.vault_rotation_tasks.rotate_encryption_key",
        "schedule": 90 * 24 * 3600,  # 90 jours
        "options": {
            "expires": 48 * 3600,  # Expire après 48h
            "jitter": 7200,  # ✅ 2.6: Jitter jusqu'à 2 heures (tasks trimestrielles)
        },
    },
    # ✅ 4.1: Rotation globale des secrets (tous les 90 jours,
    # backup de la rotation individuelle)
    "vault-rotate-all": {
        "task": "tasks.vault_rotation_tasks.rotate_all_secrets",
        "schedule": 90 * 24 * 3600,  # 90 jours
        "options": {
            "expires": 48 * 3600,  # Expire après 48h
            "jitter": 7200,  # ✅ 2.6: Jitter jusqu'à 2 heures
        },
    },
    # ✅ Curatelle: Traitement sync patient outbox (toutes les 30 secondes)
    "process-patient-sync": {
        "task": "tasks.patient_sync_tasks.process_pending_sync_events",
        "schedule": 30.0,
        "options": {
            "expires": 60,
            "jitter": 5,
        },
    },
    # ✅ Security V2: Purge audit logs expirés (quotidien à ~3h)
    "purge-audit-logs": {
        "task": "tasks.security_tasks.purge_expired_audit_logs",
        "schedule": 24 * 3600,
        "options": {
            "expires": 6 * 3600,
            "jitter": 1800,
        },
    },
    # ✅ DLQ: Cleanup automatique DLQ (quotidien)
    "dlq-cleanup-daily": {
        "task": "tasks.dlq_cleanup_task.cleanup_old_dlq_entries",
        "schedule": 24 * 3600,  # 1 jour
        "options": {
            "expires": 6 * 3600,  # Expire après 6h
            "jitter": 1800,  # ✅ 2.6: Jitter jusqu'à 30 minutes
        },
    },
    # ✅ Demo 24h: expiration auto des acces toutes les 5 minutes
    "expire-demo-accesses": {
        "task": "tasks.demo_access_tasks.expire_demo_accesses",
        "schedule": 300.0,
        "options": {
            "expires": 600,
            "jitter": 30,
        },
    },
    # ✅ Demo: contrôle fréquent du socle partagé (toutes les heures)
    "ensure-demo-reference-dataset": {
        "task": "tasks.demo_access_tasks.ensure_demo_reference_dataset",
        "schedule": 3600.0,
        "options": {
            "expires": 1800,
            "jitter": 180,
        },
    },
    # ✅ Demo: reset automatique quotidien du dataset vivant
    "reset-demo-reference-dataset-daily": {
        "task": "tasks.demo_access_tasks.reset_demo_reference_dataset",
        "schedule": 24 * 3600,
        "options": {
            "expires": 6 * 3600,
            "jitter": 1800,
        },
    },
    # V3 : rappel mensuel (aucune facture créée — emplacement notif / futur auto)
    "monthly-transport-billing-reminder": {
        "task": "tasks.transport_invoicing_automation.monthly_transport_billing_reminder_task",
        "schedule": crontab(day_of_month="1", hour="5", minute="30"),
        "options": {
            "expires": 4 * 3600,
            "jitter": 600,
        },
    },
}

# Initialize Flask app for Celery workers
_flask_app = {}


def get_flask_app():
    """Get or create Flask app instance for Celery workers."""
    if "app" not in _flask_app:
        from app import create_app

        config_name = os.getenv("FLASK_CONFIG", "production")
        # ✅ Désactiver l'initialisation des routes API dans le contexte Celery
        # pour éviter l'erreur Flask-RESTX
        # "View function mapping is overwriting an existing endpoint function: specs"
        original_skip = os.getenv("SKIP_ROUTES_INIT", "false")
        os.environ["SKIP_ROUTES_INIT"] = "true"
        try:
            _flask_app["app"] = create_app(config_name)
            logger.info(
                "Flask app created for Celery with config: %s (routes init skipped)",
                config_name,
            )
        finally:
            # Restaurer la valeur originale
            os.environ["SKIP_ROUTES_INIT"] = original_skip
    return _flask_app["app"]


class ContextTask(celery.Task):
    """Custom Celery task that runs within Flask application context."""

    def __call__(self, *args, **kwargs):
        app = get_flask_app()
        with app.app_context():
            return self.run(*args, **kwargs)


# Set the custom task class as default
celery.Task = ContextTask


def init_app(app: Flask) -> Celery:
    """Initialize Celery with Flask app context.
    Call this from create_app().
    """
    _flask_app["app"] = app

    logger.info(
        "Celery initialized with broker=%s, backend=%s, timezone=%s",
        CELERY_BROKER_URL,
        CELERY_RESULT_BACKEND,
        CELERY_TIMEZONE,
    )

    # ✅ A3: Enregistrer les handlers pour DLQ
    _register_dlq_handlers()

    # ✅ P3: Initialiser la métrique Prometheus pour la longueur de queue
    _init_queue_length_metric()

    return celery


# ✅ DLQ: Handler pour stocker métadonnées des tâches échouées avec monitoring
def _register_dlq_handlers():
    """Enregistre les handlers pour capturer les échecs et les stocker en DLQ."""
    from celery.signals import (
        task_failure,
        task_retry,
        task_success,
    )

    @task_failure.connect
    def task_failed_handler(  # pyright: ignore[reportUnusedFunction]
        sender=None, task_id=None, exception=None, traceback=None, _einfo=None, **kwargs
    ):
        """Gère les tâches échouées après max_retries."""
        task_name = getattr(sender, "name", None) if sender else "unknown"
        retries = kwargs.get("request", {}).get("retries", 0)
        max_retries = getattr(sender, "max_retries", 3) if sender else 3

        # ✅ DLQ: Enregistrer métrique Prometheus
        try:
            from services.unified_dispatch.dispatch_prometheus_metrics import (  # pyright: ignore[reportMissingImports]
                PROMETHEUS_AVAILABLE,
            )

            if PROMETHEUS_AVAILABLE:
                try:
                    from prometheus_client import (
                        Counter,
                    )

                    dlq_counter = Counter(
                        "celery_dlq_failures_total",
                        "Nombre total de tâches envoyées en DLQ",
                        ["task_name"],
                    )
                    dlq_counter.labels(task_name=task_name).inc()
                except ImportError:
                    pass
        except Exception:
            pass  # Ne pas bloquer si métriques échouent

        logger.error(
            "[Celery DLQ] Task %s failed permanently after %d/%d retries: %s",
            task_id,
            retries,
            max_retries,
            str(exception)[:200],
            extra={
                "task_id": task_id,
                "task_name": task_name,
                "exception": str(exception)[:500],
                "traceback": traceback,
                "retries": retries,
                "max_retries": max_retries,
            },
        )

        # ✅ DLQ: Stocker en DB pour visibilité et monitoring
        if task_id:
            try:
                _store_task_failure_in_db(
                    task_id=task_id,
                    task_name=task_name,
                    exception=str(exception),
                    traceback=str(traceback) if traceback else None,
                    args=kwargs.get("args") or [],
                    kwargs=kwargs.get("kwargs") or {},
                )
            except Exception as db_err:
                logger.exception(
                    "[Celery DLQ] Failed to store task failure in DB: %s", db_err
                )

    @task_retry.connect
    def task_retry_handler(  # pyright: ignore[reportUnusedFunction]
        _sender=None, task_id=None, reason=None, **kwargs
    ):
        """Log les retries."""
        retries = kwargs.get("request", {}).get("retries", 0)
        logger.warning("[Celery] Task %s retry #%d: %s", task_id, retries + 1, reason)

    @task_success.connect
    def task_succeeded_handler(  # pyright: ignore[reportUnusedFunction]
        sender=None, **_kwargs
    ):
        """Log les succès."""
        logger.debug("[Celery] Task %s succeeded", sender.name if sender else "unknown")


def _store_task_failure_in_db(
    task_id: str,
    task_name: str | None = None,
    exception: str = "",
    traceback: str | None = None,
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
):
    """Stocke une tâche échouée dans la table TaskFailure."""
    try:
        from datetime import UTC, datetime

        from ext import db
        from models import TaskFailure

        # Vérifier si la tâche existe déjà (pour éviter doublon)
        existing = TaskFailure.query.filter_by(task_id=task_id).first()

        if existing:
            # Incrémenter le compteur et mettre à jour last_seen
            existing.failure_count += 1
            existing.last_seen = datetime.now(UTC)
            existing.exception = exception[:5000]
            existing.traceback = traceback[:10000] if traceback else None
            db.session.commit()
            logger.info(
                "[Celery DLQ] Updated task failure: task_id=%s (count=%d)",
                task_id,
                existing.failure_count,
            )
        else:
            # Créer nouveau record
            failure_data = {
                "task_id": task_id,
                "task_name": task_name or "unknown",
                "exception": exception[:5000],  # Limiter taille
                "traceback": traceback[:10000] if traceback else None,
                "args": str(args)[:2000] if args else None,
                "kwargs": kwargs if kwargs else None,  # JSONB accepte dict directement
                "first_seen": datetime.now(UTC),
                "last_seen": datetime.now(UTC),
                "failure_count": 1,
            }
            failure = TaskFailure(**failure_data)

            db.session.add(failure)
            db.session.commit()

            logger.info(
                "[Celery DLQ] Stored new task failure: task_id=%s, task_name=%s",
                task_id,
                task_name,
            )

    except Exception as e:
        logger.exception("[Celery DLQ] Failed to store failure in DB: %s", e)
        # Continue sans erreur critique (ne bloque pas le worker)


# ✅ P3: Métrique Prometheus pour la longueur de queue Celery
# Utiliser un dict pour éviter l'utilisation de global statement (Ruff PLW0603)
_celery_queue_length_gauge_container: dict[str, Any | None] = {"gauge": None}


def _get_queue_length_gauge():
    """Retourne la métrique Prometheus pour la longueur de queue Celery.

    Initialise la métrique si nécessaire (lazy initialization).

    Returns:
        Gauge Prometheus ou None si prometheus_client non disponible
    """
    if _celery_queue_length_gauge_container["gauge"] is None:
        try:
            from prometheus_client import (
                Gauge,
            )

            _celery_queue_length_gauge_container["gauge"] = Gauge(
                "celery_queue_length",
                "Nombre de tâches en attente dans la queue Celery",
                ["queue_name"],
            )
            logger.info("[Celery] Prometheus metric 'celery_queue_length' initialized")
        except ImportError:
            logger.warning(
                "[Celery] prometheus_client non disponible, métrique queue_length désactivée"
            )
        except Exception as e:
            logger.warning(
                "[Celery] Erreur initialisation métrique queue_length: %s", e
            )

    return _celery_queue_length_gauge_container["gauge"]


def _init_queue_length_metric():
    """Initialise la métrique Prometheus pour la longueur de queue Celery."""
    _get_queue_length_gauge()


def get_celery_queue_lengths() -> dict[str, int]:
    """Obtient la longueur de toutes les queues Celery.

    Utilise l'API inspect de Celery pour obtenir le nombre de tâches
    actives, réservées et programmées par queue.

    Returns:
        Dict avec {queue_name: length} pour chaque queue
        Retourne {} en cas d'erreur ou si les workers ne sont pas disponibles
    """
    try:
        inspect = celery.control.inspect()

        # Obtenir les tâches actives, réservées et programmées
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}
        scheduled = inspect.scheduled() or {}

        # Compter les tâches par queue
        queue_lengths: dict[str, int] = {}

        # Parcourir toutes les tâches actives
        for worker_tasks in active.values():
            for task in worker_tasks:
                queue = task.get("delivery_info", {}).get("routing_key", "default")
                queue_lengths[queue] = queue_lengths.get(queue, 0) + 1

        # Parcourir toutes les tâches réservées
        for worker_tasks in reserved.values():
            for task in worker_tasks:
                queue = task.get("delivery_info", {}).get("routing_key", "default")
                queue_lengths[queue] = queue_lengths.get(queue, 0) + 1

        # Parcourir toutes les tâches programmées
        for worker_tasks in scheduled.values():
            for task in worker_tasks:
                queue = task.get("delivery_info", {}).get("routing_key", "default")
                queue_lengths[queue] = queue_lengths.get(queue, 0) + 1

        # Si aucune tâche trouvée mais que des workers sont actifs,
        # retourner au moins les queues connues avec 0
        if not queue_lengths:
            # Vérifier si des workers sont actifs
            stats = inspect.stats() or {}
            if stats:
                # Workers actifs mais pas de tâches = queues vides
                # Retourner les queues par défaut avec 0
                queue_lengths = {
                    "default": 0,
                    "realtime": 0,
                    "dlq": 0,
                }

        return queue_lengths

    except Exception as e:
        logger.warning(
            "[Celery] Erreur lors de la récupération de la longueur des queues: %s", e
        )
        return {}


def update_celery_queue_length_metric():
    """Met à jour la métrique Prometheus avec les longueurs actuelles des queues.

    Cette fonction doit être appelée périodiquement (ex: toutes les 30 secondes)
    ou lors de l'export des métriques Prometheus.
    """
    gauge = _get_queue_length_gauge()
    if gauge is None:
        return

    try:
        queue_lengths = get_celery_queue_lengths()

        # Mettre à jour la métrique pour chaque queue
        for queue_name, length in queue_lengths.items():
            gauge.labels(queue_name=queue_name).set(length)

        # Réinitialiser les queues qui n'ont plus de tâches
        # (pour éviter que les métriques restent à l'ancienne valeur)
        known_queues = {
            "default",
            "dispatch",
            "realtime",
            "notifications",
            "billing",
            "rl",
            "geocoding",
            "dlq",
        }
        for queue_name in known_queues:
            if queue_name not in queue_lengths:
                gauge.labels(queue_name=queue_name).set(0)

    except Exception as e:
        logger.warning(
            "[Celery] Erreur lors de la mise à jour de la métrique queue_length: %s", e
        )


from celery.signals import worker_process_init


@worker_process_init.connect
def _init_celery_worker_sentry(**_kwargs: Any) -> None:
    """Initialise Sentry dans chaque processus worker Celery."""
    try:
        from shared.sentry_init import init_sentry

        init_sentry(celery=True)
    except Exception as exc:
        logger.warning("[Celery] Sentry worker init skipped: %s", exc)
