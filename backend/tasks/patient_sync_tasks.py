# tasks/patient_sync_tasks.py
# pyright: reportCallIssue=false
"""Worker Celery pour le traitement de la file d'attente de synchronisation patient.

Traite les PatientSyncEvent (outbox) de manière :
- Idempotente (idempotency_key unique)
- Sûre (FOR UPDATE SKIP LOCKED empêche le double traitement)
- Observable (audit logs + statut mis à jour)
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from celery_app import celery
from ext import db
from models.patient_identity import (
    PatientAuditLog,
    PatientIdentityLink,
    PatientSyncEvent,
)

logger = logging.getLogger(__name__)

_BATCH_SIZE = 10


@celery.task(name="tasks.patient_sync_tasks.process_pending_sync_events")
def process_pending_sync_events() -> dict[str, Any]:
    """Traite les événements de sync en attente par batch.

    Appelé périodiquement par Celery Beat ou manuellement.
    Utilise FOR UPDATE SKIP LOCKED pour éviter les conflits entre workers.

    Returns:
        Dict avec le résumé du traitement
    """
    # Sélectionner un batch d'événements pending avec verrouillage
    events = (
        PatientSyncEvent.query
        .filter_by(status="pending")
        .order_by(PatientSyncEvent.created_at)
        .limit(_BATCH_SIZE)
        .with_for_update(skip_locked=True)
        .all()
    )

    if not events:
        return {"processed": 0, "success": 0, "failed": 0}

    # Marquer comme "processing"
    for event in events:
        event.status = "processing"
    db.session.commit()

    stats = {"processed": len(events), "success": 0, "failed": 0}

    for event in events:
        try:
            _process_single_event(event)
            stats["success"] += 1
        except Exception as exc:
            logger.exception(
                "[PatientSync] Erreur traitement event %s: %s",
                event.id,
                exc,
            )
            event.status = "failed"
            event.error = str(exc)[:2000]
            event.retry_count += 1
            if event.retry_count < event.max_retries:
                event.status = "pending"
            event.processed_at = datetime.now(UTC)
            db.session.commit()
            stats["failed"] += 1

    logger.info(
        "[PatientSync] Batch terminé: %d traités, %d succès, %d échecs",
        stats["processed"],
        stats["success"],
        stats["failed"],
    )
    return stats


def _process_single_event(event: PatientSyncEvent) -> None:
    """Traite un seul événement de synchronisation."""
    from services.patient_sync.patient_identity_service import (
        apply_sync_to_client,
        apply_sync_to_institution_patient,
        with_sync_origin,
    )

    identity = event.patient_identity
    links = PatientIdentityLink.query.filter_by(
        patient_identity_id=identity.id,
        is_active=True,
    ).all()

    errors: list[str] = []

    for link in links:
        if (
            link.entity_type == event.source_entity_type
            and link.entity_id == event.source_entity_id
        ):
            continue

        try:
            if link.entity_type == "institution_patient":
                with_sync_origin(
                    apply_sync_to_institution_patient,
                    link.entity_id,
                    event.changed_fields,
                    source_identity_id=identity.id,
                )
            elif link.entity_type == "client":
                with_sync_origin(
                    apply_sync_to_client,
                    link.entity_id,
                    event.changed_fields,
                )

            db.session.add(PatientAuditLog(
                actor_user_id=None,
                action="SYNC_APPLIED",
                entity_type=link.entity_type,
                entity_id=link.entity_id,
                metadata_json={
                    "event_id": event.id,
                    "fields": list(event.changed_fields.keys()),
                },
            ))
        except Exception as exc:
            errors.append(f"{link.entity_type}:{link.entity_id} - {exc!s}")

    target_count = sum(
        1 for lnk in links
        if not (
            lnk.entity_type == event.source_entity_type
            and lnk.entity_id == event.source_entity_id
        )
    )

    if errors:
        event.error = json.dumps(errors)
        event.retry_count += 1
        if len(errors) < target_count:
            event.status = "partial_failure"
        else:
            event.status = "failed"
        if event.retry_count < event.max_retries:
            event.status = "pending"
    else:
        event.status = "success"

    event.processed_at = datetime.now(UTC)
    db.session.commit()

    logger.info(
        "[PatientSync] Event %s: %s (targets=%d, errors=%d)",
        event.id,
        event.status,
        target_count,
        len(errors),
    )
