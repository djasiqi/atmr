"""Livraison email d'activation Lot 1 / F-03 : HMAC, supersession, finalisation."""

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from ext import db
from models.activation_email_delivery import (
    ALLOWED_TRANSITIONS,
    EMAIL_DELIVERY_FAILED,
    EMAIL_DELIVERY_KIND_INITIAL,
    EMAIL_DELIVERY_KIND_RESEND,
    EMAIL_DELIVERY_QUEUED,
    EMAIL_DELIVERY_SENDING,
    EMAIL_DELIVERY_SENT,
    SENDING_LEASE_MINUTES,
    WEBHOOK_ADVANCED_STATUSES,
    ActivationEmailDelivery,
)
from models.activation_session import ActivationSession
from services.notifications.activation_email_policy import (
    enforce_resend_policy,
    is_same_utc_day,
)
from services.notifications.activation_token import (
    ActivationTokenKeyError,
    derive_activation_token,
    hash_activation_token,
    require_activation_token_key_in_production,
)
from services.notifications.email import is_email_provider_configured

logger = logging.getLogger(__name__)

ACTIVATION_EMAIL_TTL_MINUTES = int(os.getenv("ACTIVATION_EMAIL_TTL_MINUTES", "30"))
CURRENT_TOKEN_KEY_VERSION = 1


def sanitize_email_error(message: str | None) -> str:
    """Supprime jetons, URLs et emails avant stockage interne."""
    if not message:
        return "email_send_failed"
    text_val = str(message)
    text_val = re.sub(r"https?://\S+", "[url]", text_val)
    text_val = re.sub(r"[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-.]+", "[token]", text_val)
    text_val = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "[email]",
        text_val,
    )
    return text_val[:500]


def get_activation_session_for_update(
    activation_session_pk: int,
) -> ActivationSession:
    """Recharge la session sous FOR UPDATE (autorité, pas l'ORM route)."""
    return (
        ActivationSession.query.filter_by(id=activation_session_pk)
        .populate_existing()
        .with_for_update()
        .one()
    )


def get_delivery_by_id(email_delivery_id: str) -> ActivationEmailDelivery | None:
    return ActivationEmailDelivery.query.filter_by(
        email_delivery_id=email_delivery_id
    ).first()


def sync_current_delivery_mirror(
    session: ActivationSession, delivery: ActivationEmailDelivery
) -> bool:
    """Copie le miroir sans jamais changer email_delivery_id."""
    if session.email_delivery_id != delivery.email_delivery_id:
        return False
    if delivery.superseded_at is not None:
        return False
    session.email_delivery_status = delivery.status
    session.email_delivery_kind = delivery.kind
    session.email_token_hash = delivery.email_token_hash
    session.email_token_expires_at = delivery.token_expires_at
    session.email_last_error = delivery.last_error
    session.email_provider_message_id = delivery.provider_message_id
    return True


def set_current_delivery(
    session: ActivationSession, delivery: ActivationEmailDelivery
) -> None:
    """Pointeur courant — uniquement pendant la création atomique de B."""
    session.email_delivery_id = delivery.email_delivery_id
    sync_current_delivery_mirror(session, delivery)
    session.email_last_error = None


# Compat anciens imports / tests
def _sync_session_from_delivery(
    session: ActivationSession, delivery: ActivationEmailDelivery
) -> None:
    set_current_delivery(session, delivery)


def is_sending_lease_expired(delivery: ActivationEmailDelivery) -> bool:
    if delivery.status != EMAIL_DELIVERY_SENDING:
        return False
    started = delivery.sending_started_at
    if started is None:
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return datetime.now(UTC) - started > timedelta(minutes=SENDING_LEASE_MINUTES)


def can_start_new_delivery_snapshot(
    session: ActivationSession,
) -> tuple[bool, str | None]:
    """Lecture seule — aucun effet de bord ORM (précontrôle route)."""
    if not session.email_delivery_id:
        return True, None
    delivery = get_delivery_by_id(session.email_delivery_id)
    if delivery is None:
        return True, None
    if delivery.superseded_at is not None:
        return True, None
    if delivery.status == EMAIL_DELIVERY_QUEUED:
        return False, "email_delivery_in_progress"
    if delivery.status == EMAIL_DELIVERY_SENDING and not is_sending_lease_expired(
        delivery
    ):
        return False, "email_delivery_in_progress"
    return True, None


def can_start_new_delivery(session: ActivationSession) -> tuple[bool, str | None]:
    """Alias snapshot non mutatif (F-03). Mutations lease → expire_stale_sending_under_lock."""
    return can_start_new_delivery_snapshot(session)


def expire_stale_sending_under_lock(session: ActivationSession) -> None:
    """Sous verrou session : sending lease expirée → failed historique + miroir si courant."""
    if not session.email_delivery_id:
        return
    delivery = (
        ActivationEmailDelivery.query.filter_by(
            email_delivery_id=session.email_delivery_id
        )
        .populate_existing()
        .with_for_update()
        .first()
    )
    if delivery is None:
        return
    if delivery.status != EMAIL_DELIVERY_SENDING:
        return
    if not is_sending_lease_expired(delivery):
        return
    now = datetime.now(UTC)
    updated = db.session.execute(
        text(
            """
            UPDATE activation_email_deliveries
            SET status = :failed,
                last_error = :err,
                updated_at = :now
            WHERE email_delivery_id = :did
              AND status = :sending
              AND provider_accepted_at IS NULL
            """
        ),
        {
            "failed": EMAIL_DELIVERY_FAILED,
            "err": sanitize_email_error("sending_lease_expired"),
            "now": now,
            "did": delivery.email_delivery_id,
            "sending": EMAIL_DELIVERY_SENDING,
        },
    ).rowcount
    if updated:
        db.session.refresh(delivery)
        sync_current_delivery_mirror(session, delivery)


def mark_delivery_failed(
    session: ActivationSession,
    error: str,
    *,
    email_delivery_id: str | None = None,
) -> bool:
    """CAS fail : seulement queued/sending sans provider_accepted.

    Returns:
        True si une ligne livraison a été mise à jour.
    """
    delivery_id = email_delivery_id or session.email_delivery_id
    if not delivery_id:
        return False
    now = datetime.now(UTC)
    err = sanitize_email_error(error)
    result = db.session.execute(
        text(
            """
            UPDATE activation_email_deliveries
            SET status = :failed,
                last_error = :err,
                updated_at = :now
            WHERE email_delivery_id = :did
              AND provider_accepted_at IS NULL
              AND status IN (:queued, :sending)
            """
        ),
        {
            "failed": EMAIL_DELIVERY_FAILED,
            "err": err,
            "now": now,
            "did": delivery_id,
            "queued": EMAIL_DELIVERY_QUEUED,
            "sending": EMAIL_DELIVERY_SENDING,
        },
    )
    if int(result.rowcount or 0) == 0:  # type: ignore[attr-defined]
        logger.info(
            "failure_ignored reason=already_accepted_or_terminal "
            "activation_session_id=%s email_delivery_id=%s",
            getattr(session, "activation_session_id", None),
            delivery_id,
        )
        return False
    delivery = get_delivery_by_id(delivery_id)
    if delivery is not None:
        sync_current_delivery_mirror(session, delivery)
    return True


def _supersede_previous_deliveries(
    *,
    activation_session_pk: int,
    new_delivery_id: str,
    now: datetime,
) -> None:
    db.session.execute(
        text(
            """
            UPDATE activation_email_deliveries
            SET superseded_at = :now,
                superseded_by_delivery_id = :new_id,
                updated_at = :now
            WHERE activation_session_pk = :pk
              AND superseded_at IS NULL
              AND email_delivery_id <> :new_id
            """
        ),
        {
            "now": now,
            "new_id": new_delivery_id,
            "pk": activation_session_pk,
        },
    )


def prepare_activation_email_delivery(
    session: ActivationSession,
    *,
    kind: str,
) -> tuple[str, str]:
    """Sous verrou uniquement : supersede + crée B + set_current. Retourne (id, token)."""
    if kind not in {EMAIL_DELIVERY_KIND_INITIAL, EMAIL_DELIVERY_KIND_RESEND}:
        raise ValueError(f"kind invalide: {kind}")

    delivery_id = str(uuid.uuid4())
    key_version = CURRENT_TOKEN_KEY_VERSION
    token = derive_activation_token(delivery_id, key_version=key_version)
    token_hash = hash_activation_token(token)
    now = datetime.now(UTC)
    expires = now + timedelta(minutes=ACTIVATION_EMAIL_TTL_MINUTES)

    _supersede_previous_deliveries(
        activation_session_pk=session.id,
        new_delivery_id=delivery_id,
        now=now,
    )

    delivery = ActivationEmailDelivery(
        activation_session_pk=session.id,
        email_delivery_id=delivery_id,
        kind=kind,
        status=EMAIL_DELIVERY_QUEUED,
        token_key_version=key_version,
        email_token_hash=token_hash,
        token_expires_at=expires,
        superseded_at=None,
        superseded_by_delivery_id=None,
    )
    db.session.add(delivery)
    db.session.flush()
    set_current_delivery(session, delivery)
    return delivery_id, token


def enqueue_activation_email(
    *,
    activation_session_id: str,
    email_delivery_id: str,
) -> None:
    from tasks.notification_tasks import send_activation_email_task

    send_activation_email_task.delay(
        activation_session_id=activation_session_id,
        email_delivery_id=email_delivery_id,
    )


def try_enqueue_activation_email(
    session: ActivationSession,
    *,
    kind: str,
    environment: str,
    is_testing: bool,
    email_token: str | None = None,
    email_token_hash: str | None = None,
) -> dict[str, Any]:
    """Préflight hors TX → TX atomique supersession → enqueue après commit."""
    del email_token, email_token_hash
    env = (environment or "").strip().lower()
    is_prod = env == "production"
    is_local_dev = env == "development" and not is_testing
    session_pk = int(session.id)

    # --- Préflight hors transaction (aucune supersession) ---
    try:
        require_activation_token_key_in_production()
    except ActivationTokenKeyError as e:
        return {
            "ok": False,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": None,
            "error": str(e),
            "require_502": True,
            "email_token": None,
        }

    ready, config_error = is_email_provider_configured()
    if not ready and is_prod:
        # F-03 : ne pas marquer A failed ni superseder
        return {
            "ok": False,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": None,
            "error": config_error or "email_provider_not_configured",
            "require_502": True,
            "email_token": None,
        }

    delivery_id: str | None = None
    token: str | None = None
    try:
        locked = get_activation_session_for_update(session_pk)
        if locked.email_verified_at is not None:
            db.session.commit()
            return {
                "ok": False,
                "queued": False,
                "email_sent": None,
                "debug_activation_link": None,
                "error": "email_already_verified",
                "require_502": False,
                "email_token": None,
            }

        expire_stale_sending_under_lock(locked)
        ok_new, block_reason = can_start_new_delivery_snapshot(locked)
        if not ok_new:
            db.session.rollback()
            return {
                "ok": False,
                "queued": False,
                "email_sent": None,
                "debug_activation_link": None,
                "error": block_reason or "email_delivery_in_progress",
                "require_502": False,
                "email_token": None,
            }

        if kind == EMAIL_DELIVERY_KIND_RESEND:
            daily_count = int(locked.resend_count_email or 0)
            now = datetime.now(UTC)
            if locked.last_email_sent_at and not is_same_utc_day(
                locked.last_email_sent_at, now
            ):
                daily_count = 0
            allowed, policy_error, _retry = enforce_resend_policy(
                last_sent_at=locked.last_email_sent_at,
                resend_count=daily_count,
            )
            if not allowed:
                db.session.rollback()
                return {
                    "ok": False,
                    "queued": False,
                    "email_sent": None,
                    "debug_activation_link": None,
                    "error": policy_error or "rate_limited",
                    "require_502": False,
                    "email_token": None,
                }

        if is_prod:
            require_activation_token_key_in_production()
        delivery_id, token = prepare_activation_email_delivery(locked, kind=kind)
        db.session.commit()
    except ActivationTokenKeyError as e:
        db.session.rollback()
        return {
            "ok": False,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": None,
            "error": str(e),
            "require_502": True,
            "email_token": None,
        }
    except Exception as e:
        logger.exception("[activation_email] échec préparation: %s", e)
        db.session.rollback()
        return {
            "ok": False,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": None,
            "error": sanitize_email_error(str(e)),
            "require_502": not is_local_dev and not is_testing,
            "email_token": None,
        }

    if not ready and is_local_dev and token:
        frontend = (
            os.getenv("FRONTEND_URL")
            or os.getenv("PUBLIC_FRONTEND_URL")
            or "http://localhost:3000"
        ).rstrip("/")
        return {
            "ok": True,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": f"{frontend}/activate-account?token={token}",
            "error": None,
            "require_502": False,
            "email_token": token,
        }

    try:
        enqueue_activation_email(
            activation_session_id=session.activation_session_id,
            email_delivery_id=delivery_id or "",
        )
    except Exception as e:
        logger.exception("[activation_email] échec Celery delay: %s", e)
        try:
            locked = get_activation_session_for_update(session_pk)
            mark_delivery_failed(locked, str(e), email_delivery_id=delivery_id)
            db.session.commit()
        except Exception:
            db.session.rollback()
        return {
            "ok": False,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": None,
            "error": sanitize_email_error(str(e)),
            "require_502": is_prod or (not is_local_dev and not is_testing),
            "email_token": None,
        }

    # En local : exposer le lien direct (Brevo réécrit les clics via sendibt*.com,
    # ce qui casse souvent localhost:3000 côté navigateur / anciens emails).
    debug_link = None
    if is_local_dev and token:
        frontend = (
            os.getenv("FRONTEND_URL")
            or os.getenv("PUBLIC_FRONTEND_URL")
            or "http://localhost:3000"
        ).rstrip("/")
        debug_link = f"{frontend}/activate-account?token={token}"

    return {
        "ok": True,
        "queued": True,
        "email_sent": None,
        "debug_activation_link": debug_link,
        "error": None,
        "require_502": False,
        "email_token": token,
    }


def cas_claim_sending(
    session: ActivationSession | int,
    email_delivery_id: str,
) -> str:
    """Claim atomique queued→sending sous verrou session.

    Returns:
        'proceed' | 'ignore'
    """
    session_pk = int(session) if isinstance(session, int) else int(session.id)
    locked = get_activation_session_for_update(session_pk)
    if locked.email_verified_at is not None:
        db.session.commit()
        return "ignore"

    delivery = (
        ActivationEmailDelivery.query.filter_by(email_delivery_id=email_delivery_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if delivery is None:
        db.session.commit()
        return "ignore"
    if delivery.activation_session_pk != locked.id:
        logger.info(
            "activation_email_delivery_ignored reason=not_current email_delivery_id=%s",
            email_delivery_id,
        )
        db.session.commit()
        return "ignore"
    if locked.email_delivery_id != email_delivery_id:
        logger.info(
            "activation_email_delivery_ignored reason=not_current email_delivery_id=%s",
            email_delivery_id,
        )
        db.session.commit()
        return "ignore"
    if delivery.superseded_at is not None:
        logger.info(
            "activation_email_delivery_ignored reason=not_current email_delivery_id=%s",
            email_delivery_id,
        )
        db.session.commit()
        return "ignore"

    now = datetime.now(UTC)
    expires = delivery.token_expires_at
    if expires is None:
        db.session.commit()
        return "ignore"
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    if now >= expires:
        db.session.commit()
        return "ignore"

    if delivery.status == EMAIL_DELIVERY_SENT:
        db.session.commit()
        return "ignore"
    if delivery.status in WEBHOOK_ADVANCED_STATUSES:
        db.session.commit()
        return "ignore"
    if delivery.provider_accepted_at is not None:
        db.session.commit()
        return "ignore"

    if delivery.status == EMAIL_DELIVERY_QUEUED:
        updated = db.session.execute(
            text(
                """
                UPDATE activation_email_deliveries
                SET status = :sending,
                    sending_started_at = :now,
                    updated_at = :now
                WHERE email_delivery_id = :did
                  AND status = :queued
                  AND superseded_at IS NULL
                  AND provider_accepted_at IS NULL
                  AND token_expires_at IS NOT NULL
                  AND token_expires_at > :now
                """
            ),
            {
                "sending": EMAIL_DELIVERY_SENDING,
                "queued": EMAIL_DELIVERY_QUEUED,
                "now": now,
                "did": email_delivery_id,
            },
        ).rowcount
        if int(updated or 0) == 0:
            db.session.commit()
            return "ignore"
        db.session.refresh(delivery)
        sync_current_delivery_mirror(locked, delivery)
        db.session.commit()
        return "proceed"

    if delivery.status == EMAIL_DELIVERY_SENDING:
        # Exactly-once provider hors périmètre F-03 — retry acks_late
        sync_current_delivery_mirror(locked, delivery)
        db.session.commit()
        return "proceed"

    db.session.commit()
    return "ignore"


def resolve_activation_token_for_delivery(email_delivery_id: str) -> str | None:
    """Reconstruit le jeton HMAC si livraison courante, non superseded, non expirée."""
    delivery = get_delivery_by_id(email_delivery_id)
    if delivery is None:
        return None
    if delivery.superseded_at is not None:
        return None
    session = ActivationSession.query.get(delivery.activation_session_pk)
    if session is None or session.email_delivery_id != email_delivery_id:
        return None
    expires = delivery.token_expires_at
    if expires is None:
        return None
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    if datetime.now(UTC) >= expires:
        return None
    try:
        return derive_activation_token(
            email_delivery_id, key_version=int(delivery.token_key_version or 1)
        )
    except ActivationTokenKeyError:
        logger.error(
            "[activation_email] clé manquante pour delivery_id=%s version=%s",
            email_delivery_id,
            delivery.token_key_version,
        )
        return None


get_activation_email_token = resolve_activation_token_for_delivery


def purge_activation_email_token(email_delivery_id: str) -> None:
    del email_delivery_id


def finalize_after_provider_accepted(
    session: ActivationSession | int,
    *,
    email_delivery_id: str,
    message_id: str | None,
) -> bool:
    """Historique provider même si superseded ; miroir seulement si encore courant."""
    session_pk = int(session) if isinstance(session, int) else int(session.id)
    locked = get_activation_session_for_update(session_pk)
    delivery = (
        ActivationEmailDelivery.query.filter_by(email_delivery_id=email_delivery_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if delivery is None or delivery.activation_session_pk != locked.id:
        db.session.commit()
        return False

    now = datetime.now(UTC)
    # Historique : accepter même si superseded (lien invalide de toute façon)
    result = db.session.execute(
        text(
            """
            UPDATE activation_email_deliveries
            SET
              status = CASE
                WHEN status = :sending THEN :sent
                ELSE status
              END,
              provider_message_id = COALESCE(provider_message_id, :mid),
              provider_accepted_at = :now,
              last_error = NULL,
              updated_at = :now
            WHERE email_delivery_id = :did
              AND provider_accepted_at IS NULL
            """
        ),
        {
            "sending": EMAIL_DELIVERY_SENDING,
            "sent": EMAIL_DELIVERY_SENT,
            "mid": message_id,
            "now": now,
            "did": email_delivery_id,
        },
    )
    if int(result.rowcount or 0) == 0:  # type: ignore[attr-defined]
        db.session.commit()
        return False

    db.session.refresh(delivery)

    # Effets session uniquement si encore courant et non superseded
    if locked.email_delivery_id == email_delivery_id and delivery.superseded_at is None:
        previous_sent_at = locked.last_email_sent_at
        if delivery.kind == EMAIL_DELIVERY_KIND_RESEND:
            if previous_sent_at and not is_same_utc_day(previous_sent_at, now):
                locked.resend_count_email = 1
            else:
                locked.resend_count_email = int(locked.resend_count_email or 0) + 1
        locked.last_email_sent_at = now
        locked.email_last_error = None
        sync_current_delivery_mirror(locked, delivery)
    else:
        logger.info(
            "activation_email_delivery_ignored reason=not_current email_delivery_id=%s",
            email_delivery_id,
        )

    db.session.commit()
    return True


def mark_delivery_sent(
    session: ActivationSession,
    *,
    email_delivery_id: str,
    message_id: str | None,
) -> None:
    finalize_after_provider_accepted(
        session,
        email_delivery_id=email_delivery_id,
        message_id=message_id,
    )


def apply_delivery_transition(
    delivery: ActivationEmailDelivery,
    new_status: str,
) -> bool:
    """Applique une transition selon ALLOWED_TRANSITIONS. True si appliquée."""
    current = delivery.status
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    if new_status not in allowed:
        return False
    delivery.status = new_status
    return True


def reconcile_superseded_deliveries() -> dict[str, int]:
    """Réconciliation ops : une courante max par session (pointeur valide)."""
    now = datetime.now(UTC)
    # Marquer toutes sauf la courante pointée par session (même pk)
    result = db.session.execute(
        text(
            """
            UPDATE activation_email_deliveries d
            SET superseded_at = COALESCE(d.superseded_at, :now),
                superseded_by_delivery_id = COALESCE(
                    d.superseded_by_delivery_id,
                    s.email_delivery_id
                ),
                updated_at = :now
            FROM activation_session s
            WHERE d.activation_session_pk = s.id
              AND d.superseded_at IS NULL
              AND (
                s.email_delivery_id IS NULL
                OR d.email_delivery_id <> s.email_delivery_id
                OR NOT EXISTS (
                  SELECT 1 FROM activation_email_deliveries cur
                  WHERE cur.email_delivery_id = s.email_delivery_id
                    AND cur.activation_session_pk = s.id
                )
              )
            """
        ),
        {"now": now},
    )
    db.session.commit()
    return {"superseded_rows": int(result.rowcount or 0)}  # type: ignore[attr-defined]
