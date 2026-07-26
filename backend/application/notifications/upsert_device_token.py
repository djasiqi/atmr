"""Upsert un DeviceToken par (propriétaire, device_id) avec fallback legacy driver."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError

from ext import app_logger, db
from models import DeviceToken
from services.notifications.push_token_platform import infer_fcm_platform


def _normalize_provider(token: str, provider: str | None) -> str:
    if provider in ("expo", "fcm"):
        return provider
    return "fcm" if not token.startswith("ExponentPushToken") else "expo"


def _find_row_by_unique_key(
    *,
    driver_id: int | None,
    company_id: int | None,
    device_id: str | None,
    provider: str,
) -> DeviceToken | None:
    """Récupère la ligne correspondant à la contrainte unique (owner, device_id, provider).

    Utilisé pour résoudre une race d'insertion concurrente (UniqueViolation).
    """
    device_id_str = str(device_id).strip() if device_id else None
    if not device_id_str:
        return None
    q = DeviceToken.query.filter_by(device_id=device_id_str, provider=provider)
    if driver_id is not None:
        q = q.filter_by(driver_id=driver_id)
    elif company_id is not None:
        q = q.filter_by(company_id=company_id)
    else:
        return None
    return q.first()


def _resolve_row_after_unique_violation(
    *,
    driver_id: int | None,
    company_id: int | None,
    device_id: str | None,
    provider: str,
    attempts: int = 3,
) -> DeviceToken | None:
    """Récupère la ligne existante après une course d'insertion concurrente.

    Une requête concurrente peut committer entre le SAVEPOINT rollback et notre
    SELECT ; un court backoff évite le PendingRollbackError au commit parent.
    """
    for attempt in range(attempts):
        existing = _find_row_by_unique_key(
            driver_id=driver_id,
            company_id=company_id,
            device_id=device_id,
            provider=provider,
        )
        if existing is not None:
            return existing
        if attempt + 1 < attempts:
            time.sleep(0.05 * (attempt + 1))
    return None


def _apply_token_update_to_row(
    row: DeviceToken,
    *,
    driver_id: int | None,
    company_id: int | None,
    device_id: str | None,
    token: str,
    platform: str | None,
    provider: str,
    now: datetime,
) -> DeviceToken:
    row.token = token
    row.is_active = True
    row.updated_at = now
    row.last_seen_at = now
    if device_id and not row.device_id:
        row.device_id = str(device_id).strip()
    if platform:
        row.platform = platform
    row.provider = provider
    if driver_id is not None:
        row.driver_id = driver_id
    if company_id is not None:
        row.company_id = company_id
    _deactivate_other_rows_with_same_token(
        driver_id=driver_id,
        company_id=company_id,
        token=token,
        keep_row_id=row.id,
    )
    if (
        driver_id is not None
        and provider == "fcm"
        and (platform or "").lower() == "android"
    ):
        _deactivate_stale_android_fcm_for_driver(
            driver_id=driver_id,
            keep_row_id=row.id,
        )
        _deactivate_android_expo_legacy_for_driver(
            driver_id=driver_id,
            keep_row_id=row.id,
        )
    return row


def _deactivate_other_rows_with_same_token(
    *,
    driver_id: int | None,
    company_id: int | None,
    token: str,
    keep_row_id: int | None,
) -> int:
    """Désactive toutes les autres lignes actives portant le même `token` pour l'owner.

    Cas réel : un même token FCM/Expo peut se retrouver attaché à plusieurs
    `device_id` (réinstall app, rotation Expo/Installation ID, etc.). On garde
    une seule ligne active par `token` pour éviter le fan-out qui multipliait
    les notifications. La ligne conservée est celle passée via `keep_row_id`
    (l'upsert courant).
    """
    if not token:
        return 0
    if driver_id is None and company_id is None:
        return 0
    q = DeviceToken.query.filter(
        DeviceToken.token == token,
        DeviceToken.is_active.is_(True),
    )
    if driver_id is not None:
        q = q.filter(DeviceToken.driver_id == driver_id)
    elif company_id is not None:
        q = q.filter(DeviceToken.company_id == company_id)
    if keep_row_id is not None:
        q = q.filter(DeviceToken.id != keep_row_id)
    count = q.update({"is_active": False}, synchronize_session=False)
    count = int(count or 0)
    if count > 0:
        app_logger.info(
            "[push-token] Désactivation %s ligne(s) doublon(s) du même token "
            "owner driver=%s company=%s keep_row=%s",
            count,
            driver_id,
            company_id,
            keep_row_id,
        )
    return count


def _deactivate_stale_android_fcm_for_driver(
    *,
    driver_id: int,
    keep_row_id: int | None,
) -> int:
    """Désactive les anciens tokens FCM Android (rotation device_id, réinstall)."""
    q = DeviceToken.query.filter(
        DeviceToken.driver_id == driver_id,
        DeviceToken.provider == "fcm",
        DeviceToken.platform == "android",
        DeviceToken.is_active.is_(True),
    )
    if keep_row_id is not None:
        q = q.filter(DeviceToken.id != keep_row_id)
    count = q.update({"is_active": False}, synchronize_session=False)
    count = int(count or 0)
    if count > 0:
        app_logger.info(
            "[push-token] Désactivation %s token(s) FCM Android obsolète(s) driver_id=%s",
            count,
            driver_id,
        )
    return count


def _deactivate_android_expo_legacy_for_driver(
    *,
    driver_id: int,
    keep_row_id: int | None,
) -> int:
    """Désactive les tokens Expo Android obsolètes quand un FCM Android est enregistré."""
    q = DeviceToken.query.filter(
        DeviceToken.driver_id == driver_id,
        DeviceToken.provider == "expo",
        DeviceToken.platform == "android",
        DeviceToken.is_active.is_(True),
    )
    if keep_row_id is not None:
        q = q.filter(DeviceToken.id != keep_row_id)
    count = q.update({"is_active": False}, synchronize_session=False)
    count = int(count or 0)
    if count > 0:
        app_logger.info(
            "[push-token] Désactivation %s token(s) Expo Android legacy driver_id=%s (FCM actif)",
            count,
            driver_id,
        )
    return count


def upsert_device_token(
    *,
    driver_id: int | None = None,
    company_id: int | None = None,
    device_id: str | None,
    token: str,
    platform: str | None = None,
    provider: str | None = None,
) -> DeviceToken:
    """Crée ou met à jour un DeviceToken (clé logique owner + device_id).

    - Driver : device_id recommandé ; fallback (driver_id, token) si absent.
    - Company : device_id obligatoire (ValueError si absent).
    """
    has_driver = driver_id is not None
    has_company = company_id is not None
    if has_driver == has_company:
        msg = "Exactement un de driver_id ou company_id doit être fourni."
        raise ValueError(msg)

    resolved_provider = _normalize_provider(token, provider)
    now = datetime.now(UTC)

    inferred_platform = infer_fcm_platform(
        token, platform if isinstance(platform, str) else None
    )
    if (
        resolved_provider == "fcm"
        and platform == "ios"
        and inferred_platform == "android"
    ):
        app_logger.warning(
            "[push-token] platform ios->android inferred for FCM Android token owner driver=%s company=%s",
            driver_id,
            company_id,
        )
        platform = "android"
    elif inferred_platform:
        platform = inferred_platform

    row: DeviceToken | None = None

    if company_id is not None:
        if not device_id or not str(device_id).strip():
            msg = "device_id obligatoire pour l'enregistrement push entreprise."
            raise ValueError(msg)
        row = DeviceToken.query.filter_by(
            company_id=company_id,
            device_id=str(device_id).strip(),
            provider=resolved_provider,
        ).first()
    elif driver_id is not None:
        device_id_str = str(device_id).strip() if device_id else None
        if device_id_str:
            row = DeviceToken.query.filter_by(
                driver_id=driver_id,
                device_id=device_id_str,
                provider=resolved_provider,
            ).first()
        else:
            app_logger.warning(
                "[push-token] push.device_id_missing driver_id=%s — fallback (driver_id, token)",
                driver_id,
            )
            row = DeviceToken.query.filter_by(
                driver_id=driver_id,
                token=token,
            ).first()

    if row is not None:
        _apply_token_update_to_row(
            row,
            driver_id=driver_id,
            company_id=company_id,
            device_id=device_id,
            token=token,
            platform=platform,
            provider=resolved_provider,
            now=now,
        )
        app_logger.info(
            "[push-token] Token mis à jour owner driver=%s company=%s device_id=%s provider=%s",
            driver_id,
            company_id,
            row.device_id,
            resolved_provider,
        )
        return row

    row = DeviceToken()
    row.driver_id = driver_id
    row.company_id = company_id
    row.token = token
    row.device_id = str(device_id).strip() if device_id else None
    row.platform = platform
    row.provider = resolved_provider
    row.is_active = True
    row.created_at = now
    row.updated_at = now
    row.last_seen_at = now
    db.session.add(row)
    try:
        # SAVEPOINT : isole l'INSERT pour pouvoir le rejouer sans casser la
        # transaction parente si une requête concurrente gagne la course.
        with db.session.begin_nested():
            db.session.flush()
    except IntegrityError:
        # Race condition : une requête concurrente a déjà inséré la même clé
        # logique (owner, device_id, provider). Le SAVEPOINT est annulé ; on
        # récupère la ligne existante et on la met à jour plutôt que d'échouer.
        if row in db.session:
            db.session.expunge(row)
        existing = _resolve_row_after_unique_violation(
            driver_id=driver_id,
            company_id=company_id,
            device_id=device_id,
            provider=resolved_provider,
        )
        if existing is None:
            raise
        app_logger.warning(
            "[push-token] Race d'insertion résolue (upsert concurrent) "
            "owner driver=%s company=%s device_id=%s provider=%s",
            driver_id,
            company_id,
            existing.device_id,
            resolved_provider,
        )
        return _apply_token_update_to_row(
            existing,
            driver_id=driver_id,
            company_id=company_id,
            device_id=device_id,
            token=token,
            platform=platform,
            provider=resolved_provider,
            now=now,
        )
    _deactivate_other_rows_with_same_token(
        driver_id=driver_id,
        company_id=company_id,
        token=token,
        keep_row_id=row.id,
    )
    app_logger.info(
        "[push-token] Nouveau token owner driver=%s company=%s device_id=%s provider=%s",
        driver_id,
        company_id,
        row.device_id,
        resolved_provider,
    )
    if (
        driver_id is not None
        and resolved_provider == "fcm"
        and (platform or "").lower() == "android"
    ):
        _deactivate_stale_android_fcm_for_driver(
            driver_id=driver_id,
            keep_row_id=row.id,
        )
        _deactivate_android_expo_legacy_for_driver(
            driver_id=driver_id,
            keep_row_id=row.id,
        )
    return row


def deactivate_device_tokens_for_logout(
    *,
    driver_id: int | None = None,
    company_id: int | None = None,
    device_id: str,
) -> int:
    """Désactive les tokens push pour un appareil uniquement (logout ciblé)."""
    device_id = str(device_id).strip()
    if not device_id:
        return 0

    q = DeviceToken.query.filter_by(device_id=device_id, is_active=True)
    if driver_id is not None:
        q = q.filter_by(driver_id=driver_id)
    elif company_id is not None:
        q = q.filter_by(company_id=company_id)
    else:
        return 0

    count = q.update({"is_active": False}, synchronize_session=False)
    return int(count or 0)
