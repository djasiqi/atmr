"""Upsert un DeviceToken par (propriétaire, device_id) avec fallback legacy driver."""

from __future__ import annotations

from datetime import UTC, datetime
from ext import app_logger, db
from models import DeviceToken


def _normalize_provider(token: str, provider: str | None) -> str:
    if provider in ("expo", "fcm"):
        return provider
    return "fcm" if not token.startswith("ExponentPushToken") else "expo"


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

    if (
        resolved_provider == "fcm"
        and platform == "ios"
        and token.startswith(("APA91", "APA91b"))
    ):
        app_logger.warning(
            "[push-token] platform ios->android inferred for FCM Android token owner driver=%s company=%s",
            driver_id,
            company_id,
        )
        platform = "android"

    row: DeviceToken | None = None

    if company_id is not None:
        if not device_id or not str(device_id).strip():
            msg = "device_id obligatoire pour l'enregistrement push entreprise."
            raise ValueError(msg)
        row = DeviceToken.query.filter_by(
            company_id=company_id,
            device_id=str(device_id).strip(),
        ).first()
    elif driver_id is not None:
        device_id_str = str(device_id).strip() if device_id else None
        if device_id_str:
            row = DeviceToken.query.filter_by(
                driver_id=driver_id,
                device_id=device_id_str,
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
        row.token = token
        row.is_active = True
        row.updated_at = now
        row.last_seen_at = now
        if device_id and not row.device_id:
            row.device_id = str(device_id).strip()
        if platform:
            row.platform = platform
        row.provider = resolved_provider
        if driver_id is not None:
            row.driver_id = driver_id
        if company_id is not None:
            row.company_id = company_id
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
    app_logger.info(
        "[push-token] Nouveau token owner driver=%s company=%s device_id=%s provider=%s",
        driver_id,
        company_id,
        row.device_id,
        resolved_provider,
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
