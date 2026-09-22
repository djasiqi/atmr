# backend/services/notifications/sms.py
"""Envoi SMS via Twilio Programmable Messaging.

OTP d'activation : le code est généré et stocké par LIRIE (hash SHA-256).
Twilio Verify n'est pas utilisé. Voir chantier AUTH-SMS-02 pour une migration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from services.notifications.phone_e164 import mask_phone_for_log, normalize_e164_phone

logger = logging.getLogger(__name__)

PROVIDER_NAME = "twilio_programmable_messaging"

ERROR_DISABLED = "DISABLED"
ERROR_CONFIG = "CONFIG_ERROR"
ERROR_AUTH = "AUTH_ERROR"
ERROR_SENDER = "SENDER_ERROR"
ERROR_DESTINATION = "DESTINATION_ERROR"
ERROR_REJECTION = "TWILIO_REJECTION"
ERROR_DELIVERY = "DELIVERY_FAILURE"
ERROR_SUCCESS = "SUCCESS"

_AUTH_CODES = {20001, 20003}
_SENDER_CODES = {21212, 21606, 21612, 21659}
_DESTINATION_CODES = {21211, 21214, 21408, 21610, 21614}
_REJECTION_CODES = {20404, 21608, 30007}
_DELIVERY_CODES = {30003, 30004, 30005, 30006, 30008}


@dataclass(frozen=True, slots=True)
class SmsProviderConfig:
    enabled: bool
    account_sid: str | None
    auth_token: str | None
    phone_number: str | None
    messaging_service_sid: str | None

    @property
    def has_credentials(self) -> bool:
        return bool(self.account_sid and self.auth_token)

    @property
    def has_sender(self) -> bool:
        return bool(self.phone_number or self.messaging_service_sid)

    @property
    def ready(self) -> bool:
        return self.enabled and self.has_credentials and self.has_sender

    @property
    def sender_mode(self) -> str:
        if self.messaging_service_sid:
            return "messaging_service"
        if self.phone_number:
            return "from_number"
        return "missing"


def _nonempty_env(name: str) -> str | None:
    value = (os.getenv(name) or "").strip()
    return value or None


def get_sms_config() -> SmsProviderConfig:
    """Lit la configuration SMS au moment de l'appel (pas à l'import)."""
    return SmsProviderConfig(
        enabled=os.getenv("SMS_NOTIFICATIONS_ENABLED", "false").strip().lower()
        == "true",
        account_sid=_nonempty_env("TWILIO_ACCOUNT_SID"),
        auth_token=_nonempty_env("TWILIO_AUTH_TOKEN"),
        phone_number=_nonempty_env("TWILIO_PHONE_NUMBER"),
        messaging_service_sid=_nonempty_env("TWILIO_MESSAGING_SERVICE_SID"),
    )


def mask_secret_id(
    value: str | None, *, prefix_len: int = 2, suffix_len: int = 4
) -> str:
    """Masque un identifiant (SID, message SID). Ne jamais logger la valeur brute."""
    raw = (value or "").strip()
    if not raw:
        return "MISSING"
    if len(raw) <= prefix_len + suffix_len:
        return "*" * len(raw)
    return f"{raw[:prefix_len]}{'*' * (len(raw) - prefix_len - suffix_len)}{raw[-suffix_len:]}"


def describe_sms_config(config: SmsProviderConfig | None = None) -> dict[str, Any]:
    """État de configuration sans secret (présence uniquement)."""
    cfg = config or get_sms_config()
    return {
        "provider": PROVIDER_NAME,
        "enabled": cfg.enabled,
        "ready": cfg.ready,
        "twilio_account_sid": "PRESENT" if cfg.account_sid else "MISSING",
        "twilio_auth_token": "PRESENT" if cfg.auth_token else "MISSING",
        "twilio_phone_number": "PRESENT" if cfg.phone_number else "MISSING",
        "twilio_messaging_service_sid": (
            "PRESENT" if cfg.messaging_service_sid else "MISSING"
        ),
        "sender_mode": cfg.sender_mode,
        "twilio_mode": "PROGRAMMABLE_MESSAGING",
    }


def log_sms_provider_readiness(target_logger: logging.Logger | None = None) -> None:
    """Log de démarrage : canal SMS prêt ou non, sans secret."""
    log = target_logger or logger
    snapshot = describe_sms_config()
    status = "READY" if snapshot["ready"] else "NOT_READY"
    log.info(
        "sms_provider_status=%s enabled=%s credentials=%s sender=%s "
        "sender_mode=%s provider=%s",
        status,
        snapshot["enabled"],
        (
            "PRESENT"
            if snapshot["twilio_account_sid"] == "PRESENT"
            and snapshot["twilio_auth_token"] == "PRESENT"
            else "MISSING"
        ),
        (
            "PRESENT"
            if snapshot["twilio_phone_number"] == "PRESENT"
            or snapshot["twilio_messaging_service_sid"] == "PRESENT"
            else "MISSING"
        ),
        snapshot["sender_mode"],
        snapshot["provider"],
    )


def _classify_twilio_error(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    try:
        code_int = int(code) if code is not None else None
    except (TypeError, ValueError):
        code_int = None
    try:
        status_int = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_int = None

    if code_int in _AUTH_CODES or status_int == 401:
        return ERROR_AUTH
    if code_int in _SENDER_CODES:
        return ERROR_SENDER
    if code_int in _DESTINATION_CODES:
        return ERROR_DESTINATION
    if code_int in _REJECTION_CODES:
        return ERROR_REJECTION
    if code_int in _DELIVERY_CODES:
        return ERROR_DELIVERY
    if status_int in {401, 403}:
        return ERROR_AUTH
    message = str(exc).lower()
    if "authenticate" in message or ("auth" in message and "token" in message):
        return ERROR_AUTH
    if "from" in message and "phone" in message:
        return ERROR_SENDER
    if "to" in message and "phone" in message:
        return ERROR_DESTINATION
    return ERROR_DELIVERY


def _result(
    *,
    ok: bool,
    error_class: str,
    error: str | None = None,
    disabled: bool = False,
    message_sid: str | None = None,
    provider_status: str | None = None,
    provider_error_code: str | None = None,
    destination: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "error": error,
        "error_class": error_class,
        "disabled": disabled,
        "message_sid": mask_secret_id(message_sid) if message_sid else None,
        "status": provider_status,
        "provider": PROVIDER_NAME,
        "provider_status": provider_status,
        "provider_error_code": provider_error_code,
        "destination_masked": mask_phone_for_log(destination) if destination else None,
    }


def send_sms_notification(
    phone: str,
    message: str,
    notification_type: str = "unknown",
) -> dict[str, Any]:
    """Envoie un SMS via Twilio Programmable Messaging.

    Ne jamais logger ``message`` (peut contenir un OTP) ni les credentials.
    """
    cfg = get_sms_config()
    destination = normalize_e164_phone(phone)
    if not destination:
        logger.warning(
            "sms_verification_provider_failed error_class=%s phone=%s type=%s",
            ERROR_DESTINATION,
            mask_phone_for_log(phone),
            notification_type,
        )
        return _result(
            ok=False,
            error_class=ERROR_DESTINATION,
            error="invalid_phone",
            destination=phone,
        )

    if not cfg.enabled:
        logger.info(
            "sms_verification_provider_failed error_class=%s phone=%s type=%s",
            ERROR_DISABLED,
            mask_phone_for_log(destination),
            notification_type,
        )
        return _result(
            ok=False,
            error_class=ERROR_DISABLED,
            error="SMS notifications disabled",
            disabled=True,
            destination=destination,
        )

    if not cfg.has_credentials:
        logger.error(
            "sms_verification_provider_failed error_class=%s phone=%s type=%s",
            ERROR_CONFIG,
            mask_phone_for_log(destination),
            notification_type,
        )
        return _result(
            ok=False,
            error_class=ERROR_CONFIG,
            error="Twilio credentials missing",
            destination=destination,
        )

    if not cfg.has_sender:
        logger.error(
            "sms_verification_provider_failed error_class=%s phone=%s type=%s",
            ERROR_SENDER,
            mask_phone_for_log(destination),
            notification_type,
        )
        return _result(
            ok=False,
            error_class=ERROR_SENDER,
            error="Twilio sender missing",
            destination=destination,
        )

    try:
        try:
            from twilio.rest import Client  # type: ignore
        except ImportError:
            logger.error(
                "sms_verification_provider_failed error_class=%s reason=twilio_package_missing",
                ERROR_CONFIG,
            )
            return _result(
                ok=False,
                error_class=ERROR_CONFIG,
                error="twilio package not installed",
                destination=destination,
            )

        client = Client(cfg.account_sid, cfg.auth_token)
        create_kwargs: dict[str, Any] = {
            "body": message,
            "to": destination,
        }
        if cfg.messaging_service_sid:
            create_kwargs["messaging_service_sid"] = cfg.messaging_service_sid
        else:
            create_kwargs["from_"] = cfg.phone_number

        tw_message = client.messages.create(**create_kwargs)
        sid_masked = mask_secret_id(getattr(tw_message, "sid", None))
        provider_status = getattr(tw_message, "status", None)
        logger.info(
            "sms_verification_provider_accepted phone=%s type=%s "
            "message_sid=%s provider_status=%s sender_mode=%s provider=%s",
            mask_phone_for_log(destination),
            notification_type,
            sid_masked,
            provider_status,
            cfg.sender_mode,
            PROVIDER_NAME,
        )
        return _result(
            ok=True,
            error_class=ERROR_SUCCESS,
            message_sid=getattr(tw_message, "sid", None),
            provider_status=str(provider_status) if provider_status else None,
            destination=destination,
        )

    except Exception as exc:
        error_class = _classify_twilio_error(exc)
        provider_error_code = getattr(exc, "code", None)
        logger.warning(
            "sms_verification_provider_failed error_class=%s phone=%s type=%s "
            "provider_error_code=%s provider=%s",
            error_class,
            mask_phone_for_log(destination),
            notification_type,
            provider_error_code or "-",
            PROVIDER_NAME,
        )
        return _result(
            ok=False,
            error_class=error_class,
            error="Envoi SMS refusé par le fournisseur",
            provider_error_code=(
                str(provider_error_code) if provider_error_code is not None else None
            ),
            destination=destination,
        )


def send_bulk_sms(
    recipients: list[tuple[str, str]],
    notification_type: str = "unknown",
) -> dict[str, Any]:
    """Envoie des SMS en masse."""
    success_count = 0
    failed_count = 0
    errors = []

    for phone, message in recipients:
        result = send_sms_notification(phone, message, notification_type)
        if result.get("ok"):
            success_count += 1
        else:
            failed_count += 1
            errors.append(
                {
                    "phone": mask_phone_for_log(phone),
                    "error": result.get("error"),
                    "error_class": result.get("error_class"),
                }
            )

    return {
        "ok": failed_count == 0,
        "total": len(recipients),
        "success": success_count,
        "failed": failed_count,
        "errors": errors if errors else None,
    }
