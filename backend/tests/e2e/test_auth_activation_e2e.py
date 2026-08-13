"""Tests E2E : Parcours d'activation client (email + SMS)."""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from models import ActivationSession
from models.activation_email_delivery import (
    EMAIL_DELIVERY_KIND_INITIAL,
    EMAIL_DELIVERY_QUEUED,
    EMAIL_DELIVERY_SENT,
    ActivationEmailDelivery,
)
from services.notifications.activation_token import (
    derive_activation_token,
    hash_activation_token,
)
from tests.e2e.helpers.e2e_helpers import unique_phone


def _ensure_hmac_activation_token(
    db,
    session: ActivationSession,
    *,
    expires_delta: timedelta | None = None,
) -> str:
    """Crée/aligne une livraison HMAC courante et retourne le jeton."""
    os.environ.setdefault("ACTIVATION_TOKEN_KEY_V1", "test-activation-key-v1")
    did = session.email_delivery_id or str(uuid.uuid4())
    token = derive_activation_token(did, key_version=1)
    token_hash = hash_activation_token(token)
    expires = datetime.now(UTC) + (
        expires_delta if expires_delta is not None else timedelta(minutes=30)
    )
    delivery = ActivationEmailDelivery.query.filter_by(email_delivery_id=did).first()
    if delivery is None:
        delivery = ActivationEmailDelivery(
            activation_session_pk=session.id,
            email_delivery_id=did,
            kind=EMAIL_DELIVERY_KIND_INITIAL,
            status=EMAIL_DELIVERY_QUEUED,
            token_key_version=1,
            email_token_hash=token_hash,
            token_expires_at=expires,
            superseded_at=None,
        )
        db.session.add(delivery)
    else:
        delivery.email_token_hash = token_hash
        delivery.token_expires_at = expires
        delivery.superseded_at = None
        delivery.activation_session_pk = session.id
    session.email_delivery_id = did
    session.email_token_hash = token_hash
    session.email_token_expires_at = expires
    session.email_delivery_status = EMAIL_DELIVERY_QUEUED
    session.email_delivery_kind = EMAIL_DELIVERY_KIND_INITIAL
    db.session.commit()
    return token


def _mark_current_delivery_sent(db, session: ActivationSession) -> None:
    """Libère le blocage F-03 queued/sending pour permettre un renvoi de test."""
    if not session.email_delivery_id:
        return
    delivery = ActivationEmailDelivery.query.filter_by(
        email_delivery_id=session.email_delivery_id
    ).first()
    if delivery is None:
        return
    delivery.status = EMAIL_DELIVERY_SENT
    delivery.provider_accepted_at = datetime.now(UTC)
    session.email_delivery_status = EMAIL_DELIVERY_SENT
    db.session.commit()


class TestAuthActivationFlow:
    """Tests : register -> verify-email -> verify-sms -> finalize -> login."""

    def test_e2e_activation_flow_register_to_login(self, e2e_client, db):
        unique_suffix = str(uuid.uuid4())[:8]
        password = f"UniqueTestPass123!{unique_suffix}"
        register_payload = {
            "username": f"activation_{unique_suffix}",
            "email": f"activation_{unique_suffix}@example.com",
            "password": password,
            "first_name": "Alice",
            "last_name": "Activation",
            "phone": unique_phone(),
            "address": "Rue de Test 1, 1200 Geneve",
        }

        # 1) Register -> session d'activation créée
        register_response = e2e_client.post(
            "/api/v1/auth/register",
            json=register_payload,
            headers={"Content-Type": "application/json"},
        )
        assert register_response.status_code == 201, (
            f"Register doit reussir, recu {register_response.status_code}: "
            f"{register_response.get_json()}"
        )
        register_data = register_response.get_json() or {}
        activation_session_id = register_data.get("activation_session_id")
        assert activation_session_id, "activation_session_id manquant apres register"
        assert register_data.get("email_sent") is None
        assert "activation_email_queued" in register_data

        # 2) Login bloqué avant activation complète
        login_before_response = e2e_client.post(
            "/api/v1/auth/login",
            json={
                "email": register_payload["email"],
                "password": register_payload["password"],
            },
            headers={"Content-Type": "application/json"},
        )
        assert login_before_response.status_code == 403, (
            "Le login doit etre bloque avant activation."
        )
        login_before_data = login_before_response.get_json() or {}
        assert login_before_data.get("reason") == "account_pending_activation"

        # 3) Verify-email (jeton HMAC livraison courante)
        session = ActivationSession.query.filter_by(
            activation_session_id=activation_session_id
        ).first()
        assert session is not None, "Session d'activation introuvable en base"
        email_token = _ensure_hmac_activation_token(db, session)

        verify_email_response = e2e_client.post(
            "/api/v1/auth/activation/verify-email",
            json={"token": email_token},
            headers={"Content-Type": "application/json"},
        )
        assert verify_email_response.status_code == 200, (
            f"verify-email doit reussir, recu {verify_email_response.status_code}: "
            f"{verify_email_response.get_json()}"
        )

        # 4) Préparer un code SMS connu puis verify-sms
        known_sms_code = "123456"
        session = ActivationSession.query.filter_by(
            activation_session_id=activation_session_id
        ).first()
        assert session is not None, "Session d'activation introuvable en base"
        session.sms_code_hash = hashlib.sha256(
            known_sms_code.encode("utf-8")
        ).hexdigest()
        session.sms_attempts = 0
        session.sms_locked_until = None
        db.session.commit()

        verify_sms_response = e2e_client.post(
            "/api/v1/auth/activation/verify-sms",
            json={
                "activation_session_id": activation_session_id,
                "code": known_sms_code,
            },
            headers={"Content-Type": "application/json"},
        )
        assert verify_sms_response.status_code == 200, (
            f"verify-sms doit reussir, recu {verify_sms_response.status_code}: "
            f"{verify_sms_response.get_json()}"
        )

        # 5) Finalize activation
        finalize_response = e2e_client.post(
            "/api/v1/auth/activation/finalize",
            json={"activation_session_id": activation_session_id},
            headers={"Content-Type": "application/json"},
        )
        assert finalize_response.status_code == 200, (
            f"finalize doit reussir, recu {finalize_response.status_code}: "
            f"{finalize_response.get_json()}"
        )
        finalize_data = finalize_response.get_json() or {}
        assert "user_id" in finalize_data

        # 6) Login autorisé après activation
        login_after_response = e2e_client.post(
            "/api/v1/auth/login",
            json={
                "email": register_payload["email"],
                "password": register_payload["password"],
            },
            headers={"Content-Type": "application/json"},
        )
        assert login_after_response.status_code == 200, (
            f"Le login doit reussir apres activation, recu {login_after_response.status_code}: "
            f"{login_after_response.get_json()}"
        )

    def test_e2e_activation_sms_guardrails_lock_and_resend_cooldown(
        self, e2e_client, db
    ):
        unique_suffix = str(uuid.uuid4())[:8]
        password = f"GuardrailPass123!{unique_suffix}"
        register_payload = {
            "username": f"guardrail_{unique_suffix}",
            "email": f"guardrail_{unique_suffix}@example.com",
            "password": password,
            "first_name": "Bob",
            "last_name": "Guardrail",
            "phone": unique_phone(),
            "address": "Rue de Garde 2, 1200 Geneve",
        }

        register_response = e2e_client.post(
            "/api/v1/auth/register",
            json=register_payload,
            headers={"Content-Type": "application/json"},
        )
        assert register_response.status_code == 201, (
            f"Register doit reussir, recu {register_response.status_code}: "
            f"{register_response.get_json()}"
        )
        activation_session_id = (register_response.get_json() or {}).get(
            "activation_session_id"
        )
        assert activation_session_id, "activation_session_id manquant apres register"

        # Préparer un code connu pour piloter les erreurs OTP de manière déterministe.
        known_sms_code = "123456"
        session = ActivationSession.query.filter_by(
            activation_session_id=activation_session_id
        ).first()
        assert session is not None, "Session d'activation introuvable en base"
        session.sms_code_hash = hashlib.sha256(
            known_sms_code.encode("utf-8")
        ).hexdigest()
        session.sms_attempts = 0
        session.sms_locked_until = None
        session.phone_verified_at = None
        db.session.commit()

        # 4 erreurs OTP -> 400 + remaining_attempts
        for expected_remaining in [4, 3, 2, 1]:
            wrong_otp_response = e2e_client.post(
                "/api/v1/auth/activation/verify-sms",
                json={
                    "activation_session_id": activation_session_id,
                    "code": "000000",
                },
                headers={"Content-Type": "application/json"},
            )
            assert wrong_otp_response.status_code == 400, (
                f"OTP faux doit renvoyer 400, recu {wrong_otp_response.status_code}: "
                f"{wrong_otp_response.get_json()}"
            )
            wrong_otp_data = wrong_otp_response.get_json() or {}
            details = wrong_otp_data.get("details") or {}
            assert details.get("remaining_attempts") == expected_remaining

        # 5e erreur OTP -> 429 lockout
        lock_response = e2e_client.post(
            "/api/v1/auth/activation/verify-sms",
            json={
                "activation_session_id": activation_session_id,
                "code": "000000",
            },
            headers={"Content-Type": "application/json"},
        )
        assert lock_response.status_code == 429, (
            f"Le lockout OTP doit renvoyer 429, recu {lock_response.status_code}: "
            f"{lock_response.get_json()}"
        )
        lock_data = lock_response.get_json() or {}
        lock_details = lock_data.get("details") or {}
        assert int(lock_details.get("retry_after_seconds") or 0) > 0

        # Simuler un SMS tout juste envoyé pour vérifier le cooldown resend.
        session = ActivationSession.query.filter_by(
            activation_session_id=activation_session_id
        ).first()
        assert session is not None
        session.last_sms_sent_at = datetime.now(UTC)
        session.resend_count_sms = 1
        db.session.commit()

        resend_response = e2e_client.post(
            "/api/v1/auth/activation/resend-sms",
            json={"activation_session_id": activation_session_id},
            headers={"Content-Type": "application/json"},
        )
        assert resend_response.status_code == 429, (
            f"Le cooldown resend-sms doit renvoyer 429, recu {resend_response.status_code}: "
            f"{resend_response.get_json()}"
        )
        resend_data = resend_response.get_json() or {}
        resend_details = resend_data.get("details") or {}
        assert int(resend_details.get("retry_after_seconds") or 0) > 0

    def test_e2e_activation_verify_email_expired_token(self, e2e_client, db):
        unique_suffix = str(uuid.uuid4())[:8]
        password = f"ExpiredMailPass123!{unique_suffix}"
        register_payload = {
            "username": f"expiredmail_{unique_suffix}",
            "email": f"expiredmail_{unique_suffix}@example.com",
            "password": password,
            "first_name": "Eve",
            "last_name": "Expired",
            "phone": unique_phone(),
            "address": "Rue Expiree 3, 1200 Geneve",
        }

        register_response = e2e_client.post(
            "/api/v1/auth/register",
            json=register_payload,
            headers={"Content-Type": "application/json"},
        )
        assert register_response.status_code == 201, (
            f"Register doit reussir, recu {register_response.status_code}: "
            f"{register_response.get_json()}"
        )
        activation_session_id = (register_response.get_json() or {}).get(
            "activation_session_id"
        )
        assert activation_session_id, "activation_session_id manquant apres register"

        session = ActivationSession.query.filter_by(
            activation_session_id=activation_session_id
        ).first()
        assert session is not None, "Session d'activation introuvable en base"
        session.email_verified_at = None
        email_token = _ensure_hmac_activation_token(
            db, session, expires_delta=timedelta(minutes=-1)
        )

        verify_email_response = e2e_client.post(
            "/api/v1/auth/activation/verify-email",
            json={"token": email_token},
            headers={"Content-Type": "application/json"},
        )
        assert verify_email_response.status_code == 400, (
            f"verify-email expire doit renvoyer 400, recu {verify_email_response.status_code}: "
            f"{verify_email_response.get_json()}"
        )
        verify_email_data = verify_email_response.get_json() or {}
        assert verify_email_data.get("error") == "token_expired"

    def test_e2e_activation_finalize_refused_when_sms_not_verified(
        self, e2e_client, db
    ):
        unique_suffix = str(uuid.uuid4())[:8]
        password = f"FinalizeGuardPass123!{unique_suffix}"
        register_payload = {
            "username": f"finalize_guard_{unique_suffix}",
            "email": f"finalize_guard_{unique_suffix}@example.com",
            "password": password,
            "first_name": "Nina",
            "last_name": "Finalize",
            "phone": unique_phone(),
            "address": "Rue Finalize 4, 1200 Geneve",
        }

        register_response = e2e_client.post(
            "/api/v1/auth/register",
            json=register_payload,
            headers={"Content-Type": "application/json"},
        )
        assert register_response.status_code == 201, (
            f"Register doit reussir, recu {register_response.status_code}: "
            f"{register_response.get_json()}"
        )
        activation_session_id = (register_response.get_json() or {}).get(
            "activation_session_id"
        )
        assert activation_session_id, "activation_session_id manquant apres register"

        # Email confirmé, SMS non confirmé.
        session = ActivationSession.query.filter_by(
            activation_session_id=activation_session_id
        ).first()
        assert session is not None, "Session d'activation introuvable en base"
        session.phone_verified_at = None
        email_token = _ensure_hmac_activation_token(db, session)

        verify_email_response = e2e_client.post(
            "/api/v1/auth/activation/verify-email",
            json={"token": email_token},
            headers={"Content-Type": "application/json"},
        )
        assert verify_email_response.status_code == 200, (
            f"verify-email doit reussir, recu {verify_email_response.status_code}: "
            f"{verify_email_response.get_json()}"
        )

        finalize_response = e2e_client.post(
            "/api/v1/auth/activation/finalize",
            json={"activation_session_id": activation_session_id},
            headers={"Content-Type": "application/json"},
        )
        assert finalize_response.status_code == 400, (
            f"finalize doit etre refuse sans SMS confirme, recu {finalize_response.status_code}: "
            f"{finalize_response.get_json()}"
        )
        finalize_data = finalize_response.get_json() or {}
        assert finalize_data.get("error") == "email_not_verified"
        details = finalize_data.get("details") or {}
        assert details.get("email_verified") is True
        assert details.get("phone_verified") is False

    def test_e2e_activation_verify_sms_expired_code(self, e2e_client, db):
        unique_suffix = str(uuid.uuid4())[:8]
        password = f"SmsExpiredPass123!{unique_suffix}"
        register_payload = {
            "username": f"sms_expired_{unique_suffix}",
            "email": f"sms_expired_{unique_suffix}@example.com",
            "password": password,
            "first_name": "Leo",
            "last_name": "SmsExpired",
            "phone": unique_phone(),
            "address": "Rue SMS 5, 1200 Geneve",
        }

        register_response = e2e_client.post(
            "/api/v1/auth/register",
            json=register_payload,
            headers={"Content-Type": "application/json"},
        )
        assert register_response.status_code == 201, (
            f"Register doit reussir, recu {register_response.status_code}: "
            f"{register_response.get_json()}"
        )
        activation_session_id = (register_response.get_json() or {}).get(
            "activation_session_id"
        )
        assert activation_session_id, "activation_session_id manquant apres register"

        session = ActivationSession.query.filter_by(
            activation_session_id=activation_session_id
        ).first()
        assert session is not None, "Session d'activation introuvable en base"
        session.sms_code_hash = hashlib.sha256(b"123456").hexdigest()
        session.sms_expires_at = datetime.now(UTC) - timedelta(minutes=1)
        session.sms_attempts = 0
        session.sms_locked_until = None
        session.phone_verified_at = None
        db.session.commit()

        verify_sms_response = e2e_client.post(
            "/api/v1/auth/activation/verify-sms",
            json={"activation_session_id": activation_session_id, "code": "123456"},
            headers={"Content-Type": "application/json"},
        )
        assert verify_sms_response.status_code == 400, (
            f"verify-sms expire doit renvoyer 400, recu {verify_sms_response.status_code}: "
            f"{verify_sms_response.get_json()}"
        )
        verify_sms_data = verify_sms_response.get_json() or {}
        assert verify_sms_data.get("error") == "token_expired"

    def test_e2e_activation_verify_email_invalid_token(self, e2e_client):
        # Token volontairement forgé/invalide (ni signé ni lié à une session).
        verify_email_response = e2e_client.post(
            "/api/v1/auth/activation/verify-email",
            json={"token": "invalid.forged.token"},
            headers={"Content-Type": "application/json"},
        )
        assert verify_email_response.status_code == 400, (
            f"verify-email invalide doit renvoyer 400, recu {verify_email_response.status_code}: "
            f"{verify_email_response.get_json()}"
        )
        verify_email_data = verify_email_response.get_json() or {}
        assert verify_email_data.get("error") == "token_invalid"

    def test_e2e_activation_resend_email_daily_limit_reached(self, e2e_client, db):
        unique_suffix = str(uuid.uuid4())[:8]
        password = f"ResendMailLimitPass123!{unique_suffix}"
        register_payload = {
            "username": f"resend_mail_limit_{unique_suffix}",
            "email": f"resend_mail_limit_{unique_suffix}@example.com",
            "password": password,
            "first_name": "Mia",
            "last_name": "ResendMail",
            "phone": unique_phone(),
            "address": "Rue Mail Limit 6, 1200 Geneve",
        }

        register_response = e2e_client.post(
            "/api/v1/auth/register",
            json=register_payload,
            headers={"Content-Type": "application/json"},
        )
        assert register_response.status_code == 201, (
            f"Register doit reussir, recu {register_response.status_code}: "
            f"{register_response.get_json()}"
        )
        activation_session_id = (register_response.get_json() or {}).get(
            "activation_session_id"
        )
        assert activation_session_id, "activation_session_id manquant apres register"

        session = ActivationSession.query.filter_by(
            activation_session_id=activation_session_id
        ).first()
        assert session is not None, "Session d'activation introuvable en base"
        session.email_verified_at = None
        _mark_current_delivery_sent(db, session)
        # Bypass cooldown mais dépassement certain du quota journalier.
        session.last_email_sent_at = datetime.now(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        session.resend_count_email = 9999
        db.session.commit()

        resend_response = e2e_client.post(
            "/api/v1/auth/activation/resend-email",
            json={"activation_session_id": activation_session_id},
            headers={"Content-Type": "application/json"},
        )
        assert resend_response.status_code == 429, (
            f"resend-email quota journalier doit renvoyer 429, recu {resend_response.status_code}: "
            f"{resend_response.get_json()}"
        )
        resend_data = resend_response.get_json() or {}
        assert resend_data.get("error") == "rate_limited"
        details = resend_data.get("details") or {}
        assert details.get("retry_after_seconds") == 0

    def test_e2e_activation_resend_sms_daily_limit_reached(self, e2e_client, db):
        unique_suffix = str(uuid.uuid4())[:8]
        password = f"ResendSmsLimitPass123!{unique_suffix}"
        register_payload = {
            "username": f"resend_sms_limit_{unique_suffix}",
            "email": f"resend_sms_limit_{unique_suffix}@example.com",
            "password": password,
            "first_name": "Noa",
            "last_name": "ResendSms",
            "phone": unique_phone(),
            "address": "Rue SMS Limit 7, 1200 Geneve",
        }

        register_response = e2e_client.post(
            "/api/v1/auth/register",
            json=register_payload,
            headers={"Content-Type": "application/json"},
        )
        assert register_response.status_code == 201, (
            f"Register doit reussir, recu {register_response.status_code}: "
            f"{register_response.get_json()}"
        )
        activation_session_id = (register_response.get_json() or {}).get(
            "activation_session_id"
        )
        assert activation_session_id, "activation_session_id manquant apres register"

        session = ActivationSession.query.filter_by(
            activation_session_id=activation_session_id
        ).first()
        assert session is not None, "Session d'activation introuvable en base"
        session.phone_verified_at = None
        # Bypass cooldown mais dépassement certain du quota journalier.
        session.last_sms_sent_at = datetime.now(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        session.resend_count_sms = 9999
        db.session.commit()

        resend_response = e2e_client.post(
            "/api/v1/auth/activation/resend-sms",
            json={"activation_session_id": activation_session_id},
            headers={"Content-Type": "application/json"},
        )
        assert resend_response.status_code == 429, (
            f"resend-sms quota journalier doit renvoyer 429, recu {resend_response.status_code}: "
            f"{resend_response.get_json()}"
        )
        resend_data = resend_response.get_json() or {}
        assert resend_data.get("error") == "rate_limited"
        details = resend_data.get("details") or {}
        assert details.get("retry_after_seconds") == 0

    def test_e2e_activation_resend_email_send_failure_returns_502(self, e2e_client, db):
        unique_suffix = str(uuid.uuid4())[:8]
        password = f"ResendFailPass123!{unique_suffix}"
        register_payload = {
            "username": f"resend_mail_fail_{unique_suffix}",
            "email": f"resend_mail_fail_{unique_suffix}@example.com",
            "password": password,
            "first_name": "Iris",
            "last_name": "ResendFail",
            "phone": unique_phone(),
            "address": "Rue Mail Fail 8, 1200 Geneve",
        }

        register_response = e2e_client.post(
            "/api/v1/auth/register",
            json=register_payload,
            headers={"Content-Type": "application/json"},
        )
        assert register_response.status_code == 201, (
            f"Register doit reussir, recu {register_response.status_code}: "
            f"{register_response.get_json()}"
        )
        activation_session_id = (register_response.get_json() or {}).get(
            "activation_session_id"
        )
        assert activation_session_id, "activation_session_id manquant apres register"

        session = ActivationSession.query.filter_by(
            activation_session_id=activation_session_id
        ).first()
        assert session is not None, "Session d'activation introuvable en base"
        _mark_current_delivery_sent(db, session)
        # Bypass cooldown pour forcer l'appel d'envoi.
        session.last_email_sent_at = datetime.now(UTC) - timedelta(minutes=10)
        session.resend_count_email = 0
        db.session.commit()

        with patch(
            "services.notifications.activation_email_delivery.try_enqueue_activation_email",
            return_value={
                "ok": False,
                "queued": False,
                "email_sent": None,
                "debug_activation_link": None,
                "error": "Brevo not configured",
                "require_502": True,
            },
        ):
            resend_response = e2e_client.post(
                "/api/v1/auth/activation/resend-email",
                json={"activation_session_id": activation_session_id},
                headers={"Content-Type": "application/json"},
            )

        assert resend_response.status_code == 502, (
            f"resend-email en echec SMTP doit renvoyer 502, recu {resend_response.status_code}: "
            f"{resend_response.get_json()}"
        )
        resend_data = resend_response.get_json() or {}
        assert resend_data.get("error") == "email_send_failed"
        assert resend_data.get("activation_session_id") == activation_session_id
