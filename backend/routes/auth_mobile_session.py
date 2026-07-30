"""Routes flask-restx pour sessions durables mobile — branchées sur auth_ns."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from flask import request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
)
from flask_restx import Resource

from ext import db, limiter
from models import User
from models.mobile_device_session import MobileDeviceSessionStatus
from security.mobile_device_session_service import (
    auth_capabilities,
    get_rotation_result,
    get_session_by_id,
    list_active_sessions,
    load_rotation_response,
    revoke_all_user_sessions,
    revoke_pending_idempotent,
    revoke_session,
    rotate_recovery_credential,
    store_rotation_result,
    verify_recovery_credential,
)
from security.refresh_token_service import store_refresh_token

logger = logging.getLogger(__name__)


def _user_token_version(user: User) -> int:
    return int(getattr(user, "token_version", 0) or 0)


def _issue_token_pair(user: User, session) -> dict:
    epoch = int(getattr(session, "session_epoch", 1) or 1)
    refresh_gen = int(getattr(session, "refresh_generation", 1) or 1)
    cred_gen = int(
        getattr(session, "credential_generation", session.generation) or 1
    )
    claims = {
        "role": getattr(user, "role", None),
        "aud": "atmr-api",
        "token_version": _user_token_version(user),
        "session_id": str(session.session_id),
        "session_epoch": epoch,
        "session_generation": epoch,  # compat
    }
    if getattr(user, "company_id", None):
        claims["company_id"] = user.company_id
    if session.driver_id is not None:
        claims["driver_id"] = session.driver_id

    access = create_access_token(
        identity=str(user.public_id),
        fresh=True,
        additional_claims=claims,
    )
    refresh = create_refresh_token(
        identity=str(user.public_id),
        additional_claims={
            "aud": "atmr-api",
            "token_version": _user_token_version(user),
            "session_id": str(session.session_id),
            "session_epoch": epoch,
            "refresh_generation": refresh_gen,
            "session_generation": epoch,
        },
    )
    expires_at = datetime.now(UTC) + timedelta(days=90)
    try:
        row = store_refresh_token(
            token=refresh,
            user_id=user.id,
            expires_at=expires_at,
            device_id=session.device_installation_id,
            device_name=session.device_name,
            commit=False,
        )
        if hasattr(row, "session_id"):
            row.session_id = str(session.session_id)
            row.session_generation = epoch
            db.session.add(row)
    except Exception as exc:
        logger.warning("store_refresh_token session-resume: %s", exc)

    return {
        "token": access,
        "access_token": access,
        "refresh_token": refresh,
        "session_id": str(session.session_id),
        "session_epoch": epoch,
        "credential_generation": cred_gen,
        "refresh_generation": refresh_gen,
        "session_generation": epoch,
        **auth_capabilities(),
    }


def register_mobile_session_routes(auth_ns) -> None:
    """Enregistre /session-resume, /logout-all, revoke-pending, device-sessions."""

    @auth_ns.route("/session-resume")
    class SessionResume(Resource):
        # ✅ Protection brute force : proche du refresh-token (20/min)
        @limiter.limit("20 per minute")
        def post(self):
            body = request.get_json(silent=True) or {}
            session_id = body.get("session_id")
            device_installation_id = body.get("device_installation_id")
            recovery_credential = body.get("recovery_credential")
            idempotency_key = body.get("idempotency_key") or request.headers.get(
                "Idempotency-Key"
            )
            client_generation = body.get("client_generation")

            if not session_id or not recovery_credential or not device_installation_id:
                return {
                    "error": "parametres_manquants",
                    "error_code": "invalid_request",
                }, 400

            # Verrou ligne : sérialise les tentatives de rotation concurrentes
            # (retries HTTP du même appareil) sur la même session.
            session = get_session_by_id(session_id, for_update=True)
            if session is None:
                return {
                    "error": "session_introuvable",
                    "error_code": "session_revoked",
                }, 401

            if idempotency_key:
                existing = get_rotation_result(session.session_id, str(idempotency_key))
                if existing is not None:
                    cached = load_rotation_response(existing)
                    if cached is not None:
                        return {**cached, "error_code": "refresh_duplicate"}, 200

            if not session.is_active():
                return {
                    "error": "session_revoquee",
                    "error_code": "session_revoked",
                }, 401

            if session.device_installation_id != str(device_installation_id):
                return {
                    "error": "installation_mismatch",
                    "error_code": "refresh_replay_detected",
                }, 401

            if not verify_recovery_credential(session, str(recovery_credential)):
                return {
                    "error": "credential_invalide",
                    "error_code": "session_revoked",
                }, 401

            if client_generation is not None:
                try:
                    cg = int(client_generation)
                except (TypeError, ValueError):
                    cg = None
                if cg is not None and cg < int(session.generation) - 1:
                    return {
                        "error": "generation_obsolete",
                        "error_code": "rotation_recovery_required",
                    }, 401

            user = User.query.get(session.user_id)
            if user is None:
                return {"error": "utilisateur_introuvable"}, 404

            # Compte / profil actif (même règle que refresh-token)
            from routes.auth import _check_user_profile_active

            profile_ok, profile_msg = _check_user_profile_active(user)
            if not profile_ok:
                return {
                    "error": profile_msg or "Compte désactivé",
                    "error_code": "account_disabled",
                }, 403

            request_generation = int(session.generation)
            new_recovery = rotate_recovery_credential(session)
            tokens = _issue_token_pair(user, session)
            tokens["recovery_credential"] = new_recovery

            if idempotency_key:
                store_rotation_result(
                    session=session,
                    idempotency_key=str(idempotency_key),
                    request_generation=request_generation,
                    successor_generation=int(session.generation),
                    response_payload=tokens,
                    operation_type="session_resume",
                )

            db.session.commit()
            return tokens, 200

    @auth_ns.route("/logout-all")
    class LogoutAll(Resource):
        @jwt_required()
        def post(self):
            identity = get_jwt_identity()
            user = User.query.filter_by(public_id=str(identity)).first()
            if not user:
                return {"error": "utilisateur_introuvable"}, 404
            count = revoke_all_user_sessions(
                user.id,
                reason="Logout-all utilisateur",
                status=MobileDeviceSessionStatus.revoked,
            )
            try:
                from security.refresh_token_service import revoke_all_user_tokens

                revoke_all_user_tokens(user.id, reason="Logout-all utilisateur")
            except Exception as exc:
                logger.warning("revoke_all_user_tokens: %s", exc)
            db.session.commit()
            return {"ok": True, "revoked_sessions": count}, 200

    @auth_ns.route("/sessions/<string:session_id>/revoke-pending")
    class RevokePendingSession(Resource):
        def post(self, session_id: str):
            body = request.get_json(silent=True) or {}
            secret = body.get("revocation_secret")
            if not secret:
                return {"error": "revocation_secret_requis"}, 400
            operation_id = (
                body.get("operation_id")
                or request.headers.get("Idempotency-Key")
            )
            session = get_session_by_id(session_id)
            if session is None:
                # Ne pas révéler l'existence : ACK générique si preuve fournie
                return {"ok": True, "already_absent": True}, 200

            payload, err = revoke_pending_idempotent(
                session, str(secret), operation_id=str(operation_id) if operation_id else None
            )
            if err:
                return {
                    "error": "secret_invalide",
                    "error_code": err,
                }, 401
            db.session.commit()
            return payload or {"ok": True}, 200

    @auth_ns.route("/device-sessions")
    class DeviceSessions(Resource):
        @jwt_required()
        def get(self):
            identity = get_jwt_identity()
            user = User.query.filter_by(public_id=str(identity)).first()
            if not user:
                return {"error": "utilisateur_introuvable"}, 404
            claims = get_jwt() or {}
            current_sid = claims.get("session_id")
            sessions = list_active_sessions(user.id)
            return {
                "sessions": [
                    s.serialize(is_current=str(s.session_id) == str(current_sid))
                    for s in sessions
                ],
                **auth_capabilities(),
            }, 200

    @auth_ns.route("/device-sessions/<string:session_uuid>")
    class DeviceSessionById(Resource):
        @jwt_required()
        def delete(self, session_uuid: str):
            """Révoque une session mobile par UUID (multi-appareils)."""
            identity = get_jwt_identity()
            user = User.query.filter_by(public_id=str(identity)).first()
            if not user:
                return {"error": "utilisateur_introuvable"}, 404
            session = get_session_by_id(session_uuid)
            if session is None or session.user_id != user.id:
                return {"ok": True, "already_absent": True}, 200
            if not session.is_active():
                return {"ok": True, "already_revoked": True}, 200
            revoke_session(
                session,
                reason="Revocation manuelle multi-appareils",
                revoked_by_user_id=user.id,
                status=MobileDeviceSessionStatus.revoked,
            )
            try:
                from security.refresh_token_service import revoke_tokens_for_session

                revoke_tokens_for_session(
                    str(session.session_id), reason="Revocation manuelle"
                )
            except Exception as exc:
                logger.warning("revoke_tokens_for_session: %s", exc)
            db.session.commit()
            return {"ok": True}, 200

    @auth_ns.route("/device-sessions/revoke-others")
    class DeviceSessionsRevokeOthers(Resource):
        @jwt_required()
        def post(self):
            """Révoque toutes les autres sessions actives (conserve la courante)."""
            identity = get_jwt_identity()
            user = User.query.filter_by(public_id=str(identity)).first()
            if not user:
                return {"error": "utilisateur_introuvable"}, 404
            claims = get_jwt() or {}
            current_sid = claims.get("session_id")
            except_id = None
            if current_sid:
                try:
                    import uuid as _uuid

                    except_id = _uuid.UUID(str(current_sid))
                except (ValueError, TypeError):
                    except_id = None
            count = revoke_all_user_sessions(
                user.id,
                reason="Revoke-others",
                status=MobileDeviceSessionStatus.revoked,
                except_session_id=except_id,
            )
            db.session.commit()
            return {"ok": True, "revoked_sessions": count}, 200
