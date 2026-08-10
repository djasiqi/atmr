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
from sqlalchemy.exc import IntegrityError

from ext import db, limiter
from models import User
from models.mobile_device_session import MobileDeviceSessionStatus
from security.mobile_device_session_service import (
    DeviceSessionResolutionError,
    RotationProof,
    apply_session_metadata,
    auth_capabilities,
    claim_device_session_resolution_token,
    consume_device_session_resolution_token,
    get_session_by_id,
    hash_credential,
    http_response_for_idempotency,
    is_rotation_idempotency_conflict,
    list_active_sessions,
    mark_session_confirmed,
    publish_session_revoked,
    release_device_session_resolution_claim,
    replace_device_session,
    resolve_rotation_idempotency,
    revoke_all_user_sessions,
    revoke_pending_idempotent,
    revoke_session_state,
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
    cred_gen = int(getattr(session, "credential_generation", session.generation) or 1)
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
    # Fail-closed : jamais remettre un refresh JWT sans ligne DB garantie.
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


def _session_resume_proof(
    *,
    recovery_credential: str,
    device_installation_id: str,
    request_generation: int | None = None,
) -> RotationProof:
    return RotationProof(
        proof_hash=hash_credential(str(recovery_credential)),
        device_installation_id=str(device_installation_id),
        operation_type="session_resume",
        request_generation=request_generation,
    )


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

            user = User.query.get(session.user_id)
            if user is None:
                return {"error": "utilisateur_introuvable"}, 404

            from routes.auth import _check_user_profile_active

            profile_ok, profile_msg = _check_user_profile_active(user)
            if not profile_ok:
                return {
                    "error": profile_msg or "Compte désactivé",
                    "error_code": "account_disabled",
                }, 403

            proof = _session_resume_proof(
                recovery_credential=str(recovery_credential),
                device_installation_id=str(device_installation_id),
            )

            # Receipt authentifie le retry (pas le credential courant / grâce).
            if idempotency_key:
                resolution = resolve_rotation_idempotency(
                    session.session_id, str(idempotency_key), proof=proof
                )
                mapped = http_response_for_idempotency(resolution)
                if mapped is not None:
                    return mapped

            # --- MISS : nouvelle rotation ---
            if not verify_recovery_credential(session, str(recovery_credential)):
                return {
                    "error": "credential_invalide",
                    "error_code": "session_revoked",
                }, 401

            cred_gen = int(
                getattr(session, "credential_generation", session.generation) or 1
            )
            if client_generation is not None:
                try:
                    cg = int(client_generation)
                except (TypeError, ValueError):
                    cg = None
                if cg is not None and cg < cred_gen - 1:
                    return {
                        "error": "generation_obsolete",
                        "error_code": "rotation_recovery_required",
                    }, 401

            request_generation = cred_gen
            from routes.auth import _resolve_device_session_metadata
            from security.mobile_device_session_service import apply_session_metadata

            apply_session_metadata(session, _resolve_device_session_metadata())
            new_recovery = rotate_recovery_credential(session)
            tokens = _issue_token_pair(user, session)
            tokens["recovery_credential"] = new_recovery

            if idempotency_key:
                proof = _session_resume_proof(
                    recovery_credential=str(recovery_credential),
                    device_installation_id=str(device_installation_id),
                    request_generation=request_generation,
                )
                inserted = store_rotation_result(
                    session=session,
                    idempotency_key=str(idempotency_key),
                    request_generation=request_generation,
                    successor_generation=int(session.generation),
                    response_payload=tokens,
                    operation_type="session_resume",
                    proof=proof,
                )
                if inserted is None:
                    # Collision : rollback TX perdante, reload gagnant
                    db.session.rollback()
                    session = get_session_by_id(session_id, for_update=True)
                    if session is None:
                        return {
                            "error": "session_introuvable",
                            "error_code": "session_revoked",
                        }, 401
                    resolution = resolve_rotation_idempotency(
                        session.session_id, str(idempotency_key), proof=proof
                    )
                    mapped = http_response_for_idempotency(resolution)
                    if mapped is not None:
                        return mapped
                    return {
                        "error": "rotation_idempotency_conflict",
                        "error_code": "rotation_recovery_required",
                        "retryable": False,
                    }, 401

            try:
                db.session.commit()
            except IntegrityError as exc:
                if not is_rotation_idempotency_conflict(exc):
                    db.session.rollback()
                    raise
                logger.warning(
                    "session-resume idempotency IntegrityError recovered session_id=%s",
                    session_id,
                )
                db.session.rollback()
                session = get_session_by_id(session_id, for_update=True)
                if session is None or not idempotency_key:
                    return {
                        "error": "rotation_idempotency_conflict",
                        "error_code": "rotation_recovery_required",
                        "retryable": False,
                    }, 401
                resolution = resolve_rotation_idempotency(
                    session.session_id, str(idempotency_key), proof=proof
                )
                mapped = http_response_for_idempotency(resolution)
                if mapped is not None:
                    return mapped
                return {
                    "error": "rotation_idempotency_conflict",
                    "error_code": "rotation_recovery_required",
                    "retryable": False,
                }, 401

            return tokens, 200

    @auth_ns.route("/logout-all")
    class LogoutAll(Resource):
        @jwt_required()
        def post(self):
            identity = get_jwt_identity()
            user = User.query.filter_by(public_id=str(identity)).first()
            if not user:
                return {"error": "utilisateur_introuvable"}, 404
            revoked_ids = [s.session_id for s in list_active_sessions(user.id)]
            count = revoke_all_user_sessions(
                user.id,
                reason="Logout-all utilisateur",
                status=MobileDeviceSessionStatus.revoked,
            )
            try:
                from security.refresh_token_service import revoke_all_user_tokens

                revoke_all_user_tokens(
                    user.id, reason="Logout-all utilisateur", commit=False
                )
            except Exception as exc:
                logger.warning("revoke_all_user_tokens: %s", exc)
            db.session.commit()
            for sid in revoked_ids:
                publish_session_revoked(sid)
            return {"ok": True, "revoked_sessions": count}, 200

    @auth_ns.route("/sessions/<string:session_id>/revoke-pending")
    class RevokePendingSession(Resource):
        def post(self, session_id: str):
            body = request.get_json(silent=True) or {}
            secret = body.get("revocation_secret")
            if not secret:
                return {"error": "revocation_secret_requis"}, 400
            operation_id = body.get("operation_id") or request.headers.get(
                "Idempotency-Key"
            )
            session = get_session_by_id(session_id)
            if session is None:
                # Ne pas révéler l'existence : ACK générique si preuve fournie
                return {"ok": True, "already_absent": True}, 200

            payload, err = revoke_pending_idempotent(
                session,
                str(secret),
                operation_id=str(operation_id) if operation_id else None,
            )
            if err:
                return {
                    "error": "secret_invalide",
                    "error_code": err,
                }, 401
            sid = session.session_id
            db.session.commit()
            publish_session_revoked(sid)
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
            revoke_session_state(
                session,
                reason="Revocation manuelle multi-appareils",
                revoked_by_user_id=user.id,
                status=MobileDeviceSessionStatus.revoked,
            )
            try:
                from security.refresh_token_service import revoke_tokens_for_session

                revoke_tokens_for_session(
                    str(session.session_id),
                    reason="Revocation manuelle",
                    commit=False,
                )
            except Exception as exc:
                logger.warning("revoke_tokens_for_session: %s", exc)
            sid = session.session_id
            db.session.commit()
            publish_session_revoked(sid)
            return {"ok": True}, 200

    @auth_ns.route("/device-sessions/<string:session_uuid>/confirm")
    class DeviceSessionConfirm(Resource):
        @jwt_required()
        @limiter.limit("30 per minute")
        def post(self, session_uuid: str):
            """Confirme l'adoption locale d'une session provisional (idempotent)."""
            identity = get_jwt_identity()
            user = User.query.filter_by(public_id=str(identity)).first()
            if not user:
                return {"error": "utilisateur_introuvable"}, 404
            claims = get_jwt() or {}
            jwt_sid = str(claims.get("session_id") or "")
            if not jwt_sid or jwt_sid != str(session_uuid):
                return {
                    "error": "session_mismatch",
                    "error_code": "session_mismatch",
                    "message": "Le jeton ne correspond pas à la session à confirmer.",
                }, 403
            session = get_session_by_id(session_uuid)
            if session is None or session.user_id != user.id:
                return {
                    "error": "session_not_found",
                    "error_code": "session_not_found",
                }, 404
            if not session.is_active():
                return {
                    "error": "session_revoked",
                    "error_code": "session_revoked",
                }, 401

            from routes.auth import _resolve_device_session_metadata

            apply_session_metadata(session, _resolve_device_session_metadata())
            transitioned = mark_session_confirmed(session)
            db.session.add(session)
            db.session.commit()
            return {
                "ok": True,
                "status": "confirmed" if transitioned else "already_confirmed",
                "session_id": str(session.session_id),
                "confirmed_at": session.confirmed_at.isoformat()
                if session.confirmed_at
                else None,
            }, 200

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
            # collect ids before revoke for post-commit publish
            to_revoke = [
                s.session_id
                for s in list_active_sessions(user.id)
                if except_id is None or s.session_id != except_id
            ]
            count = revoke_all_user_sessions(
                user.id,
                reason="Revoke-others",
                status=MobileDeviceSessionStatus.revoked,
                except_session_id=except_id,
            )
            # revoke_all still calls revoke_session which may publish early —
            # republish after commit for coherence
            db.session.commit()
            for sid in to_revoke:
                publish_session_revoked(sid)
            return {"ok": True, "revoked_sessions": count}, 200

    @auth_ns.route("/device-sessions/replace")
    class DeviceSessionsReplace(Resource):
        @limiter.limit("10 per minute")
        def post(self):
            """Remplace une session active via resolution_token (sans JWT).

            Transaction PostgreSQL unique ; Redis revoked + consume après COMMIT.
            """
            caps = auth_capabilities().get("capabilities") or {}
            if not caps.get("device_session_replace"):
                return {
                    "error": "resolution_unavailable",
                    "error_code": "resolution_unavailable",
                    "message": "Le remplacement d'appareil n'est pas disponible.",
                }, 503

            body = request.get_json(silent=True) or {}
            resolution_token = str(body.get("resolution_token") or "").strip()
            session_to_revoke = body.get("session_to_revoke")
            device_installation_id = request.headers.get("X-Device-ID")
            if not device_installation_id:
                return {
                    "error": "device_identity_required",
                    "error_code": "device_identity_required",
                }, 400
            if not session_to_revoke:
                return {
                    "error": "session_to_revoke_required",
                    "error_code": "session_to_revoke_required",
                }, 400

            try:
                challenge = claim_device_session_resolution_token(
                    token=resolution_token,
                    requested_device_installation_id=str(device_installation_id),
                )
            except DeviceSessionResolutionError as exc:
                status = 401 if exc.code.startswith("resolution_") else 400
                return {
                    "error": exc.code,
                    "error_code": exc.code,
                    "message": exc.message,
                }, status

            user = User.query.filter_by(id=int(challenge["user_id"])).first()
            if not user:
                release_device_session_resolution_claim(token=resolution_token)
                return {"error": "utilisateur_introuvable"}, 404

            driver_id_for_session = getattr(user, "driver_id", None)
            if driver_id_for_session is None:
                driver_obj = getattr(user, "driver", None)
                driver_id_for_session = getattr(driver_obj, "id", None)

            from routes.auth import _resolve_device_session_metadata

            try:
                mobile_session, recovery, revocation, publish_ids = (
                    replace_device_session(
                        user_id=user.id,
                        session_to_revoke=str(session_to_revoke),
                        device_installation_id=str(device_installation_id),
                        allowed_session_ids=list(
                            challenge.get("allowed_session_ids") or []
                        ),
                        driver_id=driver_id_for_session,
                        role=user.role.value if user.role else None,
                        meta=_resolve_device_session_metadata(),
                    )
                )
                tokens = _issue_token_pair(user, mobile_session)
                db.session.commit()
            except DeviceSessionResolutionError as exc:
                db.session.rollback()
                release_device_session_resolution_claim(token=resolution_token)
                return {
                    "error": exc.code,
                    "error_code": exc.code,
                    "message": exc.message,
                }, 409
            except Exception as exc:
                db.session.rollback()
                release_device_session_resolution_claim(token=resolution_token)
                logger.error("device-sessions/replace failed: %s", exc)
                return {
                    "error": "session_replace_failed",
                    "error_code": "session_replace_failed",
                    "retryable": True,
                }, 503

            # Post-commit : cache revoked (cible + reaped) + challenge consumed
            for sid in publish_ids:
                publish_session_revoked(sid)
            consume_device_session_resolution_token(token=resolution_token)

            return {
                **tokens,
                "token": tokens.get("access_token"),
                "recovery_credential": recovery,
                "revocation_secret": revocation,
                "session_id": str(mobile_session.session_id),
                "refresh_generation": int(
                    getattr(mobile_session, "refresh_generation", 1) or 1
                ),
                "credential_generation": int(
                    getattr(
                        mobile_session,
                        "credential_generation",
                        mobile_session.generation,
                    )
                    or 1
                ),
                "session_epoch": int(getattr(mobile_session, "session_epoch", 1) or 1),
                "user": {
                    "public_id": user.public_id,
                    "email": user.email,
                    "role": user.role.value if user.role else None,
                    "first_name": getattr(user, "first_name", None),
                    "last_name": getattr(user, "last_name", None),
                },
                **auth_capabilities(),
            }, 200


# Fin register_mobile_session_routes
