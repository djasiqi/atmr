"""Parcours web Google et Apple.

Le retour navigateur ne transporte ni code durable, ni jeton LIRIE.
"""

from __future__ import annotations

from datetime import UTC, datetime

from flask import current_app, make_response, redirect, request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt_identity,
    jwt_required,
    verify_jwt_in_request,
)
from flask_restx import Resource

from ext import limiter
from models import User
from routes.auth import (
    _access_expiry_metadata,
    _clear_web_auth_cookies,
    _get_password_hash_version,
    _must_complete_onboarding,
    _onboarding_reasons,
    _resolve_access_token_expires,
    _resolve_company_id,
    _resolve_refresh_token_expires,
    _user_token_version,
    auth_ns,
)
from security.refresh_token_service import store_refresh_token
from services.apple_native_session import open_native_apple_session
from services.external_auth import (
    ExternalAuthRejected,
    apple_names_from_form,
    begin,
    consume_callback,
    list_identities,
    provider_status,
    register_from_ticket,
    unlink,
)
from services.mobile_contract_issue import (
    MobileContractError,
    issue_mobile_device_contract,
)
from shared.response_helpers import json_response

TICKET_COOKIE = "external_signup"
BROWSER_COOKIE = "oauth_browser"


def _rejected(error: ExternalAuthRejected):
    return json_response(
        {"error": error.code, "message": error.message},
        error.status,
    )


def _frontend_base() -> str:
    return str(
        current_app.config.get("FRONTEND_URL") or "http://127.0.0.1:3000"
    ).rstrip("/")


def _return_url(error: str | None = None, *, consent: bool = False) -> str:
    query = "connexion=externe"
    if consent:
        query += "&inscription=1"
    if error:
        query += f"&erreur={error}"
    return f"{_frontend_base()}/login?{query}"


def _account_url(public_id: str, error: str | None = None) -> str:
    """Retour fixe vers le compte client. L'identifiant vient de la session."""
    cleaned = str(public_id or "").strip()
    if len(cleaned) != 36:
        return _return_url(error)
    query = "section=security&connexion=externe"
    if error:
        query += f"&erreur={_public_error(error)}"
    return f"{_frontend_base()}/dashboard/account/{cleaned}?{query}"


def _actor_id() -> int | None:
    try:
        verify_jwt_in_request(optional=True)
    except Exception:
        return None
    identity = get_jwt_identity()
    if not identity:
        return None
    user = User.query.filter_by(public_id=str(identity)).one_or_none()
    return int(user.id) if user is not None else None


def _issue(user: User):
    """Émet la session LIRIE existante, sans rôle ni habilitation nouveaux."""
    claims = {
        "role": user.role.value,
        "company_id": _resolve_company_id(user),
        "driver_id": getattr(user, "driver_id", None),
        "institution_id": getattr(user, "institution_id", None),
        "institution_role": getattr(user, "institution_role", None),
        "aud": "atmr-api",
        "token_version": _user_token_version(user),
    }
    from security.web_session_service import (
        create_web_session,
        is_institution_user,
        resolve_web_session_expires,
    )

    access_expires = _resolve_access_token_expires(False)
    refresh_expires = _resolve_refresh_token_expires(
        is_mobile_request=False, remember_me=False
    )
    if is_institution_user(user):
        web_session = create_web_session(
            user,
            expires_at=resolve_web_session_expires(
                remember_me=False,
                refresh_expires_delta=refresh_expires,
            ),
            remember_me=False,
        )
        claims["sid"] = str(web_session.id)
        from ext import db

        db.session.flush()
    access_token = create_access_token(
        identity=str(user.public_id),
        additional_claims=claims,
        expires_delta=access_expires,
        fresh=True,
    )
    refresh_token = create_refresh_token(
        identity=str(user.public_id),
        additional_claims={
            "aud": "atmr-api",
            "pwd_hash": _get_password_hash_version(user),
            "token_version": _user_token_version(user),
            "remember_me": False,
        },
        expires_delta=refresh_expires,
    )
    web_session_id = str(claims["sid"]) if claims.get("sid") else None
    from security.refresh_redis_rotation import (
        commit_db_after_redis,
        publish_refresh_redis,
    )
    from security.refresh_token_service import RefreshStoreUnavailableError

    try:
        store_refresh_token(
            refresh_token,
            int(user.id),
            datetime.now(UTC) + refresh_expires,
            web_session_id=web_session_id,
            commit=False,
            sync_redis=False,
        )
        redis_handle = publish_refresh_redis(
            int(user.id),
            refresh_token,
            ttl_seconds=int(refresh_expires.total_seconds()),
        )
        if current_app.config.get("TESTING"):
            from ext import db

            db.session.flush()
        else:
            commit_db_after_redis(redis_handle)
    except RefreshStoreUnavailableError:
        from ext import db

        db.session.rollback()
        return json_response(
            {
                "error": "service_unavailable",
                "message": "Stockage session indisponible. Réessayez.",
            },
            503,
        )
    body = {
        "message": "Connexion réussie",
        "user": {
            "id": user.id,
            "public_id": user.public_id,
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role.value,
            "force_password_change": bool(user.force_password_change),
            "must_complete_onboarding": _must_complete_onboarding(user),
            "onboarding_reasons": _onboarding_reasons(user),
        },
    }
    body.update(_access_expiry_metadata(access_expires))
    response = json_response(body, 200)
    _clear_web_auth_cookies(response)
    _set_auth_cookies(response, access_token, refresh_token)
    response.delete_cookie(TICKET_COOKIE, path="/")
    return response


def _set_auth_cookies(response, access_token: str, refresh_token: str) -> None:
    response.set_cookie(
        current_app.config["COOKIE_ACCESS_TOKEN_NAME"],
        access_token,
        httponly=current_app.config["COOKIE_HTTP_ONLY"],
        secure=current_app.config["COOKIE_SECURE"],
        samesite=current_app.config["COOKIE_SAME_SITE"],
        max_age=int(current_app.config["JWT_ACCESS_TOKEN_EXPIRES"].total_seconds()),
        path=current_app.config["COOKIE_PATH"],
        domain=current_app.config["COOKIE_DOMAIN"],
    )
    response.set_cookie(
        current_app.config["COOKIE_REFRESH_TOKEN_NAME"],
        refresh_token,
        httponly=current_app.config["COOKIE_HTTP_ONLY"],
        secure=current_app.config["COOKIE_SECURE"],
        samesite=current_app.config["COOKIE_SAME_SITE"],
        max_age=None,
        path=current_app.config["COOKIE_PATH"],
        domain=current_app.config["COOKIE_DOMAIN"],
    )


def _public_error(code: str) -> str:
    allowed = {
        "annule",
        "fournisseur",
        "compte_existant",
        "adresse_autre_compte",
        "compte_indisponible",
        "rejeu",
        "state_invalide",
        "jeton_invalide",
        "identite_incomplete",
        "methode_deja_liee",
        "fournisseur_indisponible",
        "nonce_invalide",
        "origine_invalide",
    }
    return code if code in allowed else "fournisseur"


def _redirect_error(code: str):
    return redirect(_return_url(_public_error(code)))


def _set_browser_cookie(response, binding: str) -> None:
    """Cookie de liaison, distinct des cookies de session.

    En HTTPS il est ``SameSite=None`` pour revenir avec le POST Apple.
    Le secret n'est pas mis dans le corps JSON.
    """
    secure = bool(current_app.config.get("COOKIE_SECURE"))
    response.set_cookie(
        BROWSER_COOKIE,
        binding,
        httponly=True,
        secure=secure,
        samesite="None" if secure else "Lax",
        max_age=600,
        path="/api/v1/auth/external",
    )


@auth_ns.route("/external/providers")
class ExternalProviders(Resource):
    def get(self):
        return provider_status(), 200


@auth_ns.route("/external/start")
class ExternalStart(Resource):
    @limiter.limit("20 per hour")
    def post(self):
        payload = request.get_json(silent=True) or {}
        intent = str(payload.get("intent") or "login")
        try:
            started = begin(
                str(payload.get("provider") or ""),
                intent=intent,
                user_id=_actor_id() if intent == "link" else None,
            )
        except ExternalAuthRejected as error:
            return _rejected(error)
        binding = started.pop("browser_binding")
        response = json_response(started, 200)
        _set_browser_cookie(response, binding)
        return response


@auth_ns.route("/external/callback")
class ExternalCallback(Resource):
    def get(self):
        return _callback()

    def post(self):
        return _callback()


def _callback():
    """Corrélation par ``state`` et par le cookie ``oauth_browser``.

    Les cookies de session ne sont pas lus. Le POST Apple est inter-site :
    seul ``oauth_browser`` est ``SameSite=None`` lorsque la réponse est
    sécurisée. Le ticket et la session sont écrits dans la réponse.
    """
    error = str(request.values.get("error") or "")
    if error in {"access_denied", "user_cancelled_authorize"}:
        return _redirect_error("annule")
    if error:
        return _redirect_error("fournisseur")
    try:
        result = consume_callback(
            str(request.values.get("code") or ""),
            str(request.values.get("state") or ""),
            apple_names=apple_names_from_form(request.form.get("user")),
            browser_binding=request.cookies.get(BROWSER_COOKIE),
        )
    except ExternalAuthRejected as error:
        if error.intent == "link" and error.role == "CLIENT" and error.public_id:
            return redirect(_account_url(error.public_id, error.code))
        return _redirect_error(error.code)
    if result["kind"] == "consent":
        response = make_response(redirect(_return_url(consent=True)))
        response.set_cookie(
            TICKET_COOKIE,
            result["ticket"],
            httponly=True,
            secure=bool(current_app.config.get("COOKIE_SECURE")),
            samesite=current_app.config.get("COOKIE_SAME_SITE") or "Lax",
            max_age=600,
            path="/",
        )
        return response
    user = result["user"]
    response = _issue(user)
    if response.status_code != 200:
        return _redirect_error("fournisseur_indisponible")
    response.status_code = 302
    if result.get("intent") == "link" and user.role.value == "CLIENT":
        response.headers["Location"] = _account_url(str(user.public_id))
    else:
        response.headers["Location"] = _return_url()
    return response


def _mobile_contract() -> bool:
    return (
        request.headers.get("X-Auth-Contract-Version") or ""
    ).strip() == "mobile-device-session-v1"


def _issue_for_client(user: User):
    if not _mobile_contract():
        return _issue(user)
    try:
        return issue_mobile_device_contract(user)
    except MobileContractError as error:
        return json_response(error.body, error.status)


@auth_ns.route("/external/apple/native")
class ExternalAppleNative(Resource):
    @limiter.limit("20 per hour")
    def post(self):
        if (request.headers.get("X-Client-Platform") or "").strip().lower() != "ios":
            return json_response(
                {
                    "error": "plateforme_refusee",
                    "error_code": "plateforme_refusee",
                },
                400,
            )
        device_id = (
            request.headers.get("X-Device-ID")
            or request.headers.get("X-Installation-ID")
            or ""
        ).strip()
        if not _mobile_contract() or not device_id:
            return json_response(
                {
                    "error": "device_identity_required",
                    "error_code": "device_identity_required",
                    "message": "Identité sécurisée de l'appareil manquante.",
                },
                400,
            )
        payload = request.get_json(silent=True) or {}
        given = str(payload.get("given_name") or "").strip()
        family = str(payload.get("family_name") or "").strip()
        names = None
        if given or family:
            names = {"given_name": given or None, "family_name": family or None}
        intent = str(payload.get("intent") or "login")
        try:
            result = open_native_apple_session(
                str(payload.get("identity_token") or ""),
                str(payload.get("raw_nonce") or ""),
                intent=intent,
                actor_id=_actor_id() if intent == "link" else None,
                names=names,
            )
        except ExternalAuthRejected as error:
            return _rejected(error)
        if result["kind"] == "consent":
            return json_response({"kind": "consent", "ticket": result["ticket"]}, 200)
        return _issue_for_client(result["user"])


@auth_ns.route("/external/register")
class ExternalRegister(Resource):
    @limiter.limit("10 per hour")
    def post(self):
        payload = request.get_json(silent=True) or {}
        ticket = str(request.cookies.get(TICKET_COOKIE) or "")
        if _mobile_contract():
            ticket = ticket or str(payload.get("ticket") or "")
        try:
            user = register_from_ticket(
                ticket,
                accept_terms=bool(payload.get("accept_terms")),
            )
        except ExternalAuthRejected as error:
            return _rejected(error)
        return _issue_for_client(user)


@auth_ns.route("/external/identities")
class ExternalIdentities(Resource):
    @jwt_required()
    def get(self):
        identity = get_jwt_identity()
        user = User.query.filter_by(public_id=str(identity)).one_or_none()
        if user is None:
            return json_response({"error": "session_invalide"}, 401)
        return list_identities(user), 200


@auth_ns.route("/external/identities/<string:provider>")
class ExternalIdentity(Resource):
    @jwt_required()
    def delete(self, provider: str):
        identity = get_jwt_identity()
        user = User.query.filter_by(public_id=str(identity)).one_or_none()
        if user is None:
            return json_response({"error": "session_invalide"}, 401)
        try:
            unlink(user, provider)
        except ExternalAuthRejected as error:
            return _rejected(error)
        return json_response({"removed": provider}, 200)
