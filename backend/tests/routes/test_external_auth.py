"""Phase 2F — Google et Apple n'ouvrent aucun portail professionnel.

Les jetons sont signés localement. L'échange HTTP fournisseur est remplacé.
Ce fichier ne déclare pas une connexion Google ou Apple réelle.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from flask_jwt_extended import create_access_token
from jwt.algorithms import ECAlgorithm, RSAAlgorithm
from sqlalchemy.orm import Session

import services.external_auth as external_auth_module
from models import Client, Company, PortalGrant, User, UserRole
from models.enums import PortalGrantStatus, PortalKind
from models.external_auth_transaction import ExternalAuthTransaction
from models.organization_application import OrganizationApplication
from models.user_external_identity import UserExternalIdentity
from services.external_auth import ExternalAuthRejected

pytestmark = pytest.mark.integration

_REAL_EXCHANGE = external_auth_module.exchange_code


@pytest.fixture(autouse=True)
def _isole_la_transaction_de_test(db, monkeypatch):
    """Garde les écritures dans la transaction du test.

    Le fixture db l'annule à la fin. On ne valide rien, donc on n'a pas
    à effacer une acceptation de conditions déjà enregistrée.
    """

    def _flush_instead(self, *args, **kwargs):
        self.flush()

    monkeypatch.setattr(Session, "commit", _flush_instead)
    monkeypatch.setattr(db.session, "remove", lambda: None)


_RSA = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_EC = ec.generate_private_key(ec.SECP256R1())
_RSA_JWK = json.loads(RSAAlgorithm.to_jwk(_RSA.public_key()))
_RSA_JWK.update({"kid": "google-test", "alg": "RS256", "use": "sig"})
_EC_JWK = json.loads(ECAlgorithm.to_jwk(_EC.public_key()))
_EC_JWK.update({"kid": "apple-test", "alg": "ES256", "use": "sig"})
_OTHER_RSA = rsa.generate_private_key(public_exponent=65537, key_size=2048)

_CONFIG = {
    "EXTERNAL_AUTH_GOOGLE_CLIENT_ID": "google-test-client",
    "EXTERNAL_AUTH_GOOGLE_CLIENT_SECRET": "secret-test",
    "EXTERNAL_AUTH_GOOGLE_REDIRECT_URI": (
        "http://127.0.0.1:5000/api/v1/auth/external/callback"
    ),
    "EXTERNAL_AUTH_APPLE_CLIENT_ID": "apple-test-client",
    "EXTERNAL_AUTH_APPLE_REDIRECT_URI": (
        "http://127.0.0.1:5000/api/v1/auth/external/callback"
    ),
    "EXTERNAL_AUTH_APPLE_TEAM_ID": "TEAMTEST",
    "EXTERNAL_AUTH_APPLE_KEY_ID": "KEYTEST",
    "EXTERNAL_AUTH_APPLE_PRIVATE_KEY": "non-utilise-par-le-test",
    "COOKIE_SECURE": False,
    "JWT_COOKIE_SECURE": False,
}


@pytest.fixture
def external_auth(app, db, monkeypatch):
    ExternalAuthTransaction.__table__.create(bind=db.engine, checkfirst=True)
    UserExternalIdentity.__table__.create(bind=db.engine, checkfirst=True)
    PortalGrant.__table__.create(bind=db.engine, checkfirst=True)
    previous = {key: app.config.get(key) for key in _CONFIG}
    present = {key: key in app.config for key in _CONFIG}
    app.config.update(_CONFIG)
    holder = {"token": "absent"}
    monkeypatch.setattr(
        "services.external_auth.exchange_code",
        lambda _provider, _code, _verifier: holder["token"],
    )
    monkeypatch.setattr(
        "services.external_auth.fetch_jwks",
        lambda _provider: {"keys": [_RSA_JWK, _EC_JWK]},
    )
    yield holder
    for key, was_present in present.items():
        if was_present:
            app.config[key] = previous[key]
        else:
            app.config.pop(key, None)


def _user(db, role: UserRole) -> User:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"ext_{suffix}"
    user.email = f"ext-{suffix}@example.com"
    user.role = role
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()
    return user


def _company(db, owner: User) -> Company:
    suffix = uuid.uuid4().hex[:8]
    company = Company()
    company.name = f"Transport {suffix}"
    company.address = "Rue du Test 1"
    company.contact_phone = "0210000000"
    company.contact_email = f"co-{suffix}@example.com"
    company.user_id = owner.id
    company.is_approved = True
    db.session.add(company)
    db.session.flush()
    return company


def _headers(app, user: User) -> dict[str, str]:
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id),
            fresh=True,
            additional_claims={"role": user.role.value, "aud": "atmr-api"},
        )
    return {"Authorization": f"Bearer {token}"}


def _sign(provider: str, claims: dict) -> str:
    if provider == "apple":
        return jwt.encode(claims, _EC, algorithm="ES256", headers={"kid": "apple-test"})
    return jwt.encode(claims, _RSA, algorithm="RS256", headers={"kid": "google-test"})


def _claims(provider: str, nonce: str, **extra) -> dict:
    now = datetime.now(UTC)
    audience = "apple-test-client" if provider == "apple" else "google-test-client"
    issuer = (
        "https://appleid.apple.com"
        if provider == "apple"
        else "https://accounts.google.com"
    )
    body = {
        "iss": issuer,
        "aud": audience,
        "sub": extra.pop("sub", f"sub-{uuid.uuid4().hex}"),
        "nonce": nonce,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
        "email": extra.pop("email", f"personne-{uuid.uuid4().hex[:8]}@example.com"),
        "email_verified": extra.pop("email_verified", True),
        "given_name": "Nora",
        "family_name": "Client",
    }
    body.update(extra)
    return body


def _start(client, provider: str = "google", intent: str = "login", headers=None):
    response = client.post(
        "/api/v1/auth/external/start",
        json={"provider": provider, "intent": intent},
        headers=headers,
    )
    assert response.status_code == 200, response.get_json()
    payload = response.get_json()
    assert "code_verifier" not in json.dumps(payload)
    assert "browser_binding" not in payload
    query = parse_qs(urlparse(payload["authorization_url"]).query)
    assert query["code_challenge_method"] == ["S256"]
    assert query["response_type"] == ["code"]
    state = query["state"][0]
    state_hash = hashlib.sha256(state.encode("utf-8")).hexdigest()
    row = ExternalAuthTransaction.query.filter_by(state_hash=state_hash).one()
    digest = hashlib.sha256(row.code_verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert query["code_challenge"] == [challenge]
    assert row.code_verifier not in payload["authorization_url"]
    return state, query["nonce"][0]


def _callback(client, state: str, *, error: str | None = None, code: str = "code-test"):
    payload = {"state": state, "code": code}
    if error:
        payload = {"error": error, "state": state}
    return client.get("/api/v1/auth/external/callback", query_string=payload)


def _prepare(client, holder, provider: str = "google", **claim_extra):
    state, nonce = _start(client, provider)
    holder["token"] = _sign(provider, _claims(provider, nonce, **claim_extra))
    return _callback(client, state)


def _cookie(response, name: str) -> str | None:
    found = None
    for cookie in response.headers.getlist("Set-Cookie"):
        if cookie.startswith(f"{name}="):
            found = cookie.split(";", 1)[0].split("=", 1)[1]
    return found or None


def _drop_session(client) -> None:
    for name in ("access_token", "refresh_token", "external_signup"):
        client.delete_cookie(name, path="/")


def test_fournisseurs_desactives_par_defaut(app, client, monkeypatch):
    for key in (
        "EXTERNAL_AUTH_GOOGLE_CLIENT_ID",
        "EXTERNAL_AUTH_GOOGLE_CLIENT_SECRET",
        "EXTERNAL_AUTH_GOOGLE_REDIRECT_URI",
        "EXTERNAL_AUTH_APPLE_CLIENT_ID",
        "EXTERNAL_AUTH_APPLE_REDIRECT_URI",
        "EXTERNAL_AUTH_APPLE_TEAM_ID",
        "EXTERNAL_AUTH_APPLE_KEY_ID",
        "EXTERNAL_AUTH_APPLE_PRIVATE_KEY",
        "EXTERNAL_AUTH_APPLE_BUNDLE_ID",
    ):
        monkeypatch.delenv(key, raising=False)
        app.config.pop(key, None)
    status = client.get("/api/v1/auth/external/providers")
    assert status.status_code == 200
    body = status.get_json()
    assert body["google"]["enabled"] is False
    assert body["apple"]["enabled"] is False
    assert body["apple"]["native"] is False
    started = client.post(
        "/api/v1/auth/external/start",
        json={"provider": "google", "intent": "login"},
    )
    assert started.status_code == 503
    assert started.get_json()["error"] == "fournisseur_indisponible"


def test_creation_client_apres_consentement_sans_habilitation(
    client, external_auth, db
):
    email = f"nouveau-{uuid.uuid4().hex[:8]}@example.com"
    pending = _prepare(
        client, external_auth, email=email, sub=f"sub-nouveau-{uuid.uuid4().hex}"
    )
    assert pending.status_code == 302
    assert "inscription=1" in pending.headers["Location"]
    assert "access_token" not in pending.headers.get("Location", "")
    assert User.query.filter_by(email=email).one_or_none() is None

    refused = client.post(
        "/api/v1/auth/external/register", json={"accept_terms": False}
    )
    assert refused.status_code == 400
    assert refused.get_json()["error"] == "consentement_requis"
    assert User.query.filter_by(email=email).one_or_none() is None

    created = client.post("/api/v1/auth/external/register", json={"accept_terms": True})
    assert created.status_code == 200
    user = User.query.filter_by(email=email).one()
    assert user.role == UserRole.CLIENT
    assert user.authentication_method == "sso"
    assert created.get_json()["user"]["role"] == "CLIENT"
    assert (
        UserExternalIdentity.query.filter_by(user_id=user.id, provider="google").count()
        == 1
    )
    assert PortalGrant.query.filter_by(user_id=user.id).count() == 0
    replay = client.post("/api/v1/auth/external/register", json={"accept_terms": True})
    assert replay.status_code == 400

    headers = _headers(client.application, user)
    company = client.get("/api/v1/companies/me/reservations", headers=headers)
    driver = client.get("/api/v1/driver/me/profile", headers=headers)
    institution = client.get("/api/v1/institutions/me", headers=headers)
    assert company.status_code in (401, 403)
    assert driver.status_code in (401, 403)
    assert institution.status_code in (401, 403)
    assert PortalGrant.query.filter_by(user_id=user.id).count() == 0


def test_meme_email_ne_fusionne_pas(client, external_auth, db):
    owner = _user(db, UserRole.CLIENT)
    subject = f"sub-autre-{uuid.uuid4().hex}"
    response = _prepare(client, external_auth, email=owner.email, sub=subject)
    assert response.status_code == 302
    assert "erreur=compte_existant" in response.headers["Location"]
    assert User.query.filter_by(email=owner.email).count() == 1
    assert UserExternalIdentity.query.filter_by(provider_sub=subject).count() == 0
    assert owner.role == UserRole.CLIENT


def test_lien_exige_une_session_et_refuse_l_autre_compte(
    client, external_auth, app, db
):
    anonymous = client.post(
        "/api/v1/auth/external/start",
        json={"provider": "google", "intent": "link"},
    )
    assert anonymous.status_code == 401

    owner = _user(db, UserRole.COMPANY)
    company = _company(db, owner)
    grant = PortalGrant(
        user_id=owner.id,
        portal=PortalKind.COMPANY.value,
        company_id=company.id,
        status=PortalGrantStatus.ACTIVE.value,
    )
    db.session.add(grant)
    db.session.flush()
    subject = f"sub-lie-{uuid.uuid4().hex}"
    state, nonce = _start(client, intent="link", headers=_headers(app, owner))
    external_auth["token"] = _sign(
        "google",
        _claims("google", nonce, sub=subject, email="autre-vue@example.com"),
    )
    linked = _callback(client, state)
    assert linked.status_code == 302
    assert "erreur=" not in linked.headers["Location"]
    db.session.refresh(owner)
    assert owner.role == UserRole.COMPANY
    assert owner.email != "autre-vue@example.com"
    assert PortalGrant.query.filter_by(user_id=owner.id, status="active").count() == 1

    _drop_session(client)
    other = _user(db, UserRole.CLIENT)
    state, nonce = _start(client, intent="link", headers=_headers(app, other))
    external_auth["token"] = _sign("google", _claims("google", nonce, sub=subject))
    stolen = _callback(client, state)
    assert stolen.status_code == 302
    assert "erreur=methode_deja_liee" in stolen.headers["Location"]
    assert UserExternalIdentity.query.filter_by(provider_sub=subject).count() == 1


def test_rejeu_signature_nonce_et_annulation(client, external_auth, caplog):
    caplog.set_level(logging.INFO)
    state, nonce = _start(client)
    external_auth["token"] = _sign(
        "google", _claims("google", nonce, sub=f"sub-rejeu-{uuid.uuid4().hex}")
    )
    first = _callback(client, state, code="code-secret-ne-pas-journaliser")
    assert first.status_code == 302
    replay = _callback(client, state, code="code-secret-ne-pas-journaliser")
    assert "erreur=rejeu" in replay.headers["Location"]
    assert "code-secret-ne-pas-journaliser" not in caplog.text

    state, nonce = _start(client)
    external_auth["token"] = jwt.encode(
        _claims("google", nonce),
        _OTHER_RSA,
        algorithm="RS256",
        headers={"kid": "google-test"},
    )
    forged = _callback(client, state)
    assert "erreur=jeton_invalide" in forged.headers["Location"]

    state, nonce = _start(client)
    external_auth["token"] = _sign("google", _claims("google", "nonce-autre"))
    mismatch = _callback(client, state)
    assert "erreur=nonce_invalide" in mismatch.headers["Location"]

    state, _nonce = _start(client)
    cancelled = _callback(client, state, error="access_denied")
    assert "erreur=annule" in cancelled.headers["Location"]
    unknown = client.get(
        "/api/v1/auth/external/callback",
        query_string={"code": "x", "state": "state-inconnu"},
    )
    assert "erreur=state_invalide" in unknown.headers["Location"]


def test_apple_sans_email_puis_changement_d_adresse(client, external_auth, db):
    missing = _prepare(
        client,
        external_auth,
        provider="apple",
        sub=f"apple-sub-nouveau-{uuid.uuid4().hex}",
        email=None,
        email_verified=False,
    )
    assert "erreur=identite_incomplete" in missing.headers["Location"]

    email = f"relay-{uuid.uuid4().hex[:8]}@privaterelay.appleid.com"
    subject = f"apple-sub-stable-{uuid.uuid4().hex}"
    created = _prepare(
        client,
        external_auth,
        provider="apple",
        sub=subject,
        email=email,
        given_name="Ada",
        family_name="Relay",
    )
    assert "inscription=1" in created.headers["Location"]
    registered = client.post(
        "/api/v1/auth/external/register", json={"accept_terms": True}
    )
    assert registered.status_code == 200
    user = User.query.filter_by(email=email).one()
    assert user.first_name == "Ada"
    assert PortalGrant.query.filter_by(user_id=user.id).count() == 0

    again = _prepare(
        client,
        external_auth,
        provider="apple",
        sub=subject,
        email=None,
        email_verified=False,
        given_name="Autre",
        family_name="Nom",
    )
    assert again.status_code == 302
    assert "erreur=" not in again.headers["Location"]
    db.session.refresh(user)
    assert user.email == email
    assert user.first_name == "Ada"
    assert user.role == UserRole.CLIENT

    changed = _prepare(
        client,
        external_auth,
        provider="apple",
        sub=subject,
        email=f"nouveau-{uuid.uuid4().hex[:8]}@privaterelay.appleid.com",
    )
    assert "erreur=" not in changed.headers["Location"]
    db.session.refresh(user)
    assert user.email == email
    identity = UserExternalIdentity.query.filter_by(provider_sub=subject).one()
    assert identity.email_at_link != email


def test_apple_form_post_nom_relais_et_annulation(client, external_auth, db, caplog):
    caplog.set_level(logging.INFO)
    started = client.post(
        "/api/v1/auth/external/start",
        json={"provider": "apple", "intent": "login"},
    )
    assert started.status_code == 200
    query = parse_qs(urlparse(started.get_json()["authorization_url"]).query)
    assert query["response_mode"] == ["form_post"]
    assert query["scope"] == ["openid email name"]
    state = query["state"][0]
    email = f"relay-{uuid.uuid4().hex[:8]}@privaterelay.appleid.com"
    subject = f"apple-sub-form-{uuid.uuid4().hex}"
    external_auth["token"] = _sign(
        "apple",
        _claims(
            "apple",
            query["nonce"][0],
            sub=subject,
            email=email,
            email_verified="true",
            given_name=None,
            family_name=None,
        ),
    )
    posted = client.post(
        "/api/v1/auth/external/callback",
        data={
            "state": state,
            "code": "code-apple-ne-pas-journaliser",
            "user": json.dumps(
                {
                    "name": {"firstName": "Iris", "lastName": "Relais"},
                    "email": "pirate@example.com",
                }
            ),
        },
    )
    assert posted.status_code == 302
    assert "inscription=1" in posted.headers["Location"]
    assert "code-apple-ne-pas-journaliser" not in posted.headers["Location"]
    assert "code-apple-ne-pas-journaliser" not in caplog.text
    assert "pirate@example.com" not in caplog.text
    created = client.post(
        "/api/v1/auth/external/register", json={"accept_terms": True}
    )
    assert created.status_code == 200
    user = User.query.filter_by(email=email).one()
    assert user.first_name == "Iris"
    assert user.last_name == "Relais"
    assert user.role == UserRole.CLIENT
    assert user.authentication_method == "sso"
    assert User.query.filter_by(email="pirate@example.com").count() == 0
    assert PortalGrant.query.filter_by(user_id=user.id).count() == 0

    again_state, again_nonce = _start(client, "apple")
    external_auth["token"] = _sign(
        "apple",
        _claims(
            "apple",
            again_nonce,
            sub=subject,
            email=None,
            email_verified=False,
            given_name=None,
            family_name=None,
        ),
    )
    again = client.post(
        "/api/v1/auth/external/callback",
        data={"state": again_state, "code": "code-apple-retour"},
    )
    assert again.status_code == 302
    assert "erreur=" not in again.headers["Location"]
    assert "inscription=1" not in again.headers["Location"]
    assert User.query.filter_by(email=email).count() == 1
    db.session.refresh(user)
    assert user.first_name == "Iris"

    cancel_state, _nonce = _start(client, "apple")
    cancelled = client.post(
        "/api/v1/auth/external/callback",
        data={"error": "user_cancelled_authorize", "state": cancel_state},
    )
    assert "erreur=annule" in cancelled.headers["Location"]


def _vider_les_cookies(client) -> None:
    """Retire les cookies de session. Conserve ``oauth_browser``."""
    for name in ("access_token", "refresh_token", "external_signup", "session"):
        client.delete_cookie(name, path="/")


def test_apple_form_post_sans_cookie_entrant(client, external_auth, app, db, caplog):
    caplog.set_level(logging.INFO)
    app.config["COOKIE_SAME_SITE"] = "Lax"
    email = f"relay-{uuid.uuid4().hex[:8]}@privaterelay.appleid.com"
    subject = f"apple-lax-{uuid.uuid4().hex}"
    state, nonce = _start(client, "apple")
    external_auth["token"] = _sign(
        "apple",
        _claims(
            "apple",
            nonce,
            sub=subject,
            email=email,
            email_verified="true",
            given_name=None,
            family_name=None,
        ),
    )
    client.set_cookie("access_token", "session-qui-ne-doit-pas-entrer", path="/")
    client.set_cookie("external_signup", "ticket-perime", path="/")
    _vider_les_cookies(client)
    assert client.get_cookie("access_token", path="/") is None
    assert client.get_cookie("external_signup", path="/") is None
    assert client.get_cookie("oauth_browser", path="/api/v1/auth/external") is not None
    posted = client.post(
        "/api/v1/auth/external/callback",
        data={
            "state": state,
            "code": "code-apple-lax-ne-pas-journaliser",
            "user": json.dumps(
                {"name": {"firstName": "Noa", "lastName": "Relais"}}
            ),
        },
        headers={"Origin": "https://appleid.apple.com"},
    )
    assert posted.status_code == 302
    assert "inscription=1" in posted.headers["Location"]
    ticket_cookie = _cookie(posted, "external_signup")
    assert ticket_cookie
    set_cookie = " ".join(posted.headers.getlist("Set-Cookie")).lower()
    assert "samesite=lax" in set_cookie
    assert "samesite=none" not in set_cookie
    assert "code-apple-lax-ne-pas-journaliser" not in caplog.text
    assert User.query.filter_by(email=email).one_or_none() is None

    _vider_les_cookies(client)
    without_ticket = client.post(
        "/api/v1/auth/external/register",
        json={"accept_terms": True},
    )
    assert without_ticket.status_code == 400
    assert without_ticket.get_json()["error"] == "inscription_expiree"

    client.set_cookie("external_signup", ticket_cookie, path="/")
    created = client.post(
        "/api/v1/auth/external/register",
        json={"accept_terms": True},
    )
    assert created.status_code == 200, created.get_json()
    user = User.query.filter_by(email=email).one()
    assert user.first_name == "Noa"
    assert user.role == UserRole.CLIENT
    assert PortalGrant.query.filter_by(user_id=user.id).count() == 0

    replay = client.post(
        "/api/v1/auth/external/callback",
        data={"state": state, "code": "code-apple-lax-ne-pas-journaliser"},
        headers={"Origin": "https://appleid.apple.com"},
    )
    assert "erreur=rejeu" in replay.headers["Location"]

    unknown = client.post(
        "/api/v1/auth/external/callback",
        data={"state": "state-inconnu", "code": "autre"},
        headers={"Origin": "https://appleid.apple.com"},
    )
    assert "erreur=state_invalide" in unknown.headers["Location"]

    expired_state, expired_nonce = _start(client, "apple")
    external_auth["token"] = _sign(
        "apple",
        _claims("apple", expired_nonce, sub=f"apple-expire-{uuid.uuid4().hex}"),
    )
    row = ExternalAuthTransaction.query.filter_by(
        state_hash=hashlib.sha256(expired_state.encode("utf-8")).hexdigest()
    ).one()
    row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    db.session.flush()
    _vider_les_cookies(client)
    expired = client.post(
        "/api/v1/auth/external/callback",
        data={"state": expired_state, "code": "code-expire"},
        headers={"Origin": "https://appleid.apple.com"},
    )
    assert "erreur=state_invalide" in expired.headers["Location"]

    cancel_state, _nonce = _start(client, "apple")
    _vider_les_cookies(client)
    cancelled = client.post(
        "/api/v1/auth/external/callback",
        data={"error": "user_cancelled_authorize", "state": cancel_state},
        headers={"Origin": "https://appleid.apple.com"},
    )
    assert "erreur=annule" in cancelled.headers["Location"]
    assert User.query.filter_by(email=email).count() == 1


def _cookie_navigateur(client) -> str:
    cookie = client.get_cookie("oauth_browser", path="/api/v1/auth/external")
    assert cookie is not None
    return cookie.value


def test_cookie_navigateur_none_uniquement_en_https(client, app, external_auth):
    app.config["COOKIE_SECURE"] = True
    response = client.post(
        "/api/v1/auth/external/start",
        json={"provider": "google", "intent": "login"},
    )
    assert response.status_code == 200
    raw = " ".join(response.headers.getlist("Set-Cookie"))
    lowered = raw.lower()
    assert "oauth_browser=" in lowered
    assert "samesite=none" in lowered
    assert "secure" in lowered
    assert "httponly" in lowered
    assert "access_token=" not in lowered
    assert "browser_binding" not in response.get_json()


def test_injection_oauth_entre_navigateurs(client, external_auth, db):
    email = f"cross-{uuid.uuid4().hex[:8]}@exemple.test"
    state, nonce = _start(client, "apple")
    external_auth["token"] = _sign(
        "apple",
        _claims("apple", nonce, email=email, email_verified="true"),
    )
    secret = _cookie_navigateur(client)
    client.delete_cookie("oauth_browser", path="/api/v1/auth/external")
    refused = client.post(
        "/api/v1/auth/external/callback",
        data={"state": state, "code": "code-navigateur-b"},
        headers={"Origin": "https://appleid.apple.com"},
    )
    assert "erreur=origine_invalide" in refused.headers["Location"]
    assert _cookie(refused, "access_token") is None
    assert _cookie(refused, "external_signup") is None
    assert User.query.filter_by(email=email).one_or_none() is None
    row = ExternalAuthTransaction.query.filter_by(
        state_hash=hashlib.sha256(state.encode("utf-8")).hexdigest()
    ).one()
    assert row.status == "pending"

    client.set_cookie("oauth_browser", secret, path="/api/v1/auth/external")
    accepted = client.post(
        "/api/v1/auth/external/callback",
        data={"state": state, "code": "code-navigateur-a"},
        headers={"Origin": "https://appleid.apple.com"},
    )
    assert "inscription=1" in accepted.headers["Location"]
    assert _cookie(accepted, "external_signup")


def test_lien_refuse_sans_cookie_du_navigateur_origine(
    client, external_auth, app, db
):
    owner = _user(db, UserRole.CLIENT)
    intruder = _user(db, UserRole.CLIENT)
    subject = f"apple-lien-{uuid.uuid4().hex}"
    state, nonce = _start(client, "apple", intent="link", headers=_headers(app, owner))
    external_auth["token"] = _sign(
        "apple",
        _claims(
            "apple",
            nonce,
            sub=subject,
            email="vue-autre@exemple.test",
            email_verified="true",
        ),
    )
    secret = _cookie_navigateur(client)
    client.delete_cookie("oauth_browser", path="/api/v1/auth/external")
    client.set_cookie("access_token", "session-du-navigateur-b", path="/")
    refused = client.post(
        "/api/v1/auth/external/callback",
        data={"state": state, "code": "code-lien-b"},
        headers={
            "Origin": "https://appleid.apple.com",
            **_headers(app, intruder),
        },
    )
    assert "erreur=origine_invalide" in refused.headers["Location"]
    assert UserExternalIdentity.query.filter_by(provider_sub=subject).count() == 0
    assert row_pending(state).status == "pending"

    client.delete_cookie("access_token", path="/")
    client.set_cookie("oauth_browser", secret, path="/api/v1/auth/external")
    accepted = client.post(
        "/api/v1/auth/external/callback",
        data={"state": state, "code": "code-lien-a"},
        headers={"Origin": "https://appleid.apple.com"},
    )
    assert accepted.status_code == 302
    assert "erreur=" not in accepted.headers["Location"]
    identity = UserExternalIdentity.query.filter_by(provider_sub=subject).one()
    assert identity.user_id == owner.id
    assert identity.user_id != intruder.id


def row_pending(state: str) -> ExternalAuthTransaction:
    return ExternalAuthTransaction.query.filter_by(
        state_hash=hashlib.sha256(state.encode("utf-8")).hexdigest()
    ).one()


def _ios_headers(device: str = "appareil-apple") -> dict[str, str]:
    return {
        "X-Auth-Contract-Version": "mobile-device-session-v1",
        "X-Device-ID": device,
        "X-Client-Platform": "ios",
        "X-Requested-With": "Expo",
    }


def _native_apple_token(nonce_claim: str, **claim_extra) -> str:
    return _sign(
        "apple",
        _claims(
            "apple",
            nonce_claim,
            aud="ch.liri.operations",
            **claim_extra,
        ),
    )


def test_apple_natif_utilise_le_contrat_mobile(client, external_auth, app, db):
    app.config["EXTERNAL_AUTH_APPLE_BUNDLE_ID"] = "ch.liri.operations"
    email = f"relay-{uuid.uuid4().hex[:8]}@privaterelay.appleid.com"
    subject = f"apple-ios-{uuid.uuid4().hex}"
    raw = f"nonce-{uuid.uuid4().hex}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    token = _native_apple_token(
        digest,
        sub=subject,
        email=email,
        email_verified="true",
        given_name=None,
        family_name=None,
    )
    missing = client.post(
        "/api/v1/auth/external/apple/native",
        json={"identity_token": token, "raw_nonce": raw, "given_name": "Iris"},
        headers={"X-Client-Platform": "ios"},
    )
    assert missing.status_code == 400
    assert missing.get_json()["error_code"] == "device_identity_required"
    assert "access_token" not in missing.get_json()

    opened = client.post(
        "/api/v1/auth/external/apple/native",
        json={
            "identity_token": token,
            "raw_nonce": raw,
            "given_name": "Iris",
            "family_name": "Relais",
            "email": "pirate@example.com",
        },
        headers=_ios_headers(),
    )
    assert opened.status_code == 200, opened.get_json()
    consent = opened.get_json()
    assert consent["kind"] == "consent"
    assert "access_token" not in consent
    assert User.query.filter_by(email=email).one_or_none() is None
    assert User.query.filter_by(email="pirate@example.com").count() == 0

    created = client.post(
        "/api/v1/auth/external/register",
        json={"accept_terms": True, "ticket": consent["ticket"]},
        headers=_ios_headers(),
    )
    assert created.status_code == 200, created.get_json()
    body = created.get_json()
    assert body["session_id"]
    assert body["refresh_token"]
    assert body["recovery_credential"]
    assert body["revocation_secret"]
    assert "access_token=" not in created.headers.get("Set-Cookie", "")
    user = User.query.filter_by(email=email).one()
    assert user.first_name == "Iris"
    assert user.role == UserRole.CLIENT
    assert user.authentication_method == "sso"
    assert Client.query.filter_by(user_id=user.id).count() == 1
    assert PortalGrant.query.filter_by(user_id=user.id).count() == 0

    raw_again = f"nonce-{uuid.uuid4().hex}"
    digest_again = hashlib.sha256(raw_again.encode("utf-8")).hexdigest()
    again_token = _native_apple_token(
        digest_again,
        sub=subject,
        email=None,
        email_verified=False,
        given_name=None,
        family_name=None,
    )
    again = client.post(
        "/api/v1/auth/external/apple/native",
        json={"identity_token": again_token, "raw_nonce": raw_again},
        headers=_ios_headers("appareil-apple-2"),
    )
    assert again.status_code == 200, again.get_json()
    session = again.get_json()
    assert session["user"]["id"] == user.id
    assert Client.query.filter_by(user_id=user.id).count() == 1
    assert "kind" not in session

    replay = client.post(
        "/api/v1/auth/external/apple/native",
        json={"identity_token": again_token, "raw_nonce": raw_again},
        headers=_ios_headers("appareil-apple-3"),
    )
    assert replay.status_code == 409
    assert replay.get_json()["error"] == "rejeu"
    assert "access_token" not in replay.get_json()

    web_token = _sign(
        "apple",
        _claims("apple", digest, aud="ch.lirie.web.auth", sub=subject),
    )
    rejected = client.post(
        "/api/v1/auth/external/apple/native",
        json={"identity_token": web_token, "raw_nonce": raw},
        headers=_ios_headers("appareil-apple-4"),
    )
    assert rejected.status_code == 400
    assert rejected.get_json()["error"] == "jeton_invalide"

    logged_out = client.post(
        "/api/v1/auth/logout",
        json={
            "refresh_token": session["refresh_token"],
            "session_id": session["session_id"],
        },
        headers={
            **_ios_headers("appareil-apple-2"),
            "Authorization": f"Bearer {session['access_token']}",
        },
    )
    assert logged_out.status_code == 200, logged_out.get_json()
    refreshed = client.post(
        "/api/v1/auth/refresh-token",
        json={"refresh_token": session["refresh_token"]},
        headers=_ios_headers("appareil-apple-2"),
    )
    assert refreshed.status_code in {400, 401}
    assert "access_token" not in (refreshed.get_json() or {})

    owner = _user(db, UserRole.CLIENT)
    raw_other = f"nonce-{uuid.uuid4().hex}"
    other_token = _native_apple_token(
        hashlib.sha256(raw_other.encode("utf-8")).hexdigest(),
        sub=f"apple-ios-{uuid.uuid4().hex}",
        email=owner.email,
        email_verified=True,
    )
    collision = client.post(
        "/api/v1/auth/external/apple/native",
        json={"identity_token": other_token, "raw_nonce": raw_other},
        headers=_ios_headers("appareil-collision"),
    )
    assert collision.status_code == 409
    assert collision.get_json()["error"] == "compte_existant"
    assert User.query.filter_by(email=owner.email).count() == 1
    assert (
        UserExternalIdentity.query.filter_by(user_id=owner.id, provider="apple").count()
        == 0
    )
    assert PortalGrant.query.filter_by(user_id=owner.id).count() == 0


def test_secret_client_apple_ne_journalise_pas_la_cle(app, caplog):
    from cryptography.hazmat.primitives import serialization

    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    app.config["EXTERNAL_AUTH_APPLE_PRIVATE_KEY"] = pem
    app.config["EXTERNAL_AUTH_APPLE_TEAM_ID"] = "TEAMTEST"
    app.config["EXTERNAL_AUTH_APPLE_KEY_ID"] = "KEYTEST"
    app.config["EXTERNAL_AUTH_APPLE_CLIENT_ID"] = "apple-test-client"
    caplog.set_level(logging.DEBUG)
    with app.app_context():
        secret = external_auth_module._apple_client_secret()
    assert "PRIVATE KEY" not in caplog.text
    assert pem not in caplog.text
    assert pem not in secret
    decoded = jwt.decode(
        secret,
        key.public_key(),
        algorithms=["ES256"],
        audience="https://appleid.apple.com",
    )
    assert decoded["iss"] == "TEAMTEST"
    assert decoded["sub"] == "apple-test-client"


def test_retrait_seulement_si_une_autre_methode_reste(client, external_auth, app, db):
    email = f"seul-{uuid.uuid4().hex[:8]}@example.com"
    subject = f"sub-seul-{uuid.uuid4().hex}"
    prepared = _prepare(client, external_auth, email=email, sub=subject)
    assert "inscription=1" in prepared.headers["Location"]
    assert (
        client.post(
            "/api/v1/auth/external/register", json={"accept_terms": True}
        ).status_code
        == 200
    )
    user = User.query.filter_by(email=email).one()
    user.totp_enabled = True
    user.totp_secret_encrypted = "secret-inchange"
    db.session.flush()
    refused = client.delete(
        "/api/v1/auth/external/identities/google",
        headers=_headers(app, user),
    )
    assert refused.status_code == 409
    assert refused.get_json()["error"] == "derniere_methode"
    assert UserExternalIdentity.query.filter_by(user_id=user.id).count() == 1
    user_id = int(user.id)
    listed = client.get(
        "/api/v1/auth/external/identities",
        headers=_headers(app, user),
    )
    assert listed.status_code == 200
    assert listed.get_json()["google"]["linked"] is True
    assert listed.get_json()["password_login"] is False

    _drop_session(client)
    returned = _prepare(client, external_auth, email=email, sub=subject)
    assert returned.status_code == 302
    assert "erreur=" not in returned.headers["Location"]
    assert "inscription=1" not in returned.headers["Location"]
    assert int(User.query.filter_by(email=email).one().id) == user_id
    assert UserExternalIdentity.query.filter_by(provider_sub=subject).count() == 1

    _drop_session(client)
    owner = _user(db, UserRole.CLIENT)
    secret = "totp-proprietaire"
    owner.totp_enabled = True
    owner.totp_secret_encrypted = secret
    db.session.flush()
    state, nonce = _start(client, intent="link", headers=_headers(app, owner))
    external_auth["token"] = _sign(
        "google",
        _claims("google", nonce, sub=f"sub-mot-de-passe-{uuid.uuid4().hex}"),
    )
    assert _callback(client, state).status_code == 302
    removed = client.delete(
        "/api/v1/auth/external/identities/google",
        headers=_headers(app, owner),
    )
    assert removed.status_code == 200
    assert UserExternalIdentity.query.filter_by(user_id=owner.id).count() == 0
    db.session.refresh(owner)
    db.session.refresh(user)
    assert owner.totp_enabled is True
    assert owner.totp_secret_encrypted == secret
    assert user.totp_enabled is True
    assert user.totp_secret_encrypted == "secret-inchange"


def _purge_email(db, email: str) -> None:
    """Ne touche pas au registre légal.

    La transaction de test est annulée par le fixture. Effacer un
    utilisateur tenterait d'effacer son acceptation, ce que la base refuse.
    """
    _ = (db, email)


def test_session_rotation_deconnexion_et_reconexion(client, external_auth, db):
    email = f"session-{uuid.uuid4().hex[:8]}@example.com"
    subject = f"sub-session-{uuid.uuid4().hex}"
    try:
        prepared = _prepare(client, external_auth, email=email, sub=subject)
        assert "inscription=1" in prepared.headers["Location"]
        created = client.post(
            "/api/v1/auth/external/register", json={"accept_terms": True}
        )
        assert created.status_code == 200
        user = User.query.filter_by(email=email).one()
        user_id = int(user.id)
        client_id = int(Client.query.filter_by(user_id=user_id).one().id)
        identity_id = int(
            UserExternalIdentity.query.filter_by(provider_sub=subject).one().id
        )
        applications = OrganizationApplication.query.count()
        assert user.role == UserRole.CLIENT
        assert PortalGrant.query.filter_by(user_id=user_id).count() == 0
        first_refresh = _cookie(created, "refresh_token")
        assert first_refresh
        client.set_cookie("refresh_token", first_refresh)
        rotated = client.post("/api/v1/auth/refresh-token", json={})
        assert rotated.status_code == 200, rotated.get_json()
        second_refresh = _cookie(rotated, "refresh_token")
        assert second_refresh
        assert second_refresh != first_refresh
        client.set_cookie("refresh_token", first_refresh)
        reused = client.post("/api/v1/auth/refresh-token", json={})
        assert reused.status_code != 200

        client.set_cookie("access_token", _cookie(created, "access_token") or "")
        client.set_cookie("refresh_token", second_refresh)
        logged_out = client.post("/api/v1/auth/logout", json={})
        assert logged_out.status_code == 200
        bootstrap = client.get("/api/v1/auth/bootstrap")
        assert bootstrap.get_json()["is_authenticated"] is False
        client.set_cookie("refresh_token", second_refresh)
        after_logout = client.post("/api/v1/auth/refresh-token", json={})
        assert after_logout.status_code != 200

        _drop_session(client)
        again = _prepare(client, external_auth, email=email, sub=subject)
        assert again.status_code == 302
        assert "inscription=1" not in again.headers["Location"]
        assert "erreur=" not in again.headers["Location"]
        same = User.query.filter_by(email=email).one()
        assert int(same.id) == user_id
        assert int(Client.query.filter_by(user_id=user_id).one().id) == client_id
        assert (
            int(UserExternalIdentity.query.filter_by(provider_sub=subject).one().id)
            == identity_id
        )
        assert User.query.filter_by(email=email).count() == 1
        assert UserExternalIdentity.query.filter_by(provider_sub=subject).count() == 1
        assert same.role == UserRole.CLIENT
        assert PortalGrant.query.filter_by(user_id=user_id).count() == 0
        assert OrganizationApplication.query.count() == applications
    finally:
        _purge_email(db, email)


def test_revocation_conservee_et_login_mot_de_passe_inchange(client, external_auth, db):
    owner = _user(db, UserRole.COMPANY)
    subject = f"sub-revoque-{uuid.uuid4().hex}"
    try:
        company = _company(db, owner)
        grant = PortalGrant(
            user_id=owner.id,
            portal=PortalKind.DRIVER.value,
            company_id=company.id,
            status=PortalGrantStatus.REVOKED.value,
            revoked_at=datetime.now(UTC),
        )
        db.session.add(grant)
        identity = UserExternalIdentity(
            user_id=owner.id,
            provider="google",
            provider_sub=subject,
            email_at_link=owner.email,
        )
        db.session.add(identity)
        db.session.flush()
        returned = _prepare(
            client, external_auth, sub=subject, email="change@example.com"
        )
        assert returned.status_code == 302
        assert "erreur=" not in returned.headers["Location"]
        db.session.refresh(owner)
        db.session.refresh(grant)
        assert owner.role == UserRole.COMPANY
        assert grant.status == PortalGrantStatus.REVOKED.value
        assert (
            PortalGrant.query.filter_by(user_id=owner.id, status="active").count() == 0
        )

        wrong = client.post(
            "/api/v1/auth/login",
            json={"email": owner.email, "password": "mauvais-mot-de-passe"},
        )
        assert wrong.status_code in (401, 403)
    finally:
        _purge_email(db, owner.email)


def test_institution_recoit_la_session_web(client, external_auth, db):
    user = _user(db, UserRole.INSTITUTION)
    subject = f"sub-institution-{uuid.uuid4().hex}"
    try:
        db.session.add(
            UserExternalIdentity(
                user_id=user.id,
                provider="google",
                provider_sub=subject,
                email_at_link=user.email,
            )
        )
        db.session.flush()
        response = _prepare(client, external_auth, sub=subject)
        assert response.status_code == 302
        access = _cookie(response, "access_token")
        assert access
        claims = jwt.decode(access, options={"verify_signature": False})
        assert claims.get("sid")
        assert claims.get("role") == "INSTITUTION"
    finally:
        _purge_email(db, user.email)


def test_fournisseur_indisponible_ne_cree_rien(client, external_auth, monkeypatch):
    before = User.query.filter(User.username.like("ext_%")).count()

    def _down(_provider, _code, _verifier):
        raise ExternalAuthRejected(
            "fournisseur_indisponible",
            "Le fournisseur est indisponible.",
            status=503,
        )

    monkeypatch.setattr("services.external_auth.exchange_code", _down)
    state, _nonce = _start(client)
    response = _callback(client, state)
    assert "erreur=fournisseur_indisponible" in response.headers["Location"]
    assert User.query.filter(User.username.like("ext_%")).count() == before


def test_jeton_expire_audience_issuer_et_algorithme(client, external_auth):
    past = int((datetime.now(UTC) - timedelta(hours=1)).timestamp())
    state, nonce = _start(client)
    external_auth["token"] = _sign("google", _claims("google", nonce, exp=past))
    expired = _callback(client, state)
    assert "erreur=jeton_invalide" in expired.headers["Location"]

    state, nonce = _start(client)
    external_auth["token"] = _sign(
        "google", _claims("google", nonce, aud="autre-client")
    )
    audience = _callback(client, state)
    assert "erreur=jeton_invalide" in audience.headers["Location"]

    state, nonce = _start(client)
    external_auth["token"] = _sign(
        "google", _claims("google", nonce, iss="https://evil.example")
    )
    issuer = _callback(client, state)
    assert "erreur=jeton_invalide" in issuer.headers["Location"]

    state, nonce = _start(client)
    external_auth["token"] = _sign(
        "google", _claims("google", nonce, azp="autre-client")
    )
    party = _callback(client, state)
    assert "erreur=jeton_invalide" in party.headers["Location"]

    state, nonce = _start(client)
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "none", "typ": "JWT"}).encode("utf-8")
    ).rstrip(b"=").decode("ascii")
    body = base64.urlsafe_b64encode(
        json.dumps(_claims("google", nonce)).encode("utf-8")
    ).rstrip(b"=").decode("ascii")
    external_auth["token"] = f"{header}.{body}."
    unsigned = _callback(client, state)
    assert "erreur=jeton_invalide" in unsigned.headers["Location"]

    accepted = _prepare(
        client,
        external_auth,
        iss="accounts.google.com",
        sub=f"sub-issuer-{uuid.uuid4().hex}",
        email=f"issuer-{uuid.uuid4().hex[:8]}@example.com",
    )
    assert "inscription=1" in accepted.headers["Location"]


def test_profil_navigateur_ignore_et_secret_non_journalise(
    app, client, external_auth, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    email = f"serveur-{uuid.uuid4().hex[:8]}@example.com"
    subject = f"sub-serveur-{uuid.uuid4().hex}"
    state, nonce = _start(client)
    external_auth["token"] = _sign(
        "google", _claims("google", nonce, email=email, sub=subject)
    )
    forged = _sign(
        "google",
        _claims(
            "google",
            nonce,
            email="pirate@example.com",
            sub="sub-pirate",
        ),
    )
    response = client.get(
        "/api/v1/auth/external/callback",
        query_string={
            "state": state,
            "code": "code-legitime-non-journalise",
            "id_token": forged,
            "email": "pirate@example.com",
        },
    )
    assert "inscription=1" in response.headers["Location"]
    assert "code-legitime-non-journalise" not in caplog.text
    assert forged not in caplog.text
    assert User.query.filter_by(email="pirate@example.com").count() == 0
    stored = ExternalAuthTransaction.query.filter_by(email=email).one()
    assert stored.provider_sub == subject

    def _boom(_request, timeout=8):
        raise URLError("indisponible")

    monkeypatch.setattr(external_auth_module, "exchange_code", _REAL_EXCHANGE)
    monkeypatch.setattr(external_auth_module, "urlopen", _boom)
    caplog.clear()
    with app.app_context(), pytest.raises(ExternalAuthRejected):
        external_auth_module.exchange_code(
            "google",
            "code-ne-pas-journaliser",
            "verificateur-ne-pas-journaliser",
        )
    assert "code-ne-pas-journaliser" not in caplog.text
    assert "verificateur-ne-pas-journaliser" not in caplog.text


def test_email_fournisseur_ne_prend_pas_un_autre_compte(client, external_auth, db):
    email = f"stable-{uuid.uuid4().hex[:8]}@example.com"
    subject = f"sub-stable-{uuid.uuid4().hex}"
    other = _user(db, UserRole.CLIENT)
    try:
        prepared = _prepare(client, external_auth, email=email, sub=subject)
        assert "inscription=1" in prepared.headers["Location"]
        assert (
            client.post(
                "/api/v1/auth/external/register", json={"accept_terms": True}
            ).status_code
            == 200
        )
        owner = User.query.filter_by(email=email).one()
        owner_id = int(owner.id)
        _drop_session(client)
        changed = _prepare(
            client, external_auth, email=other.email, sub=subject
        )
        assert changed.status_code == 302
        assert "erreur=" not in changed.headers["Location"]
        access = _cookie(changed, "access_token")
        assert access
        claims = jwt.decode(access, options={"verify_signature": False})
        assert claims["sub"] == owner.public_id
        db.session.refresh(owner)
        db.session.refresh(other)
        assert owner.email == email
        assert int(owner.id) == owner_id
        assert other.role == UserRole.CLIENT
        assert User.query.filter_by(email=other.email).count() == 1
        assert UserExternalIdentity.query.filter_by(provider_sub=subject).count() == 1
        identity = UserExternalIdentity.query.filter_by(provider_sub=subject).one()
        assert identity.email_at_link == email
        assert PortalGrant.query.filter_by(user_id=owner_id).count() == 0
    finally:
        _purge_email(db, email)


def test_compte_desactive_ne_se_rouvre_pas(client, external_auth, db):
    email = f"ferme-{uuid.uuid4().hex[:8]}@example.com"
    subject = f"sub-ferme-{uuid.uuid4().hex}"
    try:
        prepared = _prepare(client, external_auth, email=email, sub=subject)
        assert "inscription=1" in prepared.headers["Location"]
        assert (
            client.post(
                "/api/v1/auth/external/register", json={"accept_terms": True}
            ).status_code
            == 200
        )
        owner = User.query.filter_by(email=email).one()
        dossier = Client.query.filter_by(user_id=owner.id).one()
        dossier.is_active = False
        db.session.flush()
        _drop_session(client)
        refused = _prepare(client, external_auth, email=email, sub=subject)
        assert "erreur=compte_indisponible" in refused.headers["Location"]
        assert _cookie(refused, "access_token") is None
        assert User.query.filter_by(email=email).count() == 1
        assert Client.query.filter_by(user_id=owner.id).count() == 1
    finally:
        _purge_email(db, email)


def test_retrait_puis_meme_sub_ne_rattache_pas(client, external_auth, app, db):
    owner = _user(db, UserRole.CLIENT)
    subject = f"sub-retire-{uuid.uuid4().hex}"
    state, nonce = _start(client, intent="link", headers=_headers(app, owner))
    external_auth["token"] = _sign(
        "google", _claims("google", nonce, sub=subject, email=owner.email)
    )
    linked = _callback(client, state)
    assert linked.status_code == 302
    assert "erreur=" not in linked.headers["Location"]
    assert UserExternalIdentity.query.filter_by(provider_sub=subject).count() == 1
    removed = client.delete(
        "/api/v1/auth/external/identities/google",
        headers=_headers(app, owner),
    )
    assert removed.status_code == 200
    _drop_session(client)
    again = _prepare(client, external_auth, sub=subject, email=owner.email)
    assert "erreur=compte_existant" in again.headers["Location"]
    assert User.query.filter_by(email=owner.email).count() == 1
    assert UserExternalIdentity.query.filter_by(provider_sub=subject).count() == 0
    assert PortalGrant.query.filter_by(user_id=owner.id).count() == 0


def test_lien_refuse_si_adresse_dun_autre_compte(client, external_auth, app, db):
    owner = _user(db, UserRole.CLIENT)
    other = _user(db, UserRole.COMPANY)
    state, nonce = _start(client, intent="link", headers=_headers(app, owner))
    external_auth["token"] = _sign(
        "google",
        _claims(
            "google",
            nonce,
            sub=f"sub-conflit-{uuid.uuid4().hex}",
            email=other.email,
        ),
    )
    refused = _callback(client, state)
    assert "erreur=adresse_autre_compte" in refused.headers["Location"]
    assert UserExternalIdentity.query.filter_by(user_id=owner.id).count() == 0
    assert User.query.filter_by(email=other.email).one().role == UserRole.COMPANY
    listed = client.get(
        "/api/v1/auth/external/identities", headers=_headers(app, owner)
    )
    assert listed.status_code == 200
    body = listed.get_json()
    assert body["google"]["linked"] is False
    assert "provider_sub" not in json.dumps(body)
    anonymous = client.get("/api/v1/auth/external/identities")
    assert anonymous.status_code == 401


def test_session_expiree_et_login_classique(client, app, db, external_auth):
    owner = _user(db, UserRole.CLIENT)
    grants = PortalGrant.query.filter_by(user_id=owner.id).count()
    password = client.post(
        "/api/v1/auth/login",
        json={"email": owner.email, "password": "password123"},
    )
    assert password.status_code == 200, password.get_json()
    assert password.get_json()["user"]["id"] == owner.id
    assert PortalGrant.query.filter_by(user_id=owner.id).count() == grants

    _drop_session(client)
    with app.app_context():
        expired = create_access_token(
            identity=str(owner.public_id),
            expires_delta=timedelta(days=-1),
            additional_claims={"role": owner.role.value, "aud": "atmr-api"},
        )
    protected = client.get(
        "/api/v1/auth/external/identities",
        headers={"Authorization": f"Bearer {expired}"},
    )
    assert protected.status_code in (401, 422)

    collision = _prepare(
        client,
        external_auth,
        email=owner.email,
        sub=f"sub-classique-{uuid.uuid4().hex}",
    )
    assert "erreur=compte_existant" in collision.headers["Location"]
    assert owner.role == UserRole.CLIENT
    assert PortalGrant.query.filter(
        PortalGrant.user_id == owner.id,
        PortalGrant.portal.in_(
            (
                PortalKind.COMPANY.value,
                PortalKind.DRIVER.value,
                PortalKind.INSTITUTION.value,
            )
        ),
    ).count() == 0
