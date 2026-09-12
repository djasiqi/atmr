"""Preuves de classification CodeQL ``py/weak-sensitive-data-hashing``.

Ces tests verrouillent le *rôle* de chaque primitive. Ils n'autorisent
pas une migration cosmétique MD5/SHA-1 → SHA-256.
"""

from __future__ import annotations

import hashlib
import inspect
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from models.institution_api_key import (
    API_KEY_RANDOM_BYTES,
    generate_api_key,
    hash_api_key,
)
from routes import auth
from security.password_policy import PasswordPolicyService
from services.geolocation import maps as maps_mod


class TestPasswordStorageRemainsKdf:
    """Le stockage mot de passe reste un KDF dédié, pas SHA-256/SHA-1."""

    def test_set_password_utilise_un_kdf_werkzeug(self, app, db):
        from models import User

        with app.app_context():
            suffix = uuid.uuid4().hex[:10]
            user = User(
                username=f"hash_kdf_{suffix}",
                email=f"hash_kdf_{suffix}@example.com",
                password="",
            )
            user.set_password("SecurePass1!x")
            db.session.add(user)
            db.session.commit()
            stored = str(user.password)
            assert stored != "SecurePass1!x"
            assert stored.startswith("pbkdf2:") or stored.startswith("scrypt:")
            assert user.check_password("SecurePass1!x") is True
            db.session.delete(user)
            db.session.commit()


class Test245PasswordHashVersionFingerprint:
    """245 : SHA-256 d'un extrait du hash déjà stocké, pas du mot de passe."""

    def test_version_derivee_du_hash_stocke_pas_du_clair(self):
        stored = "pbkdf2:sha256:600000$salt$digestabcdef"
        version = auth._get_password_hash_version(SimpleNamespace(password=stored))
        assert version
        assert version != stored
        assert "SecurePass" not in version
        assert len(version) == auth.PASSWORD_HASH_VERSION_LENGTH

    def test_prefixe_werkzeug_ne_discrimine_pas_le_digest(self):
        """Les 16 premiers caractères d'un hash werkzeug sont le schéma.

        Deux KDF distincts avec le même algorithme/coût produisent la même
        empreinte. Ce n'est pas un hash de mot de passe : l'invalidation
        réelle des JWT repose sur ``token_version`` (Lot 0 SEC-02).
        """
        old = auth._get_password_hash_version(
            SimpleNamespace(password="pbkdf2:sha256:600000$aaa$old")
        )
        new = auth._get_password_hash_version(
            SimpleNamespace(password="pbkdf2:sha256:600000$bbb$new")
        )
        assert old == new
        assert old == auth._get_password_hash_version(
            SimpleNamespace(password="pbkdf2:sha256:600000$ccc$other")
        )

    def test_version_vide_sans_hash(self):
        assert auth._get_password_hash_version(SimpleNamespace(password="")) == ""


class Test246PasswordlessOtpModel:
    """246 : OTP 6 chiffres, TTL court, 404 hors development."""

    def test_code_six_chiffres_csprng(self):
        code = auth._create_passwordless_otp_code()
        assert len(code) == 6
        assert code.isdigit()

    def test_ttl_defaut_600_plancher_120(self, monkeypatch):
        monkeypatch.delenv("PASSWORDLESS_OTP_TTL_SECONDS", raising=False)
        assert auth._resolve_passwordless_otp_ttl_seconds() == 600
        monkeypatch.setenv("PASSWORDLESS_OTP_TTL_SECONDS", "30")
        assert auth._resolve_passwordless_otp_ttl_seconds() == 120

    def test_hors_development_interdit(self, app):
        app.config["ENVIRONMENT"] = "production"
        with app.app_context():
            assert auth._passwordless_allowed_in_environment() is False

    def test_verification_compare_digest_pas_egalite(self):
        source = inspect.getsource(auth.PasswordlessOtpVerify.post)
        assert "hmac.compare_digest" in source
        assert "hashlib.sha256" in source


class Test247HmacApiKey:
    """247 : HMAC-SHA256 réel d'une clé 256 bits, pas un hash artisanal."""

    def test_generate_entropie_256_bits(self):
        raw_key, _prefix, key_hash = generate_api_key()
        assert raw_key.startswith("lir_")
        assert len(raw_key) == len("lir_") + API_KEY_RANDOM_BYTES * 2
        assert len(key_hash) == 64

    def test_hmac_reel_et_reproductible(self):
        raw_key, _prefix, key_hash = generate_api_key()
        assert hash_api_key(raw_key) == key_hash
        source = inspect.getsource(hash_api_key)
        assert "hmac.new" in source
        assert "hashlib.sha256" in source
        assert "secret + " not in source

    def test_cles_distinctes_digests_distincts(self):
        a = generate_api_key()[2]
        b = generate_api_key()[2]
        assert a != b

    def test_mauvais_secret_change_le_digest(self, monkeypatch):
        raw_key, _prefix, original = generate_api_key()
        monkeypatch.setattr(
            "models.institution_api_key.API_KEY_HMAC_SECRET",
            "autre-secret-de-test",
        )
        from models.institution_api_key import hash_api_key as hash_again

        assert hash_again(raw_key) != original


class Test248NominatimCacheKey:
    """248 : MD5 = clé de cache, usedforsecurity=False."""

    def test_md5_annonce_non_securite(self):
        source = inspect.getsource(maps_mod.geocode_address_nominatim)
        assert "hashlib.md5" in source
        assert "usedforsecurity=False" in source
        assert "nominatim:geocode:" in source

    def test_collision_cache_ne_pas_etre_une_frontiere_auth(self):
        """Deux adresses différentes doivent produire des clés distinctes
        dans le cas nominal (pas de collision MD5 accidentelle).
        """
        a = maps_mod._normalize_address_for_cache("1 rue Test, Lausanne", "CH")
        b = maps_mod._normalize_address_for_cache("2 rue Test, Lausanne", "CH")
        ha = hashlib.md5(a.encode("utf-8"), usedforsecurity=False).hexdigest()
        hb = hashlib.md5(b.encode("utf-8"), usedforsecurity=False).hexdigest()
        assert ha != hb


class Test249HibpProtocol:
    """249 : SHA-1 imposé par HIBP, k-anonymity, pas de stockage."""

    def test_requete_prefixe_cinq_caracteres_uniquement(self):
        captured: dict[str, str] = {}

        def fake_get(url: str, timeout: int = 0):
            captured["url"] = url
            captured["timeout"] = str(timeout)
            resp = MagicMock()
            resp.status_code = 200
            resp.text = "00000:0\n"
            return resp

        with patch("security.password_policy.requests.get", side_effect=fake_get):
            is_safe, error = PasswordPolicyService.check_hibp("NotARealPassword1!")

        assert is_safe is True
        assert error is None
        assert captured["url"].startswith("https://api.pwnedpasswords.com/range/")
        prefix = captured["url"].rsplit("/", maxsplit=1)[-1]
        assert len(prefix) == 5
        assert prefix.isupper()
        assert "NotARealPassword1!" not in captured["url"]
        expected = (
            hashlib.sha1(b"NotARealPassword1!", usedforsecurity=False)
            .hexdigest()
            .upper()
        )
        assert prefix == expected[:5]
        assert expected[5:] not in captured["url"]

    def test_suffixe_compromis_detecte_sans_envoyer_le_hash_complet(self):
        password = "PwnedExample1!"
        full = (
            hashlib.sha1(password.encode("utf-8"), usedforsecurity=False)
            .hexdigest()
            .upper()
        )
        suffix = full[5:]

        def fake_get(url: str, timeout: int = 0):
            assert full not in url
            resp = MagicMock()
            resp.status_code = 200
            resp.text = f"{suffix}:42\n"
            return resp

        with patch("security.password_policy.requests.get", side_effect=fake_get):
            is_safe, error = PasswordPolicyService.check_hibp(password)
        assert is_safe is False
        assert error is not None

    def test_log_hibp_ne_journalise_que_le_compteur(self):
        source = inspect.getsource(PasswordPolicyService.check_hibp)
        assert "password_hash[:5]" in source
        assert "usedforsecurity=False" in source
        assert "HIBP (count: %s)" in source
        assert (
            "logger.warning(\n                            password_hash" not in source
        )
