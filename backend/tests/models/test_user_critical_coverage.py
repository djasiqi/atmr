"""Couverture critique de ``models.user`` (validateurs, mot de passe, chiffrement)."""

from __future__ import annotations

import builtins
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from werkzeug.security import generate_password_hash

from models.enums import GenderEnum, InstitutionRole, UserRole
from models.user import ADDRESS_MAX_LENGTH, User


@pytest.fixture
def user(app):
    """Instance User hors session, dans le contexte Flask."""
    with app.app_context():
        yield User()


def test_set_password_sans_id_ni_expiration(user, monkeypatch):
    monkeypatch.delenv("PASSWORD_EXPIRATION_DAYS", raising=False)
    user.set_password("Secret123!", force_change=True)
    assert user.force_password_change is True
    assert user.check_password("Secret123!") is True
    assert user.check_password("mauvais") is False
    assert user.password_expires_at is None


def test_set_password_historique_et_expiration(user, monkeypatch):
    monkeypatch.setenv("PASSWORD_EXPIRATION_DAYS", "30")
    added = []

    class FakeHistory:
        @staticmethod
        def add_password_to_history(user_id, old_hash):
            added.append((user_id, old_hash))

    monkeypatch.setattr(
        "security.password_history.PasswordHistoryService", FakeHistory
    )
    user.id = 42
    user.password = "ancien-hash"
    user.set_password("Nouveau123!", force_change=False)
    assert added == [(42, "ancien-hash")]
    assert user.force_password_change is False
    assert user.password_expires_at is not None
    assert user.password_expires_at > datetime.now(UTC)


def test_set_password_historique_en_echec_ne_bloque_pas(user, monkeypatch):
    class BoomHistory:
        @staticmethod
        def add_password_to_history(_user_id, _old_hash):
            raise RuntimeError("historique indisponible")

    monkeypatch.setattr(
        "security.password_history.PasswordHistoryService", BoomHistory
    )
    user.id = 7
    user.password = "ancien"
    user.set_password("Toujours123!")
    assert user.check_password("Toujours123!") is True


def test_check_password_branches_vides_et_invalides(monkeypatch):
    assert User.check_password(SimpleNamespace(password=""), "x") is False
    assert User.check_password(SimpleNamespace(password="   "), "x") is False
    assert User.check_password(SimpleNamespace(password=b""), "x") is False
    assert User.check_password(SimpleNamespace(password=b"pas-un-hash"), "x") is False
    hashed = generate_password_hash("ok")
    assert User.check_password(SimpleNamespace(password=hashed.encode()), "ok") is True
    assert (
        User.check_password(SimpleNamespace(password=bytearray(hashed.encode())), "ok")
        is True
    )

    def _raise_value_error(_hash, _password):
        raise ValueError("hash legacy")

    monkeypatch.setattr("models.user.check_password_hash", _raise_value_error)
    assert User.check_password(SimpleNamespace(password="pbkdf2:sha256:x"), "x") is False


def test_validate_phone(user):
    assert user.validate_phone("phone", None) is None
    assert user.validate_phone("phone", 41791234567) is None
    assert user.validate_phone("phone", "  ") is None
    assert user.validate_phone("phone", "+41791234567") == "+41791234567"
    with pytest.raises(ValueError, match="téléphone invalide"):
        user.validate_phone("phone", "abc")


def test_validate_birth_date(user):
    today = date.today()
    assert user.validate_birth_date("birth_date", None) is None
    assert user.validate_birth_date("birth_date", today) == today
    with pytest.raises(ValueError, match="futur"):
        user.validate_birth_date("birth_date", today + timedelta(days=1))


def test_validate_address(user):
    assert user.validate_address("address", None) is None
    assert user.validate_address("address", "Rue de Test 1") == "Rue de Test 1"
    with pytest.raises(ValueError, match="vide"):
        user.validate_address("address", "   ")
    with pytest.raises(ValueError, match="dépasser"):
        user.validate_address("address", "x" * (ADDRESS_MAX_LENGTH + 1))


def test_validate_name(user):
    assert user.validate_name("first_name", None) is None
    assert user.validate_name("first_name", "Ada") == "Ada"
    with pytest.raises(ValueError, match="first_name"):
        user.validate_name("first_name", "  ")


def test_validate_gender(user):
    assert user.validate_gender("gender", None) is None
    assert user.validate_gender("gender", "HOMME") == GenderEnum.HOMME
    assert user.validate_gender("gender", GenderEnum.FEMME) == GenderEnum.FEMME
    with pytest.raises(ValueError, match="Genre invalide"):
        user.validate_gender("gender", "inconnu")


def test_validate_role(user):
    assert user.validate_role("role", "client") == UserRole.CLIENT
    assert user.validate_role("role", UserRole.ADMIN) == UserRole.ADMIN
    with pytest.raises(ValueError, match="Invalid role"):
        user.validate_role("role", "wizard")


def test_validate_institution_role(user):
    assert user.validate_institution_role("institution_role", None) is None
    assert (
        user.validate_institution_role(
            "institution_role", InstitutionRole.ADMIN.value
        )
        == "institution_admin"
    )
    with pytest.raises(ValueError, match="Invalid institution_role"):
        user.validate_institution_role("institution_role", "superuser")


def test_validate_email(user):
    assert user.validate_email("email", None) is None
    assert user.validate_email("email", "  ") is None
    assert user.validate_email("email", "  a@b.ch ") == "a@b.ch"
    with pytest.raises(ValueError, match="email invalide"):
        user.validate_email("email", "pas-un-email")


def _fake_crypto(monkeypatch, *, decrypt_error=False):
    svc = MagicMock()
    svc.encrypt_field.side_effect = lambda v: f"enc:{v}"
    if decrypt_error:
        svc.decrypt_field.side_effect = RuntimeError("decrypt boom")
    else:
        svc.decrypt_field.side_effect = lambda v: v.removeprefix("enc:")
    monkeypatch.setattr(
        "security.crypto.get_encryption_service", lambda: svc, raising=True
    )
    return svc


def _block_crypto_import(monkeypatch):
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "security.crypto":
            raise ImportError("crypto indisponible")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


@pytest.mark.parametrize(
    ("plain_attr", "enc_attr", "prop", "fallback"),
    [
        ("phone", "phone_encrypted", "phone_secure", "+41790001111"),
        ("email", "email_encrypted", "email_secure", "user@test.ch"),
        ("first_name", "first_name_encrypted", "first_name_secure", "Ada"),
        ("last_name", "last_name_encrypted", "last_name_secure", "Lovelace"),
        ("address", "address_encrypted", "address_secure", "Rue Clair 1"),
    ],
)
def test_champs_secure_dechiffre_et_fallback(
    user, monkeypatch, plain_attr, enc_attr, prop, fallback
):
    _fake_crypto(monkeypatch)
    setattr(user, plain_attr, None)
    user.encryption_migrated = True
    setattr(user, enc_attr, "enc:secret")
    assert getattr(user, prop) == "secret"

    user.encryption_migrated = False
    setattr(user, plain_attr, fallback)
    assert getattr(user, prop) == fallback


@pytest.mark.parametrize(
    ("enc_attr", "prop"),
    [
        ("phone_encrypted", "phone_secure"),
        ("email_encrypted", "email_secure"),
        ("first_name_encrypted", "first_name_secure"),
        ("last_name_encrypted", "last_name_secure"),
        ("address_encrypted", "address_secure"),
    ],
)
def test_champs_secure_decrypt_en_echec(user, monkeypatch, enc_attr, prop):
    _fake_crypto(monkeypatch, decrypt_error=True)
    user.encryption_migrated = True
    setattr(user, enc_attr, "enc:x")
    assert getattr(user, prop) is None


@pytest.mark.parametrize(
    "prop",
    [
        "phone_secure",
        "email_secure",
        "first_name_secure",
        "last_name_secure",
        "address_secure",
    ],
)
def test_champs_secure_import_error_getter(user, monkeypatch, prop):
    _block_crypto_import(monkeypatch)
    user.phone = "+41790000000"
    user.email = "a@b.ch"
    user.first_name = "Ada"
    user.last_name = "Lovelace"
    user.address = "Rue 1"
    value = getattr(user, prop)
    assert value is not None


@pytest.mark.parametrize(
    ("enc_attr", "prop"),
    [
        ("phone_encrypted", "phone_secure"),
        ("email_encrypted", "email_secure"),
        ("first_name_encrypted", "first_name_secure"),
        ("last_name_encrypted", "last_name_secure"),
        ("address_encrypted", "address_secure"),
    ],
)
def test_champs_secure_setter_chiffre_et_vide(user, monkeypatch, enc_attr, prop):
    _fake_crypto(monkeypatch)
    setattr(user, prop, "valeur")
    assert getattr(user, enc_attr) == "enc:valeur"
    assert user.encryption_migrated is True
    setattr(user, prop, None)
    assert getattr(user, enc_attr) is None
    setattr(user, prop, "")
    assert getattr(user, enc_attr) is None


def test_phone_secure_setter_vide_ancienne_colonne(user, monkeypatch):
    _fake_crypto(monkeypatch)
    user.phone = "+41791111111"
    user.phone_secure = "+41792222222"
    assert user.phone is None
    user.phone_secure = None
    assert user.phone is None


def test_champs_secure_setter_import_error(user, monkeypatch):
    _block_crypto_import(monkeypatch)
    user.phone_secure = "+41793333333"
    assert user.phone == "+41793333333"
    user.email_secure = "fallback@test.ch"
    assert user.email == "fallback@test.ch"
    user.first_name_secure = "Ines"
    assert user.first_name == "Ines"
    user.last_name_secure = "Martin"
    assert user.last_name == "Martin"
    user.address_secure = "Chemin 2"
    assert user.address == "Chemin 2"


def test_serialize_sans_institution(user):
    user.id = 1
    user.public_id = "pub-1"
    user.username = "ada"
    user.email = "ada@test.ch"
    user.role = UserRole.CLIENT
    user.force_password_change = False
    data = user.serialize
    assert data["id"] == 1
    assert data["user_id"] == 1
    assert data["first_name"] == "Non spécifié"
    assert data["last_name"] == "Non spécifié"
    assert data["phone"] == "Non spécifié"
    assert data["address"] == "Non spécifié"
    assert data["gender"] == "Non spécifié"
    assert data["zip_code"] == "Non spécifié"
    assert data["city"] == "Non spécifié"
    assert data["role"] == "CLIENT"
    assert "institution_id" not in data
    assert user.to_dict() == data


def test_serialize_complet_avec_institution(user):
    user.id = 2
    user.public_id = "pub-2"
    user.username = "ines"
    user.email = "ines@test.ch"
    user.first_name = "Ines"
    user.last_name = "Martin"
    user.phone = "+41794444444"
    user.address = "Rue 3"
    user.birth_date = date(1990, 5, 1)
    user.gender = GenderEnum.FEMME
    user.profile_image = "img.png"
    user.role = UserRole.INSTITUTION
    user.zip_code = "1200"
    user.city = "Genève"
    user.created_at = datetime(2024, 1, 2, tzinfo=UTC)
    user.force_password_change = True
    user.institution_id = 9
    user.institution_role = InstitutionRole.ADMIN.value
    user.account_status = None
    user.job_title = "Directrice"
    data = user.serialize
    assert data["first_name"] == "Ines"
    assert data["birth_date"] == "1990-05-01"
    assert data["gender"] == "FEMME"
    assert data["institution_id"] == 9
    assert data["institution_role"] == "institution_admin"
    assert data["account_status"] == "active"
    assert data["job_title"] == "Directrice"
    assert data["created_at"] is not None


def test_serialize_role_absent(user):
    from sqlalchemy.orm.attributes import set_committed_value

    user.id = 3
    set_committed_value(user, "role", None)
    data = user.serialize
    assert data["role"] == "None"


def test_full_name_et_repr(user):
    user.username = "ada"
    user.email = "ada@test.ch"
    user.role = UserRole.CLIENT
    user.first_name = None
    user.last_name = None
    assert user.full_name == ""
    user.first_name = "Ada"
    user.last_name = "Lovelace"
    assert user.full_name == "Ada Lovelace"
    assert "ada@test.ch" in repr(user)
    assert "CLIENT" in repr(user)
