"""Couverture critique de ``models.client`` (validateurs, chiffrement, serialize)."""

from __future__ import annotations

import builtins
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm.attributes import set_committed_value

from models.client import Client
from models.enums import ClientType, ManagementMode


@pytest.fixture
def client_row(app):
    """Instance Client hors session, dans le contexte Flask."""
    with app.app_context():
        yield Client()


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


def test_align_type_invariants_portal_et_transport():
    portal = Client()
    portal.company_id = None
    portal.client_type = ClientType.TRANSPORT
    portal.management_mode = ManagementMode.MANAGED
    Client._align_type_invariants(None, None, portal)
    assert portal.client_type == ClientType.PORTAL
    assert portal.management_mode is None

    transport = Client()
    transport.company_id = 12
    transport.client_type = ClientType.PORTAL
    transport.management_mode = None
    Client._align_type_invariants(None, None, transport)
    assert transport.client_type == ClientType.TRANSPORT
    assert transport.management_mode == ManagementMode.MANAGED

    deja = Client()
    deja.company_id = 3
    deja.management_mode = ManagementMode.SELF_SERVICE
    Client._align_type_invariants(None, None, deja)
    assert deja.management_mode == ManagementMode.SELF_SERVICE


def test_validate_contact_email(client_row):
    client_row.management_mode = ManagementMode.MANAGED
    assert client_row.validate_contact_email("contact_email", None) is None
    assert client_row.validate_contact_email("contact_email", "") == ""
    assert (
        client_row.validate_contact_email("contact_email", "  ada@test.ch ")
        == "ada@test.ch"
    )
    with pytest.raises(ValueError, match="Email invalide"):
        client_row.validate_contact_email("contact_email", "pas-un-email")

    client_row.management_mode = ManagementMode.SELF_SERVICE
    with pytest.raises(ValueError, match="self-service"):
        client_row.validate_contact_email("contact_email", "")
    with pytest.raises(ValueError, match="self-service"):
        client_row.validate_contact_email("contact_email", None)
    assert (
        client_row.validate_contact_email("contact_email", "self@test.ch")
        == "self@test.ch"
    )


def test_validate_billing_address(client_row):
    client_row.company_id = None
    assert client_row.validate_billing_address("billing_address", None) is None
    assert client_row.validate_billing_address("billing_address", "Rue 1") == "Rue 1"

    client_row.company_id = 8
    client_row.domicile_address = "Domicile 12"
    assert (
        client_row.validate_billing_address("billing_address", None) == "Domicile 12"
    )
    assert client_row.validate_billing_address("billing_address", "  ") == "Domicile 12"
    assert (
        client_row.validate_billing_address("billing_address", "Facturation 1")
        == "Facturation 1"
    )

    client_row.domicile_address = None
    with pytest.raises(ValueError, match="facturation"):
        client_row.validate_billing_address("billing_address", "")
    client_row.domicile_address = "   "
    with pytest.raises(ValueError, match="facturation"):
        client_row.validate_billing_address("billing_address", None)


def test_validate_phone_numbers(client_row):
    assert client_row.validate_phone_numbers("contact_phone", None) is None
    assert client_row.validate_phone_numbers("gp_phone", "") == ""
    assert (
        client_row.validate_phone_numbers("contact_phone", "  +41791234567 ")
        == "+41791234567"
    )
    with pytest.raises(ValueError, match="contact_phone"):
        client_row.validate_phone_numbers("contact_phone", "123")
    with pytest.raises(ValueError, match="gp_phone"):
        client_row.validate_phone_numbers("gp_phone", "abc")


def test_validate_default_billed_to_type(client_row):
    client_row.default_billed_to_company_id = 44
    assert client_row.validate_default_billed_to_type("x", None) == "patient"
    assert client_row.default_billed_to_company_id is None
    assert client_row.validate_default_billed_to_type("x", " CLINIC ") == "clinic"
    assert client_row.validate_default_billed_to_type("x", "Insurance") == "insurance"
    with pytest.raises(ValueError, match="invalide"):
        client_row.validate_default_billed_to_type("x", "banque")


@pytest.mark.parametrize(
    ("plain_attr", "enc_attr", "prop", "fallback"),
    [
        ("contact_phone", "contact_phone_encrypted", "contact_phone_secure", "+41790001111"),
        ("gp_name", "gp_name_encrypted", "gp_name_secure", "Dr House"),
        ("gp_phone", "gp_phone_encrypted", "gp_phone_secure", "+41790002222"),
        (
            "billing_address",
            "billing_address_encrypted",
            "billing_address_secure",
            "Rue Facture 1",
        ),
    ],
)
def test_champs_secure_dechiffre_et_fallback(
    client_row, monkeypatch, plain_attr, enc_attr, prop, fallback
):
    _fake_crypto(monkeypatch)
    setattr(client_row, plain_attr, None)
    client_row.encryption_migrated = True
    setattr(client_row, enc_attr, "enc:secret")
    assert getattr(client_row, prop) == "secret"

    client_row.encryption_migrated = False
    setattr(client_row, plain_attr, fallback)
    assert getattr(client_row, prop) == fallback


@pytest.mark.parametrize(
    ("enc_attr", "prop"),
    [
        ("contact_phone_encrypted", "contact_phone_secure"),
        ("gp_name_encrypted", "gp_name_secure"),
        ("gp_phone_encrypted", "gp_phone_secure"),
        ("billing_address_encrypted", "billing_address_secure"),
    ],
)
def test_champs_secure_decrypt_en_echec(client_row, monkeypatch, enc_attr, prop):
    _fake_crypto(monkeypatch, decrypt_error=True)
    client_row.encryption_migrated = True
    setattr(client_row, enc_attr, "enc:x")
    assert getattr(client_row, prop) is None


@pytest.mark.parametrize(
    "prop",
    [
        "contact_phone_secure",
        "gp_name_secure",
        "gp_phone_secure",
        "billing_address_secure",
    ],
)
def test_champs_secure_import_error_getter(client_row, monkeypatch, prop):
    _block_crypto_import(monkeypatch)
    client_row.contact_phone = "+41790000000"
    client_row.gp_name = "Dr Ada"
    client_row.gp_phone = "+41791111111"
    client_row.billing_address = "Rue 1"
    value = getattr(client_row, prop)
    assert value is not None


@pytest.mark.parametrize(
    ("enc_attr", "prop"),
    [
        ("contact_phone_encrypted", "contact_phone_secure"),
        ("gp_name_encrypted", "gp_name_secure"),
        ("gp_phone_encrypted", "gp_phone_secure"),
        ("billing_address_encrypted", "billing_address_secure"),
    ],
)
def test_champs_secure_setter_chiffre_et_vide(client_row, monkeypatch, enc_attr, prop):
    _fake_crypto(monkeypatch)
    setattr(client_row, prop, "valeur")
    assert getattr(client_row, enc_attr) == "enc:valeur"
    assert client_row.encryption_migrated is True
    setattr(client_row, prop, None)
    assert getattr(client_row, enc_attr) is None
    setattr(client_row, prop, "")
    assert getattr(client_row, enc_attr) is None


def test_champs_secure_setter_import_error(client_row, monkeypatch):
    _block_crypto_import(monkeypatch)
    client_row.contact_phone_secure = "+41793333333"
    assert client_row.contact_phone == "+41793333333"
    client_row.gp_name_secure = "Dr Fallback"
    assert client_row.gp_name == "Dr Fallback"
    client_row.gp_phone_secure = "+41794444444"
    assert client_row.gp_phone == "+41794444444"
    client_row.billing_address_secure = "Chemin 2"
    assert client_row.billing_address == "Chemin 2"


def test_contact_et_gp_phone_secure_effacent_ancienne_colonne(client_row, monkeypatch):
    _fake_crypto(monkeypatch)
    client_row.contact_phone = "+41791111111"
    client_row.contact_phone_secure = "+41792222222"
    assert client_row.contact_phone is None
    client_row.gp_phone = "+41793333333"
    client_row.gp_phone_secure = "+41794444444"
    assert client_row.gp_phone is None


def test_serialize_linked_institution_et_coords(client_row):
    assert client_row._serialize_linked_institution() is None
    set_committed_value(
        client_row,
        "linked_institution",
        SimpleNamespace(
            id=9,
            name="EMS Test",
            institution_type="ems",
            address="Route 1",
            contact_email="ems@test.ch",
            contact_phone="+41790000000",
        ),
    )
    linked = client_row._serialize_linked_institution()
    assert linked is not None
    assert linked["id"] == 9
    assert linked["name"] == "EMS Test"

    billed = SimpleNamespace(serialize={"id": 3, "name": "Cie"})
    user = SimpleNamespace(
        first_name="Ada",
        last_name="Lovelace",
        username="ada",
        phone="+41790000001",
        serialize={"id": 1, "username": "ada"},
    )
    client_row.id = 15
    set_committed_value(client_row, "user", user)
    set_committed_value(client_row, "default_billed_to_company", billed)
    client_row.client_type = ClientType.PORTAL
    client_row.company_id = None
    client_row.billing_address = "Rue F"
    client_row.billing_lat = Decimal("46.2044")
    client_row.billing_lon = Decimal("6.1432")
    client_row.contact_email = "ada@test.ch"
    client_row.contact_phone = None
    client_row.domicile_address = "Domicile"
    client_row.domicile_zip = "1200"
    client_row.domicile_city = "Genève"
    client_row.domicile_lat = Decimal("46.2")
    client_row.domicile_lon = Decimal("6.14")
    client_row.door_code = "12A"
    client_row.floor = "3"
    client_row.access_notes = "Ascenseur"
    client_row.gp_name = "Dr House"
    client_row.gp_phone = "+41791111111"
    client_row.default_billed_to_type = "clinic"
    client_row.default_billed_to_company = billed
    client_row.default_billed_to_contact = "Compta"
    client_row.is_institution = True
    client_row.institution_name = "Clinique"
    client_row.linked_institution_id = 9
    client_row.residence_facility = "EMS"
    client_row.preferential_rate = Decimal("12.50")
    client_row.is_active = True
    client_row.created_at = None

    data = client_row.serialize
    assert data["full_name"] == "Ada Lovelace"
    assert data["phone"] == "+41790000001"
    assert data["billing_lat"] == pytest.approx(46.2044)
    assert data["domicile"]["lat"] == pytest.approx(46.2)
    assert data["default_billing"]["billed_to_company"]["id"] == 3
    assert data["preferential_rate"] == pytest.approx(12.5)
    assert data["linked_institution"]["name"] == "EMS Test"

    set_committed_value(client_row, "user", None)
    client_row.billing_lat = None
    client_row.billing_lon = None
    client_row.domicile_lat = None
    client_row.domicile_lon = None
    client_row.preferential_rate = None
    client_row.default_billed_to_type = None
    set_committed_value(client_row, "default_billed_to_company", None)
    empty = client_row.serialize
    assert empty["user"] is None
    assert empty["full_name"] == "Nom non renseigné"
    assert empty["billing_lat"] is None
    assert empty["preferential_rate"] is None
    assert empty["default_billing"]["billed_to_type"] == "patient"


def test_toggle_active_is_self_service_repr(client_row):
    client_row.id = 4
    client_row.user_id = 8
    client_row.client_type = ClientType.PORTAL
    client_row.is_active = True
    client_row.management_mode = ManagementMode.SELF_SERVICE
    assert client_row.is_self_service() is True
    assert client_row.toggle_active() is False
    assert client_row.is_active is False
    client_row.management_mode = ManagementMode.MANAGED
    assert client_row.is_self_service() is False
    text = repr(client_row)
    assert "Client id=4" in text
    assert "user_id=8" in text
