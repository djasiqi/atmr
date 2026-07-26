"""Tests unitaires — TransportRequestDisplayModel v1 (STOP GATE TR-01)."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

from services.institutions.transport_request_display import (
    DISPLAY_MODEL_TRANSPORT_REQUEST,
    build_transport_request_display_blocks,
)


def _tr(**kwargs):
    defaults = {
        "id": 42,
        "mission_date": date(2026, 6, 12),
        "scheduled_time": None,
        "pickup_time_confirmed": False,
        "is_round_trip": False,
        "return_time": None,
        "return_date": None,
        "return_time_confirmed": False,
        "multi_stop": False,
        "return_to_institution": False,
        "dropoff_location": "Hôpital",
        "legs": [],
        "external_reference": None,
        "contact_on_site": None,
    }
    defaults.update(kwargs)
    tr = SimpleNamespace(**defaults)
    patient = kwargs.get("patient")
    if patient is not None:
        tr.patient = patient
    institution = kwargs.get("institution")
    if institution is not None:
        tr.institution = institution
        tr.institution_id = getattr(institution, "id", None)
    else:
        tr.institution = SimpleNamespace(id=251, name="Clinique LHA")
        tr.institution_id = 251
    return tr


def _leg(seq: int, st: datetime | None, confirmed: bool, **kwargs):
    return SimpleNamespace(
        sequence_index=seq,
        scheduled_time=st,
        time_confirmed=confirmed,
        dropoff_establishment=kwargs.get("dropoff_establishment"),
        dropoff_service=kwargs.get("dropoff_service"),
    )


def test_tr01a_departure_confirmed():
    tr = _tr(
        scheduled_time=datetime(2026, 6, 12, 13, 15),
        pickup_time_confirmed=True,
    )
    blocks = build_transport_request_display_blocks(tr)
    dep = blocks["scheduling"]["departure"]
    assert dep["time_defined"] is True
    assert dep["display_time"] == "13:15"


def test_tr01b_departure_indicative():
    tr = _tr(
        scheduled_time=datetime(2026, 6, 12, 13, 15),
        pickup_time_confirmed=False,
    )
    blocks = build_transport_request_display_blocks(tr)
    dep = blocks["scheduling"]["departure"]
    assert dep["time_defined"] is False
    assert "non confirmé" in dep["display_time"]


def test_tr01c_return_undefined_summary():
    tr = _tr(
        is_round_trip=True,
        return_time=None,
        return_date=date(2026, 6, 12),
        return_time_confirmed=False,
    )
    blocks = build_transport_request_display_blocks(tr)
    ret = blocks["scheduling"]["return"]
    assert ret["display_time"] == "À définir"
    assert "Retour à définir" in blocks["scheduling"]["summary"]


def test_tr01d_multi_stop_legs():
    tr = _tr(
        scheduled_time=datetime(2026, 6, 12, 13, 0),
        pickup_time_confirmed=True,
        multi_stop=True,
        legs=[
            _leg(
                0, datetime(2026, 6, 12, 14, 0), True, dropoff_establishment="Imagerie"
            ),
            _leg(1, datetime(2026, 6, 12, 16, 0), True, dropoff_establishment="Labo"),
        ],
    )
    blocks = build_transport_request_display_blocks(tr)
    assert blocks["display_model"] == DISPLAY_MODEL_TRANSPORT_REQUEST
    assert len(blocks["legs"]) == 2
    assert blocks["legs"][0]["display_time"] == "14:00"
    assert blocks["legs"][0]["label"] == "Imagerie"


def test_tr01e_identity_institution():
    tr = _tr(
        patient=SimpleNamespace(first_name="Jean", last_name="Dupont"),
    )
    blocks = build_transport_request_display_blocks(tr)
    identity = blocks["identity"]
    assert identity["primary_label"] == "Dupont Jean"
    assert identity["secondary_label"] == "Clinique LHA"
    assert identity["display_category"] == "institution_patient"
