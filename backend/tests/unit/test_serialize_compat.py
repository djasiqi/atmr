"""Tests compatibilité sérialisation Booking (property + appel legacy)."""

from __future__ import annotations

import json

import pytest

from shared.serialize_compat import SerializeResult, as_serialize_result


def test_serialize_result_dict_access():
    payload = as_serialize_result({"id": 42, "status": "assigned"})
    assert payload["id"] == 42
    assert payload.get("status") == "assigned"


def test_serialize_result_callable_compat():
    payload = as_serialize_result({"id": 7})
    same = payload()
    assert same is payload
    assert same["id"] == 7


def test_serialize_result_json_serializable():
    payload = as_serialize_result({"id": 1, "nested": {"a": 1}})
    encoded = json.dumps(payload)
    assert '"id": 1' in encoded


def test_serialize_result_rejects_arguments():
    payload = SerializeResult({"id": 1})
    with pytest.raises(TypeError, match="argument"):
        payload("unexpected")
