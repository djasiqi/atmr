"""Tests unitaires pour recipient_utils."""

import pytest

from services.email.recipient_utils import normalize_relationship_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pere", "père"),
        ("PERE", "père"),
        ("père", "père"),
        ("mere", "mère"),
        ("MERE", "mère"),
        ("mère", "mère"),
        ("epoux", "époux"),
        ("EPOUX", "époux"),
        ("époux", "époux"),
        ("epouse", "épouse"),
        ("EPOUSE", "épouse"),
        ("épouse", "épouse"),
    ],
)
def test_normalize_relationship_label_mappings(raw, expected):
    assert normalize_relationship_label(raw) == expected


@pytest.mark.parametrize("raw", [None, ""])
def test_normalize_relationship_label_null(raw):
    assert normalize_relationship_label(raw) is None
