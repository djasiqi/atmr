"""Tests filtrage entreprises démo ↔ institution réelle."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.demo.soft_delete_guard import filter_companies_for_institution


def _company(*, email: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(contact_email=email, name=name, user=None)


def _institution(*, email: str) -> SimpleNamespace:
    return SimpleNamespace(contact_email=email, name="Institution test", users=[])


@pytest.mark.unit
def test_real_institution_excludes_demo_companies():
    real = _institution(email="admin@lha.ch")
    demo_co = _company(email="demo-transport@demo.lirie.ch", name="LIRIE Demo")
    real_co = _company(email="contact@emmenez-moi.ch", name="Emmenez Moi")

    filtered = filter_companies_for_institution([demo_co, real_co], real)

    assert [c.name for c in filtered] == ["Emmenez Moi"]


@pytest.mark.unit
def test_demo_institution_keeps_only_demo_companies():
    demo_inst = _institution(email="demo-inst@demo.lirie.ch")
    demo_co = _company(email="demo-transport@demo.lirie.ch", name="LIRIE Demo")
    real_co = _company(email="contact@emmenez-moi.ch", name="Emmenez Moi")

    filtered = filter_companies_for_institution([demo_co, real_co], demo_inst)

    assert len(filtered) == 1
    assert filtered[0].name == "LIRIE Demo"
