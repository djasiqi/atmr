"""Couverture de ``RedispatchInstitutionBookingUseCase``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from application.institutions.redispatch_institution_booking import (
    RedispatchInstitutionBookingInput,
    RedispatchInstitutionBookingUseCase,
)
from models import OfferStatus, RequestStatus


def _patch_db(monkeypatch, transport_request):
    mock_db = MagicMock()
    mock_db.session.query.return_value.filter.return_value.first.return_value = (
        transport_request
    )
    monkeypatch.setattr(
        "application.institutions.redispatch_institution_booking.db",
        mock_db,
    )
    return mock_db


def test_execute_404_sans_demande(monkeypatch):
    _patch_db(monkeypatch, None)
    result = RedispatchInstitutionBookingUseCase().execute(
        RedispatchInstitutionBookingInput(booking_id=12)
    )
    assert result.success is False
    assert result.status_code == 404
    assert result.booking_id == 12
    assert "introuvable" in (result.error or "")


def test_execute_cloture_offres_et_cree(monkeypatch):
    tr = SimpleNamespace(id=88, status="ACCEPTED", accepted_by_company_id=5)
    stale = SimpleNamespace(status=OfferStatus.PENDING.value)
    _patch_db(monkeypatch, tr)

    offer_model = MagicMock()
    offer_model.query.filter.return_value.all.return_value = [stale]
    monkeypatch.setattr(
        "application.institutions.redispatch_institution_booking.RequestOffer",
        offer_model,
    )

    uc = RedispatchInstitutionBookingUseCase()
    monkeypatch.setattr(
        uc,
        "_create_offers",
        lambda _tr, exclude_company_id=None: 3,
    )
    result = uc.execute(
        RedispatchInstitutionBookingInput(
            booking_id=12,
            previous_company_id=9,
        )
    )
    assert result.success is True
    assert result.offers_created == 3
    assert result.transport_request_id == 88
    assert tr.status == RequestStatus.SENT.value
    assert tr.accepted_by_company_id is None
    assert stale.status == OfferStatus.UNAVAILABLE.value


def test_execute_zero_offre_log_warning(monkeypatch, caplog):
    tr = SimpleNamespace(id=1, status="X", accepted_by_company_id=1)
    mock_db = _patch_db(monkeypatch, tr)
    offer_model = MagicMock()
    offer_model.query.filter.return_value.all.return_value = []
    monkeypatch.setattr(
        "application.institutions.redispatch_institution_booking.RequestOffer",
        offer_model,
    )
    uc = RedispatchInstitutionBookingUseCase()
    monkeypatch.setattr(uc, "_create_offers", lambda *_a, **_k: 0)
    with caplog.at_level("WARNING"):
        result = uc.execute(RedispatchInstitutionBookingInput(booking_id=4))
    assert result.success is True
    assert result.offers_created == 0
    mock_db.session.flush.assert_called_once()
    assert "Aucune entreprise éligible" in caplog.text


def test_create_offers_exclusion(monkeypatch):
    captured: list[dict] = []

    class _FakeSend:
        def _create_broadcast_offers(self, **kwargs):
            captured.append(kwargs)
            return 2

    monkeypatch.setattr(
        "application.institutions.send_transport_request.SendTransportRequestUseCase",
        _FakeSend,
    )
    tr = SimpleNamespace(id=3)
    created = RedispatchInstitutionBookingUseCase._create_offers(
        tr, exclude_company_id=7
    )
    assert created == 2
    assert captured[0]["transport_request"] is tr
    assert captured[0]["expires_at"] is None
    assert captured[0]["excluded_company_ids"] == [7]

    captured.clear()
    RedispatchInstitutionBookingUseCase._create_offers(tr, exclude_company_id=None)
    assert captured[0]["excluded_company_ids"] == []
