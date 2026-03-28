"""Tests pour InstitutionMetricsService — filtres RequestOffer.sent_at."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import services.metrics.institution_metrics as im
from services.metrics.institution_metrics import InstitutionMetricsService


def test_institution_metrics_source_has_no_request_offer_created_at():
    """Régression statique : aucune référence à RequestOffer.created_at (colonne absente)."""
    src = Path(im.__file__).resolve()
    text = src.read_text(encoding="utf-8")
    assert "RequestOffer.created_at" not in text


def test_compute_metrics_no_attribute_error_on_request_offer_sent_at(db):
    """Régression: RequestOffer n'a pas created_at; les filtres doivent utiliser sent_at."""
    snap = InstitutionMetricsService.compute_metrics(period_hours=24)
    assert snap.period_start is not None
    assert snap.period_end is not None


def test_compute_metrics_with_institution_filter_runs(db):
    """Filtre institution_id optionnel ne doit pas lever (requêtes join cohérentes)."""
    snap = InstitutionMetricsService.compute_metrics(
        institution_id=999999, period_hours=24
    )
    assert snap.total_offers_created == 0
    assert snap.period_end is not None


def test_compute_metrics_no_global_except_with_mocked_orm(app, monkeypatch):
    """Sans erreur dans le try, logger.exception ne doit pas être appelé (ORM mocké)."""
    tr_q = MagicMock()
    tr_q.filter.return_value = tr_q
    tr_q.count.side_effect = [2, 1]
    tr_q.all.return_value = []

    oq = MagicMock()
    oq.join.return_value = oq
    oq.filter.return_value = oq
    oq.count.side_effect = [3, 0]

    eq = MagicMock()
    eq.join.return_value = eq
    eq.filter.return_value = eq
    eq.count.return_value = 1

    ro_base = MagicMock()
    ro_base.filter.side_effect = [oq, eq]

    fb = MagicMock()
    fb.filter.return_value = fb
    fb.scalar.return_value = 0

    seq = MagicMock()
    seq.filter.return_value = seq
    seq.distinct.return_value = seq
    seq.scalar_subquery.return_value = MagicMock()

    sess_n = [0]

    def session_query(*_a, **_k):
        sess_n[0] += 1
        return fb if sess_n[0] == 1 else seq

    mock_exc = MagicMock()
    with app.app_context():
        with (
            patch.object(im.TransportRequest, "query", tr_q),
            patch.object(im.RequestOffer, "query", ro_base),
            patch.object(im.db.session, "query", session_query),
            patch.object(im.logger, "exception", mock_exc),
        ):
            snap = InstitutionMetricsService.compute_metrics(period_hours=24)

    mock_exc.assert_not_called()
    assert snap.total_requests_sent == 2
    assert snap.total_offers_created == 3
    assert snap.total_escalations == 1
    assert snap.fallback_broadcast_count == 0


def test_compute_metrics_non_empty_snapshot_when_mocks_return_counts(app):
    """Si l'ORM renvoie des comptes > 0, le snapshot ne doit pas être « vide » à tort."""
    tr_q = MagicMock()
    tr_q.filter.return_value = tr_q
    tr_q.count.side_effect = [5, 2]
    tr_q.all.return_value = []

    oq = MagicMock()
    oq.join.return_value = oq
    oq.filter.return_value = oq
    oq.count.side_effect = [4, 1]

    eq = MagicMock()
    eq.join.return_value = eq
    eq.filter.return_value = eq
    eq.count.return_value = 2

    ro_base = MagicMock()
    ro_base.filter.side_effect = [oq, eq]

    fb = MagicMock()
    fb.filter.return_value = fb
    fb.scalar.return_value = 1

    seq = MagicMock()
    seq.filter.return_value = seq
    seq.distinct.return_value = seq
    seq.scalar_subquery.return_value = MagicMock()

    sess_n = [0]

    def session_query(*_a, **_k):
        sess_n[0] += 1
        return fb if sess_n[0] == 1 else seq

    with app.app_context():
        with (
            patch.object(im.TransportRequest, "query", tr_q),
            patch.object(im.RequestOffer, "query", ro_base),
            patch.object(im.db.session, "query", session_query),
        ):
            snap = InstitutionMetricsService.compute_metrics(period_hours=24)
    assert snap.total_requests_sent >= 1
    assert snap.total_offers_created >= 1
