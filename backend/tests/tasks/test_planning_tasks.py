"""Tests tâches planning (compliance_scan_all)."""

from __future__ import annotations

from unittest.mock import MagicMock

from tasks.planning_tasks import (
    _run_compliance_scan_for_company,
    compliance_scan,
    compliance_scan_all,
)


def test_run_compliance_scan_for_company_returns_int(db):
    """Helper interne retourne un int sans lever."""
    n = _run_compliance_scan_for_company(999999)
    assert isinstance(n, int)
    assert n >= 0


def test_compliance_scan_delegates(db, app):
    """compliance_scan(company_id) utilise le même helper."""
    with app.app_context():
        n = compliance_scan.run(999999)
    assert n == _run_compliance_scan_for_company(999999)


def test_compliance_scan_all_returns_dict(app, monkeypatch):
    """compliance_scan_all agrège toutes les companies (sous app context)."""
    chain = MagicMock()
    chain.all.return_value = []
    monkeypatch.setattr(
        "tasks.planning_tasks.db.session.query",
        lambda *a, **k: chain,
    )
    with app.app_context():
        out = compliance_scan_all.run()
    assert isinstance(out, dict)
    assert out == {"companies_processed": 0, "total_shifts_touched": 0}
