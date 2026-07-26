"""Tests de non-régression : le mode MANUAL ne doit jamais déclencher de
dispatch AUTOMATIQUE, mais le bouton « Lancer le dispatch » (manuel explicite)
doit continuer à fonctionner.

Couvre :
- AutonomousDispatchManager.is_automation_allowed()
- Garde-fou principal dans queue.trigger() / trigger_on_booking_change()
- Défense en profondeur dans tasks.dispatch_tasks.run_dispatch_task()
- Invariant MANUAL ⇒ dispatch_enabled=false (use-case + modèle)
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from models import DispatchMode, DispatchTriggerOrigin
from services.unified_dispatch.core import queue as ud_queue


# ---------------------------------------------------------------------------
# is_automation_allowed()
# ---------------------------------------------------------------------------
class TestIsAutomationAllowed:
    def test_manual_never_allows_automation(self, db):
        from services.unified_dispatch.utils.autonomous import (
            get_manager_for_company,
        )
        from tests.factories import CompanyFactory

        company = CompanyFactory(
            dispatch_mode=DispatchMode.MANUAL, dispatch_enabled=False
        )
        manager = get_manager_for_company(company.id)
        assert manager.is_automation_allowed() is False

    def test_fully_auto_allows_automation(self, db):
        from services.unified_dispatch.utils.autonomous import (
            get_manager_for_company,
        )
        from tests.factories import CompanyFactory

        company = CompanyFactory(
            dispatch_mode=DispatchMode.FULLY_AUTO, dispatch_enabled=True
        )
        manager = get_manager_for_company(company.id)
        assert manager.is_automation_allowed() is True

    def test_semi_auto_follows_config(self, db):
        from services.unified_dispatch.utils.autonomous import (
            get_manager_for_company,
        )
        from tests.factories import CompanyFactory

        company = CompanyFactory(
            dispatch_mode=DispatchMode.SEMI_AUTO, dispatch_enabled=True
        )
        manager = get_manager_for_company(company.id)
        # Config par défaut : auto_dispatch.enabled = False
        assert manager.is_automation_allowed() is False


# ---------------------------------------------------------------------------
# Garde-fou dans queue.trigger()
# ---------------------------------------------------------------------------
def _patch_manager(monkeypatch, *, allowed: bool, mode: str = "manual") -> MagicMock:
    manager = MagicMock()
    manager.is_automation_allowed.return_value = allowed
    manager.mode.value = mode
    monkeypatch.setattr(
        "services.unified_dispatch.utils.autonomous.get_manager_for_company",
        lambda _company_id: manager,
    )
    return manager


class TestQueueTriggerGuard:
    def test_automated_origin_blocked_in_manual(self, monkeypatch):
        company_id = 990001
        ud_queue._STATE.pop(company_id, None)
        manager = _patch_manager(monkeypatch, allowed=False, mode="manual")
        scheduled: list[str] = []
        monkeypatch.setattr(
            ud_queue, "_schedule_run", lambda st, mode: scheduled.append(mode)
        )

        ud_queue.trigger(
            company_id,
            reason="booking_update",
            origin=DispatchTriggerOrigin.BOOKING_CHANGE,
        )

        assert scheduled == [], "Un dispatch auto ne doit PAS être planifié en MANUAL"
        manager.is_automation_allowed.assert_called_once()

    def test_manual_origin_bypasses_guard(self, monkeypatch):
        company_id = 990002
        ud_queue._STATE.pop(company_id, None)
        manager = _patch_manager(monkeypatch, allowed=False, mode="manual")
        scheduled: list[str] = []
        monkeypatch.setattr(
            ud_queue, "_schedule_run", lambda st, mode: scheduled.append(mode)
        )

        # Bouton « Lancer le dispatch » : origin par défaut MANUAL
        ud_queue.trigger(company_id, reason="manual_trigger", mode="auto")

        assert scheduled == ["auto"], "Le dispatch manuel doit toujours fonctionner"
        manager.is_automation_allowed.assert_not_called()

    def test_automated_origin_allowed_in_fully_auto(self, monkeypatch):
        company_id = 990003
        ud_queue._STATE.pop(company_id, None)
        _patch_manager(monkeypatch, allowed=True, mode="fully_auto")
        scheduled: list[str] = []
        monkeypatch.setattr(
            ud_queue, "_schedule_run", lambda st, mode: scheduled.append(mode)
        )

        ud_queue.trigger_on_booking_change(
            company_id, reason="booking_update", mode="auto"
        )

        assert scheduled == ["auto"]

    def test_origin_propagated_to_params(self, monkeypatch):
        company_id = 990004
        ud_queue._STATE.pop(company_id, None)
        _patch_manager(monkeypatch, allowed=True, mode="fully_auto")
        monkeypatch.setattr(ud_queue, "_schedule_run", lambda st, mode: None)

        ud_queue.trigger_on_booking_change(
            company_id,
            reason="booking_update",
            origin=DispatchTriggerOrigin.CANCELLATION,
        )

        st = ud_queue._get_state(company_id)
        assert st.params.get("origin") == DispatchTriggerOrigin.CANCELLATION.value
        assert "origin" in ud_queue.ALLOWED_RUN_KWARGS


# ---------------------------------------------------------------------------
# Défense en profondeur dans run_dispatch_task()
# ---------------------------------------------------------------------------
def _fake_flask_app() -> MagicMock:
    app = MagicMock()
    app.app_context.return_value.__enter__ = MagicMock(return_value=None)
    app.app_context.return_value.__exit__ = MagicMock(return_value=False)
    return app


class TestRunDispatchTaskGuard:
    def test_automated_run_skipped_in_manual(self, monkeypatch):
        from tasks import dispatch_tasks

        monkeypatch.setattr(dispatch_tasks, "get_flask_app", _fake_flask_app)
        _patch_manager(monkeypatch, allowed=False, mode="manual")
        engine_run = MagicMock()
        monkeypatch.setattr(
            "services.unified_dispatch.engine.run", engine_run, raising=False
        )

        result = dispatch_tasks.run_dispatch_task(
            company_id=1,
            for_date="2026-06-03",
            origin=DispatchTriggerOrigin.BOOKING_CHANGE.value,
        )

        assert result["meta"]["reason"] == "automation_not_allowed"
        assert result["meta"]["skipped"] is True
        engine_run.assert_not_called()

    def test_manual_run_not_skipped(self, monkeypatch):
        from tasks import dispatch_tasks

        monkeypatch.setattr(dispatch_tasks, "get_flask_app", _fake_flask_app)
        manager = _patch_manager(monkeypatch, allowed=False, mode="manual")
        engine_run = MagicMock(
            return_value={
                "assignments": [],
                "unassigned": [],
                "bookings": [],
                "drivers": [],
                "meta": {},
            }
        )
        monkeypatch.setattr(
            "services.unified_dispatch.engine.run", engine_run, raising=False
        )

        result = dispatch_tasks.run_dispatch_task(
            company_id=1,
            for_date="2026-06-03",
            origin=DispatchTriggerOrigin.MANUAL.value,
        )

        assert result.get("meta", {}).get("reason") != "automation_not_allowed"
        engine_run.assert_called_once()
        # Origine manuelle : on ne consulte même pas le gestionnaire d'automatisation
        manager.is_automation_allowed.assert_not_called()


# ---------------------------------------------------------------------------
# Invariant MANUAL ⇒ dispatch_enabled=false
# ---------------------------------------------------------------------------
class TestSetDispatchEnabledInvariant:
    def test_cannot_enable_in_manual_mode(self):
        from application.companies.set_dispatch_enabled import (
            SetDispatchEnabledUseCase,
        )

        company = MagicMock()
        company.id = 1
        company.dispatch_enabled = False
        company.dispatch_mode = DispatchMode.MANUAL

        result = SetDispatchEnabledUseCase().execute(
            company, enabled=True, reason="activate_dispatch"
        )

        assert result.ok is False
        assert result.status_code == 409
        # L'activation ne doit pas avoir été appliquée
        assert company.dispatch_enabled is False

    def test_can_disable_in_manual_mode(self):
        from application.companies.set_dispatch_enabled import (
            SetDispatchEnabledUseCase,
        )

        company = MagicMock()
        company.id = 1
        company.dispatch_enabled = True
        company.dispatch_mode = DispatchMode.MANUAL

        result = SetDispatchEnabledUseCase().execute(
            company, enabled=False, reason="deactivate_dispatch"
        )

        assert result.ok is True
        assert company.dispatch_enabled is False

    def test_can_enable_in_semi_auto(self):
        from application.companies.set_dispatch_enabled import (
            SetDispatchEnabledUseCase,
        )

        company = MagicMock()
        company.id = 1
        company.dispatch_enabled = False
        company.dispatch_mode = DispatchMode.SEMI_AUTO

        result = SetDispatchEnabledUseCase().execute(
            company, enabled=True, reason="activate_dispatch"
        )

        assert result.ok is True
        assert company.dispatch_enabled is True


class TestCompanySetDispatchModeInvariant:
    def test_switch_to_manual_disables_dispatch(self, db):
        from tests.factories import CompanyFactory

        company = CompanyFactory(
            dispatch_mode=DispatchMode.FULLY_AUTO, dispatch_enabled=True
        )

        company.set_dispatch_mode(DispatchMode.MANUAL)

        assert company.dispatch_mode == DispatchMode.MANUAL
        assert company.dispatch_enabled is False

    def test_switch_to_semi_auto_keeps_dispatch_enabled(self, db):
        from tests.factories import CompanyFactory

        company = CompanyFactory(
            dispatch_mode=DispatchMode.FULLY_AUTO, dispatch_enabled=True
        )

        company.set_dispatch_mode(DispatchMode.SEMI_AUTO)

        assert company.dispatch_mode == DispatchMode.SEMI_AUTO
        assert company.dispatch_enabled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
