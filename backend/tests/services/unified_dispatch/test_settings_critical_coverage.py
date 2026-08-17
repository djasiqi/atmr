"""Couverture critique ``services/unified_dispatch/core/settings.py`` (seuil 95 %)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from services.unified_dispatch.core import settings as ud
from services.unified_dispatch.core.settings import (
    HeuristicWeights,
    MultiObjectiveSettings,
    Settings,
)


def test_heuristic_normalized_et_zero():
    w = HeuristicWeights(
        proximity=1,
        driver_load_balance=1,
        priority=1,
        return_urgency=1,
        regular_driver_bonus=1,
    )
    n = w.normalized()
    assert (
        pytest.approx(
            sum(
                [
                    n.proximity,
                    n.driver_load_balance,
                    n.priority,
                    n.return_urgency,
                    n.regular_driver_bonus,
                ]
            )
        )
        == 1.0
    )
    zero = HeuristicWeights(0, 0, 0, 0, 0)
    assert zero.normalized() is zero


def test_efficiency_weight_et_to_dict():
    mo = MultiObjectiveSettings(fairness_weight=0.3)
    assert mo.efficiency_weight == pytest.approx(0.7)
    data = Settings().to_dict()
    assert "heuristic" in data
    assert "solver" in data
    heur = data["heuristic"]
    total = sum(heur.values())
    assert total == pytest.approx(1.0)


def test_get_env_or_default(monkeypatch):
    monkeypatch.delenv("UD_TEST_MISSING", raising=False)
    assert ud._get_env_or_default("UD_TEST_MISSING", 9) == 9
    monkeypatch.setenv("UD_TEST_INT", "42")
    assert ud._get_env_or_default("UD_TEST_INT", 0) == 42
    monkeypatch.setenv("UD_TEST_BOOL", "True")
    assert ud._get_env_or_default("UD_TEST_BOOL", False) is True
    monkeypatch.setenv("UD_TEST_STR", "osrm-host")
    assert ud._get_env_or_default("UD_TEST_STR", "x") == "osrm-host"


def test_merge_dicts_recursif_et_remplacement():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = ud._merge_dicts(base, {"a": {"c": 9}, "e": 4})
    assert merged["a"]["b"] == 1
    assert merged["a"]["c"] == 9
    assert merged["d"] == 3
    assert merged["e"] == 4
    assert ud._merge_dicts({"a": 1}, {"a": 2})["a"] == 2


def test_validate_merge_valeur_differente_et_section_absente():
    s = Settings()
    s.heuristic.driver_load_balance = 0.1
    result = ud._validate_merge_result(
        s,
        {"heuristic": {"driver_load_balance": 0.9}},
        [("heuristic.driver_load_balance", 0.7, 0.9)],
    )
    assert "heuristic.driver_load_balance" in result["critical_errors"]
    assert result["errors"]

    class _SansHeuristic:
        pass

    applied = ud._validate_merge_result(
        _SansHeuristic(),
        {"heuristic": {"proximity": 0.2}},
        [("heuristic.proximity", 0.1, 0.2)],
    )
    assert "heuristic.proximity" in applied["applied"]

    extra = ud._validate_merge_result(
        Settings(),
        {"other": True},
        [("heuristic.priority", 0.06, 0.1)],
    )
    assert "heuristic.priority" in extra["applied"]


def test_validate_critique_non_applique_et_cles_ignorees():
    result = ud._validate_merge_result(
        Settings(),
        {
            "fairness": {"fairness_weight": 0.8, "enabled": True},
            "mode": "heuristic_only",
            "run_async": True,
            "preferred_driver_id": 12,
            "reset_existing": False,
            "fast_mode": True,
            "inconnu": {"nested": 1},
        },
        [],
    )
    assert "fairness.fairness_weight" in result["critical_errors"]
    assert "mode" in result["ignored"]
    assert "inconnu.nested" in result["ignored"]


def test_merge_emergency_mapping_et_return_validation():
    base = Settings()
    merged, validation = ud.merge_overrides(
        base,
        {"emergency": {"emergency_per_stop_penalty": 777.0}},
        return_validation=True,
    )
    assert merged.emergency.emergency_penalty == 777.0
    assert isinstance(validation, dict)


def test_merge_setattr_sur_dataclass_figee():
    @dataclass(frozen=True)
    class _FrozenSolver:
        time_limit_sec: int = 60

    locked = Settings()
    locked.solver = _FrozenSolver()  # type: ignore[assignment]
    still = ud.merge_overrides(locked, {"solver": {"time_limit_sec": 120}})
    assert still.solver.time_limit_sec == 60


def test_merge_erreurs_strict_et_non_strict(monkeypatch):
    def fake_validate(*_a, **_k):
        return {
            "applied": [],
            "ignored": ["mode"],
            "errors": ["Paramètre critique non appliqué: fairness.fairness_weight"],
            "critical_errors": ["fairness.fairness_weight"],
        }

    monkeypatch.setattr(ud, "_validate_merge_result", fake_validate)
    monkeypatch.setenv("UD_SETTINGS_STRICT_VALIDATION", "false")
    ok = ud.merge_overrides(Settings(), {"heuristic": {"proximity": 0.11}})
    assert ok.heuristic.proximity == 0.11

    monkeypatch.setenv("UD_SETTINGS_STRICT_VALIDATION", "true")
    with pytest.raises(ValueError, match="critiques"):
        ud.merge_overrides(Settings(), {"heuristic": {"proximity": 0.12}})


def test_from_dict_et_from_json():
    s = ud.from_dict({"default_timezone": "UTC"})
    assert s.default_timezone == "UTC"
    s2 = ud.from_json(json.dumps({"default_timezone": "Europe/Paris"}))
    assert s2.default_timezone == "Europe/Paris"


def test_for_company_overrides_env_et_erreurs(monkeypatch):
    monkeypatch.setenv("UD_OSRM_URL", "http://osrm-test:5000")
    monkeypatch.setenv("UD_MATRIX_CACHE_TTL_SEC", "99")
    monkeypatch.setenv("UD_SOLVER_TIME_LIMIT_SEC", "12")
    monkeypatch.setenv("DISPATCH_AUTORUN_INTERVAL_SEC", "45")
    monkeypatch.setenv("DISPATCH_AUTORUN_ENABLED", "False")
    monkeypatch.setenv("UD_SAFETY_MIN_GAP_MINUTES", "7")

    plain = SimpleNamespace()
    s = ud.for_company(plain)
    assert s.matrix.osrm_url == "http://osrm-test:5000"
    assert s.matrix.cache_ttl_sec == 99
    assert s.solver.time_limit_sec == 12
    assert s.autorun.autorun_interval_sec == 45
    assert s.autorun.autorun_enabled is False
    assert s.safety.min_gap_minutes == 7

    def boom_cfg():
        raise json.JSONDecodeError("bad", "doc", 0)

    auto = SimpleNamespace(
        get_autonomous_config=boom_cfg,
        dispatch_settings='{"not json',
    )
    s2 = ud.for_company(auto)
    assert isinstance(s2, Settings)

    good = SimpleNamespace(
        get_autonomous_config=lambda: {
            "dispatch_overrides": {"solver": {"time_limit_sec": 88}}
        },
        dispatch_settings=json.dumps({"heuristic": {"proximity": 0.15}}),
    )
    s3 = ud.for_company(good)
    assert s3.solver.time_limit_sec in {12, 88}
    assert s3.heuristic.proximity == 0.15

    empty_auto = SimpleNamespace(
        get_autonomous_config=lambda: {},
        dispatch_settings=None,
    )
    assert isinstance(ud.for_company(empty_auto), Settings)


def test_driver_work_window_from_config():
    start, end = ud.driver_work_window_from_config(None)
    assert start < end
