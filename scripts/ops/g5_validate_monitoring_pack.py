#!/usr/bin/env python3
"""G5 — valide le pack monitoring (structure) sans mutation prod / sans toucher release."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

EXPECTED_RELEASE_TIP = "286737a2362eb1e38013c72d04be23fcd608210e"
REQUIRED_TOP = {
    "kind",
    "release_tip",
    "rollback_decision_owner",
    "baseline",
    "queries",
    "checkpoints",
    "thresholds",
}
REQUIRED_CHECKPOINTS = {"T+5m", "T+30m", "T+2h"}
REQUIRED_THRESHOLD_KEYS = {"immediate_rollback", "investigate_first"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = ap.parse_args()
    root = args.repo_root.resolve()
    path = root / "docs" / "ops" / "g5-monitoring-checklist.json"
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    check("G5.0 checklist exists", path.is_file(), str(path))
    if not path.is_file():
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    check("G5.0 kind", data.get("kind") == "g5-monitoring-pack", str(data.get("kind")))
    missing = REQUIRED_TOP - set(data.keys())
    check("G5.0 top keys", not missing, str(sorted(missing) or "ok"))
    check(
        "G5.0 release_tip field",
        data.get("release_tip") == EXPECTED_RELEASE_TIP,
        str(data.get("release_tip")),
    )
    cps = set((data.get("checkpoints") or {}).keys())
    check(
        "G5.2 checkpoints",
        REQUIRED_CHECKPOINTS.issubset(cps),
        str(sorted(cps)),
    )
    th = data.get("thresholds") or {}
    check(
        "G5.3 threshold keys",
        REQUIRED_THRESHOLD_KEYS.issubset(set(th.keys())),
        str(sorted(th.keys())),
    )
    check(
        "G5.3 immediate non-empty",
        len(th.get("immediate_rollback") or []) >= 8,
        str(len(th.get("immediate_rollback") or [])),
    )
    check(
        "G5.3 investigate separates expected 422",
        any("422" in str(x) for x in (th.get("investigate_first") or [])),
        "422 expected listed",
    )
    owner = data.get("rollback_decision_owner") or {}
    check(
        "G5.0 decision owner",
        bool(owner.get("role")) and bool(owner.get("authority")),
        str(owner.get("role")),
    )
    q = data.get("queries") or {}
    check(
        "G5.1 prometheus queries",
        "prometheus" in q and "loc_received_rate_5m" in (q.get("prometheus") or {}),
        "prometheus block",
    )
    check(
        "G5.4 anti-skew queries",
        "anti_skew" in q
        and "fanout_hold_assert" in (q.get("anti_skew") or {}),
        "anti_skew block",
    )
    bas = data.get("baseline") or {}
    check(
        "G5.1 baseline partial source",
        bool((bas.get("pre_release_partial") or {}).get("source")),
        str((bas.get("pre_release_partial") or {}).get("source")),
    )
    prev = root / "docs" / "ops" / "previous-release.json"
    check("G5.4 previous-release present", prev.is_file(), str(prev))
    report = root / "docs" / "ops" / "gps-p0-g5-monitoring-2026-08-16.md"
    check("G5.0 report md", report.is_file(), str(report))

    try:
        tip = subprocess.run(
            ["git", "rev-parse", "release/gps-p0-2026-08-15"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        check("G5 freeze release tip", tip == EXPECTED_RELEASE_TIP, tip)
    except Exception as e:  # noqa: BLE001
        check("G5 freeze release tip", False, str(e))

    failed = [r for r in results if not r[1]]
    print()
    print(f"TOTAL={len(results)} PASS={len(results)-len(failed)} FAIL={len(failed)}")
    if failed:
        return 1
    print("G5_VALIDATOR=VERT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
