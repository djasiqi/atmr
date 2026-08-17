#!/usr/bin/env python3
"""G4 — validation hors-prod du manifeste previous-release (skew) + contrats rollback.

N'exécute AUCUNE mutation prod. Ne touche PAS release/gps-p0-2026-08-15.

Usage (host) :
  python scripts/ops/g4_validate_previous_release.py
  python scripts/ops/g4_validate_previous_release.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPECTED_API_SHA = "927640a0995a7025edfae3d31802998948a866d5"
EXPECTED_CONSUMER_PREFIX = "390076efc61c"
EXPECTED_FANOUT_PREFIX = "16fd3e52418d"
EXPECTED_ALEMBIC = "9b6638784019"
EXPECTED_RELEASE_TIP = "286737a2362eb1e38013c72d04be23fcd608210e"
FORBIDDEN = {
    "purge_redis",
    "purge_kafka",
    "flush_queue",
    "alembic_downgrade",
    "mobile_rollback",
}


def run(cmd: list[str], cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr}")
    return p.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = ap.parse_args()
    root = args.repo_root.resolve()
    manifest_path = root / "docs" / "ops" / "previous-release.json"
    hold_path = root / "docker-compose.kafka.p0-hold.yml"
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")

    # --- G4.0 manifest exists + schema ---
    ok_exists = manifest_path.is_file()
    check("G4.0 previous-release.json", ok_exists, str(manifest_path))
    if not ok_exists:
        return 1
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    check(
        "G4.0 kind",
        data.get("kind") == "previous-release",
        f"kind={data.get('kind')}",
    )
    check(
        "G4.0 api sha",
        data.get("backend", {}).get("git_sha") == EXPECTED_API_SHA,
        data.get("backend", {}).get("git_sha", ""),
    )
    cons = data.get("tracking", {}).get("consumer", {}).get("git_sha", "")
    check(
        "G4.0 consumer sha",
        cons.startswith(EXPECTED_CONSUMER_PREFIX),
        cons,
    )
    fan = data.get("tracking", {}).get("fanout", {}).get("git_sha", "")
    check(
        "G4.0 fanout sha",
        fan.startswith(EXPECTED_FANOUT_PREFIX),
        fan,
    )
    hold = data.get("tracking", {}).get("hold", {})
    check(
        "G4.0 fanout HOLD flag",
        str(hold.get("TRACKING_PROCESSED_FANOUT_ENABLED")) == "false",
        str(hold.get("TRACKING_PROCESSED_FANOUT_ENABLED")),
    )
    check(
        "G4.0 alembic",
        data.get("alembic", {}).get("current") == EXPECTED_ALEMBIC,
        str(data.get("alembic", {}).get("current")),
    )
    forbidden = set(data.get("rollback_must_not_require") or [])
    check(
        "G4.0 no purge/alembic/mobile",
        FORBIDDEN.issubset(forbidden),
        str(sorted(forbidden)),
    )

    # --- G4.3 HOLD compose ---
    hold_txt = hold_path.read_text(encoding="utf-8") if hold_path.is_file() else ""
    check(
        "G4.3 p0-hold.yml ENABLED=false",
        'TRACKING_PROCESSED_FANOUT_ENABLED: "false"' in hold_txt,
        str(hold_path),
    )
    check(
        "G4.3 fanout desired_state",
        data.get("tracking", {}).get("fanout", {}).get("desired_state")
        == "created_not_up",
        str(data.get("tracking", {}).get("fanout", {}).get("desired_state")),
    )

    # --- G4.5 no migration in release tip vs prod API ---
    try:
        mig_diff = run(
            [
                "git",
                "diff",
                "--name-only",
                EXPECTED_API_SHA,
                EXPECTED_RELEASE_TIP,
                "--",
                "**/alembic/**",
                "**/versions/**",
            ],
            root,
        )
        check(
            "G4.5 aucune migration dans delta release",
            mig_diff == "",
            mig_diff or "(vide)",
        )
    except RuntimeError as e:
        check("G4.5 aucune migration dans delta release", False, str(e))

    # --- G4.5 alembic no downgrade ---
    check(
        "G4.5 downgrade_required=false",
        data.get("alembic", {}).get("downgrade_required_on_rollback") is False,
        str(data.get("alembic", {}).get("downgrade_required_on_rollback")),
    )

    # --- G4.4 from G3 ---
    g3 = data.get("g3", {})
    check(
        "G4.4 backend_only_rollback_safe",
        g3.get("backend_only_rollback_safe") is True,
        str(g3.get("backend_only_rollback_safe")),
    )
    check(
        "G4.4 mobile DEGRADED-SAFE",
        g3.get("mobile_p0_vs_old_backend") == "DEGRADED-SAFE",
        str(g3.get("mobile_p0_vs_old_backend")),
    )

    # --- release tip frozen ---
    try:
        tip = run(["git", "rev-parse", "release/gps-p0-2026-08-15"], root)
        check(
            "G4 freeze release tip",
            tip == EXPECTED_RELEASE_TIP,
            tip,
        )
    except RuntimeError as e:
        check("G4 freeze release tip", False, str(e))

    # --- G4.1 / G4.2 procedure fields present (dry-run contract) ---
    proc = data.get("rollback_procedure", {})
    check(
        "G4.1 procedure api_celery_ws",
        bool(proc.get("api_celery_ws")),
        str(proc.get("api_celery_ws", ""))[:80],
    )
    check(
        "G4.2 procedure consumer_outbox",
        "390076ef" in str(proc.get("consumer_outbox", "")),
        str(proc.get("consumer_outbox", ""))[:100],
    )
    check(
        "G4.6 procedure fanout hold",
        "do not compose up fanout" in str(proc.get("fanout_dlq", "")).lower()
        or "Created/stopped" in str(proc.get("fanout_dlq", "")),
        str(proc.get("fanout_dlq", ""))[:100],
    )

    # --- G4.6 match snapshot keys ---
    snap_path = (
        root / "docs" / "ops" / "_release_prod_snapshot_2026-08-15" / "snapshot.json"
    )
    if snap_path.is_file():
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        check(
            "G4.6 snapshot API match",
            snap.get("PROD_CURRENT_SHA") == data["backend"]["git_sha"],
            snap.get("PROD_CURRENT_SHA", ""),
        )
        check(
            "G4.6 snapshot consumer tag",
            "390076efc61c" in str(snap.get("TRACKING_CONSUMER_IMAGE", "")),
            str(snap.get("TRACKING_CONSUMER_IMAGE", "")),
        )
        check(
            "G4.6 snapshot fanout tag",
            "16fd3e52418d" in str(snap.get("FANOUT_IMAGE", "")),
            str(snap.get("FANOUT_IMAGE", "")),
        )
        check(
            "G4.6 snapshot alembic",
            snap.get("ALEMBIC_CURRENT") == EXPECTED_ALEMBIC,
            str(snap.get("ALEMBIC_CURRENT")),
        )
    else:
        check("G4.6 snapshot file", False, "snapshot.json missing")

    failed = [r for r in results if not r[1]]
    print()
    print(f"TOTAL={len(results)} PASS={len(results) - len(failed)} FAIL={len(failed)}")
    if failed:
        print("FAILED:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    print("G4_VALIDATOR=VERT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
