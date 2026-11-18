"""Découpage déterministe de la suite pytest pour les runners GitHub.

Le plugin n'est chargé que si ``PYTEST_SHARD`` est défini (voir ``conftest``).
Chaque test collecté appartient à un seul shard. Le job de synthèse refuse
une collection instable, un shard absent, un échec ou un trou de couverture.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

_STAT_KEYS = ("passed", "failed", "skipped", "xfailed", "xpassed", "error")


def partition(nodeids: list[str], shard: int, total: int) -> list[str]:
    """Retourne les nodeids du shard, tri stable, sans recouvrement."""
    if total < 1 or not 1 <= shard <= total:
        raise ValueError(f"shard invalide: {shard}/{total}")
    ordered = sorted(nodeids)
    return [
        nodeid for index, nodeid in enumerate(ordered) if index % total == shard - 1
    ]


def _bounds() -> tuple[int, int]:
    try:
        shard = int(os.environ["PYTEST_SHARD"])
        total = int(os.environ["PYTEST_SHARD_TOTAL"])
    except (KeyError, ValueError) as exc:
        pytest.exit(f"Variables de shard illisibles: {exc}", returncode=4)
    if total < 1 or not 1 <= shard <= total:
        pytest.exit(f"Shard invalide: {shard}/{total}", returncode=4)
    return shard, total


def pytest_configure(config: pytest.Config) -> None:
    config._ci_shard_started = time.time()  # type: ignore[attr-defined]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    shard, total = _bounds()
    collected = sorted(item.nodeid for item in items)
    selected_ids = partition(collected, shard, total)
    if not selected_ids:
        pytest.exit(f"Shard {shard}/{total} vide", returncode=1)
    selected = set(selected_ids)
    order = {nodeid: index for index, nodeid in enumerate(selected_ids)}
    items[:] = [item for item in items if item.nodeid in selected]
    items.sort(key=lambda item: order[item.nodeid])
    config._ci_shard_meta = {  # type: ignore[attr-defined]
        "shard": shard,
        "total_shards": total,
        "collected_nodeids": collected,
        "selected_nodeids": selected_ids,
    }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    meta = getattr(session.config, "_ci_shard_meta", None)
    report_path = os.environ.get("PYTEST_SHARD_REPORT")
    if meta is None or not report_path:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = getattr(reporter, "stats", {}) if reporter else {}
    counts = {key: len(stats.get(key, [])) for key in _STAT_KEYS}
    failed_nodeids = [rep.nodeid for rep in stats.get("failed", [])]
    error_nodeids = [rep.nodeid for rep in stats.get("error", [])]
    started = getattr(session.config, "_ci_shard_started", time.time())
    payload = {
        **meta,
        **counts,
        "failed_nodeids": failed_nodeids,
        "error_nodeids": error_nodeids,
        "exitstatus": int(exitstatus),
        "duration_seconds": round(time.time() - started, 3),
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_reports(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("pytest-shard-*/ci-shard-report.json"))
    if not paths:
        print(f"Aucun rapport de shard sous {root}", file=sys.stderr)
        return []
    reports = []
    for path in paths:
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def verify(root: Path) -> int:
    """Contrôle la partition et imprime le récapitulatif global."""
    reports = _load_reports(root)
    if not reports:
        return 1
    reports.sort(key=lambda item: int(item["shard"]))
    total = int(reports[0]["total_shards"])
    shards = [int(item["shard"]) for item in reports]
    ok = shards == list(range(1, total + 1))
    collected_sets = [item["collected_nodeids"] for item in reports]
    same_collection = all(item == collected_sets[0] for item in collected_sets)
    collected = collected_sets[0] if collected_sets else []
    expected = {
        shard: partition(collected, shard, total) for shard in range(1, total + 1)
    }
    print("=== Récapitulatif pytest par shard ===")
    executed = passed = failed = skipped = xfailed = xpassed = errors = 0
    duration = 0.0
    for item in reports:
        shard = int(item["shard"])
        ran = sum(int(item.get(key, 0)) for key in _STAT_KEYS)
        executed += ran
        passed += int(item.get("passed", 0))
        failed += int(item.get("failed", 0))
        skipped += int(item.get("skipped", 0))
        xfailed += int(item.get("xfailed", 0))
        xpassed += int(item.get("xpassed", 0))
        errors += int(item.get("error", 0))
        duration += float(item.get("duration_seconds", 0))
        matches = item["selected_nodeids"] == expected.get(shard)
        if (
            not matches
            or int(item.get("exitstatus", 1)) != 0
            or ran != len(item["selected_nodeids"])
        ):
            ok = False
        print(
            f"shard {shard}/{total} exit={item.get('exitstatus')} "
            f"selected={len(item['selected_nodeids'])} ran={ran} "
            f"passed={item.get('passed')} failed={item.get('failed')} "
            f"skipped={item.get('skipped')} xfailed={item.get('xfailed')} "
            f"xpassed={item.get('xpassed')} errors={item.get('error')} "
            f"duration_s={item.get('duration_seconds')} partition_ok={matches}"
        )
        for nodeid in item.get("failed_nodeids") or []:
            print(f"  FAILED {nodeid}")
        for nodeid in item.get("error_nodeids") or []:
            print(f"  ERROR {nodeid}")
    union: list[str] = []
    for item in reports:
        union.extend(item["selected_nodeids"])
    if len(union) != len(set(union)) or sorted(union) != sorted(collected):
        ok = False
    if not same_collection:
        ok = False
        print("Collection différente entre shards", file=sys.stderr)
    print(
        f"TOTAL collected={len(collected)} executed={executed} "
        f"passed={passed} failed={failed} skipped={skipped} "
        f"xfailed={xfailed} xpassed={xpassed} errors={errors} "
        f"duration_s={round(duration, 3)}"
    )
    if not ok:
        print("Contrôle des shards en échec", file=sys.stderr)
        return 1
    print("Contrôle des shards OK")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "verify":
        print(
            "Usage: python tests/ci_shard.py verify <dossier-artefacts>",
            file=sys.stderr,
        )
        return 2
    return verify(Path(argv[1]))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
