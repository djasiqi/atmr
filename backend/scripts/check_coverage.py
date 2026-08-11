#!/usr/bin/env python3
"""Analyse la couverture de tests à partir d'un coverage.xml (Cobertura).

Responsabilité : juger ce qui a déjà été mesuré. Les omits / le périmètre
appartiennent uniquement à ``backend/.coveragerc`` — ce script ne redéfinit
jamais le dénominateur.

Usage::

    # Rapport seul (exit 0)
    python scripts/check_coverage.py --coverage-xml coverage.xml

    # Gate GLOBAL uniquement
    python scripts/check_coverage.py --coverage-xml coverage.xml --fail-under 70

    # Gate CRITIQUES uniquement
    python scripts/check_coverage.py --coverage-xml coverage.xml --require-critical

    # Les deux gates
    python scripts/check_coverage.py --coverage-xml coverage.xml \\
        --fail-under 70 --require-critical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from defusedxml import ElementTree as ET
except ImportError:
    ET = None

# Référentiel d'affichage quand --fail-under n'est pas fourni (non bloquant)
DEFAULT_REPORT_THRESHOLD = 70.0

# Seuils spécifiques (chemin d'exécution produit, chemins normalisés)
CRITICAL_MODULE_THRESHOLDS: dict[str, float] = {
    "routes/auth.py": 95.0,
    # Dispatch — chemins d'implémentation (exécution produit)
    "services/unified_dispatch/core/engine.py": 95.0,
    "services/unified_dispatch/optimization/solver.py": 95.0,
    "services/unified_dispatch/optimization/heuristics.py": 95.0,
    "services/unified_dispatch/utils/autonomous.py": 95.0,
    "services/unified_dispatch/optimization/assignment_applier.py": 95.0,
    "services/unified_dispatch/core/settings.py": 95.0,
    "services/unified_dispatch/core/queue.py": 95.0,
    # Shims de compat (réexport) — doivent rester importables / couverts
    "services/unified_dispatch/solver.py": 95.0,
    "services/unified_dispatch/autonomous_manager.py": 95.0,
    "services/unified_dispatch/apply.py": 95.0,
    "services/unified_dispatch/settings.py": 95.0,
    "services/geolocation/osrm.py": 90.0,
    # Runtime RL (API dispatch / suggestions)
    "routes/dispatch/rl_helpers.py": 80.0,
    "services/ml/rl/suggestion_generator.py": 80.0,
}

DEFAULT_CRITICAL_THRESHOLD = 80.0

CRITICAL_MODULES: set[str] = {
    "routes/bookings.py",
    "routes/companies.py",
    "routes/auth.py",
    "routes/admin.py",
    "routes/dispatch_routes.py",
    "routes/payments.py",
    "routes/dispatch/rl_helpers.py",
    "services/unified_dispatch/core/engine.py",
    "services/unified_dispatch/core/queue.py",
    "services/unified_dispatch/core/settings.py",
    "services/unified_dispatch/optimization/solver.py",
    "services/unified_dispatch/optimization/heuristics.py",
    "services/unified_dispatch/optimization/assignment_applier.py",
    "services/unified_dispatch/utils/autonomous.py",
    "services/unified_dispatch/solver.py",
    "services/unified_dispatch/autonomous_manager.py",
    "services/unified_dispatch/apply.py",
    "services/unified_dispatch/settings.py",
    "services/geolocation/osrm.py",
    "services/ml/rl/suggestion_generator.py",
    "security/crypto.py",
    "security/audit_log.py",
    "services/api_slo.py",
    "services/unified_dispatch/metrics/slo.py",
    "middleware/metrics.py",
    "db.py",
    "models/booking.py",
    "models/client.py",
    "models/driver.py",
    "models/user.py",
}

# Racine backend (scripts/ → parent)
_BACKEND_ROOT = Path(__file__).resolve().parents[1]


def normalize_module_path(filename: str) -> str:
    """Normalise un chemin Cobertura vers une clé relative au package produit."""
    path = filename.replace("\\", "/").lstrip("./")
    while "//" in path:
        path = path.replace("//", "/")
    for prefix in ("backend/",):
        if path.startswith(prefix):
            path = path[len(prefix) :]
    return path.lstrip("/")


def get_module_threshold(module_path: str) -> float:
    """Retourne le seuil critique requis, ou 0.0 si le module n'est pas critique."""
    path = normalize_module_path(module_path)
    if path in CRITICAL_MODULE_THRESHOLDS:
        return CRITICAL_MODULE_THRESHOLDS[path]

    for critical_path, threshold in CRITICAL_MODULE_THRESHOLDS.items():
        if critical_path.endswith("/") and path.startswith(critical_path):
            return threshold

    if path in CRITICAL_MODULES:
        return DEFAULT_CRITICAL_THRESHOLD

    return 0.0


def find_existing_test_candidates(source_path: str) -> list[str]:
    """Retourne des chemins de tests existants liés au module source (relatifs à backend/)."""
    stem = Path(source_path).stem
    parts = Path(source_path).parts
    candidates: list[str] = []

    patterns = [
        f"tests/test_{stem}.py",
        f"tests/**/test_{stem}.py",
        f"tests/**/test_*{stem}*.py",
    ]
    if len(parts) >= 2:
        # ex. services/unified_dispatch/apply.py → tests/unified_dispatch/test_apply.py
        pkg = parts[-2]
        patterns.extend(
            [
                f"tests/{pkg}/test_{stem}.py",
                f"tests/**/{pkg}/test_{stem}.py",
                f"tests/test_{pkg}_{stem}.py",
            ]
        )

    seen: set[str] = set()
    for pattern in patterns:
        for match in _BACKEND_ROOT.glob(pattern):
            if not match.is_file():
                continue
            rel = match.relative_to(_BACKEND_ROOT).as_posix()
            if rel not in seen:
                seen.add(rel)
                candidates.append(rel)
    return sorted(candidates)


def parse_coverage_xml(xml_path: Path) -> dict[str, Any]:
    """Parse coverage.xml et agrège les stats par module (périmètre = XML tel quel)."""
    if not xml_path.exists():
        print(f"❌ Fichier coverage.xml introuvable: {xml_path}", file=sys.stderr)
        sys.exit(1)

    if ET is None:
        print("❌ defusedxml.ElementTree non disponible", file=sys.stderr)
        sys.exit(1)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    total_lines = 0
    total_covered = 0
    modules: dict[str, dict[str, Any]] = {}

    for package in root.findall(".//package"):
        for class_elem in package.findall(".//class"):
            filename = class_elem.get("filename", "") or ""
            module_path = normalize_module_path(filename)
            if not module_path:
                continue

            line_elems = list(class_elem.findall(".//line"))
            if line_elems:
                total = len(line_elems)
                covered = sum(
                    1 for line in line_elems if int(line.get("hits", "0")) > 0
                )
            else:
                covered = int(class_elem.get("lines-covered", "0"))
                total = int(class_elem.get("lines-valid", "0"))

            if module_path not in modules:
                modules[module_path] = {
                    "lines": 0,
                    "covered": 0,
                    "missing": [],
                    "percentage": 0.0,
                }

            modules[module_path]["lines"] += total
            modules[module_path]["covered"] += covered
            total_lines += total
            total_covered += covered

            for line in line_elems:
                if line.get("hits", "0") == "0":
                    line_number = int(line.get("number", "0"))
                    modules[module_path]["missing"].append(line_number)

    for stats in modules.values():
        if stats["lines"] > 0:
            stats["percentage"] = (stats["covered"] / stats["lines"]) * 100
        stats["missing"].sort()

    total_percentage = (total_covered / total_lines * 100) if total_lines > 0 else 0.0

    return {
        "total": {
            "lines": total_lines,
            "covered": total_covered,
            "percentage": total_percentage,
        },
        "modules": modules,
    }


def analyze_coverage(
    coverage_data: dict[str, Any],
    *,
    report_threshold: float = DEFAULT_REPORT_THRESHOLD,
) -> dict[str, Any]:
    """Analyse les données de couverture (rapport ; le gating est dans main())."""
    modules = coverage_data["modules"]
    total_percentage = coverage_data["total"]["percentage"]

    low_coverage_modules = {
        path: stats
        for path, stats in modules.items()
        if stats["percentage"] < report_threshold
    }

    critical_low_coverage: dict[str, Any] = {}
    for path, stats in modules.items():
        threshold = get_module_threshold(path)
        if threshold > 0 and stats["percentage"] < threshold:
            critical_low_coverage[path] = {
                **stats,
                "required_threshold": threshold,
            }

    # Critiques absents du XML (omis ou jamais instrumentés) : signalés mais
    # hors périmètre mesuré — ne pas les inventer dans le dénominateur.
    critical_missing_from_xml = sorted(
        path
        for path in CRITICAL_MODULES | set(CRITICAL_MODULE_THRESHOLDS)
        if path not in modules
    )

    sorted_modules = sorted(modules.items(), key=lambda x: x[1]["percentage"])[:20]

    untested_modules = {
        path: stats
        for path, stats in modules.items()
        if stats["percentage"] == 0.0 and stats["lines"] > 10
    }

    return {
        "summary": {
            "total_percentage": total_percentage,
            "total_lines": coverage_data["total"]["lines"],
            "total_covered": coverage_data["total"]["covered"],
            "report_threshold": report_threshold,
            "meets_report_threshold": total_percentage >= report_threshold,
            "modules_count": len(modules),
            "low_coverage_count": len(low_coverage_modules),
            "critical_low_coverage_count": len(critical_low_coverage),
            "untested_count": len(untested_modules),
            "critical_missing_from_xml_count": len(critical_missing_from_xml),
        },
        "low_coverage_modules": dict(
            sorted(low_coverage_modules.items(), key=lambda x: x[1]["percentage"])
        ),
        "critical_low_coverage": dict(
            sorted(critical_low_coverage.items(), key=lambda x: x[1]["percentage"])
        ),
        "critical_missing_from_xml": critical_missing_from_xml,
        "untested_modules": dict(
            sorted(untested_modules.items(), key=lambda x: x[1]["lines"], reverse=True)
        ),
        "worst_20_modules": [{"path": path, **stats} for path, stats in sorted_modules],
    }


def _print_module_recommendations(path: str, stats: dict[str, Any]) -> None:
    required = stats.get("required_threshold")
    current = stats["percentage"]
    print(f"   Source: {path}")
    if required is not None:
        print(f"   Coverage: {current:.1f}% (requis: {required:.1f}%)")
    else:
        print(f"   Coverage: {current:.1f}% ({stats['lines']} lignes)")
    candidates = find_existing_test_candidates(path)
    if candidates:
        print("   Tests candidats existants:")
        for candidate in candidates[:5]:
            print(f"   - {candidate}")
    else:
        print("   Aucun test direct identifié.")
    print()


def print_report(report: dict[str, Any]) -> None:
    """Affiche un rapport formaté dans le terminal."""
    summary = report["summary"]

    print("=" * 80)
    print("📊 RAPPORT COUVERTURE TESTS")
    print("=" * 80)
    print()

    print(f"Couverture globale: {summary['total_percentage']:.2f}%")
    print(f"  - Lignes totales: {summary['total_lines']:,}")
    print(f"  - Lignes couvertes: {summary['total_covered']:,}")
    print(f"  - Référentiel rapport: {summary['report_threshold']:.1f}%")
    print(
        "  - Statut référentiel: "
        f"{'✅ ATTEINT' if summary['meets_report_threshold'] else '❌ EN DESSOUS'}"
    )
    print()

    print(f"Modules analysés: {summary['modules_count']}")
    print(
        f"Modules < {summary['report_threshold']:.1f}%: {summary['low_coverage_count']}"
    )
    print(
        "Modules critiques sous seuil: "
        f"{summary['critical_low_coverage_count']}"
    )
    print(f"Modules non testés (0%): {summary['untested_count']}")
    if summary.get("critical_missing_from_xml_count"):
        print(
            "Modules critiques absents du XML (omis / hors mesure): "
            f"{summary['critical_missing_from_xml_count']}"
        )
    print()

    if report["critical_low_coverage"]:
        print("=" * 80)
        print("🚨 MODULES CRITIQUES EN DESSOUS DU SEUIL REQUIS (PRIORITÉ HAUTE)")
        print("=" * 80)
        for path, stats in list(report["critical_low_coverage"].items())[:10]:
            missing_count = len(stats.get("missing", []))
            required = stats.get("required_threshold", DEFAULT_CRITICAL_THRESHOLD)
            current = stats["percentage"]
            gap = required - current
            print(
                f"  {path:60s} {current:6.2f}% / {required:5.1f}% requis "
                f"({stats['covered']}/{stats['lines']} lignes, "
                f"{missing_count} manquantes, gap: {gap:.1f}%)"
            )
        if len(report["critical_low_coverage"]) > 10:
            print(f"  ... et {len(report['critical_low_coverage']) - 10} autres")
        print()

    if report["worst_20_modules"]:
        print("=" * 80)
        print("⚠️  TOP 20 MODULES AVEC LE MOINS DE COUVERTURE")
        print("=" * 80)
        for item in report["worst_20_modules"]:
            path = item["path"]
            percentage = item["percentage"]
            lines = item["lines"]
            missing = len(item.get("missing", []))
            is_critical = get_module_threshold(path) > 0
            indicator = "🚨" if is_critical else "  "
            print(
                f"{indicator} {path:60s} {percentage:6.2f}% "
                f"({lines} lignes, {missing} manquantes)"
            )
        print()

    if report["untested_modules"]:
        print("=" * 80)
        print("❌ MODULES NON TESTÉS (0% couverture)")
        print("=" * 80)
        for path, stats in list(report["untested_modules"].items())[:15]:
            print(f"  {path:60s} {stats['lines']:4d} lignes")
        if len(report["untested_modules"]) > 15:
            print(f"  ... et {len(report['untested_modules']) - 15} autres")
        print()

    print("=" * 80)
    print("💡 RECOMMANDATIONS")
    print("=" * 80)

    if not summary["meets_report_threshold"]:
        gap = summary["report_threshold"] - summary["total_percentage"]
        print(f"❌ Couverture globale en dessous du référentiel de {gap:.2f}%")
        print(
            f"   Objectif: {summary['report_threshold']:.1f}%, "
            f"Actuel: {summary['total_percentage']:.2f}%"
        )
        print()

    if report["critical_low_coverage"]:
        print(
            f"🚨 {len(report['critical_low_coverage'])} modules critiques "
            "sous seuil → PRIORITÉ HAUTE"
        )
        for path, stats in list(report["critical_low_coverage"].items())[:5]:
            _print_module_recommendations(path, stats)

    if report["untested_modules"]:
        print(
            f"⚠️  {len(report['untested_modules'])} modules non testés → PRIORITÉ MOYENNE"
        )
        for path, stats in list(report["untested_modules"].items())[:5]:
            _print_module_recommendations(path, stats)

    print("Pour générer coverage.xml (CI) :")
    print(
        "  pytest backend/tests -v --cov=backend "
        "--cov-config=backend/.coveragerc "
        "--cov-report=xml:backend/coverage.xml"
    )
    print()


def main() -> int:
    """Point d'entrée : rapport + gates optionnels indépendants."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyse coverage.xml. "
            "Sans --fail-under ni --require-critical : rapport seul (exit 0)."
        )
    )
    parser.add_argument(
        "--coverage-xml",
        type=Path,
        default=Path("coverage.xml"),
        help="Chemin vers coverage.xml (défaut: coverage.xml)",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help=(
            "Gate GLOBAL : échoue si couverture globale < seuil. "
            "Absent = pas de gate global (défaut)."
        ),
    )
    parser.add_argument(
        "--require-critical",
        action="store_true",
        help="Gate CRITIQUES : échoue si un module critique est sous son seuil.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Sauvegarder rapport JSON dans ce fichier",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Mode silencieux (pas de sortie console)",
    )

    args = parser.parse_args()

    report_threshold = (
        args.fail_under
        if args.fail_under is not None
        else DEFAULT_REPORT_THRESHOLD
    )

    coverage_data = parse_coverage_xml(args.coverage_xml)
    report = analyze_coverage(coverage_data, report_threshold=report_threshold)

    if not args.quiet:
        print_report(report)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not args.quiet:
            print(f"✅ Rapport JSON sauvegardé: {args.json}")

    exit_code = 0

    if args.fail_under is not None:
        global_ok = report["summary"]["total_percentage"] >= args.fail_under
        if not global_ok:
            exit_code = 1
            if not args.quiet:
                print(
                    f"\n❌ ÉCHEC gate GLOBAL: "
                    f"{report['summary']['total_percentage']:.2f}% "
                    f"< {args.fail_under:.1f}%",
                    file=sys.stderr,
                )

    if args.require_critical:
        critical_ok = len(report["critical_low_coverage"]) == 0
        if not critical_ok:
            exit_code = 1
            if not args.quiet:
                print(
                    "\n❌ ÉCHEC gate CRITIQUES: "
                    f"{len(report['critical_low_coverage'])} module(s) sous seuil",
                    file=sys.stderr,
                )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
