#!/usr/bin/env python3
"""✅ 3.1: Script pour analyser la couverture de tests et identifier modules à tester.

Usage:
    # Générer rapport coverage
    pytest --cov=. --cov-report=xml --cov-report=term-missing

    # Analyser coverage existant
    python scripts/check_coverage.py --coverage-xml coverage.xml

    # Vérifier seuils
    python scripts/check_coverage.py --coverage-xml coverage.xml --fail-under 70

Crée un rapport JSON et affiche:
- Modules avec couverture < 70% (globale)
- Modules critiques avec couverture < 80%
- Top 20 modules avec le moins de couverture
- Recommandations de fichiers à tester en priorité
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import xml.etree.ElementTree as ET
except ImportError:
    ET = None

# Seuils de coverage spécifiques par module (selon plan d'audit)
# Les modules listés ici ont des seuils plus élevés que le seuil par défaut (80%)
CRITICAL_MODULE_THRESHOLDS = {
    # Modules avec seuil ≥95%
    "routes/auth.py": 95.0,
    "services/unified_dispatch/engine.py": 95.0,
    "services/unified_dispatch/solver.py": 95.0,
    "services/unified_dispatch/heuristics.py": 95.0,
    "services/unified_dispatch/autonomous_manager.py": 95.0,
    "services/unified_dispatch/queue.py": 95.0,
    "services/unified_dispatch/data.py": 95.0,
    "services/unified_dispatch/apply.py": 95.0,
    "services/unified_dispatch/settings.py": 95.0,
    # Modules avec seuil ≥90% (osrm_client.py = alias compat, implémentation dans geolocation/osrm.py)
    "services/geolocation/osrm.py": 90.0,
    # Modules avec seuil ≥85%
    "services/rl/dispatch_env.py": 85.0,
}

# Seuil par défaut pour les autres modules critiques
DEFAULT_CRITICAL_THRESHOLD = 80.0

# Modules critiques (doivent avoir ≥ 80% couverture par défaut, ou seuil spécifique si défini)
CRITICAL_MODULES = {
    # Routes API critiques
    "routes/bookings.py",
    "routes/companies.py",
    "routes/auth.py",
    "routes/admin.py",
    "routes/dispatch_routes.py",
    "routes/payments.py",
    # Services critiques métier
    "services/unified_dispatch/engine.py",
    "services/unified_dispatch/solver.py",
    "services/unified_dispatch/heuristics.py",
    "services/unified_dispatch/autonomous_manager.py",
    "services/unified_dispatch/queue.py",
    "services/unified_dispatch/data.py",
    "services/unified_dispatch/apply.py",
    "services/unified_dispatch/settings.py",
    # Services externes critiques (OSRM : implémentation dans geolocation/osrm.py)
    "services/geolocation/osrm.py",
    # RL
    "services/rl/dispatch_env.py",
    # Sécurité
    "security/crypto.py",
    "security/audit_log.py",
    # Services critiques
    "services/api_slo.py",
    "services/unified_dispatch/slo.py",
    "middleware/metrics.py",
    # Database & ORM
    "db.py",
    "models/booking.py",
    "models/client.py",
    "models/driver.py",
    "models/user.py",
}


def get_module_threshold(module_path: str) -> float:
    """Retourne le seuil de coverage requis pour un module.

    Args:
        module_path: Chemin du module (ex: "routes/auth.py")

    Returns:
        Seuil de coverage en pourcentage (défaut: DEFAULT_CRITICAL_THRESHOLD)
    """
    # Vérifier d'abord les correspondances exactes
    if module_path in CRITICAL_MODULE_THRESHOLDS:
        return CRITICAL_MODULE_THRESHOLDS[module_path]

    # Vérifier les correspondances par préfixe (pour les répertoires comme unified_dispatch/)
    for critical_path, threshold in CRITICAL_MODULE_THRESHOLDS.items():
        if critical_path.endswith("/") and module_path.startswith(critical_path):
            return threshold
        if module_path.startswith(critical_path.replace(".py", "/")):
            return threshold

    # Vérifier si c'est un module critique (seuil par défaut)
    is_critical = any(
        module_path == critical
        or module_path.endswith("/" + critical)
        or module_path.startswith(critical + "/")
        for critical in CRITICAL_MODULES
    )

    if is_critical:
        return DEFAULT_CRITICAL_THRESHOLD

    # Module non critique : pas de seuil spécifique
    return 0.0


# Modules à ignorer (temporairement)
IGNORED_MODULES = {
    # Scripts non testables
    "scripts/",
    "migrations/",
    "manage.py",
    "wsgi.py",
    "app.py",  # App factory - tests via routes
    "celery_app.py",  # Tests via tasks
    # Modules avec imports conditionnels
    "shared/otel_setup.py",  # OpenTelemetry optionnel
}


def parse_coverage_xml(xml_path: Path) -> dict[str, Any]:
    """Parse le fichier coverage.xml et retourne un dict avec stats par module.

    Args:
        xml_path: Chemin vers coverage.xml

    Returns:
        Dict avec structure:
        {
            'total': {'lines': int, 'covered': int, 'percentage': float},
            'modules': {
                'module_path': {'lines': int, 'covered': int, 'missing': list, 'percentage': float}
            }
        }
    """
    if not xml_path.exists():
        print(f"❌ Fichier coverage.xml introuvable: {xml_path}", file=sys.stderr)
        sys.exit(1)

    if ET is None:
        print("❌ xml.etree.ElementTree non disponible", file=sys.stderr)
        sys.exit(1)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    total_lines = 0
    total_covered = 0
    modules: dict[str, dict[str, Any]] = {}

    for package in root.findall(".//package"):
        # package_name non utilisé pour l'instant mais peut être utile pour logging
        # _ = package.get("name", "")

        for class_elem in package.findall(".//class"):
            # class_name non utilisé, filename suffit pour identifier le module
            filename = class_elem.get("filename", "")

            # Construire chemin module relatif
            if filename.startswith(Path.cwd().name + "/"):
                module_path = filename[len(Path.cwd().name) + 1 :]
            else:
                module_path = filename

            # Ignorer modules excluded
            if any(module_path.startswith(ignored) for ignored in IGNORED_MODULES):
                continue

            # ✅ Cobertura XML: selon la version, `lines-valid/lines-covered` peut
            # ne pas être présent au niveau <class>. On calcule donc à partir des
            # <line hits="..."> pour être robuste.
            line_elems = list(class_elem.findall(".//line"))
            if line_elems:
                total = len(line_elems)
                covered = sum(
                    1 for line in line_elems if int(line.get("hits", "0")) > 0
                )
            else:
                # Fallback sur attributs si disponibles
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

            # Récupérer lignes manquantes
            for line in line_elems:
                if line.get("hits", "0") == "0":
                    line_number = int(line.get("number", "0"))
                    modules[module_path]["missing"].append(line_number)

    # Calculer pourcentage pour chaque module
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
    coverage_data: dict[str, Any], fail_under: float = 70.0
) -> dict[str, Any]:
    """Analyse les données de couverture et génère un rapport.

    Args:
        coverage_data: Dict retourné par parse_coverage_xml
        fail_under: Seuil minimum de couverture (par défaut 70%)

    Returns:
        Dict avec rapport d'analyse
    """
    modules = coverage_data["modules"]
    total_percentage = coverage_data["total"]["percentage"]

    # Modules en dessous du seuil global
    low_coverage_modules = {
        path: stats
        for path, stats in modules.items()
        if stats["percentage"] < fail_under
    }

    # Modules critiques en dessous de leur seuil requis
    critical_low_coverage = {}
    for path, stats in modules.items():
        # Obtenir le seuil requis pour ce module
        threshold = get_module_threshold(path)

        # Vérifier si module critique et si en dessous du seuil
        if threshold > 0 and stats["percentage"] < threshold:
            critical_low_coverage[path] = {
                **stats,
                "required_threshold": threshold,
            }

    # Top 20 modules avec le moins de couverture
    sorted_modules = sorted(modules.items(), key=lambda x: x[1]["percentage"])[:20]

    # Modules non testés (0%)
    untested_modules = {
        path: stats
        for path, stats in modules.items()
        if stats["percentage"] == 0.0 and stats["lines"] > 10  # Ignorer petits fichiers
    }

    return {
        "summary": {
            "total_percentage": total_percentage,
            "total_lines": coverage_data["total"]["lines"],
            "total_covered": coverage_data["total"]["covered"],
            "fail_under": fail_under,
            "meets_threshold": total_percentage >= fail_under,
            "modules_count": len(modules),
            "low_coverage_count": len(low_coverage_modules),
            "critical_low_coverage_count": len(critical_low_coverage),
            "untested_count": len(untested_modules),
        },
        "low_coverage_modules": dict(
            sorted(low_coverage_modules.items(), key=lambda x: x[1]["percentage"])
        ),
        "critical_low_coverage": dict(
            sorted(critical_low_coverage.items(), key=lambda x: x[1]["percentage"])
        ),
        "untested_modules": dict(
            sorted(untested_modules.items(), key=lambda x: x[1]["lines"], reverse=True)
        ),
        "worst_20_modules": [{"path": path, **stats} for path, stats in sorted_modules],
    }


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
    print(f"  - Seuil minimum: {summary['fail_under']:.1f}%")
    print(
        f"  - Statut: {'✅ ATTEINT' if summary['meets_threshold'] else '❌ EN DESSOUS'}"
    )
    print()

    print(f"Modules analysés: {summary['modules_count']}")
    print(f"Modules < {summary['fail_under']:.1f}%: {summary['low_coverage_count']}")
    print(
        f"Modules critiques en dessous du seuil requis: {summary['critical_low_coverage_count']}"
    )
    print(f"Modules non testés (0%): {summary['untested_count']}")
    print()

    # Modules critiques en dessous de leur seuil requis
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
                f"({stats['covered']}/{stats['lines']} lignes, {missing_count} manquantes, gap: {gap:.1f}%)"
            )
        if len(report["critical_low_coverage"]) > 10:
            print(f"  ... et {len(report['critical_low_coverage']) - 10} autres")
        print()

    # Top 20 modules avec faible couverture
    if report["worst_20_modules"]:
        print("=" * 80)
        print("⚠️  TOP 20 MODULES AVEC LE MOINS DE COUVERTURE")
        print("=" * 80)
        for item in report["worst_20_modules"]:
            path = item["path"]
            percentage = item["percentage"]
            lines = item["lines"]
            missing = len(item.get("missing", []))

            # Indicateur si critique
            is_critical = any(
                path == critical
                or path.endswith("/" + critical)
                or path.startswith(critical + "/")
                for critical in CRITICAL_MODULES
            )
            indicator = "🚨" if is_critical else "  "

            print(
                f"{indicator} {path:60s} {percentage:6.2f}% ({lines} lignes, {missing} manquantes)"
            )
        print()

    # Modules non testés
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

    if not summary["meets_threshold"]:
        gap = summary["fail_under"] - summary["total_percentage"]
        print(f"❌ Couverture globale en dessous du seuil de {gap:.2f}%")
        print(
            f"   Objectif: {summary['fail_under']:.1f}%, Actuel: {summary['total_percentage']:.2f}%"
        )
        print()

    if report["critical_low_coverage"]:
        print(
            f"🚨 {len(report['critical_low_coverage'])} modules critiques en dessous du seuil requis → PRIORITÉ HAUTE"
        )
        print("   Créer tests pour ces modules en premier:")
        for path, stats in list(report["critical_low_coverage"].items())[:5]:
            required = stats.get("required_threshold", DEFAULT_CRITICAL_THRESHOLD)
            current = stats["percentage"]
            print(
                f"   - tests/test_{Path(path).stem}.py (actuel: {current:.1f}%, requis: {required:.1f}%)"
            )
        print()

    if report["untested_modules"]:
        print(
            f"⚠️  {len(report['untested_modules'])} modules non testés → PRIORITÉ MOYENNE"
        )
        print("   Créer tests pour modules importants:")
        # Prioriser modules avec le plus de lignes
        for path, stats in sorted(
            list(report["untested_modules"].items())[:5],
            key=lambda x: x[1]["lines"],
            reverse=True,
        ):
            print(f"   - tests/test_{Path(path).stem}.py ({stats['lines']} lignes)")
        print()

    print("Pour générer coverage.xml:")
    print("  pytest --cov=. --cov-report=xml --cov-report=term-missing")
    print()


def main() -> int:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Analyse la couverture de tests et identifie modules à améliorer"
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
        default=70.0,
        help="Seuil minimum de couverture en pourcentage (défaut: 70.0)",
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

    # Parser coverage XML
    coverage_data = parse_coverage_xml(args.coverage_xml)

    # Analyser
    report = analyze_coverage(coverage_data, fail_under=args.fail_under)

    # Afficher rapport
    if not args.quiet:
        print_report(report)

    # Sauvegarder JSON si demandé
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not args.quiet:
            print(f"✅ Rapport JSON sauvegardé: {args.json}")

    # Exit code selon seuil global ET modules critiques
    # Le script échoue si:
    # 1. Le seuil global n'est pas atteint, OU
    # 2. Des modules critiques ne respectent pas leurs seuils spécifiques
    meets_global_threshold = report["summary"]["meets_threshold"]
    meets_critical_thresholds = len(report["critical_low_coverage"]) == 0

    if not args.quiet and not meets_critical_thresholds:
        print(
            "\n❌ ÉCHEC: Des modules critiques ne respectent pas leurs seuils spécifiques",
            file=sys.stderr,
        )

    return 0 if (meets_global_threshold and meets_critical_thresholds) else 1


if __name__ == "__main__":
    sys.exit(main())
