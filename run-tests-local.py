#!/usr/bin/env python3
"""
Script de validation locale - Tests Refactoring B1
Vérifie la syntaxe et les imports sans Docker
"""

import sys
from pathlib import Path
import py_compile


def test_syntax(files):
    """Teste la syntaxe Python de tous les fichiers migrés"""
    print("Test 1/4: Syntaxe Python des fichiers migres")
    errors = []
    for file in files:
        try:
            py_compile.compile(file, doraise=True)
            print(f"  OK {file.relative_to('backend')}")
        except py_compile.PyCompileError as e:
            errors.append((file, str(e)))
            print(f"  ERR {file.relative_to('backend')}: {e}")

    if errors:
        print(f"\nERR {len(errors)} erreurs de syntaxe")
        return False
    print(f"\nOK Tous les fichiers compilent ({len(files)} fichiers)")
    return True


def test_imports():
    """Teste les imports du module unified_dispatch"""
    print("\nTest 2/4: Imports module unified_dispatch")
    try:
        # Ajout du chemin backend au PYTHONPATH
        backend_path = Path("backend")
        sys.path.insert(0, str(backend_path.absolute()))

        # Test imports critiques
        tests = [
            ("core.types", "from services.unified_dispatch.core import types"),
            (
                "core.exceptions",
                "from services.unified_dispatch.core import exceptions",
            ),
            ("core.settings", "from services.unified_dispatch.core import settings"),
            ("data.loader", "from services.unified_dispatch.data import loader"),
            (
                "optimization.solver",
                "from services.unified_dispatch.optimization import solver",
            ),
        ]

        errors = []
        for module_name, import_stmt in tests:
            try:
                exec(import_stmt)
                print(f"  OK {module_name}")
            except Exception as e:
                errors.append((module_name, str(e)))
                print(f"  ERR {module_name}: {e}")

        if errors:
            print(
                f"\nWARN {len(errors)} erreurs d'imports (possibles dependances manquantes)"
            )
            return False
        print(f"\nOK Tous les imports fonctionnent ({len(tests)} testes)")
        return True
    except Exception as e:
        print(f"\nWARN Erreur globale: {e}")
        return False


def test_structure():
    """Vérifie la structure des modules"""
    print("\nTest 3/4: Structure modules unified_dispatch")
    backend = Path("backend/services/unified_dispatch")

    expected_modules = [
        "core",
        "data",
        "optimization",
        "ml",
        "metrics",
        "validation",
        "shadow_mode",
        "utils",
        "orchestration",
        "locking",
    ]

    missing = []
    for module in expected_modules:
        module_path = backend / module
        if not module_path.exists():
            missing.append(module)
            print(f"  ERR {module}/ manquant")
        else:
            init_file = module_path / "__init__.py"
            if init_file.exists():
                print(f"  OK {module}/ avec __init__.py")
            else:
                print(f"  WARN {module}/ sans __init__.py")

    if missing:
        print(f"\nERR {len(missing)} modules manquants")
        return False
    print(f"\nOK Tous les modules presents ({len(expected_modules)} modules)")
    return True


def count_tests():
    """Compte les fichiers de tests"""
    print("\nTest 4/4: Fichiers de tests disponibles")
    test_dir = Path("backend/tests/services/unified_dispatch")

    if not test_dir.exists():
        print("  ERR Repertoire tests/ n'existe pas")
        return False

    test_files = list(test_dir.rglob("test_*.py"))
    print(f"  OK {len(test_files)} fichiers de tests trouves")
    for test_file in test_files[:14]:  # Afficher les 14 premiers
        print(
            f"     - {test_file.relative_to('backend/tests/services/unified_dispatch')}"
        )

    return True


def main():
    print("=" * 60)
    print("VALIDATION LOCALE - Refactoring B1")
    print("=" * 60)

    # Collecte fichiers migrés
    unified_dispatch = Path("backend/services/unified_dispatch")

    # Fichiers P0+P1+P2 migrés
    migrated_files = [
        # Core
        unified_dispatch / "core/types.py",
        unified_dispatch / "core/exceptions.py",
        unified_dispatch / "core/settings.py",
        unified_dispatch / "core/problem_state.py",
        unified_dispatch / "core/queue.py",
        unified_dispatch / "core/engine.py",
        # Data
        unified_dispatch / "data/loader.py",
        unified_dispatch / "data/clustering.py",
        unified_dispatch / "data/warm_start.py",
        # Optimization
        unified_dispatch / "optimization/solver.py",
        unified_dispatch / "optimization/assignment_applier.py",
        unified_dispatch / "optimization/heuristics.py",
        unified_dispatch / "optimization/pareto_front.py",
        unified_dispatch / "optimization/score_fusion.py",
        unified_dispatch / "optimization/warm_start_tracker.py",
        # ML
        unified_dispatch / "ml/rl_optimizer.py",
        unified_dispatch / "ml/predictor.py",
        unified_dispatch / "ml/delay_predictor.py",
        unified_dispatch / "ml/rl_kpi_monitor.py",
        unified_dispatch / "ml/ab_tracking.py",
        unified_dispatch / "ml/ab_router.py",
        # Metrics
        unified_dispatch / "metrics/dispatch.py",
        unified_dispatch / "metrics/prometheus.py",
        unified_dispatch / "metrics/slo.py",
        unified_dispatch / "metrics/performance.py",
        unified_dispatch / "metrics/errors.py",
        unified_dispatch / "metrics/osrm_cache.py",
        # Validation
        unified_dispatch / "validation/constraints.py",
        unified_dispatch / "validation/assignment.py",
        # Shadow Mode
        unified_dispatch / "shadow_mode/orchestrator.py",
        unified_dispatch / "shadow_mode/manager.py",
        # Utils
        unified_dispatch / "utils/transactions.py",
        unified_dispatch / "utils/realtime.py",
        unified_dispatch / "utils/suggestions.py",
        unified_dispatch / "utils/autonomous.py",
    ]

    # Filtrer fichiers existants
    existing_files = [f for f in migrated_files if f.exists()]
    print(f"\nFichiers a valider: {len(existing_files)}/{len(migrated_files)}\n")

    # Exécution tests
    results = []
    results.append(("Syntaxe Python", test_syntax(existing_files)))
    results.append(("Imports modules", test_imports()))
    results.append(("Structure", test_structure()))
    results.append(("Fichiers tests", count_tests()))

    # Résumé
    print("\n" + "=" * 60)
    print("RESUME")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status} - {name}")

    print(f"\nResultat: {passed}/{total} tests passent")

    if passed == total:
        print("\nTOUS LES TESTS LOCAUX PASSENT!")
        print("\nNote: Les tests unitaires complets (pytest) necessitent Docker")
        print("   Mais la syntaxe, les imports et la structure sont valides")
        return 0
    else:
        print("\nECHEC: Certains tests ont echoue")
        return 1


if __name__ == "__main__":
    sys.exit(main())
