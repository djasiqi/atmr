#!/usr/bin/env python3
"""
analyze-imports.py - Analyse les imports de unified_dispatch dans le codebase

Ce script aide à identifier quels fichiers doivent être mis à jour
lors du refactoring B1.

Usage:
    python analyze-imports.py
    python analyze-imports.py --file types.py
    python analyze-imports.py --module core
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path


def find_imports_in_file(file_path):
    """Trouve tous les imports de unified_dispatch dans un fichier"""
    imports = []

    try:
        with Path(file_path).open(encoding="utf-8") as f:
            content = f.read()

        # Patterns d'import
        patterns = [
            r"from\s+services\.unified_dispatch\s+import\s+(.+)",
            r"from\s+services\.unified_dispatch\.(\S+)\s+import\s+(.+)",
            r"import\s+services\.unified_dispatch\.(\S+)",
            r"from\s+\.\.unified_dispatch\s+import\s+(.+)",
            r"from\s+\.unified_dispatch\s+import\s+(.+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            if matches:
                imports.extend(matches)

    except Exception as e:
        print(f"⚠️  Erreur lecture {file_path}: {e}")

    return imports


def analyze_codebase(base_dir="backend", target_file=None, target_module=None):
    """Analyse tous les fichiers Python du codebase"""

    results = defaultdict(list)

    # Parcourir tous les fichiers .py
    for root, dirs, files in os.walk(base_dir):
        # Ignorer certains dossiers
        dirs[:] = [
            d
            for d in dirs
            if d not in ["__pycache__", ".git", "node_modules", "venv", ".venv"]
        ]

        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = Path(root) / file
            rel_path = file_path.relative_to(base_dir)

            imports = find_imports_in_file(file_path)

            if imports:
                # Filtrer si nécessaire
                if target_file:
                    imports = [imp for imp in imports if target_file in str(imp)]
                if target_module:
                    imports = [imp for imp in imports if target_module in str(imp)]

                if imports:
                    results[rel_path] = imports

    return results


def print_results(results, target=None):
    """Affiche les résultats de l'analyse"""

    if not results:
        print(f"✅ Aucun import trouvé{' pour ' + target if target else ''}")
        return

    print(f"\n📊 Imports de unified_dispatch trouvés: {len(results)} fichiers\n")

    # Grouper par module
    by_module = defaultdict(list)

    for file_path, imports in results.items():
        module = file_path.split("/")[0] if "/" in file_path else "root"
        by_module[module].append((file_path, imports))

    # Afficher par module
    for module, files in sorted(by_module.items()):
        print(f"📦 Module: {module} ({len(files)} fichiers)")

        for file_path, imports in sorted(files):
            print(f"  📄 {file_path}")
            for imp in imports:
                print(f"     → {imp}")
        print()


def generate_migration_plan(results):
    """Génère un plan de migration basé sur les imports"""

    print("\n📋 Plan de Migration Suggéré\n")
    print("Les fichiers suivants devront être mis à jour lors de la migration:")
    print()

    # Compter les imports par fichier source
    import_counts = defaultdict(int)

    # Constantes pour les seuils de priorité
    HIGH_PRIORITY_THRESHOLD = 10
    MEDIUM_PRIORITY_THRESHOLD = 5

    for _file_path, imports in results.items():
        for imp in imports:
            import_counts[str(imp)] += 1

    # Trier par nombre d'imports (plus utilisé = plus prioritaire)
    sorted_imports = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)

    for i, (import_str, count) in enumerate(sorted_imports[:20], 1):
        priority = (
            "P0"
            if count > HIGH_PRIORITY_THRESHOLD
            else "P1"
            if count > MEDIUM_PRIORITY_THRESHOLD
            else "P2"
        )
        print(f"{i:2d}. [{priority}] {import_str:40s} ({count:3d} usages)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse les imports de unified_dispatch"
    )
    parser.add_argument(
        "--file", help="Filtrer par nom de fichier (ex: types.py)", default=None
    )
    parser.add_argument(
        "--module", help="Filtrer par module (ex: core, data)", default=None
    )
    parser.add_argument(
        "--plan", action="store_true", help="Générer un plan de migration"
    )
    parser.add_argument(
        "--base-dir", default="backend", help="Répertoire de base à analyser"
    )

    args = parser.parse_args()

    print("🔍 Analyse des imports de unified_dispatch...")
    print(f"📂 Répertoire: {args.base_dir}")

    if args.file:
        print(f"🎯 Fichier cible: {args.file}")
    if args.module:
        print(f"🎯 Module cible: {args.module}")

    print()

    results = analyze_codebase(
        base_dir=args.base_dir, target_file=args.file, target_module=args.module
    )

    print_results(results, target=args.file or args.module)

    if args.plan and results:
        generate_migration_plan(results)

    print(f"\n✅ Analyse terminée ({len(results)} fichiers)")


if __name__ == "__main__":
    main()
