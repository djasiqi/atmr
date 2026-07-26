#!/usr/bin/env python3
"""
Script pour détecter les exceptions trop larges dans le codebase.

Usage:
    python scripts/detect_broad_exceptions.py
    python scripts/detect_broad_exceptions.py --fix  # Mode interactif pour corriger
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple

# Couleurs pour output
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"


class BroadExceptionDetector(ast.NodeVisitor):
    """Visiteur AST pour détecter les exceptions trop larges."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.issues: List[Tuple[int, str, str]] = []

    def visit_ExceptHandler(self, node):
        """Visite les handlers d'exception."""
        # Détecter except: (sans type)
        if node.type is None:
            self.issues.append(
                (
                    node.lineno,
                    "except:",
                    "Exception handler sans type spécifique (attrape toutes les exceptions)",
                )
            )
        # Détecter except Exception:
        elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
            self.issues.append(
                (
                    node.lineno,
                    "except Exception:",
                    "Exception handler trop large (attrape toutes les exceptions)",
                )
            )
        # Détecter except (Exception, ...):
        elif isinstance(node.type, ast.Tuple):
            for elt in node.type.elts:
                if isinstance(elt, ast.Name) and elt.id == "Exception":
                    self.issues.append(
                        (
                            node.lineno,
                            "except (..., Exception, ...):",
                            "Exception handler inclut Exception (trop large)",
                        )
                    )
                    break

        self.generic_visit(node)


def scan_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """Scanne un fichier Python pour détecter les exceptions trop larges."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, filename=str(file_path))
            detector = BroadExceptionDetector(file_path)
            detector.visit(tree)
            return detector.issues
    except SyntaxError as e:
        print(f"{YELLOW}⚠️  Erreur de syntaxe dans {file_path}:{e}{RESET}")
        return []
    except Exception as e:
        print(f"{RED}❌ Erreur lors du scan de {file_path}: {e}{RESET}")
        return []


def scan_directory(directory: Path) -> dict:
    """Scanne un répertoire récursivement."""
    results = {}
    total_issues = 0

    # Ignorer certains répertoires (dette progressive + bruit local)
    ignore_dirs = {
        "__pycache__",
        "venv",
        ".venv",
        "node_modules",
        ".git",
        "migrations",
        ".cursor-server",
        ".local",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
        "tests",
        "scripts",
    }

    for py_file in directory.rglob("*.py"):
        # Ignorer les fichiers dans les répertoires à ignorer
        if any(ignore_dir in py_file.parts for ignore_dir in ignore_dirs):
            continue
        if py_file.name in {"test_b2_imports.py", "conftest.py"}:
            continue

        issues = scan_file(py_file)
        if issues:
            results[str(py_file.relative_to(directory))] = issues
            total_issues += len(issues)

    return results, total_issues


def print_report(results: dict, total_issues: int):
    """Affiche le rapport de détection."""
    print(f"\n{'=' * 80}")
    print(f"{YELLOW}📊 RAPPORT DE DÉTECTION DES EXCEPTIONS TROP LARGES{RESET}")
    print(f"{'=' * 80}\n")

    if not results:
        print(f"{GREEN}✅ Aucune exception trop large détectée !{RESET}\n")
        return

    print(f"{RED}❌ Total d'occurrences détectées : {total_issues}{RESET}\n")

    # Trier par nombre d'issues
    sorted_results = sorted(results.items(), key=lambda x: len(x[1]), reverse=True)

    for file_path, issues in sorted_results:
        print(f"{YELLOW}📁 {file_path}{RESET} ({len(issues)} occurrence(s))")
        for line_num, pattern, description in issues:
            print(f"  {RED}Ligne {line_num:4d}:{RESET} {pattern}")
            print(f"    └─ {description}")
        print()

    # Top 10 fichiers les plus problématiques
    print(f"\n{YELLOW}🔝 Top 10 fichiers les plus problématiques :{RESET}")
    for i, (file_path, issues) in enumerate(sorted_results[:10], 1):
        print(f"  {i:2d}. {file_path} ({len(issues)} occurrence(s))")


def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Détecte les exceptions trop larges dans le codebase"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="backend",
        help="Chemin du répertoire à scanner (défaut: backend)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Mode interactif pour corriger (non implémenté pour l'instant)",
    )

    args = parser.parse_args()

    directory = Path(args.path)
    if not directory.exists():
        print(f"{RED}❌ Répertoire introuvable : {directory}{RESET}")
        sys.exit(1)

    print(f"{GREEN}🔍 Scan en cours...{RESET}")
    results, total_issues = scan_directory(directory)

    print_report(results, total_issues)

    if total_issues > 0:
        print(f"\n{YELLOW}💡 Conseil :{RESET}")
        print(
            "  - Consultez docs/PLAN_CORRECTION_EXCEPTIONS_LARGES.md pour le plan de correction"
        )
        print("  - Remplacez les exceptions larges par des exceptions spécifiques")
        print("  - Utilisez les helpers dans shared/error_handling.py")
        print(
            f"\n{YELLOW}⚠️  Rapport informatif ({total_issues} occ.) — exit 0 (dette progressive).{RESET}"
        )
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
