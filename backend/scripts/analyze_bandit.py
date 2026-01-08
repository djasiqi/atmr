#!/usr/bin/env python3
"""Script pour analyser le rapport Bandit et afficher les problèmes HIGH
prioritaires."""

import json
import sys
from collections import Counter
from pathlib import Path


def main():
    report_path = Path(__file__).parent.parent / "bandit_report.json"

    if not report_path.exists():
        print(f"[X] Rapport Bandit introuvable: {report_path}")
        print("Executez d'abord: bandit -r . -c .bandit -f json -o bandit_report.json")
        sys.exit(1)

    with report_path.open() as f:
        report = json.load(f)

    results = report.get("results", [])

    # Statistiques globales
    high = [r for r in results if r["issue_severity"] == "HIGH"]
    medium = [r for r in results if r["issue_severity"] == "MEDIUM"]
    low = [r for r in results if r["issue_severity"] == "LOW"]

    print("=" * 80)
    print("RAPPORT BANDIT - STATISTIQUES")
    print("=" * 80)
    print(f"HIGH:   {len(high):4d}")
    print(f"MEDIUM: {len(medium):4d}")
    print(f"LOW:    {len(low):4d}")
    print(f"TOTAL:  {len(results):4d}")
    print()

    # Grouper les problèmes HIGH par type
    high_by_test = Counter(r["test_id"] for r in high)

    print("=" * 80)
    print("PROBLEMES HIGH PAR TYPE")
    print("=" * 80)
    for test_id, count in high_by_test.most_common():
        print(f"  {test_id:20s} : {count:4d} occurrences")
    print()

    # Afficher les 30 premiers problèmes HIGH avec détails
    print("=" * 80)
    print("TOP 30 PROBLEMES HIGH (details)")
    print("=" * 80)

    for i, issue in enumerate(high[:30], 1):
        filename = issue.get("filename", "unknown")
        line = issue.get("line_number", "?")
        test_id = issue.get("test_id", "?")
        issue_text = issue.get("issue_text", "")
        confidence = issue.get("issue_confidence", "?")

        print(f"\n{i}. {test_id} ({confidence} confidence)")
        print(f"   File: {filename}:{line}")
        print(f"   Issue: {issue_text}")

    print("\n" + "=" * 80)
    print(f"Rapport complet : {report_path}")
    print("=" * 80)

    # Retourner un code d'erreur si des HIGH existent
    if high:
        print(f"\n[X] {len(high)} problemes HIGH detectes")
        return 1

    print("\n[OK] Aucun probleme HIGH detecte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
