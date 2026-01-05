#!/usr/bin/env python3
"""Script de comparaison des résultats de dispatch.

Compare les résultats de deux exécutions de dispatch et génère un rapport.
Utile pour valider que le refactoring n'a pas changé le comportement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def compare_results(result1: dict[str, Any], result2: dict[str, Any]) -> dict[str, Any]:
    """Compare deux résultats de dispatch.

    Args:
        result1: Premier résultat (référence)
        result2: Deuxième résultat (à comparer)

    Returns:
        Dict contenant les différences trouvées
    """
    differences: dict[str, Any] = {
        "identical": True,
        "differences": [],
        "summary": {},
    }

    # Comparer les clés principales
    keys_to_compare = [
        "assignments",
        "unassigned",
        "bookings",
        "drivers",
        "meta",
        "debug",
    ]

    for key in keys_to_compare:
        val1 = result1.get(key)
        val2 = result2.get(key)

        if val1 != val2:
            differences["identical"] = False

            if isinstance(val1, list) and isinstance(val2, list):
                # Comparer les listes
                len_diff = len(val1) - len(val2)
                if len_diff != 0:
                    differences["differences"].append(
                        {
                            "key": key,
                            "type": "length_difference",
                            "value1": len(val1),
                            "value2": len(val2),
                            "difference": len_diff,
                        }
                    )
                else:
                    # Comparer les éléments
                    for i, (item1, item2) in enumerate(zip(val1, val2, strict=False)):
                        if item1 != item2:
                            differences["differences"].append(
                                {
                                    "key": f"{key}[{i}]",
                                    "type": "item_difference",
                                    "value1": item1,
                                    "value2": item2,
                                }
                            )
            elif isinstance(val1, dict) and isinstance(val2, dict):
                # Comparer les dictionnaires
                keys1 = set(val1.keys())
                keys2 = set(val2.keys())

                if keys1 != keys2:
                    differences["differences"].append(
                        {
                            "key": key,
                            "type": "key_difference",
                            "keys_only_in_1": list(keys1 - keys2),
                            "keys_only_in_2": list(keys2 - keys1),
                        }
                    )

                # Comparer les valeurs communes
                for common_key in keys1 & keys2:
                    if val1[common_key] != val2[common_key]:
                        differences["differences"].append(
                            {
                                "key": f"{key}.{common_key}",
                                "type": "value_difference",
                                "value1": val1[common_key],
                                "value2": val2[common_key],
                            }
                        )
            else:
                differences["differences"].append(
                    {
                        "key": key,
                        "type": "value_difference",
                        "value1": val1,
                        "value2": val2,
                    }
                )

    # Générer un résumé
    differences["summary"] = {
        "assignments_count_1": len(result1.get("assignments", [])),
        "assignments_count_2": len(result2.get("assignments", [])),
        "unassigned_count_1": len(result1.get("unassigned", [])),
        "unassigned_count_2": len(result2.get("unassigned", [])),
        "bookings_count_1": len(result1.get("bookings", [])),
        "bookings_count_2": len(result2.get("bookings", [])),
        "drivers_count_1": len(result1.get("drivers", [])),
        "drivers_count_2": len(result2.get("drivers", [])),
        "total_differences": len(differences["differences"]),
    }

    return differences


def generate_report(
    differences: dict[str, Any], output_file: Path | None = None
) -> str:
    """Génère un rapport de comparaison.

    Args:
        differences: Résultat de la comparaison
        output_file: Fichier de sortie (optionnel)

    Returns:
        Rapport sous forme de string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("RAPPORT DE COMPARAISON DES RÉSULTATS DE DISPATCH")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Résumé
    summary = differences["summary"]
    report_lines.append("RÉSUMÉ")
    report_lines.append("-" * 80)
    report_lines.append(
        f"Assignments: {summary['assignments_count_1']} vs {summary['assignments_count_2']}"
    )
    report_lines.append(
        f"Unassigned: {summary['unassigned_count_1']} vs {summary['unassigned_count_2']}"
    )
    report_lines.append(
        f"Bookings: {summary['bookings_count_1']} vs {summary['bookings_count_2']}"
    )
    report_lines.append(
        f"Drivers: {summary['drivers_count_1']} vs {summary['drivers_count_2']}"
    )
    report_lines.append(f"Total différences: {summary['total_differences']}")
    report_lines.append("")

    # Statut
    if differences["identical"]:
        report_lines.append("✅ RÉSULTATS IDENTIQUES")
    else:
        report_lines.append("⚠️  DIFFÉRENCES DÉTECTÉES")
    report_lines.append("")

    # Détails des différences
    if differences["differences"]:
        report_lines.append("DÉTAILS DES DIFFÉRENCES")
        report_lines.append("-" * 80)
        for i, diff in enumerate(differences["differences"], 1):
            report_lines.append(f"\n{i}. {diff['key']} ({diff['type']})")
            if "value1" in diff:
                report_lines.append(f"   Valeur 1: {diff['value1']}")
            if "value2" in diff:
                report_lines.append(f"   Valeur 2: {diff['value2']}")
            if "difference" in diff:
                report_lines.append(f"   Différence: {diff['difference']}")

    report_lines.append("")
    report_lines.append("=" * 80)

    report = "\n".join(report_lines)

    if output_file:
        output_file.write_text(report, encoding="utf-8")
        print(f"Rapport sauvegardé dans {output_file}")

    return report


def main():
    """Point d'entrée principal."""
    if len(sys.argv) < 3:
        print(
            "Usage: python compare_dispatch_results.py <result1.json> <result2.json> [output.txt]"
        )
        sys.exit(1)

    result1_file = Path(sys.argv[1])
    result2_file = Path(sys.argv[2])
    output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    if not result1_file.exists():
        print(f"Erreur: {result1_file} n'existe pas")
        sys.exit(1)

    if not result2_file.exists():
        print(f"Erreur: {result2_file} n'existe pas")
        sys.exit(1)

    # Charger les résultats
    result1 = json.loads(result1_file.read_text(encoding="utf-8"))
    result2 = json.loads(result2_file.read_text(encoding="utf-8"))

    # Comparer
    differences = compare_results(result1, result2)

    # Générer le rapport
    report = generate_report(differences, output_file)
    print(report)

    # Code de sortie
    sys.exit(0 if differences["identical"] else 1)


if __name__ == "__main__":
    main()
