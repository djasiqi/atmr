#!/usr/bin/env python3
"""Analyse du fichier XLSB (format binaire Excel)."""
import sys
from pathlib import Path

import pyxlsb  # pyright: ignore[reportMissingImports]

excel_file = Path("transport_annee_complete.xlsb")

if not excel_file.exists():
    print("❌ Fichier non trouvé: {excel_file}")
    sys.exit(1)

print("=" * 80)
print("📊 ANALYSE DU FICHIER XLSB (1 ANNÉE)")
print("=" * 80)
print("📂 Fichier : {excel_file.absolute()}")
print()

# Lire le fichier XLSB
try:
    with pyxlsb.open_workbook(excel_file) as wb:
        print("📑 Feuilles disponibles : {wb.sheets}")
        print()

        # Analyser la première feuille
        sheet_name = wb.sheets[0]
        print("📄 Analyse de la feuille : {sheet_name}")
        print()

        rows = []
        with wb.get_sheet(sheet_name) as sheet:
            for row in sheet.rows():
                rows.append([cell.v for cell in row])

        print("📊 Nombre de lignes : {len(rows)}")
        print()

        if rows:
            print("📋 En-têtes (première ligne) :")
            for _i, _col in enumerate(rows[0], 1):
                print("  {i}. {col}")
            print()

            print("👀 Aperçu des 3 premières lignes de données :")
            for _i, row in enumerate(rows[1:4], 1):
                print("\nLigne {i}:")
                for _j, val in enumerate(row):
                    if val:
                        print("  {rows[0][j]}: {val}")
            print()

        print("📈 Statistiques :")
        print("  - Total lignes de données : {len(rows) - 1}")
        print()

except Exception:
    print("❌ Erreur de lecture : {e}")
    import traceback
    traceback.print_exc()

