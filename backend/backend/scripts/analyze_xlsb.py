#!/usr/bin/env python3
# ruff: noqa: T201
"""Analyse du fichier XLSB (format binaire Excel)"""
import sys
from pathlib import Path

import pyxlsb

excel_file = Path("transport_annee_complete.xlsb")

if not excel_file.exists():
    print(f"❌ Fichier non trouvé: {excel_file}")
    sys.exit(1)

print("=" * 80)
print("📊 ANALYSE DU FICHIER XLSB (1 ANNÉE)")
print("=" * 80)
print(f"📂 Fichier : {excel_file.absolute()}")
print()

# Lire le fichier XLSB
try:
    with pyxlsb.open_workbook(excel_file) as wb:
        print(f"📑 Feuilles disponibles : {wb.sheets}")
        print()
        
        # Analyser la première feuille
        sheet_name = wb.sheets[0]
        print(f"📄 Analyse de la feuille : {sheet_name}")
        print()
        
        rows = []
        with wb.get_sheet(sheet_name) as sheet:
            for row in sheet.rows():
                rows.append([cell.v for cell in row])
        
        print(f"📊 Nombre de lignes : {len(rows)}")
        print()
        
        if rows:
            print("📋 En-têtes (première ligne) :")
            for i, col in enumerate(rows[0], 1):
                print(f"  {i}. {col}")
            print()
            
            print("👀 Aperçu des 3 premières lignes de données :")
            for i, row in enumerate(rows[1:4], 1):
                print(f"\nLigne {i}:")
                for j, val in enumerate(row):
                    if val:
                        print(f"  {rows[0][j]}: {val}")
            print()
            
        print(f"📈 Statistiques :")
        print(f"  - Total lignes de données : {len(rows) - 1}")
        print()
        
except Exception as e:
    print(f"❌ Erreur de lecture : {e}")
    import traceback
    traceback.print_exc()

