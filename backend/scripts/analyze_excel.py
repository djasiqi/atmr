#!/usr/bin/env python3
"""Analyse du fichier Excel de transport historique."""
import sys
from pathlib import Path

import pandas as pd

excel_file = Path("transport.xlsx")

if not excel_file.exists():
    print("❌ Fichier non trouvé: {excel_file}")
    sys.exit(1)

print("=" * 80)
print("📊 ANALYSE DU FICHIER EXCEL")
print("=" * 80)
print("📂 Fichier : {excel_file.absolute()}")
print()

# Lire le fichier Excel
try:
    # Essayer de lire toutes les feuilles
    xlsx = pd.ExcelFile(excel_file)
    print("📑 Feuilles disponibles : {xlsx.sheet_names}")
    print()

    # Analyser chaque feuille
    for sheet_name in xlsx.sheet_names:
        print("{'=' * 80}")
        print("📄 FEUILLE : {sheet_name}")
        print("{'=' * 80}")

        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        print("📊 Nombre de lignes : {len(df)}")
        print("📊 Nombre de colonnes : {len(df.columns)}")
        print()

        print("📋 Colonnes :")
        for _i, _col in enumerate(df.columns, 1):
            print("  {i}. {col}")
        print()

        print("👀 Aperçu des 3 premières lignes :")
        print(df.head(3).to_string())
        print()

        # Statistiques
        print("📈 Statistiques :")
        if "Date" in df.columns or "date" in df.columns:
            date_col = "Date" if "Date" in df.columns else "date"
            print("  - Dates : {df[date_col].min()} → {df[date_col].max()}")

        print("  - Valeurs nulles : {df.isnull().sum().sum()}")
        print()

except Exception:
    print("❌ Erreur de lecture : {e}")
    import traceback
    traceback.print_exc()

