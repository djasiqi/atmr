#!/usr/bin/env python3
# ruff: noqa: T201
"""Analyse du fichier Excel de transport historique"""
import pandas as pd
import sys
from pathlib import Path

excel_file = Path("transport.xlsx")

if not excel_file.exists():
    print(f"❌ Fichier non trouvé: {excel_file}")
    sys.exit(1)

print("=" * 80)
print("📊 ANALYSE DU FICHIER EXCEL")
print("=" * 80)
print(f"📂 Fichier : {excel_file.absolute()}")
print()

# Lire le fichier Excel
try:
    # Essayer de lire toutes les feuilles
    xlsx = pd.ExcelFile(excel_file)
    print(f"📑 Feuilles disponibles : {xlsx.sheet_names}")
    print()
    
    # Analyser chaque feuille
    for sheet_name in xlsx.sheet_names:
        print(f"{'=' * 80}")
        print(f"📄 FEUILLE : {sheet_name}")
        print(f"{'=' * 80}")
        
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        print(f"📊 Nombre de lignes : {len(df)}")
        print(f"📊 Nombre de colonnes : {len(df.columns)}")
        print()
        
        print("📋 Colonnes :")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        print()
        
        print("👀 Aperçu des 3 premières lignes :")
        print(df.head(3).to_string())
        print()
        
        # Statistiques
        print("📈 Statistiques :")
        if 'Date' in df.columns or 'date' in df.columns:
            date_col = 'Date' if 'Date' in df.columns else 'date'
            print(f"  - Dates : {df[date_col].min()} → {df[date_col].max()}")
        
        print(f"  - Valeurs nulles : {df.isnull().sum().sum()}")
        print()
        
except Exception as e:
    print(f"❌ Erreur de lecture : {e}")
    import traceback
    traceback.print_exc()

