#!/usr/bin/env python3
"""Monitoring de la conversion XLSB 1 année."""
import re
import sys
from pathlib import Path

log_file = Path("data/rl/conversion_full_year.log")

if not log_file.exists():
    print("❌ Fichier de log non trouvé.")
    sys.exit(1)

print("=" * 80)
print("📊 MONITORING CONVERSION 1 ANNÉE COMPLÈTE")
print("=" * 80)
print()

with Path(log_file, encoding="utf-8").open() as f:
    content = f.read()

# Feuilles traitées
sheets_done = len(re.findall(r"Traitement feuille", content))
print("📄 Feuilles traitées : {sheets_done}/12")

# Courses traitées
courses = re.findall(r"(\d+) courses traitées au total", content)
if courses:
    total_courses = courses[-1]
    print("📦 Courses traitées  : {total_courses}")

# Géocodage
geocoding_success = len(re.findall(r"✅ Géocodage réussi", content))
geocoding_failed = len(re.findall(r"⚠️  Géocodage échoué", content))

if geocoding_success + geocoding_failed > 0:
    success_rate = (geocoding_success / (geocoding_success + geocoding_failed)) * 100
    print("🗺️  Géocodage        : {geocoding_success} réussis, {geocoding_failed} échoués ({success_rate")

# Terminé ?
if "STATISTIQUES FINALES" in content:
    print()
    print("✅ CONVERSION TERMINÉE !")
    print()

    dispatches = re.search(r"Total dispatches\s*:\s*(\d+)", content)
    bookings = re.search(r"Total courses\s*:\s*(\d+)", content)
    avg_gap = re.search(r"Écart moyen\s*:\s*([\d.]+)", content)

    if dispatches:
        print("📊 Résultats :")
        print("  - Dispatches créés : {dispatches.group(1)}")
    if bookings:
        print("  - Total courses    : {bookings.group(1)}")
    if avg_gap:
        print("  - Écart moyen      : {avg_gap.group(1)} courses")
    print()
    print("🚀 Prêt pour réentraînement v3 (15,000 épisodes) !")
else:
    print()
    print("⏳ Conversion en cours (1-2h estimé)...")
    print()
    print("📊 Pour suivre :")
    print("   docker exec atmr-api-1 tail -f data/rl/conversion_full_year.log")

print()
print("=" * 80)

