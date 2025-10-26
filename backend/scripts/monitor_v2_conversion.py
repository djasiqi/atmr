#!/usr/bin/env python3
"""Monitoring de la conversion XLSB v2."""
import re
import sys
from pathlib import Path

log_file = Path("data/rl/conversion_full_year_v2.log")

if not log_file.exists():
    print("❌ Fichier de log v2 non trouvé.")
    sys.exit(1)

print("=" * 80)
print("📊 MONITORING CONVERSION V2 (1 ANNÉE COMPLÈTE)")
print("=" * 80)
print()

with Path(log_file, encoding="utf-8").open() as f:
    content = f.read()

# Feuilles traitées
sheets_done = len(re.findall(r"Traitement feuille", content))
print("📄 Feuilles traitées : {sheets_done}/12")

# Courses traitées (dernière occurrence)
courses_matches = re.findall(r"(\d+) traitées", content)
if courses_matches:
    current_courses = courses_matches[-1]
    print("📦 Courses traitées  : {current_courses}")

    # Estimation
    total_est = 2500
    progress = (int(current_courses) / total_est) * 100
    print("📈 Progression       : {progress")

    remaining = total_est - int(current_courses)
    time_remaining_min = (remaining * 2) / 60  # 2 sec per course
    print("⏱️  Temps restant     : ~{int(time_remaining_min)} min")

# Géocodage
geocoding_success = len(re.findall(r"✅", content))
geocoding_failed = len(re.findall(r"⚠️", content))

if geocoding_success + geocoding_failed > 0:
    success_rate = (geocoding_success / (geocoding_success + geocoding_failed)) * 100
    print("🗺️  Géocodage        : {geocoding_success} réussis, {geocoding_failed} échoués ({success_rate")

# Terminé ?
if "RÉSUMÉ CONVERSION V2" in content:
    print()
    print("✅ CONVERSION TERMINÉE !")
    print()

    dispatches = re.search(r"Dispatches créés\s*:\s*(\d+)", content)
    bookings = re.search(r"Courses totales\s*:\s*(\d+)", content)

    if dispatches:
        print("📊 Résultats :")
        print("  - Dispatches créés : {dispatches.group(1)}")
    if bookings:
        print("  - Total courses    : {bookings.group(1)}")
    print()
    print("🚀 Lancer réentraînement v3 (15,000 épisodes) !")
else:
    print()
    print("⏳ Conversion en cours...")
    print()
    print("📊 Pour suivre :")
    print("   docker exec atmr-api-1 tail -f data/rl/conversion_full_year_v2.log")

print()
print("=" * 80)

