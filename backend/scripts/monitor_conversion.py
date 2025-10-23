#!/usr/bin/env python3
# ruff: noqa: T201
"""Script de monitoring de la conversion Excel"""
import re
from pathlib import Path

log_file = Path("data/rl/conversion_output.log")

if not log_file.exists():
    print("❌ Fichier de log non trouvé. La conversion n'a pas encore démarré.")
    exit(1)

print("=" * 80)
print("📊 MONITORING CONVERSION EXCEL")
print("=" * 80)
print()

# Lire le fichier de log
with open(log_file, encoding="utf-8") as f:
    content = f.read()

# Extraire la progression
progress = re.findall(r"Traité (\d+)/(\d+) courses", content)
if progress:
    current, total = progress[-1]
    pct = (int(current) / int(total)) * 100
    print(f"📈 Progression : {current}/{total} courses ({pct:.1f}%)")
else:
    print("⏳ Conversion en cours de démarrage...")

# Extraire les statistiques de géocodage
geocoding_success = len(re.findall(r"✅ Géocodage réussi", content))
geocoding_failed = len(re.findall(r"⚠️  Géocodage échoué", content))

if geocoding_success + geocoding_failed > 0:
    success_rate = (geocoding_success / (geocoding_success + geocoding_failed)) * 100
    print()
    print("🗺️  Géocodage :")
    print(f"  - Réussi : {geocoding_success}")
    print(f"  - Échoué : {geocoding_failed}")
    print(f"  - Taux   : {success_rate:.1f}%")

# Vérifier si terminé
if "DISPATCHES CRÉÉS" in content:
    print()
    print("✅ CONVERSION TERMINÉE !")
    print()

    # Extraire les statistiques finales
    dispatches = re.search(r"Total dispatches\s*:\s*(\d+)", content)
    bookings = re.search(r"Total bookings\s*:\s*(\d+)", content)
    avg_gap = re.search(r"Écart moyen\s*:\s*([\d.]+)", content)

    if dispatches:
        print("📊 Résultats :")
        print(f"  - Dispatches créés : {dispatches.group(1)}")
    if bookings:
        print(f"  - Total courses    : {bookings.group(1)}")
    if avg_gap:
        print(f"  - Écart moyen      : {avg_gap.group(1)} courses")
    print()
    print("🚀 Prochaine étape : Réentraîner le modèle RL !")
else:
    print()
    print("⏳ Conversion en cours...")
    print()
    print("📊 Pour suivre en temps réel :")
    print("   docker exec atmr-api-1 tail -f data/rl/conversion_output.log")

print()
print("=" * 80)

