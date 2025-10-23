#!/usr/bin/env python3
"""Script de monitoring de l'entraînement RL"""
import re
import time
from pathlib import Path

log_file = Path("data/rl/training_output.log")

if not log_file.exists():
    print("❌ Fichier de log non trouvé. L'entraînement n'a pas encore démarré.")
    exit(1)

print("=" * 80)
print("📊 MONITORING ENTRAÎNEMENT RL")
print("=" * 80)
print()

# Lire le fichier de log
with open(log_file, encoding="utf-8") as f:
    content = f.read()

# Extraire les épisodes
episodes = re.findall(r"Episode (\d+)/(\d+)", content)
if episodes:
    current, total = episodes[-1]
    progress = (int(current) / int(total)) * 100
    print(f"📈 Progression : {current}/{total} épisodes ({progress:.1f}%)")
else:
    print("⏳ Entraînement en cours de démarrage...")

# Extraire les dernières métriques
metrics = re.findall(
    r"Avg Reward\s*:\s*([-+]?\d+\.\d+).*?"
    r"Avg Load Gap\s*:\s*(\d+\.\d+).*?"
    r"Avg Distance\s*:\s*(\d+\.\d+)",
    content,
    re.DOTALL,
)

if metrics:
    reward, gap, distance = metrics[-1]
    print()
    print("📊 Dernières métriques (100 derniers épisodes):")
    print(f"   - Récompense moyenne : {reward}")
    print(f"   - Écart moyen        : {gap} courses")
    print(f"   - Distance moyenne   : {distance} km")
    print()
    
    # Trouver le meilleur écart
    best_gaps = re.findall(r"gap=(\d+\.\d+)", content)
    if best_gaps:
        best = min(float(g) for g in best_gaps)
        print(f"🏆 Meilleur écart atteint : {best:.2f} courses")
        print()

# Vérifier si terminé
if "ENTRAÎNEMENT TERMINÉ" in content:
    print("✅ ENTRAÎNEMENT TERMINÉ !")
    print()
    print("📂 Modèle sauvegardé dans : data/rl/models/dispatch_optimized_v1.pth")
else:
    print("⏳ Entraînement en cours...")
    print()
    print("📊 Pour suivre en temps réel :")
    print("   tail -f data/rl/training_output.log")

print()
print("=" * 80)

