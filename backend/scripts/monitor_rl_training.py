#!/usr/bin/env python3
"""Script de monitoring de l'entraînement RL."""
import re
import sys
from pathlib import Path

log_file = Path("data/rl/training_output.log")

if not log_file.exists():
    sys.exit(1)


# Lire le fichier de log
with Path(log_file, encoding="utf-8").open() as f:
    content = f.read()

# Extraire les épisodes
episodes = re.findall(r"Episode (\d+)/(\d+)", content)
if episodes:
    current, total = episodes[-1]
    progress = (int(current) / int(total)) * 100
else:
    pass

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

    # Trouver le meilleur écart
    best_gaps = re.findall(r"gap=(\d+\.\d+)", content)
    if best_gaps:
        best = min(float(g) for g in best_gaps)

# Vérifier si terminé
if "ENTRAÎNEMENT TERMINÉ" in content:
    pass
else:
    pass


