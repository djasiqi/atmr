#!/usr/bin/env bash
# Re-génère les artefacts ML avec la version sklearn du runtime (évite InconsistentVersionWarning).
#
# Les modèles sont embarqués dans l'image backend (pas de .pkl versionnés dans ce repo).
# Exécuter dans l'environnement d'entraînement CI ou localement avant release :
#
#   pip install -r backend/requirements-rl.txt  # ou requirements ML
#   python -c "import sklearn; print(sklearn.__version__)"
#   # Re-entraîner / re-exporter vers data/ml/models/ puis rebuild image backend
#
# Vérification post-build :
#   docker compose run --rm backend python -c "import sklearn; print(sklearn.__version__)"

set -euo pipefail

echo "Ce script documente le processus de re-pickle sklearn."
echo "Aucun .pkl n'est stocké dans le dépôt git — régénérer via le pipeline RL/ML habituel."
echo "Runtime cible : sklearn 1.9.x (aligné requirements backend)."
