#!/bin/bash
# Script pour lancer l'entraînement RL en arrière-plan avec logging

echo "🚀 Lancement de l'entraînement RL (5000 épisodes)..."
echo "📂 Les logs seront sauvegardés dans: data/rl/training_output.log"
echo ""

# Créer le répertoire si nécessaire
mkdir -p data/rl/models

# Lancer l'entraînement en arrière-plan avec nohup
nohup python -u backend/scripts/rl_train_offline.py > data/rl/training_output.log 2>&1 &

# Récupérer le PID
PID=$!
echo "✅ Entraînement lancé en arrière-plan (PID: $PID)"
echo "$PID" > data/rl/training.pid
echo ""
echo "📊 Pour suivre la progression :"
echo "   tail -f data/rl/training_output.log"
echo ""
echo "🛑 Pour arrêter l'entraînement :"
echo "   kill $PID"
echo ""
echo "⏱️  Durée estimée : 2-3 heures"
echo "📈 Le modèle sera sauvegardé tous les 100 épisodes dans: data/rl/models/"

