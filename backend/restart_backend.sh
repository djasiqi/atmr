#!/bin/bash
# Script de redémarrage rapide du backend Flask

echo "🔧 Redémarrage du backend Flask ATMR..."
echo ""

# Aller dans le répertoire backend
cd "$(dirname "$0")" || exit 1

# Vérifier si un processus Flask est en cours
echo "🔍 Vérification des processus Flask en cours..."
if pgrep -f "flask run" > /dev/null || pgrep -f "python app.py" > /dev/null; then
    echo "⚠️  Un processus Flask est déjà en cours. Arrêt en cours..."
    pkill -f "flask run"
    pkill -f "python app.py"
    sleep 2
    echo "✅ Processus Flask arrêté"
else
    echo "ℹ️  Aucun processus Flask actif détecté"
fi

echo ""
echo "🚀 Démarrage du backend Flask..."
echo ""
echo "📍 Répertoire de travail: $(pwd)"
echo "🐍 Version Python:"
python --version
echo ""

# Activer l'environnement virtuel si présent
if [ -d "venv" ]; then
    echo "🔧 Activation de l'environnement virtuel..."
    source venv/bin/activate
    echo "✅ Environnement virtuel activé"
elif [ -d "../venv" ]; then
    echo "🔧 Activation de l'environnement virtuel..."
    source ../venv/bin/activate
    echo "✅ Environnement virtuel activé"
else
    echo "⚠️  Aucun environnement virtuel détecté"
fi

echo ""
echo "🎯 Lancement de Flask..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Démarrer Flask
python app.py

# Si le script est arrêté (Ctrl+C), nettoyer
echo ""
echo "🛑 Backend arrêté"
