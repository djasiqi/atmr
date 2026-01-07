#!/bin/bash
# Script Bash pour valider les règles architecturales
# Usage: ./run-architecture-lint.sh

echo "🏗️ Validation des règles architecturales ATMR"
echo "================================================"
echo ""

# Vérifier si Semgrep est installé
if ! command -v semgrep &> /dev/null; then
    echo "❌ Semgrep n'est pas installé."
    echo "   Installation: pip install semgrep"
    exit 1
fi

echo "✅ Semgrep détecté"
echo ""

# Chemin vers les règles
RULES_PATH=".semgrep/rules/architecture.yml"

if [ ! -f "$RULES_PATH" ]; then
    echo "❌ Fichier de règles introuvable: $RULES_PATH"
    exit 1
fi

echo "📋 Règles chargées: $RULES_PATH"
echo ""

# Scanner les bounded contexts
echo "🔍 Scan des Bounded Contexts..."
echo "   - bookings/"
echo "   - drivers/"
echo "   - dispatch/"
echo "   - companies/"
echo ""

SCAN_PATHS="bookings drivers dispatch companies"

# Exécuter Semgrep
if semgrep --config="$RULES_PATH" $SCAN_PATHS --error --no-rewrite-rule-ids; then
    echo ""
    echo "✅ Aucune violation détectée !"
    echo ""
    exit 0
else
    echo ""
    echo "⚠️ Violations détectées. Voir ci-dessus pour les détails."
    echo ""
    echo "📖 Voir: docs/ARCHITECTURE_RULES.md"
    echo ""
    exit 1
fi

