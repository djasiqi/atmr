#!/bin/bash
# Script bash pour exécuter les tests pytest avec les options optimisées
# Usage: ./run_tests.sh [options]
#
# Options:
#   -x          Arrêter au premier échec (mode développement)
#   -k PATTERN  Exécuter seulement les tests correspondant au pattern
#   -m MARKER   Exécuter seulement les tests avec le marqueur (ex: -m unit)
#   --cov       Activer le coverage (défaut: activé)
#   --no-cov    Désactiver le coverage
#   --html      Ouvrir le rapport HTML de coverage après les tests

set -e

# Variables par défaut
STOP_ON_FIRST_FAIL=false
PATTERN=""
MARKER=""
COVERAGE=true
OPEN_HTML=false

# Parser les arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -x)
            STOP_ON_FIRST_FAIL=true
            shift
            ;;
        -k)
            PATTERN="$2"
            shift 2
            ;;
        -m)
            MARKER="$2"
            shift 2
            ;;
        --cov)
            COVERAGE=true
            shift
            ;;
        --no-cov)
            COVERAGE=false
            shift
            ;;
        --html)
            OPEN_HTML=true
            shift
            ;;
        *)
            echo "Option inconnue: $1"
            echo "Usage: $0 [-x] [-k PATTERN] [-m MARKER] [--cov|--no-cov] [--html]"
            exit 1
            ;;
    esac
done

# Construire la commande pytest
PYTEST_ARGS=("backend/tests")

# Options de base
if [ "$STOP_ON_FIRST_FAIL" = true ]; then
    PYTEST_ARGS+=("-x")
fi

if [ -n "$PATTERN" ]; then
    PYTEST_ARGS+=("-k" "$PATTERN")
fi

if [ -n "$MARKER" ]; then
    PYTEST_ARGS+=("-m" "$MARKER")
fi

# Options de coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS+=("--cov=backend")
    PYTEST_ARGS+=("--cov-report=xml:backend/coverage.xml")
    PYTEST_ARGS+=("--cov-report=html:backend/htmlcov")
    PYTEST_ARGS+=("--cov-report=term-missing")
fi

echo "🧪 Exécution des tests pytest..."
echo "Command: pytest ${PYTEST_ARGS[*]}"

# Exécuter pytest
pytest "${PYTEST_ARGS[@]}"

# Ouvrir le rapport HTML si demandé
if [ "$OPEN_HTML" = true ] && [ "$COVERAGE" = true ]; then
    HTML_PATH="backend/htmlcov/index.html"
    if [ -f "$HTML_PATH" ]; then
        echo ""
        echo "📊 Ouverture du rapport de coverage..."
        if command -v xdg-open > /dev/null; then
            xdg-open "$HTML_PATH"
        elif command -v open > /dev/null; then
            open "$HTML_PATH"
        else
            echo "⚠️  Impossible d'ouvrir automatiquement le rapport HTML"
            echo "   Ouvrez manuellement: $HTML_PATH"
        fi
    else
        echo "⚠️  Rapport HTML de coverage non trouvé: $HTML_PATH"
    fi
fi

