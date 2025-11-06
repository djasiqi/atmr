#!/usr/bin/env bash
# enable_chaos.sh
# Script utilitaire pour activer le chaos engineering en développement/test
# ⚠️ NE JAMAIS UTILISER EN PRODUCTION !

set -euo pipefail

CHAOS_TYPE="${1:-all}"

echo "⚠️  ACTIVATION DU CHAOS ENGINEERING"
echo "Type: $CHAOS_TYPE"
echo ""

# Vérifier que ce n'est pas en production
ENV_CHECK="${FLASK_ENV:-development}"
if [ "$ENV_CHECK" = "production" ]; then
    echo "❌ ERREUR: Tentative d'activer le chaos en PRODUCTION !"
    echo "   Le chaos ne doit JAMAIS être activé en production."
    exit 1
fi

# Options disponibles
case "$CHAOS_TYPE" in
    all)
        export CHAOS_ENABLED=true
        export CHAOS_OSRM_DOWN=false  # Pas down par défaut (juste lent)
        export CHAOS_DB_READ_ONLY=false
        echo "✅ Chaos activé (mode général)"
        echo "   - Latence/erreurs réseau: configurable via injector"
        ;;
    osrm-down)
        export CHAOS_ENABLED=true
        export CHAOS_OSRM_DOWN=true
        export CHAOS_DB_READ_ONLY=false
        echo "✅ Chaos activé: OSRM DOWN"
        ;;
    db-readonly)
        export CHAOS_ENABLED=true
        export CHAOS_OSRM_DOWN=false
        export CHAOS_DB_READ_ONLY=true
        echo "✅ Chaos activé: DB READ-ONLY"
        ;;
    disable)
        export CHAOS_ENABLED=false
        export CHAOS_OSRM_DOWN=false
        export CHAOS_DB_READ_ONLY=false
        echo "✅ Chaos désactivé"
        ;;
    *)
        echo "Usage: $0 [all|osrm-down|db-readonly|disable]"
        echo ""
        echo "Options:"
        echo "  all        - Activer chaos général (latence/erreurs)"
        echo "  osrm-down  - Simuler OSRM down"
        echo "  db-readonly - Simuler DB read-only"
        echo "  disable    - Désactiver tout le chaos"
        exit 1
        ;;
esac

echo ""
echo "📝 Pour appliquer les changements, redémarrer les services:"
echo "   docker-compose restart api celery-worker"
echo ""
echo "⚠️  Vérifier les logs pour confirmer l'activation du chaos."

