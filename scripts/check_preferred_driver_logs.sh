#!/bin/bash
# Script pour filtrer les logs backend concernant le chauffeur préféré
# Usage: ./scripts/check_preferred_driver_logs.sh [dispatch_run_id] [preferred_driver_id]

DISPATCH_RUN_ID=${1:-334}
PREFERRED_DRIVER_ID=${2:-2}

echo "🔍 Recherche des logs pour dispatch_run_id=$DISPATCH_RUN_ID, preferred_driver_id=$PREFERRED_DRIVER_ID"
echo "======================================================================"
echo ""

# Option 1: Si logs dans Docker
if command -v docker &> /dev/null; then
    echo "📋 Logs depuis Docker (celery-worker):"
    echo "-----------------------------------"
    docker logs celery-worker 2>&1 | grep -i "preferred_driver\|🎯.*préféré\|preferred" | tail -50
    
    echo ""
    echo "📋 Logs depuis Docker (api):"
    echo "-----------------------------------"
    docker logs api 2>&1 | grep -i "preferred_driver\|🎯.*préféré\|preferred" | tail -50
fi

# Option 2: Si logs dans fichiers
if [ -d "logs" ]; then
    echo ""
    echo "📋 Logs depuis fichiers:"
    echo "-----------------------------------"
    find logs -name "*.log" -type f -exec grep -l "preferred_driver\|🎯.*préféré" {} \; | head -5 | while read file; do
        echo "Fichier: $file"
        grep -i "preferred_driver\|🎯.*préféré\|preferred" "$file" | tail -20
        echo ""
    done
fi

echo ""
echo "======================================================================"
echo "✅ Recherche terminée"
echo ""
echo "Messages clés à vérifier:"
echo "  - [Dispatch] 🎯 Chauffeur préféré CONFIGURÉ"
echo "  - [HEURISTIC] 🎯 assign() entry: preferred_driver_id"
echo "  - [HEURISTIC] 🎯 Bonus préférence FORT appliqué"
echo "  - [HEURISTIC] ✅ Booking → Chauffeur préféré"
echo "  - [FALLBACK] 🎯 Chauffeur préféré détecté"

