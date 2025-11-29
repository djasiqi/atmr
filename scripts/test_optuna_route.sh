#!/bin/bash
# Script pour tester la route Optuna après déploiement
# Usage: ./scripts/test_optuna_route.sh

set -e

echo "🧪 Test de la route Optuna..."
echo ""

BACKEND_CONTAINER="atmr-backend"
ROUTE="/api/v1/admin/optuna/optimize"

# 1. Vérifier que le backend est running
echo "1️⃣  Vérification que le backend est running..."
if ! docker ps | grep -q "$BACKEND_CONTAINER.*Up"; then
    echo "❌ Le backend n'est pas running"
    exit 1
fi
echo "✅ Backend est running"

# 2. Vérifier que la route existe dans Flask
echo ""
echo "2️⃣  Vérification que la route existe dans Flask..."
ROUTE_EXISTS=$(docker exec "$BACKEND_CONTAINER" python3 -c "
from app import create_app
app = create_app('production')
with app.app_context():
    routes = [str(rule) for rule in app.url_map.iter_rules()]
    optuna_routes = [r for r in routes if 'optuna' in r.lower()]
    print('YES' if optuna_routes else 'NO')
    for r in optuna_routes:
        print(f'  {r}')
" 2>/dev/null || echo "ERROR")

if echo "$ROUTE_EXISTS" | grep -q "YES\|optuna"; then
    echo "✅ Route Optuna trouvée dans Flask"
    echo "$ROUTE_EXISTS" | grep "optuna" || true
else
    echo "❌ Route Optuna NON trouvée dans Flask"
    echo "💡 La route n'est pas enregistrée dans l'application"
    exit 1
fi

# 3. Tester la route (sans authentification pour voir le type d'erreur)
echo ""
echo "3️⃣  Test de la route (sans authentification)..."

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "http://localhost:5000${ROUTE}" \
    -H "Content-Type: application/json" \
    -d '{}' 2>/dev/null || echo "000")

echo "HTTP Status Code: $HTTP_CODE"

case "$HTTP_CODE" in
    404)
        echo "❌ 404 Not Found - La route n'existe pas"
        echo "💡 Vérifiez que la nouvelle image contient bien la route"
        ;;
    403)
        echo "⚠️  403 Forbidden - La route existe mais accès refusé"
        echo "💡 Probablement bloqué par IP whitelist ou permissions"
        echo "💡 C'est normal, la route fonctionne !"
        ;;
    401)
        echo "✅ 401 Unauthorized - La route existe mais demande authentification"
        echo "✅ C'est normal ! La route fonctionne correctement"
        ;;
    422|400)
        echo "⚠️  $HTTP_CODE - La route existe mais erreur de validation"
        echo "💡 C'est normal, il manque des paramètres requis"
        ;;
    000)
        echo "❌ Impossible de se connecter au backend"
        echo "💡 Vérifiez que le backend écoute sur le port 5000"
        ;;
    *)
        echo "⚠️  Status $HTTP_CODE - Réponse inattendue"
        ;;
esac

# 4. Vérifier les logs pour voir si la route a été appelée
echo ""
echo "4️⃣  Dernières lignes des logs du backend..."
docker logs "$BACKEND_CONTAINER" --tail 10 | grep -i "optuna\|admin" || echo "  Pas de logs Optuna récents"

echo ""
echo "✅ Test terminé !"
echo ""
echo "📋 Résumé :"
echo "   - Si vous voyez 401 ou 403 : La route fonctionne ✅"
echo "   - Si vous voyez 404 : La route n'existe pas ❌"
echo ""
echo "💡 Pour tester avec authentification :"
echo "   curl -X POST http://localhost:5000/api/v1/admin/optuna/optimize \\"
echo "     -H 'Authorization: Bearer YOUR_TOKEN' \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"company_id\": 1, \"n_trials\": 10}'"

