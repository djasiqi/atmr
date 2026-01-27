#!/bin/bash
# ✅ Script de test pour vérifier l'allowlist IP de /metrics
#
# Usage:
#   ./scripts/test-metrics-allowlist.sh <host> [prometheus_ip]
#
# Exemples:
#   ./scripts/test-metrics-allowlist.sh https://api.example.com
#   ./scripts/test-metrics-allowlist.sh http://localhost:5000 172.17.0.1

set -e

HOST="${1:-http://localhost:5000}"
PROMETHEUS_IP="${2:-}"

METRICS_ENDPOINT="${HOST}/api/v1/prometheus/metrics"

echo "🔍 Test allowlist IP pour /metrics"
echo "=================================="
echo "Host: ${HOST}"
echo "Endpoint: ${METRICS_ENDPOINT}"
echo ""

# ✅ Test 1: Depuis un poste non autorisé (doit être 403/404)
echo "📋 Test 1: Accès depuis IP non autorisée (attendu: 403 ou 404)"
echo "------------------------------------------------------------"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -I "${METRICS_ENDPOINT}" 2>&1 || echo "000")
if [ "${RESPONSE}" = "403" ] || [ "${RESPONSE}" = "404" ]; then
    echo "✅ PASS: Accès refusé (HTTP ${RESPONSE})"
else
    echo "❌ FAIL: Accès autorisé (HTTP ${RESPONSE}) - L'allowlist ne fonctionne pas !"
    exit 1
fi
echo ""

# ✅ Test 2: Depuis la machine Prometheus (si IP fournie)
if [ -n "${PROMETHEUS_IP}" ]; then
    echo "📋 Test 2: Accès depuis IP Prometheus (attendu: 200)"
    echo "----------------------------------------------------"
    echo "⚠️  Note: Ce test nécessite d'être exécuté depuis la machine Prometheus"
    echo "   ou avec un proxy qui simule l'IP source ${PROMETHEUS_IP}"
    echo "   Pour tester réellement, exécutez depuis Prometheus:"
    echo "   curl -I ${METRICS_ENDPOINT}"
    echo ""
else
    echo "📋 Test 2: Accès depuis IP Prometheus (skippé - IP non fournie)"
    echo "----------------------------------------------------"
    echo "   Pour tester, exécutez depuis la machine Prometheus:"
    echo "   curl -I ${METRICS_ENDPOINT}"
    echo ""
fi

# ✅ Test 3: Vérifier que le contenu est bien des métriques Prometheus
echo "📋 Test 3: Vérification contenu (si accès autorisé)"
echo "---------------------------------------------------"
echo "⚠️  Note: Ce test nécessite un accès autorisé"
echo "   Vérifiez manuellement que le contenu contient:"
echo "   - '# HELP invoice_pdf_generation_ms'"
echo "   - '# TYPE invoice_pdf_generation_ms histogram'"
echo "   - 'invoice_pdf_generation_ms_count'"
echo ""

# ✅ Test 4: Vérifier headers de sécurité
echo "📋 Test 4: Headers de sécurité"
echo "-------------------------------"
HEADERS=$(curl -s -I "${METRICS_ENDPOINT}" 2>&1 || echo "")
if echo "${HEADERS}" | grep -q "X-Content-Type-Options"; then
    echo "✅ PASS: Header X-Content-Type-Options présent"
else
    echo "⚠️  WARN: Header X-Content-Type-Options absent (optionnel)"
fi
echo ""

echo "✅ Tests terminés"
echo ""
echo "📝 Checklist manuelle:"
echo "  [ ] Test 1: Accès refusé depuis IP non autorisée (403/404)"
echo "  [ ] Test 2: Accès autorisé depuis Prometheus (200)"
echo "  [ ] Test 3: Contenu Prometheus valide (si accès autorisé)"
echo "  [ ] Test 4: Headers de sécurité présents"
