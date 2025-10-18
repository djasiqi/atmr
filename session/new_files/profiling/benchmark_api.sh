#!/bin/bash
# Benchmark API endpoints with wrk (HTTP benchmarking tool)
# Usage: ./benchmark_api.sh [JWT_TOKEN]

set -e

# Configuration
API_URL="${API_URL:-http://localhost:5000}"
JWT_TOKEN="${1:-$JWT_TOKEN}"
DURATION="${DURATION:-30s}"
THREADS="${THREADS:-4}"
CONNECTIONS="${CONNECTIONS:-100}"

if [[ -z "$JWT_TOKEN" ]]; then
    echo "❌ JWT_TOKEN requis"
    echo "Usage: $0 <JWT_TOKEN>"
    echo "   ou: JWT_TOKEN=xxx $0"
    exit 1
fi

echo "🔥 ATMR API Benchmarks"
echo "======================"
echo "API: $API_URL"
echo "Duration: $DURATION"
echo "Threads: $THREADS"
echo "Connections: $CONNECTIONS"
echo ""

# ============================================================
# Test 1: GET /api/bookings (query with filters)
# ============================================================
echo "📊 Test 1: GET /api/bookings?date=2025-10-20"
echo "-------------------------------------------"
wrk -t$THREADS -c$CONNECTIONS -d$DURATION --latency \
  -H "Authorization: Bearer $JWT_TOKEN" \
  "$API_URL/api/bookings?date=2025-10-20"

echo ""
echo "✅ Critères d'acceptation (APRÈS patches):"
echo "   - Latency p50: <70ms"
echo "   - Latency p95: <95ms"
echo "   - Latency p99: <180ms"
echo "   - Requests/sec: >450"
echo ""
read -p "Appuyez sur Entrée pour continuer..."

# ============================================================
# Test 2: GET /api/drivers
# ============================================================
echo ""
echo "📊 Test 2: GET /api/drivers"
echo "--------------------------"
wrk -t$THREADS -c$CONNECTIONS -d$DURATION --latency \
  -H "Authorization: Bearer $JWT_TOKEN" \
  "$API_URL/api/drivers"

echo ""
echo "✅ Critères d'acceptation:"
echo "   - Latency p95: <80ms"
echo "   - Requests/sec: >600"
echo ""
read -p "Appuyez sur Entrée pour continuer..."

# ============================================================
# Test 3: GET /api/companies/me
# ============================================================
echo ""
echo "📊 Test 3: GET /api/companies/me"
echo "--------------------------------"
wrk -t$THREADS -c$CONNECTIONS -d$DURATION --latency \
  -H "Authorization: Bearer $JWT_TOKEN" \
  "$API_URL/api/companies/me"

echo ""
echo "✅ Critères d'acceptation:"
echo "   - Latency p95: <100ms"
echo "   - Requests/sec: >800"
echo ""

# ============================================================
# Test 4: POST /api/auth/login (stress test)
# ============================================================
echo ""
echo "📊 Test 4: POST /api/auth/login"
echo "-------------------------------"
echo "⚠️  Note: Test avec connexions modérées (c=10) car login coûteux (bcrypt)"

# Create Lua script for POST
cat > /tmp/login_post.lua <<'EOF'
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"email":"test@test.com","password":"password123"}'
EOF

wrk -t2 -c10 -d30s --latency \
  -s /tmp/login_post.lua \
  "$API_URL/api/auth/login"

rm /tmp/login_post.lua

echo ""
echo "✅ Critères d'acceptation:"
echo "   - Latency p95: <300ms (bcrypt intentionnellement lent)"
echo "   - Requests/sec: >60"
echo "   - Error rate: <1%"
echo ""

# ============================================================
# Résumé
# ============================================================
echo ""
echo "=========================================="
echo "📈 RÉSUMÉ DES BENCHMARKS"
echo "=========================================="
echo ""
echo "Objectifs globaux (APRÈS patches):"
echo "  - GET /api/bookings : p95 < 120ms (-62% vs avant)"
echo "  - GET /api/drivers : p95 < 80ms"
echo "  - GET /api/companies/me : p95 < 100ms"
echo "  - POST /api/auth/login : p95 < 300ms"
echo ""
echo "Si tous les objectifs atteints : ✅ VALIDÉ"
echo ""

