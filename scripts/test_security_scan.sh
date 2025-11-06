#!/bin/bash
# Script de test local pour bandit et semgrep
# Usage: ./scripts/test_security_scan.sh

set -euo pipefail

echo "🔒 Test scans de sécurité (Bandit + Semgrep)"
echo "=============================================="

# Installer outils si nécessaire
if ! command -v bandit &> /dev/null; then
    echo "📦 Installation de bandit..."
    pip install bandit
fi

if ! command -v semgrep &> /dev/null; then
    echo "📦 Installation de semgrep..."
    pip install semgrep
fi

mkdir -p artifacts

cd backend

echo ""
echo "1️⃣  Bandit (SAST Python)..."
bandit -r . -f json -o ../artifacts/bandit.json || true
echo "   ✅ Rapport JSON généré: artifacts/bandit.json"

# Scan avec affichage
echo "   Scan des vulnérabilités high/critical:"
if bandit -r . --severity-level high -q; then
    echo "   ✅ Aucune vulnérabilité high/critical trouvée"
else
    echo "   ⚠️  Vulnérabilités détectées (voir ci-dessus)"
fi

echo ""
echo "2️⃣  Semgrep (règles OWASP)..."
semgrep --config p/ci --config p/security-audit . --json -o ../artifacts/semgrep.json || true
echo "   ✅ Rapport JSON généré: artifacts/semgrep.json"

# Scan avec affichage
echo "   Scan des règles de sécurité:"
if semgrep --config p/ci --config p/security-audit . --error; then
    echo "   ✅ Aucune violation de sécurité trouvée"
else
    echo "   ⚠️  Violations détectées (voir ci-dessus)"
fi

echo ""
echo "=============================================="
echo "✅ Tests terminés. Rapports dans artifacts/"
echo "  - artifacts/bandit.json"
echo "  - artifacts/semgrep.json"

