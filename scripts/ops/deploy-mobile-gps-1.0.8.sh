#!/usr/bin/env bash
# Déploiement mobile Lirie 1.0.8 — chaîne GPS durable (flags progressifs).
#
# Usage (depuis la racine du repo) :
#   bash scripts/ops/deploy-mobile-gps-1.0.8.sh phase1          # build store + submit
#   bash scripts/ops/deploy-mobile-gps-1.0.8.sh phase2-ota      # OTA cascade
#   bash scripts/ops/deploy-mobile-gps-1.0.8.sh phase3-ota      # OTA FSM
#   bash scripts/ops/deploy-mobile-gps-1.0.8.sh apk             # APK interne QA
#
# Prérequis : eas-cli, compte Expo, credentials Play/App Store configurés.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP_DIR="${ROOT}/mobile/unified-app"
PHASE="${1:-phase1}"

cd "${APP_DIR}"

echo "=== Déploiement GPS mobile 1.0.8 — phase=${PHASE} ==="
echo "Version app.json : $(node -p "require('./app.json').expo.version")"

case "${PHASE}" in
  phase1)
    echo "--- Phase 1 : build store (self-heal ON, cascade OFF, FSM OFF) ---"
    echo "Profil EAS : production"
    eas build --platform all --profile production --non-interactive
    echo ""
    echo "Après validation QA interne (APK preview ou production-apk) :"
    echo "  eas submit --platform all --profile production --latest"
    ;;
  phase2-ota)
    echo "--- Phase 2 : OTA recovery cascade (48h après phase 1 stable) ---"
    eas update --channel production-gps-phase2 --message "GPS phase2: recovery cascade" --non-interactive
    ;;
  phase3-ota)
    echo "--- Phase 3 : OTA FSM shadow (7j après phase 2 stable) ---"
    eas update --channel production-gps-phase3 --message "GPS phase3: FSM enabled" --non-interactive
    ;;
  apk)
    echo "--- APK interne QA (panel tracking activé) ---"
    eas build --platform android --profile production-apk --non-interactive
    ;;
  *)
    echo "Phase inconnue : ${PHASE}"
    echo "Valeurs : phase1 | phase2-ota | phase3-ota | apk"
    exit 1
    ;;
esac

echo ""
echo "✅ Commande terminée. Enchaîner avec :"
echo "  bash scripts/prod-tracking-gps-validation.sh   (checklist prod 8 points)"
