#!/usr/bin/env bash
# Runbook P1-4 : purger les topics legacy v1 (driver.location.raw/processed/dlq sans .v2)
# ⚠️ DESTRUCTIF — exécution MANUELLE en prod uniquement après validation.
#
# Usage :
#   ./scripts/kafka-purge-legacy-v1.sh check
#   CONFIRM_PURGE=YES ./scripts/kafka-purge-legacy-v1.sh purge
set -euo pipefail

BOOTSTRAP="${BOOTSTRAP_SERVERS:-kafka-broker-1:29092}"
LEGACY_TOPICS=(
  "driver.location.raw"
  "driver.location.processed"
  "driver.location.dlq"
)
V2_TOPICS=(
  "driver.location.raw.v2"
  "driver.location.processed.v2"
  "driver.location.dlq.v2"
)
ACTION="${1:-check}"

echo "== Purge topics legacy v1 (P1-4) =="
echo "bootstrap=${BOOTSTRAP}"

echo "== Topics v2 actifs (doivent exister) =="
for t in "${V2_TOPICS[@]}"; do
  kafka-topics --bootstrap-server "${BOOTSTRAP}" --describe --topic "${t}" 2>/dev/null \
    && echo "  [OK] ${t}" || echo "  [MISSING] ${t}"
done

echo "== Consumer groups sur legacy v1 (doivent être vides ou absents) =="
for t in "${LEGACY_TOPICS[@]}"; do
  echo "--- ${t} ---"
  kafka-consumer-groups --bootstrap-server "${BOOTSTRAP}" \
    --describe --all-groups 2>/dev/null | grep "${t}" || echo "  (aucun consumer)"
done

case "${ACTION}" in
  check)
    echo ""
    echo "Vérification terminée. Si aucun consumer actif sur v1 et v2 opérationnels :"
    echo "  CONFIRM_PURGE=YES ./scripts/kafka-purge-legacy-v1.sh purge"
    ;;
  purge)
    if [[ "${CONFIRM_PURGE:-}" != "YES" ]]; then
      echo "Refusé : définir CONFIRM_PURGE=YES pour confirmer la suppression."
      exit 1
    fi
    for t in "${LEGACY_TOPICS[@]}"; do
      echo "Suppression ${t}..."
      kafka-topics --bootstrap-server "${BOOTSTRAP}" --delete --topic "${t}" 2>/dev/null \
        && echo "  [DELETED] ${t}" || echo "  [SKIP] ${t} (absent ou erreur)"
    done
    echo "Purge terminée."
    ;;
  *)
    echo "Action inconnue: ${ACTION} (check|purge)"
    exit 1
    ;;
esac
