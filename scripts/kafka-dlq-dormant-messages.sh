#!/usr/bin/env bash
# Runbook P1-3 : traiter les messages dormants sur driver.location.dlq.v2
# (typiquement 2 messages sur partition 1 — exécution MANUELLE en prod).
#
# Usage (depuis un conteneur broker ou hôte avec accès Kafka) :
#   BOOTSTRAP_SERVERS=kafka-broker-1:29092 ./scripts/kafka-dlq-dormant-messages.sh inspect
#   BOOTSTRAP_SERVERS=kafka-broker-1:29092 ./scripts/kafka-dlq-dormant-messages.sh archive
#
# ⚠️ Ne pas exécuter en CI — validation SSH manuelle requise.
set -euo pipefail

BOOTSTRAP="${BOOTSTRAP_SERVERS:-kafka-broker-1:29092}"
TOPIC_DLQ="${KAFKA_TOPIC_DRIVER_LOCATION_DLQ:-driver.location.dlq.v2}"
ARCHIVE_DIR="${KAFKA_DLQ_ARCHIVE_DIR:-/tmp/kafka-dlq-archive}"
ACTION="${1:-inspect}"

mkdir -p "${ARCHIVE_DIR}"

echo "== DLQ dormant messages runbook =="
echo "bootstrap=${BOOTSTRAP} topic=${TOPIC_DLQ} action=${ACTION}"

echo "== Offsets consumer groups sur ${TOPIC_DLQ} =="
kafka-consumer-groups --bootstrap-server "${BOOTSTRAP}" \
  --describe --all-groups 2>/dev/null | grep "${TOPIC_DLQ}" || true

echo "== Messages DLQ (max 20, from-beginning) =="
kafka-console-consumer \
  --bootstrap-server "${BOOTSTRAP}" \
  --topic "${TOPIC_DLQ}" \
  --from-beginning \
  --timeout-ms 5000 \
  --max-messages 20 2>/dev/null | tee "${ARCHIVE_DIR}/dlq-sample-$(date +%Y%m%d-%H%M%S).jsonl" || true

case "${ACTION}" in
  inspect)
    echo "Inspection terminée. Analyser le fichier archivé dans ${ARCHIVE_DIR}."
    echo "Si messages obsolètes (>7j) et déjà traités manuellement : voir action=archive."
    ;;
  archive)
    echo "Archivage terminé (lecture seule). Pour purger après validation humaine :"
    echo "  kafka-delete-records --bootstrap-server ${BOOTSTRAP} \\"
    echo "    --offset-json-file <offsets-before-end>.json"
    echo "⚠️ Purge destructive — ne pas exécuter sans accord ops."
    ;;
  *)
    echo "Action inconnue: ${ACTION} (inspect|archive)"
    exit 1
    ;;
esac

echo ""
echo "== R7 — Post-alerte TrackingKafkaDlqForceCommit =="
echo "Si force_commit s'est produit sur l'ingest (pas la DLQ topic) :"
echo "  - La position GPS est DÉJÀ PERDUE (commit après échec DLQ)."
echo "  - Aucune récupération / rejeu possible."
echo "  - Enquête : corréler driver_id dans logs CRITICAL ingest_consumer."
echo "  - Vérifier que la prochaine position mobile comble le trou."
echo "  - Si trou > 5 min : informer exploitants (lacune tracé)."
