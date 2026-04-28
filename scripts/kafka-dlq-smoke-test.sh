#!/usr/bin/env bash
# Vérifie la présence des topics tracking + envoie un message volontairement invalide
# (driver_id non entier) pour valider le chemin DLQ. À lancer dans un conteneur
# connecté au réseau atmr (ex. broker) :
#
#   docker exec -i atmr-kafka-broker-1 bash -s < scripts/kafka-dlq-smoke-test.sh
# ou, depuis l’hôte (Git Bash / WSL) :
#   BOOTSTRAP_SERVERS=localhost:9092 ./scripts/kafka-dlq-smoke-test.sh
set -euo pipefail

BOOTSTRAP="${BOOTSTRAP_SERVERS:-kafka-broker-1:29092}"
TOPIC_RAW="${KAFKA_TOPIC_DRIVER_LOCATION_RAW:-driver.location.raw}"
TOPIC_DLQ="${KAFKA_TOPIC_DRIVER_LOCATION_DLQ:-driver.location.dlq}"

echo "== Topics (grep driver.location) =="
kafka-topics --bootstrap-server "${BOOTSTRAP}" --list 2>/dev/null | grep 'driver\.location' || {
  echo "Aucun topic driver.location listé (bootstrap=${BOOTSTRAP})"
  exit 1
}

echo "== Produit 1 message invalide sur ${TOPIC_RAW} =="
# Payload volontairement rejeté par ingest_consumer (driver_id non int)
echo '{"driver_id":"smoke-test-invalid","payload":{}}' | kafka-console-producer \
  --bootstrap-server "${BOOTSTRAP}" \
  --topic "${TOPIC_RAW}" 2>/dev/null

echo "== Attendre 5s (consumer → DLQ) =="
sleep 5

echo "== (Optionnel) premier enregistrement sur ${TOPIC_DLQ} =="
kafka-console-consumer \
  --bootstrap-server "${BOOTSTRAP}" \
  --topic "${TOPIC_DLQ}" \
  --from-beginning \
  --max-messages 1 \
  2>/dev/null | head -1 || true

echo "Fumée terminée. Vérifier: docker logs atmr-tracking-kafka-consumer-1 2>&1 | tail -30"
