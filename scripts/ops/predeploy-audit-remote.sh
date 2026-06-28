#!/usr/bin/env bash
# Audit read-only pre-deploy — ne pas afficher les valeurs des secrets.
set -uo pipefail
ROOT=/srv/atmr
cd "$ROOT" 2>/dev/null || { echo "FAIL: $ROOT inaccessible"; exit 1; }

check_key() {
  local file="$1" key="$2"
  if [ ! -f "$file" ]; then echo "${key}:FILE_MISSING"; return; fi
  local line val
  line=$(grep -m1 "^${key}=" "$file" 2>/dev/null || true)
  if [ -z "$line" ]; then echo "${key}:MISSING"; return; fi
  val="${line#*=}"
  val="${val%\"}"
  val="${val#\"}"
  if [ -z "$val" ]; then echo "${key}:EMPTY"; else echo "${key}:SET"; fi
}

echo "=== SERVEUR $(hostname) — pre-deploy audit (read-only) ==="
echo ""

echo "--- Fichiers racine ---"
for f in .env.production .env.production.local .env firebase-service-account.json; do
  if [ -f "$f" ]; then
    printf "%s: present (%s bytes)\n" "$f" "$(wc -c < "$f")"
  else
    echo "$f: ABSENT"
  fi
done
echo ""

echo "--- .env.production.local (structure) ---"
if [ -f .env.production.local ]; then
  lines=$(wc -l < .env.production.local)
  dup=$(grep -c '^EMAIL_PROVIDER_MODE=' .env.production.local 2>/dev/null || echo 0)
  dup_kafka=$(grep -c '^KAFKA_ENABLED=' .env.production.local 2>/dev/null || echo 0)
  echo "lignes: $lines"
  echo "EMAIL_PROVIDER_MODE dans local: $dup (attendu 0 si surcharges seules)"
  echo "KAFKA_ENABLED dans local: $dup_kafka (attendu 0 si surcharges seules)"
  echo "topics v2 dans local:"
  grep -E '^KAFKA_TOPIC_.*\.v2' .env.production.local 2>/dev/null | sed 's/=.*$/' || echo "  (aucun topic .v2 explicite dans local)"
  echo "cles sensibles dans local (presence):"
  for k in OPENWEATHER_API_KEY BREVO_SMTP_PASSWORD ADMIN_IP_WHITELIST ALERTING_EMAIL_WEBHOOK_URL; do
    check_key .env.production.local "$k"
  done
else
  echo "ABSENT — deploy utilisera uniquement CI + fragment"
fi
echo ""

echo "--- .env.production effectif (presence only) ---"
if [ -f .env.production ]; then
  for k in KAFKA_ENABLED TRACKING_INGEST_ASYNC_ENABLED TRACKING_PROCESSED_FANOUT_ENABLED WS_KAFKA_CONSUMER_ENABLED TRACKING_INGEST_PERSIST_ENABLED \
           KAFKA_TOPIC_DRIVER_LOCATION_RAW KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED KAFKA_TOPIC_DRIVER_LOCATION_DLQ \
           APP_ENCRYPTION_KEY_B64 MASTER_ENCRYPTION_KEY SENTRY_DSN \
           BREVO_API_KEY BREVO_SMTP_PASSWORD SMTP_HOST SMTP_USERNAME SMTP_PASSWORD EMAIL_PROVIDER_MODE EMAIL_PROVIDER; do
    check_key .env.production "$k"
  done
else
  echo ".env.production ABSENT"
fi
echo ""

echo "--- firebase-service-account.json ---"
if [ -f firebase-service-account.json ]; then
  if python3 -c "import json; d=json.load(open('firebase-service-account.json')); assert d.get('type')=='service_account' and d.get('project_id')" 2>/dev/null; then
    proj=$(python3 -c "import json; print(json.load(open('firebase-service-account.json')).get('project_id','?'))")
    echo "JSON valide, project_id=$proj"
  else
    echo "FICHIER PRESENT mais JSON invalide ou incomplet"
  fi
  perms=$(stat -c %a firebase-service-account.json 2>/dev/null || echo "?")
  echo "permissions: $perms"
else
  echo "ABSENT sur disque hote"
fi
echo ""

echo "--- Conteneurs (firebase + secrets runtime) ---"
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^atmr-backend$'; then
  if docker exec atmr-backend test -r /app/firebase-service-account.json 2>/dev/null; then
    echo "atmr-backend: firebase monte et lisible"
  else
    echo "atmr-backend: firebase NON lisible dans le conteneur"
  fi
  for k in APP_ENCRYPTION_KEY_B64 MASTER_ENCRYPTION_KEY SENTRY_DSN FIREBASE_SERVICE_ACCOUNT_PATH; do
    if docker exec atmr-backend printenv "$k" 2>/dev/null | grep -q .; then
      echo "container $k: SET"
    else
      echo "container $k: EMPTY/MISSING"
    fi
  done
else
  echo "atmr-backend: conteneur non running"
fi
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^atmr-celery-worker$'; then
  if docker exec atmr-celery-worker test -r /app/firebase-service-account.json 2>/dev/null; then
    echo "atmr-celery-worker: firebase monte et lisible"
  else
    echo "atmr-celery-worker: firebase NON lisible"
  fi
else
  echo "atmr-celery-worker: non running"
fi
echo ""

echo "--- Kafka stack (etat rapide) ---"
docker ps --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null | grep -E 'kafka|tracking|zookeeper|atmr-backend|atmr-ws' || echo "(aucun conteneur matching)"
