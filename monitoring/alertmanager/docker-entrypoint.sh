#!/bin/sh
# Entrypoint pour Alertmanager avec substitution de variables d'environnement
# Utilise envsubst si dispo, sinon un sed simple.

set -e

CONFIG_SRC="/etc/alertmanager/alertmanager.yml"
CONFIG_DST="/tmp/alertmanager.yml"

# Valeurs par défaut
: "${SMTP_HOST:=localhost}"
: "${SMTP_PORT:=587}"
: "${ALERTMANAGER_FROM_EMAIL:=alerts@atmr.local}"
: "${SMTP_USERNAME:=}"
: "${SMTP_PASSWORD:=}"
: "${ALERT_EMAIL_TO:=}"
: "${ALERTMANAGER_EXTERNAL_URL:=http://localhost:9093}"

if [ ! -f "$CONFIG_SRC" ]; then
  echo "❌ Fichier de configuration non trouvé: $CONFIG_SRC" >&2
  exit 1
fi

echo "📝 Substitution des variables d'environnement dans alertmanager.yml..."

cp "$CONFIG_SRC" "$CONFIG_DST"

if command -v envsubst >/dev/null 2>&1; then
  echo "✅ Utilisation de envsubst pour la substitution..."
  export SMTP_HOST SMTP_PORT ALERTMANAGER_FROM_EMAIL SMTP_USERNAME SMTP_PASSWORD ALERT_EMAIL_TO ALERTMANAGER_EXTERNAL_URL SMTP_REQUIRE_TLS
  envsubst '${SMTP_HOST} ${SMTP_PORT} ${ALERTMANAGER_FROM_EMAIL} ${SMTP_USERNAME} ${SMTP_PASSWORD} ${ALERT_EMAIL_TO} ${ALERTMANAGER_EXTERNAL_URL} ${SMTP_REQUIRE_TLS}' \
    < "$CONFIG_SRC" > "$CONFIG_DST"
else
  echo "⚠️  envsubst non disponible, utilisation de sed simple..."

  ALERT_EMAIL_TO_VALUE="${ALERT_EMAIL_TO:-noreply@atmr.local}"
  ALERTMANAGER_FROM_EMAIL_VALUE="${ALERTMANAGER_FROM_EMAIL:-alerts@atmr.local}"

  sed -i \
    -e "s|'\${SMTP_HOST}'|'$SMTP_HOST'|g" \
    -e "s|\${SMTP_HOST}|$SMTP_HOST|g" \
    -e "s|'\${SMTP_PORT}'|'$SMTP_PORT'|g" \
    -e "s|\${SMTP_PORT}|$SMTP_PORT|g" \
    -e "s|'\${ALERTMANAGER_FROM_EMAIL}'|'$ALERTMANAGER_FROM_EMAIL_VALUE'|g" \
    -e "s|\${ALERTMANAGER_FROM_EMAIL}|$ALERTMANAGER_FROM_EMAIL_VALUE|g" \
    -e "s|'\${SMTP_USERNAME}'|'$SMTP_USERNAME'|g" \
    -e "s|\${SMTP_USERNAME}|$SMTP_USERNAME|g" \
    -e "s|'\${SMTP_PASSWORD}'|'$SMTP_PASSWORD'|g" \
    -e "s|\${SMTP_PASSWORD}|$SMTP_PASSWORD|g" \
    -e "s|\${ALERT_EMAIL_TO}|$ALERT_EMAIL_TO_VALUE|g" \
    -e "s|'\${ALERTMANAGER_EXTERNAL_URL}'|'$ALERTMANAGER_EXTERNAL_URL'|g" \
    -e "s|\${ALERTMANAGER_EXTERNAL_URL}|$ALERTMANAGER_EXTERNAL_URL|g" \
    "$CONFIG_DST"
fi

echo "✅ Configuration générée: $CONFIG_DST"

if command -v alertmanager >/dev/null 2>&1; then
  ALERTMANAGER_BIN="$(command -v alertmanager)"
elif [ -x "/bin/alertmanager" ]; then
  ALERTMANAGER_BIN="/bin/alertmanager"
elif [ -x "/usr/bin/alertmanager" ]; then
  ALERTMANAGER_BIN="/usr/bin/alertmanager"
else
  echo "❌ Binaire Alertmanager non trouvé" >&2
  exit 1
fi

echo "✅ Binaire Alertmanager trouvé: $ALERTMANAGER_BIN"

echo "🔍 Validation de la configuration..."
if "$ALERTMANAGER_BIN" --help 2>&1 | grep -q -- '--check-config'; then
  if ! "$ALERTMANAGER_BIN" \
      --config.file="$CONFIG_DST" \
      --storage.path=/alertmanager \
      --check-config; then
    echo "❌ Erreur de validation de la configuration Alertmanager" >&2
    exit 1
  fi
  echo "✅ Configuration valide"
else
  echo "⚠️  --check-config non supporté par cette version, démarrage direct..."
fi

echo "🚀 Démarrage d'Alertmanager..."

exec "$ALERTMANAGER_BIN" \
  --config.file="$CONFIG_DST" \
  --storage.path=/alertmanager \
  --web.external-url="$ALERTMANAGER_EXTERNAL_URL"
