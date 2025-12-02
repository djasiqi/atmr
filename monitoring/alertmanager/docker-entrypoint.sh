#!/bin/sh
# Entrypoint pour Alertmanager avec substitution de variables d'environnement
# Utilise envsubst si dispo, sinon un sed simple.

set -e

CONFIG_SRC="/etc/alertmanager/alertmanager.yml"
CONFIG_DST="/tmp/alertmanager.yml"

# Valeurs par défaut
: "${SLACK_WEBHOOK_URL:=}"
: "${SMTP_HOST:=localhost}"
: "${SMTP_PORT:=587}"
: "${ALERTMANAGER_FROM_EMAIL:=alerts@atmr.local}"
: "${SMTP_USERNAME:=}"
: "${SMTP_PASSWORD:=}"
: "${ALERT_EMAIL_TO:=}"
: "${ALERTMANAGER_EXTERNAL_URL:=http://localhost:9093}"

# Déterminer si TLS est requis selon le port
# Port 465 = SSL implicite (non supporté directement par Alertmanager, utiliser 587)
# Port 587 = STARTTLS (recommandé)
# Port 25 = Pas de TLS (non recommandé)
if [ "$SMTP_PORT" = "465" ]; then
  echo "⚠️  Port 465 détecté. Alertmanager supporte mieux le port 587 avec STARTTLS."
  echo "⚠️  Considérez utiliser le port 587 pour GoDaddy."
  SMTP_REQUIRE_TLS="true"
elif [ "$SMTP_PORT" = "587" ]; then
  SMTP_REQUIRE_TLS="true"
else
  SMTP_REQUIRE_TLS="false"
fi

if [ ! -f "$CONFIG_SRC" ]; then
  echo "❌ Fichier de configuration non trouvé: $CONFIG_SRC" >&2
  exit 1
fi

echo "📝 Substitution des variables d'environnement dans alertmanager.yml..."

# On commence par copier brut
cp "$CONFIG_SRC" "$CONFIG_DST"

# Si envsubst est dispo, on l'utilise
if command -v envsubst >/dev/null 2>&1; then
  echo "✅ Utilisation de envsubst pour la substitution..."
  export SLACK_WEBHOOK_URL SMTP_HOST SMTP_PORT ALERTMANAGER_FROM_EMAIL SMTP_USERNAME SMTP_PASSWORD ALERT_EMAIL_TO ALERTMANAGER_EXTERNAL_URL SMTP_REQUIRE_TLS
  envsubst '${SLACK_WEBHOOK_URL} ${SMTP_HOST} ${SMTP_PORT} ${ALERTMANAGER_FROM_EMAIL} ${SMTP_USERNAME} ${SMTP_PASSWORD} ${ALERT_EMAIL_TO} ${ALERTMANAGER_EXTERNAL_URL} ${SMTP_REQUIRE_TLS}' \
    < "$CONFIG_SRC" > "$CONFIG_DST"
  
  # Post-traitement : si SLACK_WEBHOOK_URL est vide, supprimer la config Slack
  if [ -z "$SLACK_WEBHOOK_URL" ] || [ -z "$(echo "$SLACK_WEBHOOK_URL" | tr -d '[:space:]')" ]; then
    echo "⚠️  SLACK_WEBHOOK_URL vide, suppression des lignes Slack dans la config..."
    sed -i '/^[[:space:]]*slack_api_url:/d' "$CONFIG_DST"
    # Utiliser awk dans un fichier temporaire pour éviter les problèmes d'échappement
    cat > /tmp/remove_slack.awk << 'AWKEOF'
BEGIN { in_slack = 0; slack_indent = 0 }
/^[[:space:]]*slack_configs:/ {
  in_slack = 1
  slack_indent = 0
  pos = 1
  while (pos <= length($0) && (substr($0, pos, 1) == " " || substr($0, pos, 1) == "\t")) {
    slack_indent++
    pos++
  }
  next
}
in_slack == 1 {
  current_indent = 0
  pos = 1
  while (pos <= length($0) && (substr($0, pos, 1) == " " || substr($0, pos, 1) == "\t")) {
    current_indent++
    pos++
  }
  if (/^[[:space:]]*email_configs:/ && current_indent <= slack_indent) {
    in_slack = 0
    print
    next
  }
  if (current_indent <= slack_indent && length($0) > 0 && !/^[[:space:]]*$/) {
    in_slack = 0
    print
    next
  }
  next
}
{ print }
AWKEOF
    awk -f /tmp/remove_slack.awk "$CONFIG_DST" > "$CONFIG_DST.tmp" && mv "$CONFIG_DST.tmp" "$CONFIG_DST"
    rm -f /tmp/remove_slack.awk
  fi
else
  echo "⚠️  envsubst non disponible, utilisation de sed simple..."

  # Substitution basique des variables connues
  sed -i \
    -e "s|\${SMTP_HOST}|$SMTP_HOST|g" \
    -e "s|\${SMTP_PORT}|$SMTP_PORT|g" \
    -e "s|\${ALERTMANAGER_FROM_EMAIL}|$ALERTMANAGER_FROM_EMAIL|g" \
    -e "s|\${SMTP_USERNAME}|$SMTP_USERNAME|g" \
    -e "s|\${SMTP_PASSWORD}|$SMTP_PASSWORD|g" \
    -e "s|\${ALERT_EMAIL_TO}|$ALERT_EMAIL_TO|g" \
    -e "s|\${ALERTMANAGER_EXTERNAL_URL}|$ALERTMANAGER_EXTERNAL_URL|g" \
    -e "s|\${SMTP_REQUIRE_TLS}|$SMTP_REQUIRE_TLS|g" \
    "$CONFIG_DST"

  # Gestion de SLACK_WEBHOOK_URL :
  if [ -n "$SLACK_WEBHOOK_URL" ]; then
    sed -i -e "s|\${SLACK_WEBHOOK_URL}|$SLACK_WEBHOOK_URL|g" "$CONFIG_DST"
  else
    echo "⚠️  SLACK_WEBHOOK_URL vide, suppression des lignes Slack dans la config..."
    # 1) supprimer slack_api_url dans global
    sed -i '/^[[:space:]]*slack_api_url:/d' "$CONFIG_DST"
    # 2) supprimer blocs slack_configs : supprimer la ligne slack_configs: et toutes les lignes suivantes indentées
    # jusqu'à ce qu'on trouve email_configs: ou une ligne non-indentée
    # Utiliser un script awk dans un fichier temporaire pour éviter les problèmes d'échappement
    cat > /tmp/remove_slack.awk << 'AWKEOF'
BEGIN { in_slack = 0; slack_indent = 0 }
/^[[:space:]]*slack_configs:/ {
  in_slack = 1
  slack_indent = 0
  pos = 1
  while (pos <= length($0) && (substr($0, pos, 1) == " " || substr($0, pos, 1) == "\t")) {
    slack_indent++
    pos++
  }
  next
}
in_slack == 1 {
  current_indent = 0
  pos = 1
  while (pos <= length($0) && (substr($0, pos, 1) == " " || substr($0, pos, 1) == "\t")) {
    current_indent++
    pos++
  }
  if (/^[[:space:]]*email_configs:/ && current_indent <= slack_indent) {
    in_slack = 0
    print
    next
  }
  if (current_indent <= slack_indent && length($0) > 0 && !/^[[:space:]]*$/) {
    in_slack = 0
    print
    next
  }
  next
}
{ print }
AWKEOF
    awk -f /tmp/remove_slack.awk "$CONFIG_DST" > "$CONFIG_DST.tmp" && mv "$CONFIG_DST.tmp" "$CONFIG_DST"
    rm -f /tmp/remove_slack.awk
  fi
fi

echo "✅ Configuration générée: $CONFIG_DST"

# Trouver le binaire alertmanager
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
