#!/bin/sh
# Entrypoint pour Alertmanager avec substitution de variables d'environnement
# Utilise envsubst pour remplacer ${VAR} dans alertmanager.yml

set -e

# Définir les valeurs par défaut pour les variables manquantes
export SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
export SMTP_HOST="${SMTP_HOST:-localhost}"
export SMTP_PORT="${SMTP_PORT:-587}"
export ALERTMANAGER_FROM_EMAIL="${ALERTMANAGER_FROM_EMAIL:-alerts@atmr.local}"
export SMTP_USERNAME="${SMTP_USERNAME:-}"
export SMTP_PASSWORD="${SMTP_PASSWORD:-}"
export ALERT_EMAIL_TO="${ALERT_EMAIL_TO:-}"
export ALERTMANAGER_EXTERNAL_URL="${ALERTMANAGER_EXTERNAL_URL:-http://localhost:9093}"

# Fichier source et destination
CONFIG_SRC="/etc/alertmanager/alertmanager.yml"
CONFIG_DST="/tmp/alertmanager.yml"

# Fonction de substitution avec sed (fallback si envsubst n'est pas disponible)
substitute_with_sed() {
  local file="$1"
  sed -e "s|\${SLACK_WEBHOOK_URL}|${SLACK_WEBHOOK_URL}|g" \
      -e "s|\${SMTP_HOST}|${SMTP_HOST}|g" \
      -e "s|\${SMTP_PORT}|${SMTP_PORT}|g" \
      -e "s|\${ALERTMANAGER_FROM_EMAIL}|${ALERTMANAGER_FROM_EMAIL}|g" \
      -e "s|\${SMTP_USERNAME}|${SMTP_USERNAME}|g" \
      -e "s|\${SMTP_PASSWORD}|${SMTP_PASSWORD}|g" \
      -e "s|\${ALERT_EMAIL_TO}|${ALERT_EMAIL_TO}|g" \
      -e "s|\${ALERTMANAGER_EXTERNAL_URL}|${ALERTMANAGER_EXTERNAL_URL}|g" \
      "$file"
}

# Si le fichier source existe, faire la substitution
if [ -f "$CONFIG_SRC" ]; then
  echo "📝 Substitution des variables d'environnement dans alertmanager.yml..."
  
  # Essayer d'utiliser envsubst si disponible, sinon utiliser sed
  if command -v envsubst >/dev/null 2>&1; then
    echo "✅ Utilisation de envsubst pour la substitution..."
    envsubst '${SLACK_WEBHOOK_URL} ${SMTP_HOST} ${SMTP_PORT} ${ALERTMANAGER_FROM_EMAIL} ${SMTP_USERNAME} ${SMTP_PASSWORD} ${ALERT_EMAIL_TO} ${ALERTMANAGER_EXTERNAL_URL}' < "$CONFIG_SRC" > "$CONFIG_DST"
  else
    echo "⚠️  envsubst non disponible, utilisation de sed pour la substitution..."
    substitute_with_sed "$CONFIG_SRC" > "$CONFIG_DST"
  fi
  
  # Post-traitement : supprimer les lignes avec des valeurs vides qui causent des erreurs
  # Si SLACK_WEBHOOK_URL est vide, commenter la ligne slack_api_url
  if [ -z "$SLACK_WEBHOOK_URL" ]; then
    echo "⚠️  SLACK_WEBHOOK_URL non défini, désactivation de Slack..."
    # Utiliser sed avec fichier temporaire (portable sur toutes les distributions)
    sed 's/^  slack_api_url:.*$/  # slack_api_url: "" # Désactivé (SLACK_WEBHOOK_URL non défini)/' "$CONFIG_DST" > "$CONFIG_DST.tmp" && mv "$CONFIG_DST.tmp" "$CONFIG_DST"
  fi
  
  # Si SMTP_HOST est localhost (valeur par défaut), s'assurer que smtp_smarthost est correct
  if [ "$SMTP_HOST" = "localhost" ]; then
    echo "⚠️  SMTP_HOST non défini, utilisation de localhost (emails ne fonctionneront pas)..."
  fi
  
  echo "✅ Configuration générée: $CONFIG_DST"
  CONFIG_FILE="$CONFIG_DST"
else
  echo "❌ Fichier de configuration non trouvé: $CONFIG_SRC"
  exit 1
fi

# Valider la configuration avant de lancer Alertmanager
echo "🔍 Validation de la configuration..."
/alertmanager --config.file="$CONFIG_FILE" --storage.path=/alertmanager --check-config

if [ $? -ne 0 ]; then
  echo "❌ Erreur de validation de la configuration Alertmanager"
  exit 1
fi

echo "✅ Configuration valide, démarrage d'Alertmanager..."

# Remplacer --config.file dans les arguments et lancer Alertmanager
exec /alertmanager --config.file="$CONFIG_FILE" --storage.path=/alertmanager --web.external-url="${ALERTMANAGER_EXTERNAL_URL:-http://localhost:9093}"
