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
# Si SLACK_WEBHOOK_URL est vide, supprimer la ligne au lieu de la substituer
substitute_with_sed() {
  local file="$1"
  local output=""
  
  # Si SLACK_WEBHOOK_URL est vide, supprimer la ligne slack_api_url directement
  if [ -z "$SLACK_WEBHOOK_URL" ] || [ -z "$(echo "$SLACK_WEBHOOK_URL" | tr -d '[:space:]')" ]; then
    # Supprimer la ligne slack_api_url avant la substitution
    sed '/^[[:space:]]*slack_api_url:/d' "$file" | \
    sed -e "s|\${SMTP_HOST}|${SMTP_HOST}|g" \
        -e "s|\${SMTP_PORT}|${SMTP_PORT}|g" \
        -e "s|\${ALERTMANAGER_FROM_EMAIL}|${ALERTMANAGER_FROM_EMAIL}|g" \
        -e "s|\${SMTP_USERNAME}|${SMTP_USERNAME}|g" \
        -e "s|\${SMTP_PASSWORD}|${SMTP_PASSWORD}|g" \
        -e "s|\${ALERT_EMAIL_TO}|${ALERT_EMAIL_TO}|g" \
        -e "s|\${ALERTMANAGER_EXTERNAL_URL}|${ALERTMANAGER_EXTERNAL_URL}|g"
  else
    # Substitution normale avec toutes les variables
    sed -e "s|\${SLACK_WEBHOOK_URL}|${SLACK_WEBHOOK_URL}|g" \
        -e "s|\${SMTP_HOST}|${SMTP_HOST}|g" \
        -e "s|\${SMTP_PORT}|${SMTP_PORT}|g" \
        -e "s|\${ALERTMANAGER_FROM_EMAIL}|${ALERTMANAGER_FROM_EMAIL}|g" \
        -e "s|\${SMTP_USERNAME}|${SMTP_USERNAME}|g" \
        -e "s|\${SMTP_PASSWORD}|${SMTP_PASSWORD}|g" \
        -e "s|\${ALERT_EMAIL_TO}|${ALERT_EMAIL_TO}|g" \
        -e "s|\${ALERTMANAGER_EXTERNAL_URL}|${ALERTMANAGER_EXTERNAL_URL}|g" \
        "$file"
  fi
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
  # Si SLACK_WEBHOOK_URL est vide, supprimer TOUTE la configuration Slack
  # (slack_api_url dans global ET tous les slack_configs dans les receivers)
  if [ -z "$SLACK_WEBHOOK_URL" ] || [ -z "$(echo "$SLACK_WEBHOOK_URL" | tr -d '[:space:]')" ]; then
    echo "⚠️  SLACK_WEBHOOK_URL non défini ou vide, suppression complète de la configuration Slack..."
    
    # 1. Supprimer slack_api_url de la section global
    sed '/^[[:space:]]*slack_api_url:/d' "$CONFIG_DST" > "$CONFIG_DST.tmp" && mv "$CONFIG_DST.tmp" "$CONFIG_DST"
    
    # 2. Supprimer tous les blocs slack_configs des receivers
    # Méthode simple : supprimer toutes les lignes contenant "slack_configs" et les lignes suivantes indentées
    # jusqu'à ce qu'on trouve "email_configs:" ou une ligne non-indentée
    # Utiliser un script awk simple sans caractères spéciaux problématiques
    awk '
      BEGIN { skip = 0; indent_level = 0 }
      /slack_configs:/ {
        skip = 1
        # Compter les espaces au début de la ligne
        indent_level = 0
        while (substr($0, indent_level + 1, 1) == " " || substr($0, indent_level + 1, 1) == "\t") {
          indent_level++
        }
        next
      }
      skip == 1 {
        # Compter les espaces de la ligne actuelle
        current_indent = 0
        while (substr($0, current_indent + 1, 1) == " " || substr($0, current_indent + 1, 1) == "\t") {
          current_indent++
        }
        # Si on revient au même niveau ou moins ET que ce n'est pas une ligne vide, on sort du bloc
        if (length($0) > 0 && current_indent <= indent_level) {
          skip = 0
          print
        }
        # Sinon, on ignore la ligne (elle fait partie du bloc slack_configs)
        next
      }
      skip == 0 {
        print
      }
    ' "$CONFIG_DST" > "$CONFIG_DST.tmp" && mv "$CONFIG_DST.tmp" "$CONFIG_DST"
    
    echo "✅ Configuration Slack complètement supprimée (slack_api_url + tous les slack_configs)"
  else
    # Vérifier aussi après substitution si la valeur est vide
    if grep -qE "^[[:space:]]*slack_api_url:[[:space:]]*(''|\"\")" "$CONFIG_DST"; then
      echo "⚠️  SLACK_WEBHOOK_URL est vide après substitution, suppression complète de la configuration Slack..."
      sed '/^[[:space:]]*slack_api_url:/d' "$CONFIG_DST" > "$CONFIG_DST.tmp" && mv "$CONFIG_DST.tmp" "$CONFIG_DST"
      # Utiliser la même méthode awk que ci-dessus
      awk '
        BEGIN { skip = 0; indent_level = 0 }
        /slack_configs:/ {
          skip = 1
          indent_level = 0
          while (substr($0, indent_level + 1, 1) == " " || substr($0, indent_level + 1, 1) == "\t") {
            indent_level++
          }
          next
        }
        skip == 1 {
          current_indent = 0
          while (substr($0, current_indent + 1, 1) == " " || substr($0, current_indent + 1, 1) == "\t") {
            current_indent++
          }
          if (length($0) > 0 && current_indent <= indent_level) {
            skip = 0
            print
          }
          next
        }
        skip == 0 {
          print
        }
      ' "$CONFIG_DST" > "$CONFIG_DST.tmp" && mv "$CONFIG_DST.tmp" "$CONFIG_DST"
    fi
  fi
  
  # Supprimer les lignes vides multiples
  sed '/^[[:space:]]*$/N;/^\n$/d' "$CONFIG_DST" > "$CONFIG_DST.tmp" && mv "$CONFIG_DST.tmp" "$CONFIG_DST"
  
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

# Trouver le binaire Alertmanager (peut être dans /bin, /usr/bin, ou accessible via PATH)
ALERTMANAGER_BIN=""
if [ -x "/bin/alertmanager" ]; then
  ALERTMANAGER_BIN="/bin/alertmanager"
elif [ -x "/usr/bin/alertmanager" ]; then
  ALERTMANAGER_BIN="/usr/bin/alertmanager"
elif [ -x "/alertmanager" ]; then
  ALERTMANAGER_BIN="/alertmanager"
elif command -v alertmanager >/dev/null 2>&1; then
  ALERTMANAGER_BIN=$(command -v alertmanager)
else
  echo "❌ Binaire Alertmanager non trouvé"
  exit 1
fi

echo "✅ Binaire Alertmanager trouvé: $ALERTMANAGER_BIN"

# Valider la configuration avant de lancer Alertmanager (si --check-config est supporté)
# Note: Certaines versions d'Alertmanager ne supportent pas --check-config
echo "🔍 Validation de la configuration..."
if "$ALERTMANAGER_BIN" --help 2>&1 | grep -q "check-config"; then
  if ! "$ALERTMANAGER_BIN" --config.file="$CONFIG_FILE" --storage.path=/alertmanager --check-config; then
    echo "❌ Erreur de validation de la configuration Alertmanager"
    exit 1
  fi
  echo "✅ Configuration valide"
else
  echo "⚠️  --check-config non supporté par cette version, démarrage direct..."
fi

echo "✅ Démarrage d'Alertmanager..."

# Remplacer --config.file dans les arguments et lancer Alertmanager
exec "$ALERTMANAGER_BIN" --config.file="$CONFIG_FILE" --storage.path=/alertmanager --web.external-url="${ALERTMANAGER_EXTERNAL_URL:-http://localhost:9093}"
