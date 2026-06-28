#!/usr/bin/env bash
# Nettoie /srv/atmr/.env.production.local : conserve secrets + surcharges serveur,
# supprime le bloc dupliqué de scripts/env.production.defaults.fragment et les topics
# Kafka .v2 déjà présents dans le fragment.
#
# Usage (sur le serveur) :
#   cd /srv/atmr
#   bash scripts/ops/clean-env-production-local.sh --dry-run
#   bash scripts/ops/clean-env-production-local.sh
#
# Variables :
#   ATMR_DEPLOY_ROOT — défaut /srv/atmr
set -euo pipefail

ROOT="${ATMR_DEPLOY_ROOT:-/srv/atmr}"
TARGET="${ROOT}/.env.production.local"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

read_env_val() {
  local file="$1" key="$2"
  if [[ ! -f "$file" ]]; then
    return 1
  fi
  local line
  line="$(grep -m1 "^${key}=" "$file" 2>/dev/null || true)"
  [[ -n "$line" ]] || return 1
  printf '%s' "${line#*=}"
}

# Clés conservées depuis l'existant (secrets + surcharges opérationnelles).
PRESERVE_KEYS=(
  OPENWEATHER_API_KEY
  OPENWEATHER_CACHE_TTL
  BREVO_SMTP_PASSWORD
  ALERTING_EMAIL_WEBHOOK_URL
  ADMIN_IP_WHITELIST
  JWT_LEGACY_SECRET_KEYS
  JWT_LEGACY_SECRET_KEY
  LEGACY_ENCRYPTION_KEYS
  WS_DRIVER_LOCATION_BATCH_LIMIT
  WS_DRIVER_LOCATION_BATCH_WINDOW_SEC
  WS_DRIVER_LOCATION_LIMIT
  WS_DRIVER_LOCATION_WINDOW_SEC
  SMS_NOTIFICATIONS_ENABLED
  TWILIO_ACCOUNT_SID
  TWILIO_AUTH_TOKEN
  TWILIO_PHONE_NUMBER
)

if [[ ! -f "$TARGET" ]]; then
  echo "Absent : ${TARGET} — rien à nettoyer." >&2
  exit 0
fi

declare -A VALUES=()
for key in "${PRESERVE_KEYS[@]}"; do
  if val="$(read_env_val "$TARGET" "$key" 2>/dev/null || true)"; then
    VALUES["$key"]="$val"
  fi
done

# Préférer la dernière occurrence si doublons (fusion append).
for key in "${PRESERVE_KEYS[@]}"; do
  if [[ -z "${VALUES[$key]+x}" ]]; then
    last="$(grep "^${key}=" "$TARGET" 2>/dev/null | tail -1 || true)"
    if [[ -n "$last" ]]; then
      VALUES["$key"]="${last#*=}"
    fi
  fi
done

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

cat >"$tmp" <<'HEADER'
# =============================================================================
# Surcharges serveur — .env.production.local (fusionné en dernier par deploy-production.sh)
# Secrets + overrides opérationnels UNIQUEMENT. Le fragment CI couvre le reste.
# Généré / nettoyé par scripts/ops/clean-env-production-local.sh
# =============================================================================

HEADER

append_kv() {
  local key="$1" val="$2" comment="${3:-}"
  if [[ -n "$comment" ]]; then
    printf '\n# %s\n' "$comment" >>"$tmp"
  fi
  printf '%s=%s\n' "$key" "$val" >>"$tmp"
}

# --- OpenWeather
if [[ -n "${VALUES[OPENWEATHER_API_KEY]:-}" ]]; then
  append_kv OPENWEATHER_API_KEY "${VALUES[OPENWEATHER_API_KEY]}" "OpenWeather (secret serveur)"
fi
if [[ -n "${VALUES[OPENWEATHER_CACHE_TTL]:-}" ]]; then
  append_kv OPENWEATHER_CACHE_TTL "${VALUES[OPENWEATHER_CACHE_TTL]}"
fi

# --- Alertes / admin / Brevo SMTP
if [[ -n "${VALUES[BREVO_SMTP_PASSWORD]:-}" ]]; then
  append_kv BREVO_SMTP_PASSWORD "${VALUES[BREVO_SMTP_PASSWORD]}" "Brevo SMTP (si brevo_smtp)"
fi
if [[ -n "${VALUES[ALERTING_EMAIL_WEBHOOK_URL]:-}" ]]; then
  append_kv ALERTING_EMAIL_WEBHOOK_URL "${VALUES[ALERTING_EMAIL_WEBHOOK_URL]}"
fi
if [[ -n "${VALUES[ADMIN_IP_WHITELIST]:-}" ]]; then
  append_kv ADMIN_IP_WHITELIST "${VALUES[ADMIN_IP_WHITELIST]}"
fi

# --- Legacy crypto (conserver même vide si présent dans l'ancien fichier)
for key in JWT_LEGACY_SECRET_KEYS JWT_LEGACY_SECRET_KEY LEGACY_ENCRYPTION_KEYS; do
  if grep -q "^${key}=" "$TARGET" 2>/dev/null; then
    append_kv "$key" "${VALUES[$key]:-}"
  fi
done

# --- WS rate limit (surcharge opérationnelle prod)
ws_override=0
for key in WS_DRIVER_LOCATION_BATCH_LIMIT WS_DRIVER_LOCATION_BATCH_WINDOW_SEC WS_DRIVER_LOCATION_LIMIT WS_DRIVER_LOCATION_WINDOW_SEC; do
  if [[ -n "${VALUES[$key]:-}" ]]; then
    ws_override=1
  fi
done
if ((ws_override)); then
  printf '\n# --- WebSocket driver_location_batch (surcharge serveur) ---\n' >>"$tmp"
  for key in WS_DRIVER_LOCATION_BATCH_LIMIT WS_DRIVER_LOCATION_BATCH_WINDOW_SEC WS_DRIVER_LOCATION_LIMIT WS_DRIVER_LOCATION_WINDOW_SEC; do
    if [[ -n "${VALUES[$key]:-}" ]]; then
      append_kv "$key" "${VALUES[$key]}"
    fi
  done
fi

# --- Twilio (optionnel)
twilio=0
for key in SMS_NOTIFICATIONS_ENABLED TWILIO_ACCOUNT_SID TWILIO_AUTH_TOKEN TWILIO_PHONE_NUMBER; do
  if [[ -n "${VALUES[$key]:-}" ]]; then
    twilio=1
  fi
done
if ((twilio)); then
  printf '\n# --- Twilio SMS ---\n' >>"$tmp"
  for key in SMS_NOTIFICATIONS_ENABLED TWILIO_ACCOUNT_SID TWILIO_AUTH_TOKEN TWILIO_PHONE_NUMBER; do
    if [[ -n "${VALUES[$key]:-}" ]]; then
      append_kv "$key" "${VALUES[$key]}"
    fi
  done
fi

old_lines=$(wc -l <"$TARGET")
new_lines=$(wc -l <"$tmp")
removed=$((old_lines - new_lines))

echo "=== clean-env-production-local ==="
echo "Cible   : ${TARGET}"
echo "Avant   : ${old_lines} lignes"
echo "Après   : ${new_lines} lignes (~${removed} lignes fragment/topics dupliqués supprimées)"
echo ""
echo "Clés conservées (présence, sans valeur) :"
for key in "${PRESERVE_KEYS[@]}"; do
  if [[ -n "${VALUES[$key]:-}" ]] || grep -q "^${key}=" "$TARGET" 2>/dev/null; then
    if [[ -n "${VALUES[$key]:-}" ]]; then
      echo "  ${key}: SET"
    else
      echo "  ${key}: (vide — omis ou ligne vide)"
    fi
  fi
done
echo ""
echo "Supprimé (désormais fourni par scripts/env.production.defaults.fragment) :"
echo "  - bloc contact / cookies / rate limit / postgres redis / gunicorn / firebase path"
echo "  - EMAIL_PROVIDER_MODE, FRONTEND_URL, …"
echo "  - KAFKA_TOPIC_* .v2 (déjà dans le fragment depuis Phase 1 LIRIE)"
echo ""

if ((DRY_RUN)); then
  echo "[dry-run] Aperçu (secrets masqués) :"
  sed -E 's/^(OPENWEATHER_API_KEY|BREVO_SMTP_PASSWORD|TWILIO_AUTH_TOKEN|TWILIO_ACCOUNT_SID)=.*/\1=***/' "$tmp"
  echo ""
  echo "[dry-run] Aucune écriture. Relancer sans --dry-run pour appliquer."
  exit 0
fi

backup="${TARGET}.bak.$(date -u +%Y%m%dT%H%M%SZ)"
cp -a "$TARGET" "$backup"
chmod 600 "$backup" "$tmp"
mv "$tmp" "$TARGET"
chmod 600 "$TARGET"
trap - EXIT
echo "Backup : ${backup}"
echo "✅ ${TARGET} nettoyé."
echo ""
echo "Prochaine étape : regénérer .env.production sans redeploy complet :"
echo "  cd ${ROOT} && cp .env.production .env.production.pre-clean.bak && \\"
echo "  bash scripts/deploy-production.sh  # ou attendre le prochain deploy CI"
echo "Ou fusion manuelle : retirer le bloc fragment du .env.production actuel si doublons persistent."
