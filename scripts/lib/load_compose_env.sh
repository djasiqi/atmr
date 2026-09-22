#!/usr/bin/env bash
# Charge un fichier .env style Docker Compose dans le shell **sans l'exécuter**.
# Contrairement à `source`, les valeurs avec espaces (ex. "100 per 15 minutes")
# ne sont pas interprétées comme des commandes.

load_compose_env_file() {
  local file="$1"
  local line key val
  if [ -z "${file:-}" ] || [ ! -f "$file" ]; then
    echo "Fichier env introuvable: ${file:-}" >&2
    return 1
  fi
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in
      '' | \#*) continue ;;
    esac
    case "$line" in
      export\ *) line="${line#export }" ;;
    esac
    case "$line" in
      *=*) ;;
      *) continue ;;
    esac
    key="${line%%=*}"
    val="${line#*=}"
    if ! [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      continue
    fi
    if [[ "$val" == \"*\" ]]; then
      val="${val#\"}"
      val="${val%\"}"
    elif [[ "$val" == \'*\' ]]; then
      val="${val#\'}"
      val="${val%\'}"
    fi
    printf -v "$key" '%s' "$val"
    export "${key?}"
  done <"$file"
}
