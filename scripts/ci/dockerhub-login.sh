#!/usr/bin/env bash
# Authentifie le daemon Docker auprès de Docker Hub.
# Retry uniquement les erreurs réseau transitoires (RST Cloudflare, EOF, timeout).
# Les 401/403 (mauvais token) sont fatals immédiatement.
set -euo pipefail

USERNAME="${DOCKERHUB_USERNAME:-}"
TOKEN="${DOCKERHUB_TOKEN:-}"
MAX_ATTEMPTS="${DOCKERHUB_LOGIN_ATTEMPTS:-5}"

if [ -z "${USERNAME}" ] || [ -z "${TOKEN}" ]; then
  echo "❌ DOCKERHUB_USERNAME ou DOCKERHUB_TOKEN vide"
  exit 1
fi

is_fatal_auth() {
  echo "$1" | grep -qiE \
    'unauthorized|incorrect username|incorrect password|401|403|access denied|bad credentials|authentication required'
}

for i in $(seq 1 "${MAX_ATTEMPTS}"); do
  echo "🔐 Login Docker Hub tentative ${i}/${MAX_ATTEMPTS}..."
  set +e
  output="$(printf '%s' "${TOKEN}" | docker login -u "${USERNAME}" --password-stdin 2>&1)"
  status=$?
  set -e
  if [ "${status}" -eq 0 ]; then
    echo "✅ Login Docker Hub réussi (tentative ${i})"
    exit 0
  fi
  echo "${output}"
  if is_fatal_auth "${output}"; then
    echo "❌ Identifiants Docker Hub refusés (pas un incident réseau)"
    exit 1
  fi
  if [ "${i}" -lt "${MAX_ATTEMPTS}" ]; then
    delay=$((i * 8))
    echo "⚠️  Échec transitoire (reset/timeout auth.docker.io). Nouvelle tentative dans ${delay}s..."
    sleep "${delay}"
  fi
done

echo "❌ Login Docker Hub impossible après ${MAX_ATTEMPTS} tentatives"
exit 1
