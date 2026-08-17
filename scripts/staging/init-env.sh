#!/usr/bin/env bash
# Génère .env.staging à partir de env.staging.example (secrets locaux, jamais prod).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXAMPLE="env.staging.example"
TARGET=".env.staging"

[[ -f "$EXAMPLE" ]] || {
  echo "manque $EXAMPLE" >&2
  exit 1
}

if [[ -f "$TARGET" ]]; then
  echo "$TARGET existe déjà — ne pas écraser."
  exit 0
fi

rand_hex() {
  openssl rand -hex "${1:-32}"
}

fernet_key() {
  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" \
    2>/dev/null \
    || python -c "import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())" \
    2>/dev/null \
    || docker compose exec -T atmr_api python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
}

cp "$EXAMPLE" "$TARGET"
pass="$(rand_hex 16)"
secret="$(rand_hex 32)"
jwt="$(rand_hex 32)"
master="$(rand_hex 32)"
enc="$(fernet_key)"

# Remplace uniquement les sentinelles — jamais d'autres fichiers.
python - "$TARGET" "$pass" "$secret" "$jwt" "$master" "$enc" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
password, secret, jwt, master, enc = sys.argv[2:7]
text = path.read_text(encoding="utf-8")
repl = {
    "POSTGRES_PASSWORD=CHANGE_ME_GENERATE": f"POSTGRES_PASSWORD={password}",
    "SECRET_KEY=CHANGE_ME_GENERATE": f"SECRET_KEY={secret}",
    "JWT_SECRET_KEY=CHANGE_ME_GENERATE": f"JWT_SECRET_KEY={jwt}",
    "MASTER_ENCRYPTION_KEY=CHANGE_ME_GENERATE": f"MASTER_ENCRYPTION_KEY={master}",
    "APP_ENCRYPTION_KEY_B64=CHANGE_ME_GENERATE": f"APP_ENCRYPTION_KEY_B64={enc}",
}
for old, new in repl.items():
    if old not in text:
        raise SystemExit(f"sentinelle manquante: {old}")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
print(f"écrit {path} (secrets générés localement)")
PY
