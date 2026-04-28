# Applique les migrations Alembic dans le conteneur atmr_api (réseau Docker correct).
# Prérequis : stack démarrée (docker compose up -d).
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$compose = if ($env:COMPOSE_FILE) { $env:COMPOSE_FILE } else { "docker-compose.yml" }
& docker compose -f $compose exec -T `
  -e DISABLE_EVENTLET=1 `
  atmr_api alembic -c /app/migrations/alembic.ini upgrade head
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
