Param(
  [string]$ComposeFile = "docker-compose.production.yml",
  [string]$BackendService = "backend",
  [string]$BackendContainer = "",
  [string]$PgbouncerService = "pgbouncer",
  [string]$PgbouncerContainer = "atmr-pgbouncer"
)

$ErrorActionPreference = "Stop"

Write-Host "== Verification backend -> PgBouncer =="

function Resolve-ContainerFromCompose {
  param(
    [string]$ServiceName
  )

  $containerId = docker compose -f $ComposeFile ps -q $ServiceName
  if ($containerId) {
    return $containerId.Trim()
  }
  return ""
}

$resolvedBackend = $BackendContainer
if ([string]::IsNullOrWhiteSpace($resolvedBackend)) {
  $resolvedBackend = Resolve-ContainerFromCompose -ServiceName $BackendService
}
if ([string]::IsNullOrWhiteSpace($resolvedBackend)) {
  throw "Backend container introuvable. Renseigner -BackendContainer ou -BackendService."
}

$resolvedPgbouncer = $PgbouncerContainer
if ([string]::IsNullOrWhiteSpace($resolvedPgbouncer)) {
  $resolvedPgbouncer = Resolve-ContainerFromCompose -ServiceName $PgbouncerService
}
if ([string]::IsNullOrWhiteSpace($resolvedPgbouncer)) {
  throw "PgBouncer container introuvable. Renseigner -PgbouncerContainer ou -PgbouncerService."
}

Write-Host "`n[1/4] Status containers"
docker compose -f $ComposeFile ps $BackendService $PgbouncerService

Write-Host "`n[2/4] Environment backend (DB host/port)"
$envLines = docker inspect $resolvedBackend --format "{{range .Config.Env}}{{println .}}{{end}}"
$envLines |
  Select-String -Pattern "^POSTGRES_HOST=|^POSTGRES_PORT=|^DATABASE_URL=|^SQLALCHEMY_DATABASE_URI="

$dbHostLine = $envLines | Select-String -Pattern "^POSTGRES_HOST="
if ($dbHostLine -and $dbHostLine.Line -notmatch "POSTGRES_HOST=pgbouncer") {
  Write-Host "FAIL: POSTGRES_HOST n'est pas pgbouncer."
  exit 1
}
if (-not $dbHostLine) {
  $databaseUrlLine = $envLines | Select-String -Pattern "^DATABASE_URL="
  if ($databaseUrlLine -and $databaseUrlLine.Line -match "@([^:/]+):") {
    $databaseHost = $Matches[1]
    if ($databaseHost -ne "pgbouncer") {
      Write-Host "FAIL: DATABASE_URL pointe vers '$databaseHost' au lieu de 'pgbouncer'."
      exit 1
    }
  }
}

Write-Host "`n[3/4] Connectivity backend -> PgBouncer"
docker exec $resolvedBackend python -c "import socket; s=socket.create_connection(('pgbouncer', 6432), 5); s.close(); print('backend_to_pgbouncer=ok')"

Write-Host "`n[4/4] Logs backend (errors de resolution/connexion)"
$logs = (cmd /c "docker logs $resolvedBackend --since 15m 2>&1" | Out-String)
$errors = $logs | Select-String -Pattern "failed to resolve host 'postgres'|OperationalError|could not connect to server"
if ($errors) {
  Write-Host "FAIL: erreurs detectees dans les logs backend"
  $errors | ForEach-Object { Write-Host $_.Line }
  exit 1
}

Write-Host "PASS: aucune erreur de resolution/connexion DB detectee sur les 15 dernieres minutes."
