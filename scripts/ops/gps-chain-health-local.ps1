# Sante chaine GPS locale (Docker + API localhost).
# Usage:
#   .\scripts\ops\gps-chain-health-local.ps1
#   $env:DRIVER_ID=7135; .\scripts\ops\gps-chain-health-local.ps1
param(
    [string]$ApiBase = "http://127.0.0.1:5000",
    [string]$DriverId = $env:DRIVER_ID,
    [int]$DlqWindowMin = 10
)

$Fail = 0

function Ok($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red; $Fail = 1 }
function Section($title) { Write-Host ""; Write-Host "========== $title ==========" -ForegroundColor Cyan }

function Test-ContainerRunning($name) {
    $names = docker ps --format "{{.Names}}"
    return ($names -match "^$name$")
}

Section "1. Conteneurs Docker"
$required = @(
    "atmr-atmr_api",
    "atmr-redis-1",
    "atmr-kafka-broker-1",
    "atmr-tracking-kafka-consumer-1",
    "atmr-tracking-processed-fanout-1"
)
foreach ($c in $required) {
    if (Test-ContainerRunning $c) { Ok "running: $c" } else { Fail "absent ou arrete: $c" }
}

Section "2. API locale"
try {
    Invoke-RestMethod -Uri "$ApiBase/health" -TimeoutSec 5 | Out-Null
    Ok "API health $ApiBase/health"
} catch {
    Fail "API health inaccessible ($ApiBase)"
}

try {
    $ready = Invoke-RestMethod -Uri "$ApiBase/api/v1/ready" -TimeoutSec 5
    if ($ready.status -eq "ready") { Ok "API ready" } else { Warn "API non ready" }
} catch {
    Warn "API ready inaccessible"
}

Section "3. Workers tracking"
if (Test-ContainerRunning "atmr-tracking-kafka-consumer-1") {
    $ce = docker exec atmr-tracking-kafka-consumer-1 env 2>$null
    foreach ($var in @("FLASK_CONFIG", "APP_ENCRYPTION_KEY_B64", "REDIS_URL", "TRACKING_INGEST_PERSIST_ENABLED")) {
        if ($ce -match "^$var=") { Ok "consumer: $var defini" } else { Fail "consumer: $var manquant" }
    }
    $logs = docker logs atmr-tracking-kafka-consumer-1 --since "${DlqWindowMin}m" 2>&1
    $dlq = ($logs | Select-String "DLQ confirmed").Count
    if ($dlq -eq 0) { Ok "aucune DLQ consumer ($DlqWindowMin min)" }
    else { Fail "$dlq DLQ confirmee(s) - voir data/kafka-dlq/kafka_dlq_events.jsonl" }
}

if (Test-ContainerRunning "atmr-tracking-processed-fanout-1") {
    $fe = docker exec atmr-tracking-processed-fanout-1 env 2>$null
    if ($fe -match "^TRACKING_PROCESSED_FANOUT_ENABLED=true") { Ok "fanout enabled" }
    else { Fail "fanout TRACKING_PROCESSED_FANOUT_ENABLED absent/false" }
    if ($fe -match "^REDIS_URL=") { Ok "fanout REDIS_URL defini" }
    else { Fail "fanout REDIS_URL manquant" }
}

Section "4. Redis chauffeur"
if ($DriverId -and (Test-ContainerRunning "atmr-redis-1")) {
    docker exec atmr-redis-1 redis-cli HGETALL "driver:${DriverId}:loc"
    Ok "driver:${DriverId}:loc consulte"
} else {
    Warn "DRIVER_ID non defini - ex: env:DRIVER_ID=7135"
}

Section "Verdict"
if ($Fail -eq 0) {
    Ok "Chaine locale GPS - configuration saine"
} else {
    Fail "Problemes detectes - corriger les [FAIL] ci-dessus"
    exit 1
}
