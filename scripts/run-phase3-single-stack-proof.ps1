param(
  [string]$Owner = "SRE Lead",
  [string]$EnvReference = ".env.production.local",
  [string]$ComposeEnvFile = ".env.production",
  [string]$ApiBaseUrl = "http://localhost:5000",
  [int]$ApiReadinessTimeoutSec = 180,
  [int]$ApiReadinessPollIntervalSec = 5,
  [string]$FirebaseServiceAccountPath = "firebase-service-account.json",
  [switch]$EnableRedisCluster,
  [switch]$SkipCleanStart,
  [switch]$NonInteractive,
  [switch]$Ci,
  [switch]$KeepGeneratedEnvFile
)

$ErrorActionPreference = "Stop"
if ($Ci) { $NonInteractive = $true }

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outputDir = Join-Path "scripts" "output"
$reportPath = Join-Path $outputDir ("phase3-proof-{0}.md" -f $timestamp)
$jsonPath = Join-Path $outputDir ("phase3-proof-{0}.json" -f $timestamp)
$logPath = Join-Path $outputDir ("phase3-proof-{0}.log" -f $timestamp)

$productionCompose = "docker-compose.production.yml"
$kafkaCompose = "docker-compose.kafka.yml"
$redisClusterCompose = "docker-compose.redis-cluster.yml"
$forbiddenCompose = "docker-compose.yml"

$checks = New-Object System.Collections.Generic.List[object]
$failureStep = ""
$failureMessage = ""
$verdict = "PASS"
$exitCode = 0
$generatedComposeEnvFile = $false

function Initialize-OutputDirectory {
  if (-not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
  }
}

function Write-Log {
  param([string]$Message)
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
  if (-not ($Ci -and $Message.StartsWith("START "))) {
    Write-Host $line
  }
  Add-Content -LiteralPath $logPath -Value $line
}

function Set-FailureState {
  param(
    [string]$Step,
    [string]$Message,
    [string]$FailureKind
  )

  if ($script:failureStep) { return }

  $script:failureStep = $Step
  $script:failureMessage = $Message

  switch ($FailureKind) {
    "BLOCKED" {
      $script:verdict = "BLOCKED"
      $script:exitCode = 2
    }
    "SCRIPT_ERROR" {
      $script:verdict = "SCRIPT_ERROR"
      $script:exitCode = 3
    }
    default {
      $script:verdict = "FAIL"
      $script:exitCode = 1
    }
  }
}

function Add-Check {
  param(
    [string]$Name,
    [string]$Status,
    [string]$Evidence
  )
  $checks.Add([pscustomobject]@{
      name     = $Name
      status   = $Status
      evidence = $Evidence
    })
}

function Invoke-Step {
  param(
    [string]$Name,
    [scriptblock]$Action,
    [ValidateSet("FAIL", "BLOCKED", "SCRIPT_ERROR")][string]$FailureKind = "FAIL"
  )

  Write-Log ("START {0}" -f $Name)
  try {
    & $Action
    Add-Check -Name $Name -Status "PASS" -Evidence "OK"
    Write-Log ("PASS {0}" -f $Name)
  }
  catch {
    $message = $_.Exception.Message
    Add-Check -Name $Name -Status $FailureKind -Evidence $message
    Set-FailureState -Step $Name -Message $message -FailureKind $FailureKind
    Write-Log ("{0} {1} :: {2}" -f $FailureKind, $Name, $message)
    throw
  }
}

function Assert-FileExists {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    throw "Fichier requis introuvable: $Path"
  }
}

function Assert-FileIsRegular {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "Le chemin doit etre un fichier regulier: $Path"
  }
}

function Initialize-ComposeEnvFile {
  if (Test-Path -LiteralPath $ComposeEnvFile) {
    return
  }
  if (-not (Test-Path -LiteralPath $EnvReference)) {
    throw "Fichier env compose manquant ($ComposeEnvFile) et reference introuvable ($EnvReference)."
  }
  Copy-Item -LiteralPath $EnvReference -Destination $ComposeEnvFile -Force
  $script:generatedComposeEnvFile = $true
  Write-Log "Fichier env compose genere: $ComposeEnvFile (source: $EnvReference)"
}

function Assert-LastExitCode {
  param([string]$Context)
  if ($LASTEXITCODE -ne 0) {
    throw "$Context a retourne un code non-zero: $LASTEXITCODE"
  }
}

function Get-RunningComposeServices {
  param([string]$ComposeFile)
  $services = docker compose -f $ComposeFile ps --services --status running 2>$null
  if (-not $services) { return @() }
  return ($services -split "`r?`n" | Where-Object { $_.Trim().Length -gt 0 })
}

function Assert-HttpStatus200 {
  param(
    [string]$Url,
    [string]$Label,
    [switch]$AllowHttpsRedirect
  )
  $probe = curl.exe -s -o NUL -w "%{http_code} %{redirect_url}" -I --max-redirs 0 $Url
  if ($LASTEXITCODE -ne 0) {
    throw "$Label probe HTTP a echoue (curl exit=$LASTEXITCODE)"
  }

  $parts = $probe.Trim() -split "\s+", 2
  $statusCode = if ($parts.Count -ge 1) { $parts[0] } else { "" }
  $redirectUrl = if ($parts.Count -ge 2) { $parts[1] } else { "" }

  if ($statusCode -eq "200") {
    return
  }

  if ($AllowHttpsRedirect -and ($statusCode -in @("301", "302", "307", "308"))) {
    if (-not [string]::IsNullOrWhiteSpace($redirectUrl) -and $redirectUrl.StartsWith("https://")) {
      Write-Log "$Label redirection HTTPS acceptee ($statusCode -> $redirectUrl)"
      return
    }
  }

  throw "$Label a retourne $statusCode (redirect=$redirectUrl)"
}

function Wait-Http200 {
  param(
    [string]$Url,
    [string]$Label,
    [int]$TimeoutSec,
    [int]$PollIntervalSec
  )

  $start = Get-Date
  while ($true) {
    try {
      $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
      if ($response.StatusCode -eq 200) {
        return
      }
    }
    catch {
      # retry
    }

    $elapsed = ((Get-Date) - $start).TotalSeconds
    if ($elapsed -ge $TimeoutSec) {
      throw "$Label non pret apres ${TimeoutSec}s"
    }
    Start-Sleep -Seconds $PollIntervalSec
  }
}

function Assert-ContainerHealthyFromCompose {
  param(
    [string]$ComposeFile,
    [string]$ServiceName,
    [int]$TimeoutSec = 120,
    [int]$PollIntervalSec = 4
  )
  $id = (docker compose -f $ComposeFile ps -q $ServiceName 2>$null)
  if ([string]::IsNullOrWhiteSpace($id)) {
    throw "Service introuvable: $ServiceName dans $ComposeFile"
  }
  Assert-ContainerHealthy -ContainerName $id.Trim() -TimeoutSec $TimeoutSec -PollIntervalSec $PollIntervalSec
}

function Assert-ContainerHealthy {
  param(
    [string]$ContainerName,
    [int]$TimeoutSec = 120,
    [int]$PollIntervalSec = 4
  )

  $start = Get-Date
  while ($true) {
    $status = (docker inspect $ContainerName --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}" 2>$null).Trim()
    if (($status -eq "healthy") -or ($status -eq "running")) {
      return
    }
    if (($status -eq "exited") -or ($status -eq "dead")) {
      throw "Container $ContainerName en etat terminal (status=$status)"
    }

    $elapsed = ((Get-Date) - $start).TotalSeconds
    if ($elapsed -ge $TimeoutSec) {
      throw "Container $ContainerName non healthy/running apres ${TimeoutSec}s (status=$status)"
    }
    Start-Sleep -Seconds $PollIntervalSec
  }
}

function Assert-WsRedisUp {
  $raw = docker exec atmr-ws-service python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8001/health', timeout=5).read().decode())"
  $obj = $raw | ConvertFrom-Json
  if (-not $obj.redis_up) {
    throw "ws-service health indique redis_up=false"
  }
}

function Write-Report {
  $failureLine = if ($failureStep) { $failureStep } else { "Aucune" }
  $checksTable = @()
  foreach ($item in $checks) {
    $checksTable += "| $($item.name) | $($item.status) | $($item.evidence -replace '\|','/') |"
  }

  $markdown = @"
# Phase3 Runtime Proof Report

- Run ID: $timestamp
- Date/Heure: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz")
- Owner: $Owner
- Stack de reference: $productionCompose + $kafkaCompose
- Verdict global: **$verdict**
- Premiere etape en echec: $failureLine
- Script: scripts/run-phase3-single-stack-proof.ps1
- Mode: Ci=$Ci, NonInteractive=$NonInteractive

## Checks executes

| Check | Resultat | Preuve |
| --- | --- | --- |
$($checksTable -join "`n")

## Bloc matrice (copier-coller)

- Commande executee: powershell -ExecutionPolicy Bypass -File scripts/run-phase3-single-stack-proof.ps1 -Owner "$Owner"
- Resultat: $verdict
- Preuve: $reportPath
- Rollback associe: docker compose -f $productionCompose down; docker compose -f $kafkaCompose down; docker compose -f $redisClusterCompose down
- Ecart ouvert restant: <a completer si FAIL/BLOCKED>
"@
  Set-Content -LiteralPath $reportPath -Value $markdown -Encoding UTF8

  $jsonReport = [pscustomobject]@{
    run_id          = $timestamp
    owner           = $Owner
    timestamp       = (Get-Date -Format "yyyy-MM-ddTHH:mm:sszzz")
    status          = $verdict
    exit_code       = $exitCode
    failed_step     = $failureStep
    failure_message = $failureMessage
    stack_reference = @($productionCompose, $kafkaCompose)
    checks          = $checks
    artifacts       = [pscustomobject]@{
      markdown = $reportPath
      json     = $jsonPath
      log      = $logPath
    }
    flags = [pscustomobject]@{
      ci              = [bool]$Ci
      non_interactive = [bool]$NonInteractive
      skip_clean      = [bool]$SkipCleanStart
      enable_redis_cluster = [bool]$EnableRedisCluster
    }
  }
  $jsonReport | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding UTF8
}

Initialize-OutputDirectory
Set-Content -LiteralPath $logPath -Value ("Phase3 single-stack proof log ({0})" -f $timestamp) -Encoding UTF8

try {
  Invoke-Step -Name "Preflight Docker" -FailureKind "SCRIPT_ERROR" -Action {
    $null = docker version
    Assert-LastExitCode -Context "docker version"
    $null = docker compose version
    Assert-LastExitCode -Context "docker compose version"
  }

  Invoke-Step -Name "Preflight fichiers requis" -FailureKind "BLOCKED" -Action {
    Assert-FileExists -Path $productionCompose
    Assert-FileExists -Path $kafkaCompose
    Assert-FileExists -Path $redisClusterCompose
    Assert-FileExists -Path $EnvReference
    if ($ApiReadinessTimeoutSec -lt 10) {
      throw "ApiReadinessTimeoutSec doit etre >= 10."
    }
    if ($ApiReadinessPollIntervalSec -lt 1) {
      throw "ApiReadinessPollIntervalSec doit etre >= 1."
    }
    Assert-FileExists -Path "scripts/verify-backend-pgbouncer.ps1"
    Assert-FileExists -Path "scripts/smoke-scalability-proof.ps1"
    Assert-FileIsRegular -Path $FirebaseServiceAccountPath
  }

  Invoke-Step -Name "Preflight environnement ambigu" -FailureKind "BLOCKED" -Action {
    $forbiddenRunning = Get-RunningComposeServices -ComposeFile $forbiddenCompose
    if ($forbiddenRunning.Count -gt 0) {
      $joined = $forbiddenRunning -join ", "
      throw "Stack interdite active ($forbiddenCompose): $joined"
    }
  }

  Invoke-Step -Name "Prepare compose env file" -FailureKind "BLOCKED" -Action {
    Initialize-ComposeEnvFile
  }

  if (-not $SkipCleanStart) {
    Invoke-Step -Name "Clean start stacks Phase3" -Action {
      docker compose -f $productionCompose down
      Assert-LastExitCode -Context "docker compose production down"
      docker compose -f $kafkaCompose down
      Assert-LastExitCode -Context "docker compose kafka down"
      docker compose -f $redisClusterCompose down
      Assert-LastExitCode -Context "docker compose redis-cluster down"
    }
  }

  Invoke-Step -Name "Bootstrap production core" -Action {
    docker compose -f $productionCompose up -d postgres pgbouncer redis
    Assert-LastExitCode -Context "docker compose production up core"
    docker compose -f $productionCompose up -d backend ws-service celery-worker celery-beat
    Assert-LastExitCode -Context "docker compose production up apps"
  }

  Invoke-Step -Name "Bootstrap kafka extension" -Action {
    docker compose -f $kafkaCompose up -d zookeeper kafka tracking-kafka-consumer
    Assert-LastExitCode -Context "docker compose kafka up"
  }

  if ($EnableRedisCluster) {
    Invoke-Step -Name "Bootstrap redis cluster optionnel" -Action {
      powershell -ExecutionPolicy Bypass -File "scripts/init-redis-cluster.ps1"
      Assert-LastExitCode -Context "init-redis-cluster"
    }
  }

  Invoke-Step -Name "Check endpoints API" -Action {
    Wait-Http200 -Url "$ApiBaseUrl/health" -Label "/health readiness" -TimeoutSec $ApiReadinessTimeoutSec -PollIntervalSec $ApiReadinessPollIntervalSec
    Assert-HttpStatus200 -Url "$ApiBaseUrl/health" -Label "/health"
    Assert-HttpStatus200 -Url "$ApiBaseUrl/api/v1/realtime-gateway/canary" -Label "/api/v1/realtime-gateway/canary" -AllowHttpsRedirect
  }

  Invoke-Step -Name "Check backend via PgBouncer" -Action {
    powershell -ExecutionPolicy Bypass -File "scripts/verify-backend-pgbouncer.ps1" `
      -ComposeFile $productionCompose `
      -BackendService "backend" `
      -PgbouncerService "pgbouncer" `
      -PgbouncerContainer "atmr-pgbouncer"
    Assert-LastExitCode -Context "verify-backend-pgbouncer"
  }

  Invoke-Step -Name "Check ws-service redis_up" -Action {
    Assert-WsRedisUp
  }

  Invoke-Step -Name "Check services healthy" -Action {
    Assert-ContainerHealthy -ContainerName "atmr-postgres"
    Assert-ContainerHealthy -ContainerName "atmr-pgbouncer"
    Assert-ContainerHealthy -ContainerName "atmr-redis"
    Assert-ContainerHealthyFromCompose -ComposeFile $productionCompose -ServiceName "backend"
    Assert-ContainerHealthy -ContainerName "atmr-ws-service"
    Assert-ContainerHealthy -ContainerName "atmr-kafka"
    Assert-ContainerHealthy -ContainerName "atmr-tracking-kafka-consumer-1"
  }

  Invoke-Step -Name "Smoke script officiel" -Action {
    powershell -ExecutionPolicy Bypass -File "scripts/smoke-scalability-proof.ps1"
    Assert-LastExitCode -Context "smoke-scalability-proof"
  }
}
catch {
  if (-not $failureStep) {
    Set-FailureState -Step "Unhandled exception" -Message $_.Exception.Message -FailureKind "SCRIPT_ERROR"
    Add-Check -Name "Unhandled exception" -Status "SCRIPT_ERROR" -Evidence $_.Exception.Message
  }
}
finally {
  Write-Report
  if ($generatedComposeEnvFile -and -not $KeepGeneratedEnvFile) {
    Remove-Item -LiteralPath $ComposeEnvFile -Force -ErrorAction SilentlyContinue
    Write-Log "Fichier env compose supprime apres run: $ComposeEnvFile"
  }
  Write-Host ""
  Write-Host "Verdict global: $verdict"
  if ($failureStep) {
    Write-Host "Premiere etape en echec: $failureStep"
  }
  Write-Host "Rapport: $reportPath"
  Write-Host "JSON: $jsonPath"
  Write-Host "Log: $logPath"
}

exit $exitCode
