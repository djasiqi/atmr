# D5 canary — analyse post-hoc logcat (étroit)
# Usage:
#   .\analyze_d5_canary.ps1 -LogcatPath .\d5_canary\logcat_continuous.txt
#   .\analyze_d5_canary.ps1 -LogcatPath .\logcat.txt -OutDir .\d5_canary
param(
  [Parameter(Mandatory = $true)][string]$LogcatPath,
  [string]$OutDir = "",
  [int]$StormWindowMs = 500
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $LogcatPath)) { throw "Logcat introuvable: $LogcatPath" }
if (-not $OutDir) { $OutDir = Split-Path -Parent (Resolve-Path $LogcatPath) }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$lines = Get-Content -LiteralPath $LogcatPath -ErrorAction Stop
$unregister = @($lines | Select-String -Pattern "Unregistering\s+'background-location-task'|Unregistering task.*background-location" -AllMatches)
$register = @($lines | Select-String -Pattern "Registering\s+'background-location-task'|registerTaskAsync|hasRegistered.*background-location" -AllMatches)
$finished = @($lines | Select-String -Pattern "Finished (background )?task 'background-location-task'|Finished task 'background-location-task'" -AllMatches)
$missing = @($lines | Select-String -Pattern "Could not find a location task" -AllMatches)
$abandoned = @($lines | Select-String -Pattern "tracking\.lifecycle\.stop\.abandoned|pre_native_abort_guard" -AllMatches)
$executed = @($lines | Select-String -Pattern "tracking\.lifecycle\.stop\.executed" -AllMatches)
$requested = @($lines | Select-String -Pattern "tracking\.lifecycle\.stop\.requested|tracking\.background\.stop_requested" -AllMatches)
$l2 = @($lines | Select-String -Pattern "recovery_level[=:]L2|self_heal_restart" -AllMatches)
$transientPending = @($lines | Select-String -Pattern "transient_loss\.pending" -AllMatches)
$transientConfirmed = @($lines | Select-String -Pattern "transient_loss\.confirmed" -AllMatches)

function Get-TsMs([string]$line) {
  # Formats courants: "08-16 21:18:49.975" ou ISO
  if ($line -match '(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\.(\d{3})') {
    $h = [int]$Matches[3]; $m = [int]$Matches[4]; $s = [int]$Matches[5]; $ms = [int]$Matches[6]
    return ((($h * 60 + $m) * 60 + $s) * 1000 + $ms)
  }
  return $null
}

$stormPairs = 0
$regTs = @()
foreach ($m in $register) {
  $t = Get-TsMs $m.Line
  if ($null -ne $t) { $regTs += $t }
}
foreach ($m in $unregister) {
  $t = Get-TsMs $m.Line
  if ($null -eq $t) { continue }
  foreach ($rt in $regTs) {
    if ([Math]::Abs($rt - $t) -le $StormWindowMs) { $stormPairs++; break }
  }
}

$verdict = "PASS"
$reasons = @()
if ($missing.Count -gt 0) { $verdict = "FAIL"; $reasons += "missing_location_task=$($missing.Count)" }
if ($stormPairs -gt 0) { $verdict = "FAIL"; $reasons += "register_unregister_storm_pairs=$stormPairs" }
# Unregister brut > 0 n'est pas FAIL auto (STOP légitimes) — flag REVIEW si élevé sans abandoned/executed ratio
if ($unregister.Count -gt 3 -and $abandoned.Count -eq 0 -and $finished.Count -lt 5) {
  $verdict = "REVIEW"
  $reasons += "unregister_elevated_without_ownership_signals"
}

$report = @"
# D5 canary analyze — $(Get-Date -Format o)

Logcat: $LogcatPath
StormWindowMs: $StormWindowMs

## Compteurs

| Signal | Count |
|--------|------:|
| Unregister background-location-task | $($unregister.Count) |
| Register (approx) | $($register.Count) |
| Storm pairs (≤${StormWindowMs}ms) | $stormPairs |
| Finished background-location-task | $($finished.Count) |
| Could not find a location task | $($missing.Count) |
| stop.requested (JS/native) | $($requested.Count) |
| stop.abandoned | $($abandoned.Count) |
| stop.executed | $($executed.Count) |
| self_heal / L2 hits | $($l2.Count) |
| transient_loss.pending | $($transientPending.Count) |
| transient_loss.confirmed | $($transientConfirmed.Count) |

## Verdict auto

**$verdict**

Reasons: $(if ($reasons.Count) { $reasons -join '; ' } else { 'none' })

## Interprétation manuelle obligatoire

- Classer chaque Unregister : LEGITIME (logout / mission terminale) vs INATTENDU (IN_PROGRESS).
- Smoking gun PASS ownership : stop.requested → abandoned (stale) → Finished continue.
- Smoking gun FAIL D5 : Unregister↔Register storm + cut Finished/PUT/LOC.

## Gate figée

``````
unexpected Unregister = 0  (manuel)
storm = 0
missing task = 0
Finished cadence = continue
PUT/LOC = continue (hors de ce script — check SSH/PG)
DISTRIBUTION = NO-GO jusqu'à CANARY VALIDATED
``````
"@

$outFile = Join-Path $OutDir "analyze_verdict.txt"
$report | Out-File -FilePath $outFile -Encoding utf8
Write-Host $report
Write-Host "Wrote $outFile"
