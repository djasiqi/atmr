# Canary LOC diurne post-release — runner léger
# Usage:
#   .\run_canary_diurne.ps1 -AdbSerial "100.x.x.x:PORT" -DriverId 19
# Prérequis: device online, app loguée, SSH deploy configuré via .local.deploy.env

param(
  [Parameter(Mandatory = $true)][string]$AdbSerial,
  [int]$DriverId = 19,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_release_exec_canary_diurne_2026-08-16",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env"
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline.txt"
function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) $msg"
  Add-Content -Path $timeline -Value $line
  Write-Host $line
}

# Load SERVER_USER / SERVER_HOST
Get-Content $DeployEnv | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  if ($_ -match '^\s*export\s+(\w+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'")) }
  elseif ($_ -match '^\s*(\w+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'")) }
}
$sshTarget = "$($env:SERVER_USER)@$($env:SERVER_HOST)"

$adb = $AdbPath
$d = $AdbSerial
$st = (& $adb -s $d get-state 2>&1 | Out-String).Trim()
if ($st -ne "device") {
  throw "Device $d state='$st' (attendu device). Reconnecte wireless debugging puis relance."
}
TLog "CANARY_START driver=$DriverId device=$d"

function Capture-Logcat([string]$label) {
  $pidApp = (& $adb -s $d shell pidof ch.liri.operations 2>$null | Out-String).Trim()
  $file = Join-Path $OutDir "logcat_$label.txt"
  if ($pidApp) {
    & $adb -s $d logcat -d --pid=$pidApp -t 3000 *:S ReactNativeJS:I | Out-File $file -Encoding utf8
  } else {
    "NO_PID" | Out-File $file -Encoding utf8
  }
  $authSkip = @(Select-String -Path $file -Pattern "auth_not_usable" -EA SilentlyContinue).Count
  $errFg = @(Select-String -Path $file -Pattern "ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED" -EA SilentlyContinue).Count
  $nativeErr = @(Select-String -Path $file -Pattern "native_start_error" -EA SilentlyContinue).Count
  $both = @(Select-String -Path $file -Pattern 'start_in_flight.: 1' -EA SilentlyContinue | Where-Object { $_.Line -match 'stop_in_flight.: 1' }).Count
  $genNull = @(Select-String -Path $file -Pattern "generation.: null|generation=null" -EA SilentlyContinue).Count
  $hol = @(Select-String -Path $file -Pattern "ingested_non_persisted|HOL|enqueue_blocked" -EA SilentlyContinue).Count
  $msg = "SIG_$label auth_not_usable=$authSkip err_fg=$errFg native_start_error=$nativeErr overlap=$both gen_null=$genNull holish=$hol pid=$pidApp"
  $msg | Tee-Object -FilePath (Join-Path $OutDir "summary_$label.txt")
  TLog $msg
}

function Snap-Prod([string]$label) {
  $remote = @"
set -euo pipefail
RPW=`$(grep -E '^REDIS_PASSWORD=' /srv/atmr/.env.production | head -1 | cut -d= -f2- | sed 's/^["\x27]//;s/["\x27]`$//')
echo LABEL $label
echo NOW `$(date -u -Iseconds)
echo CLAIM_KEYS=`$(docker exec atmr-redis redis-cli -a "`$RPW" --no-auth-warning --scan --pattern 'atmr:driver_location:event:*' | wc -l)
docker exec atmr-backend-1 python -c "
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app=create_app(); ctx=app.app_context(); ctx.push()
from models import db
since=datetime.now(timezone.utc)-timedelta(minutes=30)
did=$DriverId
health=list(db.session.execute(text('''
  SELECT recorded_at, app_state, fgs_running, native_task_running,
         last_fix_age_seconds, native_last_fix_age_seconds,
         constraint_reason, native_start_error, health_class, observability_class
  FROM driver_device_health_events
  WHERE driver_id=:did AND recorded_at>=:since
  ORDER BY recorded_at DESC LIMIT 15
'''),{'did':did,'since':since}).mappings())
print('HEALTH_N', len(health))
print('NATIVE_ERR', sum(1 for h in health if h.get('native_start_error')))
for h in health[:6]:
  print('H', h.get('recorded_at'), 'app=', h.get('app_state'), 'fgs=', h.get('fgs_running'),
        'fix=', h.get('last_fix_age_seconds'), 'nfix=', h.get('native_last_fix_age_seconds'),
        'hc=', h.get('health_class') or h.get('observability_class'),
        'err=', (h.get('native_start_error') or '')[:80])
locs=list(db.session.execute(text('''
  SELECT created_at, mission_id FROM driver_location_events
  WHERE driver_id=:did AND created_at>=:since
  ORDER BY created_at DESC LIMIT 20
'''),{'did':did,'since':since}).fetchall())
print('LOC_N', len(locs))
for r in locs[:8]:
  print('LOC', r[0], 'mission=', r[1])
"
"@
  $tmp = Join-Path $env:TEMP "snap_diurne_$label.sh"
  [System.IO.File]::WriteAllText($tmp, $remote.Replace("`r`n", "`n"))
  scp -o BatchMode=yes $tmp "${sshTarget}:/tmp/snap_diurne_$label.sh" | Out-Null
  $out = ssh -o BatchMode=yes $sshTarget "bash /tmp/snap_diurne_$label.sh" 2>&1
  $out | Out-File (Join-Path $OutDir "snap_$label.txt") -Encoding utf8
  TLog ("SNAP_$label " + (($out | Select-Object -First 8) -join " | "))
}

# Baseline
& $adb -s $d logcat -c 2>$null
Capture-Logcat "PRE"
Snap-Prod "PRE"

TLog "MANUAL: reste en FG 2-3 min (GPS on, app visible) puis Entrée"
Read-Host "FG done"
Capture-Logcat "FG"
Snap-Prod "FG"

TLog "MANUAL: HOME / BG 2-3 min puis Entrée"
Read-Host "BG done"
Capture-Logcat "BG"
Snap-Prod "BG"

TLog "MANUAL: lock 60-90s puis unlock, Entrée"
Read-Host "LOCK done"
Capture-Logcat "LOCK"
Snap-Prod "LOCK"

TLog "MANUAL: déplacement réel 3-5 min puis Entrée"
Read-Host "MOVE done"
Capture-Logcat "MOVE"
Snap-Prod "MOVE"

TLog "CANARY_END — scorera summary_*.txt + snap_*.txt (LOC_N doit progresser)"
