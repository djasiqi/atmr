# Smoke Play Internal Testing — runner (discriminant vs sideload)
# Usage:
#   .\run_smoke_play_internal.ps1 -AdbSerial "RFCW20QC53W" -DriverId 20135
# Prérequis: app installée DEPUIS Play Internal, versionCode=126, pas de Metro.

param(
  [Parameter(Mandatory = $true)][string]$AdbSerial,
  [int]$DriverId = 20135,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_release_exec_mobile_builds_2026-08-16",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [int]$FgSeconds = 120,
  [int]$HomeSeconds = 60,
  [int]$LockSeconds = 60,
  [int]$PostSeconds = 120,
  [int]$MinFgLoc = 4
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$prefix = "smoke_play"
$timeline = Join-Path $OutDir "${prefix}_timeline.txt"
function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) $msg"
  try { Add-Content -Path $timeline -Value $line -EA SilentlyContinue } catch {}
  Write-Host $line
}

Get-Content $DeployEnv | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  if ($_ -match '^\s*export\s+(\w+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'")) }
  elseif ($_ -match '^\s*(\w+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'")) }
}
$sshTarget = "$($env:SERVER_USER)@$($env:SERVER_HOST)"
$adb = $AdbPath
$d = $AdbSerial
$st = (& $adb -s $d get-state 2>&1 | Out-String).Trim()
if ($st -ne "device") { throw "Device $d state='$st'" }

# Guard: version + no Metro reverse
$pkg = (& $adb -s $d shell dumpsys package ch.liri.operations 2>$null | Out-String)
if ($pkg -notmatch "versionCode=126") { throw "versionCode attendu 126 introuvable" }
if ($pkg -notmatch "versionName=1.0.11") { throw "versionName attendu 1.0.11 introuvable" }
$rev = (& $adb -s $d reverse --list 2>$null | Out-String)
if ($rev -match "tcp:8081|tcp:15100") {
  TLog "WARN adb reverse present (ne pas utiliser Metro): $($rev.Trim())"
}
# Installer / installerPackageName hint (Play vs adb)
$inst = (& $adb -s $d shell pm path ch.liri.operations 2>$null | Out-String).Trim()
$installer = (& $adb -s $d shell cmd package get-installer-package-name ch.liri.operations 2>$null | Out-String).Trim()
TLog "SMOKE_PLAY_START driver=$DriverId device=$d installer=$installer path=$inst version=1.0.11/126"

$snapPy = @"
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
import sys
label = sys.argv[1] if len(sys.argv) > 1 else 'X'
did = int(sys.argv[2]) if len(sys.argv) > 2 else 20135
app = create_app(); ctx = app.app_context(); ctx.push()
from models import db
since = datetime.now(timezone.utc) - timedelta(minutes=45)
print('LABEL', label)
print('NOW', datetime.now(timezone.utc).isoformat())
active = list(db.session.execute(text('''
  SELECT id, status::text FROM booking
  WHERE driver_id=:did AND status::text IN ('ASSIGNED','ACCEPTED','EN_ROUTE','IN_PROGRESS')
  ORDER BY id DESC LIMIT 5
'''), {'did': did}).fetchall())
print('ACTIVE', [(a[0], a[1]) for a in active])
health = list(db.session.execute(text('''
  SELECT recorded_at, app_state, tracking_active, fgs_running, native_task_running,
         last_fix_age_seconds, native_last_fix_age_seconds,
         constraint_reason, native_start_error, trigger_reason, native_start_phase
  FROM driver_device_health_events
  WHERE driver_id=:did AND recorded_at>=:since
  ORDER BY recorded_at DESC LIMIT 25
'''), {'did': did, 'since': since}).mappings())
print('HEALTH_N', len(health))
print('NATIVE_ERR', sum(1 for h in health if h.get('native_start_error')))
print('FGS_NOT_RUNNING_N', sum(1 for h in health if (h.get('constraint_reason') or '') == 'fgs_not_running'))
for h in health[:12]:
    print(
      'H', h.get('recorded_at'),
      'app=', h.get('app_state'),
      'trk=', h.get('tracking_active'),
      'fgs=', h.get('fgs_running'),
      'ntask=', h.get('native_task_running'),
      'fix_age=', h.get('last_fix_age_seconds'),
      'task_invoke_age=', h.get('native_last_fix_age_seconds'),
      'cstr=', h.get('constraint_reason'),
      'trig=', h.get('trigger_reason'),
      'phase=', h.get('native_start_phase'),
      'err=', (h.get('native_start_error') or '')[:80],
    )
locs = list(db.session.execute(text('''
  SELECT created_at, mission_id FROM driver_location_events
  WHERE driver_id=:did AND created_at>=:since
  ORDER BY created_at DESC LIMIT 40
'''), {'did': did, 'since': since}).fetchall())
print('LOC_N', len(locs))
for r in locs[:15]:
    print('LOC', r[0], 'mission=', r[1])
# LOC count last 3 minutes (phase freshness)
recent = datetime.now(timezone.utc) - timedelta(minutes=3)
loc3 = db.session.execute(text('''
  SELECT count(*) FROM driver_location_events
  WHERE driver_id=:did AND created_at>=:s
'''), {'did': did, 's': recent}).scalar()
print('LOC_LAST_3M', loc3)
"@
$snapLocal = Join-Path $env:TEMP "smoke_play_snap_helper.py"
[System.IO.File]::WriteAllText($snapLocal, $snapPy.Replace("`r`n", "`n"))
scp -o BatchMode=yes $snapLocal "${sshTarget}:/tmp/smoke_play_snap_helper.py" | Out-Null
ssh -o BatchMode=yes $sshTarget "docker cp /tmp/smoke_play_snap_helper.py atmr-backend-1:/tmp/smoke_play_snap_helper.py" | Out-Null

function Capture-Logcat([string]$label) {
  $pidApp = (& $adb -s $d shell pidof ch.liri.operations 2>$null | Out-String).Trim()
  $file = Join-Path $OutDir "${prefix}_logcat_$label.txt"
  if ($pidApp) {
    & $adb -s $d logcat -d --pid=$pidApp -t 6000 *:S ReactNativeJS:I | Out-File $file -Encoding utf8
  } else {
    "NO_PID" | Out-File $file -Encoding utf8
  }
  $authSkip = @(Select-String -Path $file -Pattern "auth_not_usable" -EA SilentlyContinue).Count
  $errFg = @(Select-String -Path $file -Pattern "ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED" -EA SilentlyContinue).Count
  $nativeErr = @(Select-String -Path $file -Pattern "native_start_error" -EA SilentlyContinue).Count
  $both = @(Select-String -Path $file -Pattern 'start_in_flight.: 1' -EA SilentlyContinue | Where-Object { $_.Line -match 'stop_in_flight.: 1' }).Count
  $genNull = @(Select-String -Path $file -Pattern "generation.: null|generation=null" -EA SilentlyContinue).Count
  $metro = @(Select-String -Path $file -Pattern "Metro|DevLauncher|localhost:8081|:15100" -EA SilentlyContinue).Count
  $nloStart = @(Select-String -Path $file -Pattern "nlo_start_" -EA SilentlyContinue).Count
  $nloStop = @(Select-String -Path $file -Pattern "nlo_stop_" -EA SilentlyContinue).Count
  $msg = "SIG_$label auth_not_usable=$authSkip err_fg=$errFg native_start_error=$nativeErr overlap=$both gen_null=$genNull metroish=$metro nlo_start=$nloStart nlo_stop=$nloStop pid=$pidApp"
  $msg | Tee-Object -FilePath (Join-Path $OutDir "${prefix}_summary_$label.txt")
  TLog $msg
  # Extract nlo lines for evidence
  Select-String -Path $file -Pattern "nlo_start_|nlo_stop_|fgs_not_running|tracking\.(watch|lifecycle)|constraint" -EA SilentlyContinue |
    Select-Object -Last 40 |
    ForEach-Object { $_.Line } |
    Out-File (Join-Path $OutDir "${prefix}_nlo_$label.txt") -Encoding utf8
}

function Snap-Prod([string]$label) {
  $out = ssh -o BatchMode=yes $sshTarget "docker exec atmr-backend-1 python /tmp/smoke_play_snap_helper.py $label $DriverId" 2>&1
  $out | Out-File (Join-Path $OutDir "${prefix}_snap_$label.txt") -Encoding utf8
  $useful = @($out | Where-Object { $_ -match '^(LABEL|NOW|ACTIVE|HEALTH|NATIVE|FGS_|H |LOC)' })
  TLog ("SNAP_$label " + (($useful | Select-Object -First 14) -join " | "))
}

function Ensure-Foreground {
  & $adb -s $d shell am start -n ch.liri.operations/.MainActivity 2>$null | Out-Null
  Start-Sleep 2
}
function Go-Home {
  & $adb -s $d shell input keyevent KEYCODE_HOME 2>$null | Out-Null
  Start-Sleep 1
}
function Lock-Device {
  & $adb -s $d shell input keyevent KEYCODE_SLEEP 2>$null | Out-Null
  Start-Sleep 1
}
function Unlock-Device {
  & $adb -s $d shell input keyevent KEYCODE_WAKEUP 2>$null | Out-Null
  Start-Sleep 1
  & $adb -s $d shell input keyevent 82 2>$null | Out-Null
  Start-Sleep 1
  & $adb -s $d shell input swipe 540 2000 540 400 250 2>$null | Out-Null
  Start-Sleep 2
}

& $adb -s $d logcat -c 2>$null
Ensure-Foreground
Capture-Logcat "PRE"
Snap-Prod "PRE"

TLog "PHASE_FG ${FgSeconds}s (target >= $MinFgLoc LOC)"
Ensure-Foreground
Start-Sleep $FgSeconds
Capture-Logcat "FG"
Snap-Prod "FG"

TLog "PHASE_HOME ${HomeSeconds}s"
Go-Home
Start-Sleep $HomeSeconds
Capture-Logcat "HOME"
Snap-Prod "HOME"

TLog "PHASE_BACK_APP"
Ensure-Foreground
Start-Sleep 20
Capture-Logcat "BACK"
Snap-Prod "BACK"

TLog "PHASE_LOCK ${LockSeconds}s"
Ensure-Foreground
Start-Sleep 5
Lock-Device
Start-Sleep $LockSeconds
Unlock-Device
Ensure-Foreground
Start-Sleep 25
Capture-Logcat "LOCK"
Snap-Prod "LOCK"

TLog "PHASE_POST ${PostSeconds}s"
Ensure-Foreground
Start-Sleep $PostSeconds
Capture-Logcat "POST"
Snap-Prod "POST"

TLog "SMOKE_PLAY_END - scorer FG/HOME/LOCK LOC + FGS_NOT_RUNNING_N + signaux"
