# Smoke Android production binary — standalone (no Metro)
param(
  [Parameter(Mandatory = $true)][string]$AdbSerial,
  [int]$DriverId = 20135,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_release_exec_mobile_builds_2026-08-16",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [int]$FgSeconds = 90,
  [int]$BgSeconds = 90,
  [int]$LockSeconds = 70
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "smoke_timeline.txt"
function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) $msg"
  Add-Content -Path $timeline -Value $line
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

TLog "SMOKE_START driver=$DriverId device=$d standalone=1 tip=286737a2 versionCode=126"

# Upload snap helper once (avoids quoting hell in ssh python -c)
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
  SELECT recorded_at, app_state, fgs_running, native_task_running,
         last_fix_age_seconds, native_last_fix_age_seconds,
         constraint_reason, native_start_error, health_class, observability_class
  FROM driver_device_health_events
  WHERE driver_id=:did AND recorded_at>=:since
  ORDER BY recorded_at DESC LIMIT 20
'''), {'did': did, 'since': since}).mappings())
print('HEALTH_N', len(health))
print('NATIVE_ERR', sum(1 for h in health if h.get('native_start_error')))
for h in health[:8]:
    print('H', h.get('recorded_at'), 'app=', h.get('app_state'), 'fgs=', h.get('fgs_running'),
          'ntask=', h.get('native_task_running'),
          'fix=', h.get('last_fix_age_seconds'), 'nfix=', h.get('native_last_fix_age_seconds'),
          'hc=', h.get('health_class') or h.get('observability_class'),
          'err=', (h.get('native_start_error') or '')[:80])
locs = list(db.session.execute(text('''
  SELECT created_at, mission_id FROM driver_location_events
  WHERE driver_id=:did AND created_at>=:since
  ORDER BY created_at DESC LIMIT 30
'''), {'did': did, 'since': since}).fetchall())
print('LOC_N', len(locs))
for r in locs[:12]:
    print('LOC', r[0], 'mission=', r[1])
"@
$snapLocal = Join-Path $env:TEMP "smoke_snap_helper.py"
[System.IO.File]::WriteAllText($snapLocal, $snapPy.Replace("`r`n", "`n"))
scp -o BatchMode=yes $snapLocal "${sshTarget}:/tmp/smoke_snap_helper.py" | Out-Null
ssh -o BatchMode=yes $sshTarget "docker cp /tmp/smoke_snap_helper.py atmr-backend-1:/tmp/smoke_snap_helper.py" | Out-Null

function Capture-Logcat([string]$label) {
  $pidApp = (& $adb -s $d shell pidof ch.liri.operations 2>$null | Out-String).Trim()
  $file = Join-Path $OutDir "smoke_logcat_$label.txt"
  if ($pidApp) {
    & $adb -s $d logcat -d --pid=$pidApp -t 5000 *:S ReactNativeJS:I | Out-File $file -Encoding utf8
  } else {
    "NO_PID" | Out-File $file -Encoding utf8
  }
  $authSkip = @(Select-String -Path $file -Pattern "auth_not_usable" -EA SilentlyContinue).Count
  $errFg = @(Select-String -Path $file -Pattern "ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED" -EA SilentlyContinue).Count
  $nativeErr = @(Select-String -Path $file -Pattern "native_start_error" -EA SilentlyContinue).Count
  $both = @(Select-String -Path $file -Pattern 'start_in_flight.: 1' -EA SilentlyContinue | Where-Object { $_.Line -match 'stop_in_flight.: 1' }).Count
  $genNull = @(Select-String -Path $file -Pattern "generation.: null|generation=null" -EA SilentlyContinue).Count
  $metro = @(Select-String -Path $file -Pattern "Metro|DevLauncher|localhost:8081|:15100" -EA SilentlyContinue).Count
  $msg = "SIG_$label auth_not_usable=$authSkip err_fg=$errFg native_start_error=$nativeErr overlap=$both gen_null=$genNull metroish=$metro pid=$pidApp"
  $msg | Tee-Object -FilePath (Join-Path $OutDir "smoke_summary_$label.txt")
  TLog $msg
}

function Snap-Prod([string]$label) {
  $out = ssh -o BatchMode=yes $sshTarget "docker exec atmr-backend-1 python /tmp/smoke_snap_helper.py $label $DriverId" 2>&1
  $out | Out-File (Join-Path $OutDir "smoke_snap_$label.txt") -Encoding utf8
  $useful = @($out | Where-Object { $_ -match '^(LABEL|NOW|ACTIVE|HEALTH|NATIVE|H |LOC)' })
  TLog ("SNAP_$label " + (($useful | Select-Object -First 12) -join " | "))
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

TLog "PHASE_FG ${FgSeconds}s"
Ensure-Foreground
Start-Sleep $FgSeconds
Capture-Logcat "FG"
Snap-Prod "FG"

TLog "PHASE_BG ${BgSeconds}s"
Go-Home
Start-Sleep $BgSeconds
Capture-Logcat "BG"
Snap-Prod "BG"

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

TLog "SMOKE_END"
