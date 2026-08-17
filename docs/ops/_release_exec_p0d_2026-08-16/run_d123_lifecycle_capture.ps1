# P0-D D1/D2/D3 — capture START/STOP/process-state Prod 126 (diagnostic only)
# Usage:
#   .\run_d123_lifecycle_capture.ps1 -AdbSerial "192.168.1.33:31803" -DriverId 20135

param(
  [Parameter(Mandatory = $true)][string]$AdbSerial,
  [int]$DriverId = 20135,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_release_exec_p0d_2026-08-16\d123_lifecycle",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [int]$WaitLoginSeconds = 90,
  [int]$FgStableSeconds = 40,
  [int]$HomeSeconds = 30
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline.txt"
function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) $msg"
  Add-Content -Path $timeline -Value $line -ErrorAction SilentlyContinue
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

$pkg = (& $adb -s $d shell dumpsys package ch.liri.operations 2>$null | Out-String)
if ($pkg -notmatch "versionCode=126") { throw "versionCode 126 required" }
if ($pkg -match "DEBUGGABLE") { throw "DEBUGGABLE detected - need release binary" }
& $adb -s $d reverse --remove-all 2>$null | Out-Null

$snapPyPath = Join-Path $OutDir "_snap_helper.py"
@'
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
import sys
label = sys.argv[1] if len(sys.argv) > 1 else "X"
did = int(sys.argv[2]) if len(sys.argv) > 2 else 20135
app = create_app()
app.app_context().push()
from models import db
since = datetime.now(timezone.utc) - timedelta(minutes=20)
print("LABEL", label)
print("NOW", datetime.now(timezone.utc).isoformat())
health = list(db.session.execute(text("""
  SELECT recorded_at, app_state, tracking_active, fgs_running, native_task_running,
         last_fix_age_seconds, native_last_fix_age_seconds,
         constraint_reason, native_start_error, trigger_reason, native_start_phase
  FROM driver_device_health_events
  WHERE driver_id=:did AND recorded_at>=:since
  ORDER BY recorded_at DESC LIMIT 30
"""), {"did": did, "since": since}).mappings())
print("HEALTH_N", len(health))
print("FGS_TRUE_N", sum(1 for h in health if h.get("fgs_running") is True))
print("FGS_FALSE_N", sum(1 for h in health if h.get("fgs_running") is False))
for h in health[:15]:
    print(
      "H", h.get("recorded_at"),
      "app=", h.get("app_state"),
      "trk=", h.get("tracking_active"),
      "fgs=", h.get("fgs_running"),
      "ntask=", h.get("native_task_running"),
      "fix_age=", h.get("last_fix_age_seconds"),
      "task_age=", h.get("native_last_fix_age_seconds"),
      "cstr=", h.get("constraint_reason"),
      "trig=", h.get("trigger_reason"),
      "phase=", h.get("native_start_phase"),
      "err=", (h.get("native_start_error") or "")[:100],
    )
locs = list(db.session.execute(text("""
  SELECT created_at, mission_id FROM driver_location_events
  WHERE driver_id=:did AND created_at>=:since
  ORDER BY created_at DESC LIMIT 20
"""), {"did": did, "since": since}).fetchall())
print("LOC_N", len(locs))
for r in locs[:10]:
    print("LOC", r[0], "mission=", r[1])
'@ | Set-Content -Path $snapPyPath -Encoding utf8
scp -o BatchMode=yes $snapPyPath "${sshTarget}:/tmp/d123_snap.py" | Out-Null
ssh -o BatchMode=yes $sshTarget "docker cp /tmp/d123_snap.py atmr-backend-1:/tmp/d123_snap.py" | Out-Null

function Snap-Prod([string]$label) {
  $out = ssh -o BatchMode=yes $sshTarget "docker exec atmr-backend-1 python /tmp/d123_snap.py $label $DriverId" 2>&1
  $out | Out-File (Join-Path $OutDir "snap_$label.txt") -Encoding utf8
  $useful = @($out | Where-Object { $_ -match '^(LABEL|NOW|HEALTH|FGS_|H |LOC)' })
  TLog ("SNAP_$label " + (($useful | Select-Object -First 12) -join " | "))
}

function Dump-Fgs([string]$label) {
  $svc = & $adb -s $d shell dumpsys activity services ch.liri.operations 2>$null | Out-String
  $svc | Out-File (Join-Path $OutDir "svc_$label.txt") -Encoding utf8
  $hit = @($svc -split "`n" | Where-Object {
    $_ -match "LocationTaskService|getFgsAllowStart|getFgsAllowWiu|startRequested|startForegroundCount|infoAllowStartForeground|isForeground|createdFromFg|destroyTime"
  })
  $hit | Out-File (Join-Path $OutDir "fgs_$label.txt") -Encoding utf8
  TLog ("FGS_$label " + (($hit | Select-Object -First 16) -join " || "))
}

function Dump-Proc([string]$label) {
  $p = & $adb -s $d shell dumpsys activity processes 2>$null | Out-String
  $hit = @($p -split "`n" | Where-Object { $_ -match "ch\.liri\.operations" } | Select-Object -First 25)
  $hit | Out-File (Join-Path $OutDir "proc_$label.txt") -Encoding utf8
  TLog ("PROC_$label " + (($hit | Select-Object -First 8) -join " || "))
}

function Capture-AmSlice([string]$label) {
  $file = Join-Path $OutDir "am_$label.txt"
  & $adb -s $d logcat -d -t 5000 2>$null |
    Select-String -Pattern "LocationTask|ForegroundService|startForeground|Stopping service|Destroying service|Background start|not allowed|ch\.liri\.operations|ExpoLocation|DENIED|denied|FGS" |
    ForEach-Object { $_.Line } |
    Out-File $file -Encoding utf8
  $n = @(Get-Content $file -ErrorAction SilentlyContinue).Count
  TLog "AM_$label lines=$n"
}

function Ui-Shot([string]$label) {
  & $adb -s $d shell screencap -p /sdcard/d123.png 2>$null | Out-Null
  & $adb -s $d pull /sdcard/d123.png (Join-Path $OutDir "ui_$label.png") 2>$null | Out-Null
}

TLog "D123_START driver=$DriverId device=$d out=$OutDir"
& $adb -s $d logcat -c 2>$null | Out-Null

# Soft relaunch (keep session if possible)
& $adb -s $d shell am start -n ch.liri.operations/.MainActivity 2>$null | Out-Null
Start-Sleep 3
TLog "LAUNCH"
Ui-Shot "launch"
Dump-Fgs "launch"
Dump-Proc "launch"
Snap-Prod "launch"

TLog "WAIT_FGS_LOC up to ${WaitLoginSeconds}s"
$deadline = (Get-Date).AddSeconds($WaitLoginSeconds)
$ready = $false
while ((Get-Date) -lt $deadline) {
  Snap-Prod "poll"
  $poll = Get-Content (Join-Path $OutDir "snap_poll.txt") -Raw -ErrorAction SilentlyContinue
  if ($poll -match "fgs=\s*True") {
    $ready = $true
    TLog "READY_FGS_TRUE"
    break
  }
  Start-Sleep 12
}
if (-not $ready) { TLog "WARN_FGS_NOT_TRUE_continuing" }

TLog "PHASE_FG_STABLE ${FgStableSeconds}s"
& $adb -s $d shell am start -n ch.liri.operations/.MainActivity 2>$null | Out-Null
Start-Sleep $FgStableSeconds
Dump-Fgs "BEFORE_HOME"
Dump-Proc "BEFORE_HOME"
Snap-Prod "BEFORE_HOME"
Capture-AmSlice "BEFORE_HOME"
Ui-Shot "BEFORE_HOME"

TLog "PHASE_HOME"
& $adb -s $d shell input keyevent KEYCODE_HOME 2>$null | Out-Null
Start-Sleep 2
Dump-Fgs "HOME_2s"
Dump-Proc "HOME_2s"
Capture-AmSlice "HOME_2s"
Snap-Prod "HOME_2s"

Start-Sleep 13
Dump-Fgs "HOME_15s"
Dump-Proc "HOME_15s"
Capture-AmSlice "HOME_15s"
Snap-Prod "HOME_15s"

$remain = [Math]::Max(0, $HomeSeconds - 15)
Start-Sleep $remain
Dump-Fgs "HOME_30s"
Dump-Proc "HOME_30s"
Capture-AmSlice "HOME_30s"
Snap-Prod "HOME_30s"

TLog "PHASE_BACK"
& $adb -s $d shell am start -n ch.liri.operations/.MainActivity 2>$null | Out-Null
Start-Sleep 20
Dump-Fgs "BACK"
Dump-Proc "BACK"
Capture-AmSlice "BACK"
Snap-Prod "BACK"
Ui-Shot "BACK"

& $adb -s $d logcat -d -t 10000 2>$null |
  Select-String -Pattern "LocationTask|ForegroundService|startForeground|Stopping service|Destroying service|Background start|not allowed|ch\.liri\.operations|ExpoLocation|DENIED|denied|FGS|stopLocation|startLocation" |
  ForEach-Object { $_.Line } |
  Out-File (Join-Path $OutDir "am_session_full.txt") -Encoding utf8

TLog "D123_END"
Write-Host "Artefacts: $OutDir"
