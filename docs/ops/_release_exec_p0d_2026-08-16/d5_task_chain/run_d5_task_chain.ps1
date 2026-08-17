# P0-D D5 — post-cut mobile chain (read-only, backend frozen)
# Question: after cut, is background-location-task still invoked with Location payload,
# and how far does JS progress before PUT stops?
param(
  [string]$AdbSerial = "RFCW20QC53W",
  [int]$DriverId = 20135,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_release_exec_p0d_2026-08-16\d5_task_chain",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [int]$FgSeconds = 90,
  [int]$HomeSeconds = 180
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline.txt"
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
if ((& $adb -s $d get-state 2>&1 | Out-String).Trim() -ne "device") { throw "device offline" }

$pkg = (& $adb -s $d shell dumpsys package ch.liri.operations | Select-String "versionCode=|versionName=" | Select-Object -First 2) -join " "
TLog "D5_START driver=$DriverId $pkg BACKEND=FROZEN"

function Snap-Fgs([string]$label) {
  $raw = & $adb -s $d shell dumpsys activity services LocationTaskService 2>$null | Out-String
  $raw | Out-File (Join-Path $OutDir "fgs_$label.txt") -Encoding utf8
  $allow = if ($raw -match "getFgsAllowStart[^\r\n]*=\s*(\S+)") { $Matches[1] } else { "?" }
  $fg = if ($raw -match "isForeground=(\w+)") { $Matches[1] } else { "?" }
  $sr = if ($raw -match "startRequested=(\w+)") { $Matches[1] } else { "?" }
  TLog "FGS_$label isForeground=$fg startRequested=$sr allow=$allow"
}

function Snap-Fused([string]$label) {
  $raw = & $adb -s $d shell dumpsys activity service com.google.android.gms/.location.LocationService 2>$null | Out-String
  if (-not $raw -or $raw.Length -lt 40) {
    $raw = & $adb -s $d shell dumpsys location 2>$null | Out-String
  }
  $raw | Out-File (Join-Path $OutDir "fused_$label.txt") -Encoding utf8
  $ws = @(Select-String -InputObject $raw -Pattern "WorkSource.*10905|ch\.liri\.operations|F318B210|HIGH_ACCURACY" -AllMatches).Count
  TLog "FUSED_$label hits=$ws"
}

function Capture-Native([string]$label) {
  $file = Join-Path $OutDir "native_$label.txt"
  & $adb -s $d logcat -d -t 25000 2>$null |
    Select-String -Pattern "LocationTaskConsumer|LocationTaskService|TaskService|TaskManager|FusedLocation|GmsLocation|Location unavailable|Finished|background-location-task|onLocationResult|onLocationAvailability|executeTask|Background started FGS|FGS: Denied|Stopping service|Destroying service|stopSelf|ReactNativeJS" |
    ForEach-Object { $_.Line } |
    Out-File $file -Encoding utf8
  $fin = @(Select-String -Path $file -Pattern "Finished.*background-location-task|TaskService.*Finished" -EA SilentlyContinue).Count
  $unavail = @(Select-String -Path $file -Pattern "Location unavailable" -EA SilentlyContinue).Count
  $telemInvoke = @(Select-String -Path $file -Pattern "task_invoked|tracking\.background\.task" -EA SilentlyContinue).Count
  $telemSkip = @(Select-String -Path $file -Pattern "task\.skipped|enqueue_blocked|sqlite_headless" -EA SilentlyContinue).Count
  TLog "NATIVE_$label finishedish=$fin unavailable=$unavail telem_invokeish=$telemInvoke telem_skipish=$telemSkip"
}

function Snap-Put([string]$label) {
  $out = ssh -o BatchMode=yes $sshTarget "docker logs atmr-backend-1 --since 4m 2>&1 | grep 'PUT /api/v1/driver/me/location' | tail -30" 2>&1
  $out | Out-File (Join-Path $OutDir "put_$label.txt") -Encoding utf8
  $n = @($out | Where-Object { $_ -match 'PUT /api/v1/driver/me/location' }).Count
  $last = @($out | Where-Object { $_ -match 'PUT /api/v1/driver/me/location' } | Select-Object -Last 1)
  TLog "PUT_$label n_tail=$n last=$last"
}

function Snap-Loc([string]$label) {
  $py = @"
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app=create_app(); app.app_context().push()
from models import db
did=$DriverId
since=datetime.now(timezone.utc)-timedelta(minutes=15)
rows=list(db.session.execute(text('''
  SELECT created_at, recorded_at, sequence_id FROM driver_location_events
  WHERE driver_id=:d AND created_at>=:s ORDER BY created_at DESC LIMIT 8
'''),{'d':did,'s':since}).fetchall())
print('LABEL','$label')
print('NOW',datetime.now(timezone.utc).isoformat())
print('N',len(rows))
for r in rows:
  print('LOC', r[0], 'rec', r[1], 'seq', r[2])
"@
  $localPy = Join-Path $OutDir "_snap_loc.py"
  [IO.File]::WriteAllText($localPy, ($py -replace "`r`n","`n"))
  scp -o BatchMode=yes $localPy "${sshTarget}:/tmp/d5_snap_loc.py" | Out-Null
  ssh -o BatchMode=yes $sshTarget "docker cp /tmp/d5_snap_loc.py atmr-backend-1:/tmp/d5_snap_loc.py && docker exec atmr-backend-1 python /tmp/d5_snap_loc.py" 2>&1 |
    Tee-Object (Join-Path $OutDir "loc_$label.txt") |
    ForEach-Object { if ($_ -match '^(LABEL|NOW|N|LOC)') { TLog "LOC_$label $_" } }
}

& $adb -s $d logcat -c 2>$null | Out-Null
& $adb -s $d shell am start -n ch.liri.operations/.MainActivity 2>$null | Out-Null
Start-Sleep 3

TLog "PHASE_FG ${FgSeconds}s"
Snap-Fgs "FG0"
Snap-Fused "FG0"
Start-Sleep $FgSeconds
Capture-Native "FG"
Snap-Fgs "FG1"
Snap-Put "FG"
Snap-Loc "FG"

TLog "PHASE_HOME ${HomeSeconds}s cut_watch"
& $adb -s $d shell input keyevent KEYCODE_HOME 2>$null | Out-Null
$homeAt = Get-Date
TLog "HOME_AT $($homeAt.ToUniversalTime().ToString('o'))"

# sample every 30s
$deadline = $homeAt.AddSeconds($HomeSeconds)
$i = 0
while ((Get-Date) -lt $deadline) {
  Start-Sleep 30
  $i++
  Capture-Native "HOME_$i"
  Snap-Fgs "HOME_$i"
  Snap-Put "HOME_$i"
  Snap-Loc "HOME_$i"
}

Capture-Native "HOME_END"
Snap-Fused "HOME_END"
Snap-Fgs "HOME_END"
Snap-Put "HOME_END"
Snap-Loc "HOME_END"
TLog "D5_END"
