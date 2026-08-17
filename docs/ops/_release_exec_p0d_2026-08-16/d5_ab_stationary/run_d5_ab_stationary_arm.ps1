# P0-D D5 A/B — un bras stationnaire (read-only)
param(
  [Parameter(Mandatory=$true)][ValidateSet("PROD","DEV")][string]$Arm,
  [string]$AdbSerial = "100.81.106.54:43223",
  [int]$DriverId = 20135,
  [int]$Seconds = 240,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_release_exec_p0d_2026-08-16\d5_ab_stationary",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env"
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$prefix = Join-Path $OutDir ($Arm.ToLower())
$timeline = "${prefix}_timeline.txt"
function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) [$Arm] $msg"
  Add-Content -Path $timeline -Value $line
  Write-Host $line
}

Get-Content $DeployEnv | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  if ($_ -match '^\s*export\s+(\w+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'")) }
  elseif ($_ -match '^\s*(\w+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'")) }
}
$ssh = "$($env:SERVER_USER)@$($env:SERVER_HOST)"
$adb = $AdbPath
$d = $AdbSerial
if ((& $adb -s $d get-state 2>&1 | Out-String).Trim() -ne "device") { throw "device offline" }

$pkg = (& $adb -s $d shell dumpsys package ch.liri.operations | Select-String "versionCode=|versionName=" | Select-Object -First 2) -join " "
$deb = (& $adb -s $d shell dumpsys package ch.liri.operations | Select-String "flags=\[.*DEBUGGABLE" | Select-Object -First 1)
TLog "START $pkg debuggable=$deb seconds=$Seconds BACKEND=FROZEN"

# dumpsys request snapshot (Expo / WorkSource)
$locDump = & $adb -s $d shell dumpsys location 2>$null | Out-String
$locDump | Out-File "${prefix}_dumpsys_location.txt" -Encoding utf8
$reqLines = $locDump -split "`n" | Where-Object { $_ -match "ch\.liri\.operations|WorkSource\{10905|Request\[|minUpdate|HIGH_ACCURACY|ProviderRequest" }
$reqLines | Out-File "${prefix}_request_excerpt.txt" -Encoding utf8
TLog "REQUEST_LINES n=$($reqLines.Count)"

& $adb -s $d logcat -c 2>$null | Out-Null
& $adb -s $d shell am start -n ch.liri.operations/.MainActivity 2>$null | Out-Null
Start-Sleep 5

TLog "SAMPLE ${Seconds}s stationary"
Start-Sleep $Seconds

& $adb -s $d logcat -d -t 30000 2>$null |
  Select-String -Pattern "LocationTaskConsumer|Location unavailable|background-location-task|TaskService|FusedLocation|too close|too fast|GmsPassiveListener_FLP|ReactNativeJS.*(tracking|task_|enqueue)" |
  ForEach-Object { $_.Line } |
  Out-File "${prefix}_native.txt" -Encoding utf8

$f = "${prefix}_native.txt"
$fin = @(Select-String -Path $f -Pattern "background-location-task" -EA SilentlyContinue).Count
$un = @(Select-String -Path $f -Pattern "Location unavailable" -EA SilentlyContinue).Count
$tc = @(Select-String -Path $f -Pattern "too close" -EA SilentlyContinue).Count
$tf = @(Select-String -Path $f -Pattern "too fast" -EA SilentlyContinue).Count
$flp = @(Select-String -Path $f -Pattern "GmsPassiveListener_FLP" -EA SilentlyContinue).Count
$reg = @(Select-String -Path "${prefix}_request_excerpt.txt" -Pattern "WorkSource\{10905|ch\.liri\.operations" -EA SilentlyContinue).Count
TLog "NATIVE finishedish=$fin unavailable=$un too_close=$tc too_fast=$tf flp=$flp request_hits=$reg"

$puts = ssh -o BatchMode=yes $ssh "docker logs atmr-backend-1 --since 6m 2>&1 | grep 'PUT /api/v1/driver/me/location' | grep -v Darwin | tail -30" 2>&1
$puts | Out-File "${prefix}_put.txt" -Encoding utf8
TLog "PUT n=$(@($puts | Where-Object { $_ -match 'PUT' }).Count)"

$py = @"
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app=create_app(); app.app_context().push()
from models import db
since=datetime.now(timezone.utc)-timedelta(minutes=10)
rows=list(db.session.execute(text('''
  SELECT created_at, recorded_at, sequence_id FROM driver_location_events
  WHERE driver_id=:d AND created_at>=:s ORDER BY created_at DESC LIMIT 10
'''),{'d':$DriverId,'s':since}).fetchall())
print('NOW',datetime.now(timezone.utc).isoformat())
print('N',len(rows))
for r in rows:
  print('LOC', r[0], 'rec', r[1], 'seq', r[2])
"@
[IO.File]::WriteAllText("${prefix}_snap.py", ($py -replace "`r`n","`n"))
scp -o BatchMode=yes "${prefix}_snap.py" "${ssh}:/tmp/d5_ab_snap.py" | Out-Null
ssh -o BatchMode=yes $ssh "docker cp /tmp/d5_ab_snap.py atmr-backend-1:/tmp/d5_ab_snap.py && docker exec atmr-backend-1 python /tmp/d5_ab_snap.py" 2>&1 |
  Tee-Object "${prefix}_loc.txt" |
  ForEach-Object { if ($_ -match '^(NOW|N|LOC)') { TLog $_ } }

# post dumpsys
$locDump2 = & $adb -s $d shell dumpsys location 2>$null | Out-String
$locDump2 | Out-File "${prefix}_dumpsys_location_end.txt" -Encoding utf8
TLog "END"
