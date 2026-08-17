# D5 C2 canary - churn FG <-> HOME (binary 127)
# Backend observation only. No force-stop. Mission stays IN_PROGRESS.
param(
  [string]$AdbSerial = "100.81.106.54:43223",
  [int]$DriverId = 20135,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_release_exec_p0d_2026-08-16\d5_canary",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [int]$Cycles = 8,
  [int]$FgSec = 40,
  [int]$HomeSec = 40,
  [int]$PollSec = 20
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline_C2.txt"
$csv = Join-Path $OutDir "samples_C2.csv"
$logcatFile = Join-Path $OutDir "logcat_C2_continuous.txt"
"ts,phase,cycle,elapsed_s,fgCount,binds,isFg,startReq,finished_delta,unregister_delta,put_n,loc_n,note" | Out-File $csv -Encoding utf8

function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) [C2] $msg"
  try { [IO.File]::AppendAllText($timeline, $line + [Environment]::NewLine) } catch {}
  Write-Host $line
}

if (Test-Path $DeployEnv) {
  Get-Content $DeployEnv | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    if ($_ -match '^\s*export\s+(\w+)=(.*)$') {
      Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'"))
    }
    elseif ($_ -match '^\s*(\w+)=(.*)$') {
      Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'"))
    }
  }
}
$ssh = "$($env:SERVER_USER)@$($env:SERVER_HOST)"
$adb = $AdbPath
$d = $AdbSerial
$pkg = "ch.liri.operations"
$state = (& $adb -s $d get-state 2>&1 | Out-String).Trim()
if ($state -ne "device") { throw "device offline" }

$pkgDump = & $adb -s $d shell dumpsys package $pkg 2>&1 | Out-String
$vc = if ($pkgDump -match "versionCode=(\d+)") { [int]$Matches[1] } else { -1 }
$vn = if ($pkgDump -match "versionName=([^\s]+)") { $Matches[1] } else { "?" }
TLog "PREFLIGHT versionName=$vn versionCode=$vc"
if ($vc -ne 127) { TLog "ABORT expected versionCode 127"; exit 2 }

function Get-SvcMetrics {
  $raw = & $adb -s $d shell dumpsys activity services LocationTaskService 2>$null | Out-String
  $fg = if ($raw -match "startForegroundCount=(\d+)") { [int]$Matches[1] } else { -1 }
  $isFg = if ($raw -match "isForeground=(true|false)") { $Matches[1] } else { "NA" }
  $startReq = if ($raw -match "startRequested=(true|false)") { $Matches[1] } else { "NA" }
  $binds = 0
  $inPer = $false
  foreach ($line in ($raw -split "`n")) {
    if ($line -match "Per-process Connections:") { $inPer = $true; continue }
    if ($inPer -and $line -match "ConnectionRecord\{") { $binds++ }
    if ($inPer -and $line -match "All Connections:") { break }
  }
  return [pscustomobject]@{ FgCount = $fg; Binds = $binds; IsForeground = $isFg; StartRequested = $startReq }
}

function Get-NativeDelta {
  $tmp = Join-Path $OutDir "_nat_tmp_C2.txt"
  & $adb -s $d logcat -d -t 4000 2>$null |
    Select-String -Pattern "Unregistering|Registering|Finished task 'background-location-task'|Could not find a location task|transient_loss|tracking\.lifecycle|stop_requested" |
    ForEach-Object { $_.Line } |
    Out-File $tmp -Encoding utf8
  $fin = @(Select-String -Path $tmp -Pattern "Finished task 'background-location-task'" -EA SilentlyContinue).Count
  $unreg = @(Select-String -Path $tmp -Pattern "Unregistering" -EA SilentlyContinue).Count
  $transient = @(Select-String -Path $tmp -Pattern "transient_loss" -EA SilentlyContinue).Count
  return [pscustomobject]@{ Finished = $fin; Unregister = $unreg; Transient = $transient }
}

function Get-PutLoc {
  $putN = -1
  $locN = -1
  if (-not $env:SERVER_HOST) { return [pscustomobject]@{ PutN = $putN; LocN = $locN } }
  $puts = ssh -o BatchMode=yes -o ConnectTimeout=8 $ssh "docker logs atmr-backend-1 --since 60s 2>&1 | grep 'PUT /api/v1/driver/me/location' | grep -v Darwin | wc -l" 2>&1
  if ("$puts" -match '(\d+)') { $putN = [int]$Matches[1] }
  $py = @"
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app=create_app(); app.app_context().push()
from models import db
DRIVER_ID=$DriverId
since=datetime.now(timezone.utc)-timedelta(seconds=60)
n=db.session.execute(text('SELECT count(*) FROM driver_location_events WHERE driver_id=:d AND created_at>=:s'),{'d':DRIVER_ID,'s':since}).scalar()
print('N', int(n or 0))
"@
  $pyFile = Join-Path $OutDir "_loc_snap_C2.py"
  [IO.File]::WriteAllText($pyFile, ($py -replace "`r`n", "`n"))
  scp -o BatchMode=yes -o ConnectTimeout=8 $pyFile "${ssh}:/tmp/d5_c2_snap.py" 2>$null | Out-Null
  $locOut = ssh -o BatchMode=yes -o ConnectTimeout=12 $ssh "docker cp /tmp/d5_c2_snap.py atmr-backend-1:/tmp/d5_c2_snap.py && docker exec atmr-backend-1 python /tmp/d5_c2_snap.py 2>/dev/null" 2>&1
  if ("$locOut" -match '(?m)^N (\d+)') { $locN = [int]$Matches[1] }
  return [pscustomobject]@{ PutN = $putN; LocN = $locN }
}

function Sample([string]$phase, [int]$cycle, [int]$elapsed) {
  $svc = Get-SvcMetrics
  $nat = Get-NativeDelta
  $pl = Get-PutLoc
  $ts = Get-Date -Format o
  $row = "$ts,$phase,$cycle,$elapsed,$($svc.FgCount),$($svc.Binds),$($svc.IsForeground),$($svc.StartRequested),$($nat.Finished),$($nat.Unregister),$($pl.PutN),$($pl.LocN),transient=$($nat.Transient)"
  [IO.File]::AppendAllText($csv, $row + [Environment]::NewLine)
  TLog "SAMPLE phase=$phase c=$cycle t=+${elapsed}s fg=$($svc.FgCount) binds=$($svc.Binds) isFg=$($svc.IsForeground) startReq=$($svc.StartRequested) finished=$($nat.Finished) unreg=$($nat.Unregister) transient=$($nat.Transient) PUT=$($pl.PutN) LOC=$($pl.LocN)"
  return $svc
}

function Run-Phase([string]$phase, [int]$cycle, [int]$sec) {
  TLog "PHASE_BEGIN $phase cycle=$cycle sec=$sec"
  if ($phase -eq "HOME") {
    & $adb -s $d shell input keyevent KEYCODE_HOME 2>$null | Out-Null
  } else {
    & $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null
  }
  $t0 = Get-Date
  Sample $phase $cycle 0 | Out-Null
  $elapsed = 0
  while ($elapsed -lt $sec) {
    Start-Sleep $PollSec
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds
    if ($elapsed -gt $sec) { $elapsed = $sec }
    Sample $phase $cycle $elapsed | Out-Null
  }
  TLog "PHASE_END $phase cycle=$cycle"
}

& $adb -s $d logcat -c 2>$null | Out-Null
if (Test-Path $logcatFile) { Remove-Item $logcatFile -Force }
$errLog = Join-Path $OutDir "logcat_C2_stderr.txt"
$logcatProc = Start-Process -FilePath $adb -ArgumentList @("-s", $d, "logcat", "-v", "threadtime") -RedirectStandardOutput $logcatFile -RedirectStandardError $errLog -PassThru -WindowStyle Hidden
TLog "LOGCAT_PID=$($logcatProc.Id) file=$logcatFile"
TLog "BACKEND=OBSERVATION_ONLY DRIVER=$DriverId Cycles=$Cycles FgSec=$FgSec HomeSec=$HomeSec"

& $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null
Start-Sleep 3
$ready = Get-SvcMetrics
TLog "READY_CHECK isFg=$($ready.IsForeground) startReq=$($ready.StartRequested) fgCount=$($ready.FgCount)"
if ($ready.StartRequested -ne "true") {
  TLog "WARN tracking not clearly active - continuing C2"
}

for ($c = 1; $c -le $Cycles; $c++) {
  Run-Phase "FG" $c $FgSec
  Run-Phase "HOME" $c $HomeSec
}

try { Stop-Process -Id $logcatProc.Id -Force -ErrorAction SilentlyContinue } catch {}
TLog "LOGCAT_STOPPED"
TLog "END_C2_OK"
exit 0
