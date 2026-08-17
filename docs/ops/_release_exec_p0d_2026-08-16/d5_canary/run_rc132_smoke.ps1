# RC132 smoke — FG stable puis HOME (minimal, pas C1–C4)
param(
  [string]$AdbSerial = "192.168.1.33:34343",
  [int]$DriverId = 20135,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_release_exec_p0d_2026-08-16\d5_canary",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [int]$MinVersionCode = 132,
  [int]$FgSec = 90,
  [int]$HomeSec = 180,
  [int]$PollSec = 30,
  [string]$RunTag = "RC132_SMOKE"
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline_$RunTag.txt"
$csv = Join-Path $OutDir "samples_$RunTag.csv"
$logcatFile = Join-Path $OutDir "logcat_${RunTag}_continuous.txt"
"ts,phase,elapsed_s,fgCount,binds,isFg,startReq,finished_delta,unregister_delta,put_n,loc_n" | Out-File $csv -Encoding utf8

function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) [$RunTag] $msg"
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
TLog "PREFLIGHT versionName=$vn versionCode=$vc minRequired=$MinVersionCode"
if ($vc -lt $MinVersionCode) { TLog "ABORT need versionCode >= $MinVersionCode"; exit 2 }

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
  $tmp = Join-Path $OutDir "_nat_tmp_$RunTag.txt"
  & $adb -s $d logcat -d -t 8000 2>$null |
    Select-String -Pattern "Unregistering task 'background-location-task'|Finished task 'background-location-task'|Registered task 'background-location-task'" |
    ForEach-Object { $_.Line } |
    Out-File $tmp -Encoding utf8
  $fin = @(Select-String -Path $tmp -Pattern "Finished task 'background-location-task'" -EA SilentlyContinue).Count
  $unreg = @(Select-String -Path $tmp -Pattern "Unregistering task 'background-location-task'" -EA SilentlyContinue).Count
  $reg = @(Select-String -Path $tmp -Pattern "Registered task 'background-location-task'" -EA SilentlyContinue).Count
  return [pscustomobject]@{ Finished = $fin; Unregister = $unreg; Registered = $reg }
}

function Get-PutLoc {
  $putN = -1
  $locN = -1
  if (-not $env:SERVER_HOST) { return [pscustomobject]@{ PutN = $putN; LocN = $locN } }
  $puts = ssh -o BatchMode=yes -o ConnectTimeout=8 $ssh "docker logs atmr-backend-1 --since 60s 2>&1 | grep 'PUT /api/v1/driver/me/location' | grep -v Darwin | wc -l" 2>&1
  if ("$puts" -match '(\d+)') { $putN = [int]$Matches[1] }
  $pyFile = Join-Path $OutDir "_loc_snap_$RunTag.py"
  $py = @"
from app import create_app
app=create_app()
with app.app_context():
 from sqlalchemy import text
 from models import db
 n=db.session.execute(text('SELECT count(*) FROM driver_location_events WHERE driver_id=$DriverId AND created_at>=now()-interval ''60 seconds''')).scalar()
 print('N', int(n or 0))
"@
  [IO.File]::WriteAllText($pyFile, ($py -replace "`r`n", "`n"))
  scp -o BatchMode=yes -o ConnectTimeout=8 $pyFile "${ssh}:/tmp/d5_rc132_snap.py" 2>$null | Out-Null
  $locOut = ssh -o BatchMode=yes -o ConnectTimeout=12 $ssh "docker cp /tmp/d5_rc132_snap.py atmr-backend-1:/tmp/d5_rc132_snap.py && docker exec atmr-backend-1 python /tmp/d5_rc132_snap.py 2>/dev/null" 2>&1
  if ("$locOut" -match '(?m)^N (\d+)') { $locN = [int]$Matches[1] }
  return [pscustomobject]@{ PutN = $putN; LocN = $locN }
}

function Sample([string]$phase, [int]$elapsed) {
  $svc = Get-SvcMetrics
  $nat = Get-NativeDelta
  $pl = Get-PutLoc
  $ts = Get-Date -Format o
  $row = "$ts,$phase,$elapsed,$($svc.FgCount),$($svc.Binds),$($svc.IsForeground),$($svc.StartRequested),$($nat.Finished),$($nat.Unregister),$($pl.PutN),$($pl.LocN)"
  [IO.File]::AppendAllText($csv, $row + [Environment]::NewLine)
  TLog "SAMPLE phase=$phase t=+${elapsed}s fg=$($svc.FgCount) binds=$($svc.Binds) isFg=$($svc.IsForeground) startReq=$($svc.StartRequested) finished=$($nat.Finished) unreg=$($nat.Unregister) reg=$($nat.Registered) PUT=$($pl.PutN) LOC=$($pl.LocN)"
}

function Run-Phase([string]$phase, [int]$sec) {
  TLog "PHASE_BEGIN $phase sec=$sec"
  $t0 = Get-Date
  Sample $phase 0
  $elapsed = 0
  while ($elapsed -lt $sec) {
    Start-Sleep $PollSec
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds
    if ($elapsed -gt $sec) { $elapsed = $sec }
    Sample $phase $elapsed
  }
  TLog "PHASE_END $phase"
}

& $adb -s $d logcat -c 2>$null | Out-Null
if (Test-Path $logcatFile) { Remove-Item $logcatFile -Force }
$errLog = Join-Path $OutDir "logcat_${RunTag}_stderr.txt"
$logcatProc = Start-Process -FilePath $adb -ArgumentList @("-s", $d, "logcat", "-v", "threadtime") -RedirectStandardOutput $logcatFile -RedirectStandardError $errLog -PassThru -WindowStyle Hidden
TLog "LOGCAT_PID=$($logcatProc.Id)"
TLog "BACKEND=OBSERVATION_ONLY DRIVER=$DriverId"

& $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null
Start-Sleep 4
$ready = Get-SvcMetrics
TLog "READY_CHECK isFg=$($ready.IsForeground) startReq=$($ready.StartRequested) fgCount=$($ready.FgCount)"
if ($ready.IsForeground -ne "true" -or $ready.StartRequested -ne "true") {
  TLog "ABORT FGS not alive"
  try { Stop-Process -Id $logcatProc.Id -Force -ErrorAction SilentlyContinue } catch {}
  exit 3
}

Run-Phase "FG" $FgSec

TLog "HOME press"
& $adb -s $d shell input keyevent KEYCODE_HOME 2>$null | Out-Null
Start-Sleep 3
Run-Phase "HOME" $HomeSec

try { Stop-Process -Id $logcatProc.Id -Force -ErrorAction SilentlyContinue } catch {}
$end = Get-SvcMetrics
TLog "END_FGS isFg=$($end.IsForeground) startReq=$($end.StartRequested) fgCount=$($end.FgCount)"
TLog "LOGCAT_STOPPED"
TLog "END_SMOKE_OK"
exit 0
