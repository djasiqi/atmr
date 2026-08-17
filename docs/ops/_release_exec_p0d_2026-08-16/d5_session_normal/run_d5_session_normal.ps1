# P0-D D5 — Session normale Prod126 (sans force-stop)
# Observe fgCount/binds toutes les 30s : FG5 → HOME5 → FG5 → HOME5
param(
  [string]$AdbSerial = "100.81.106.54:43223",
  [int]$DriverId = 20135,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_release_exec_p0d_2026-08-16\d5_session_normal",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [int]$PhaseSec = 300,
  [int]$PollSec = 30,
  [int]$WaitReadySec = 120
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline.txt"
$csv = Join-Path $OutDir "samples.csv"
"ts,phase,elapsed_s,fgCount,binds,isFg,startReq,finished_delta,unavailable_delta,put_n,loc_n,note" | Out-File $csv -Encoding utf8

function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) [SESS] $msg"
  try { [IO.File]::AppendAllText($timeline, $line + [Environment]::NewLine) } catch {}
  Write-Host $line
}

if (Test-Path $DeployEnv) {
  Get-Content $DeployEnv | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    if ($_ -match '^\s*export\s+(\w+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'")) }
    elseif ($_ -match '^\s*(\w+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'")) }
  }
}
$ssh = "$($env:SERVER_USER)@$($env:SERVER_HOST)"
$adb = $AdbPath
$d = $AdbSerial
$pkg = "ch.liri.operations"
if ((& $adb -s $d get-state 2>&1 | Out-String).Trim() -ne "device") { throw "device offline" }

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
  if ($fg -lt 0 -and $raw -notmatch "LocationTaskService") {
    return [pscustomobject]@{ FgCount = -1; Binds = -1; IsForeground = "absent"; StartRequested = "absent"; RawLen = $raw.Length }
  }
  return [pscustomobject]@{ FgCount = $fg; Binds = $binds; IsForeground = $isFg; StartRequested = $startReq; RawLen = $raw.Length }
}

function Get-NativeDelta {
  $tmp = Join-Path $OutDir "_nat_tmp.txt"
  & $adb -s $d logcat -d -t 4000 2>$null |
    Select-String -Pattern "LocationTaskConsumer|Location unavailable|Finished task 'background-location-task'|TaskService|nlo_start|nlo_stop|recover|start_requested|start_failed|startLocationUpdates|stopLocationUpdates|tracking\.background|LocationTaskService" |
    ForEach-Object { $_.Line } |
    Out-File $tmp -Encoding utf8
  $fin = @(Select-String -Path $tmp -Pattern "Finished task 'background-location-task'" -EA SilentlyContinue).Count
  $un = @(Select-String -Path $tmp -Pattern "Location unavailable" -EA SilentlyContinue).Count
  return [pscustomobject]@{ Finished = $fin; Unavailable = $un; Path = $tmp }
}

function Get-PutLoc {
  $putN = 0; $locN = 0
  if (-not $env:SERVER_HOST) { return [pscustomobject]@{ PutN = -1; LocN = -1 } }
  $puts = ssh -o BatchMode=yes -o ConnectTimeout=10 $ssh "docker logs atmr-backend-1 --since 90s 2>&1 | grep 'PUT /api/v1/driver/me/location' | grep -v Darwin | wc -l" 2>&1
  if ($puts -match '(\d+)') { $putN = [int]$Matches[1] }
  $tpl = Join-Path $OutDir "snap_loc_template.py"
  if (-not (Test-Path $tpl)) {
    @"
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app=create_app(); app.app_context().push()
from models import db
DRIVER_ID=__DRIVER_ID__
since=datetime.now(timezone.utc)-timedelta(seconds=90)
n=db.session.execute(text('SELECT count(*) FROM driver_location_events WHERE driver_id=:d AND created_at>=:s'),{'d':DRIVER_ID,'s':since}).scalar()
print('N', int(n or 0))
"@ | Out-File $tpl -Encoding utf8
  }
  $py = (Get-Content -Raw $tpl).Replace("__DRIVER_ID__", "$DriverId")
  $pyFile = Join-Path $OutDir "_loc_snap.py"
  [IO.File]::WriteAllText($pyFile, ($py -replace "`r`n", "`n"))
  scp -o BatchMode=yes -o ConnectTimeout=10 $pyFile "${ssh}:/tmp/d5_sess_snap.py" 2>$null | Out-Null
  $locOut = ssh -o BatchMode=yes -o ConnectTimeout=15 $ssh "docker cp /tmp/d5_sess_snap.py atmr-backend-1:/tmp/d5_sess_snap.py && docker exec atmr-backend-1 python /tmp/d5_sess_snap.py" 2>&1
  if ($locOut -match '(?m)^N (\d+)') { $locN = [int]$Matches[1] }
  return [pscustomobject]@{ PutN = $putN; LocN = $locN }
}

function Sample([string]$phase, [int]$elapsed, [int]$prevFg, [int]$prevBinds) {
  $svc = Get-SvcMetrics
  $nat = Get-NativeDelta
  $pl = Get-PutLoc
  $note = ""
  if ($prevFg -ge 0 -and $svc.FgCount -gt $prevFg) { $note = "FGCOUNT_UP_${prevFg}_to_$($svc.FgCount)" }
  if ($prevBinds -ge 0 -and $svc.Binds -gt $prevBinds) {
    if ($note) { $note += ";" }
    $note += "BINDS_UP_${prevBinds}_to_$($svc.Binds)"
  }
  $ts = Get-Date -Format o
  $row = "$ts,$phase,$elapsed,$($svc.FgCount),$($svc.Binds),$($svc.IsForeground),$($svc.StartRequested),$($nat.Finished),$($nat.Unavailable),$($pl.PutN),$($pl.LocN),$note"
  [IO.File]::AppendAllText($csv, $row + [Environment]::NewLine)
  TLog "SAMPLE phase=$phase t=+${elapsed}s fg=$($svc.FgCount) binds=$($svc.Binds) isFg=$($svc.IsForeground) startReq=$($svc.StartRequested) finishedish=$($nat.Finished) unavail=$($nat.Unavailable) PUT=$($pl.PutN) LOC=$($pl.LocN) $note"
  if ($note -match "BINDS_UP|FGCOUNT_UP") {
    $stamp = Get-Date -Format "HHmmss"
    Copy-Item $nat.Path (Join-Path $OutDir ("TRIGGER_${stamp}_native.txt")) -Force
    & $adb -s $d shell dumpsys activity services LocationTaskService 2>$null | Out-File (Join-Path $OutDir ("TRIGGER_${stamp}_svc.txt")) -Encoding utf8
    TLog "TRIGGER_CAPTURED $note -> TRIGGER_${stamp}_*"
  }
  return $svc
}

# Wait ready — no force-stop ever
TLog "START session-normal NO_FORCE_STOP poll=${PollSec}s phases=${PhaseSec}s BACKEND=FROZEN"
& $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null
$deadline = (Get-Date).AddSeconds($WaitReadySec)
$ready = $false
while ((Get-Date) -lt $deadline) {
  $m = Get-SvcMetrics
  if ($m.IsForeground -eq "true" -and $m.StartRequested -eq "true" -and $m.FgCount -ge 1) {
    $ready = $true
    TLog "READY fg=$($m.FgCount) binds=$($m.Binds)"
    break
  }
  Start-Sleep 5
}
if (-not $ready) {
  TLog "NOT_READY abort"
  exit 2
}

# Confirm FLP
$loc = & $adb -s $d shell dumpsys location 2>$null | Out-String
$flp = ($loc -split "`n" | Where-Object { $_ -match "ProviderRequest\[@\+8s0ms, HIGH_ACCURACY, WorkSource\{1090\d+ ch\.liri" } | Select-Object -First 1)
TLog "FLP $($flp.Trim())"

& $adb -s $d logcat -c 2>$null | Out-Null
$prevFg = -1
$prevBinds = -1

$phases = @(
  @{ Name = "FG1"; Action = "FG" },
  @{ Name = "HOME1"; Action = "HOME" },
  @{ Name = "FG2"; Action = "FG" },
  @{ Name = "HOME2"; Action = "HOME" }
)

foreach ($ph in $phases) {
  TLog "PHASE_BEGIN $($ph.Name) action=$($ph.Action)"
  if ($ph.Action -eq "HOME") {
    & $adb -s $d shell input keyevent KEYCODE_HOME 2>$null | Out-Null
  } else {
    & $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null
  }
  $t0 = Get-Date
  $elapsed = 0
  $svc = Sample $ph.Name 0 $prevFg $prevBinds
  $prevFg = $svc.FgCount
  $prevBinds = $svc.Binds
  while ($elapsed -lt $PhaseSec) {
    Start-Sleep $PollSec
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds
    if ($elapsed -gt $PhaseSec) { $elapsed = $PhaseSec }
    $svc = Sample $ph.Name $elapsed $prevFg $prevBinds
    $prevFg = $svc.FgCount
    $prevBinds = $svc.Binds
  }
  TLog "PHASE_END $($ph.Name) final fg=$prevFg binds=$prevBinds"
}

TLog "END_OK final fg=$prevFg binds=$prevBinds"
exit 0
