# P0-D D5 — Cold start Prod126 capture (read-only)
param(
  [string]$AdbSerial = "100.81.106.54:43223",
  [int]$DriverId = 20135,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_release_exec_p0d_2026-08-16\d5_cold_start",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [string]$Apk = "c:\Users\jasiq\atmr\docs\ops\_release_exec_mobile_builds_2026-08-16\operations-app-1.0.11-126-286737a-universal.apk",
  [switch]$SkipInstall,
  [int]$WaitReadySec = 600
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline.txt"
function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) [COLD] $msg"
  try {
    [IO.File]::AppendAllText($timeline, $line + [Environment]::NewLine)
  } catch {
    Start-Sleep -Milliseconds 200
    try { [IO.File]::AppendAllText($timeline, $line + [Environment]::NewLine) } catch { }
  }
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

function Get-Uid {
  $m = & $adb -s $d shell dumpsys package $pkg 2>$null | Select-String "userId=(\d+)" | Select-Object -First 1
  if ($m) { return [int]$m.Matches[0].Groups[1].Value }
  return 0
}

function Snapshot-Service([string]$label) {
  $svcPath = Join-Path $OutDir ("snap_{0}_LocationTaskService.txt" -f $label)
  $raw = & $adb -s $d shell dumpsys activity services LocationTaskService 2>$null | Out-String
  if (-not $raw -or $raw.Length -lt 80) {
    $raw = & $adb -s $d shell dumpsys activity services $pkg 2>$null | Out-String
  }
  $raw | Out-File $svcPath -Encoding utf8
  $fg = if ($raw -match "startForegroundCount=(\d+)") { $Matches[1] } else { "NA" }
  $isFg = if ($raw -match "isForeground=(true|false)") { $Matches[1] } else { "NA" }
  $startReq = if ($raw -match "startRequested=(true|false)") { $Matches[1] } else { "NA" }
  $lastAct = if ($raw -match "lastActivity=([^\s]+)") { $Matches[1] } else { "NA" }
  $binds = 0
  $inPer = $false
  foreach ($line in ($raw -split "`n")) {
    if ($line -match "Per-process Connections:") { $inPer = $true; continue }
    if ($inPer -and $line -match "^\s+ConnectionRecord\{") { $binds++ }
    if ($inPer -and $line -match "All Connections:") { break }
  }
  return [pscustomobject]@{
    Label = $label; FgCount = $fg; Binds = $binds; IsForeground = $isFg; StartRequested = $startReq; LastActivity = $lastAct
  }
}

function Snapshot-NativeDelta([string]$label, [datetime]$sinceLocal) {
  $natPath = Join-Path $OutDir ("snap_{0}_native.txt" -f $label)
  & $adb -s $d logcat -d -t 20000 2>$null |
    Select-String -Pattern "LocationTaskConsumer|Location unavailable|background-location-task|TaskService|FusedLocation|too close|too fast|GmsPassiveListener_FLP|nlo_start|nlo_stop|recover|tracking\.(background|cold_start)|start_requested|start_failed|LocationTaskService" |
    ForEach-Object { $_.Line } |
    Out-File $natPath -Encoding utf8
  $f = $natPath
  $finished = @(Select-String -Path $f -Pattern "Finished task 'background-location-task'" -EA SilentlyContinue).Count
  $unavail = @(Select-String -Path $f -Pattern "Location unavailable" -EA SilentlyContinue).Count
  $nloStart = @(Select-String -Path $f -Pattern "nlo_start" -EA SilentlyContinue).Count
  $nloStop = @(Select-String -Path $f -Pattern "nlo_stop" -EA SilentlyContinue).Count
  $recover = @(Select-String -Path $f -Pattern "recover" -EA SilentlyContinue).Count
  $startReqTel = @(Select-String -Path $f -Pattern "start_requested|tracking\.background\.start" -EA SilentlyContinue).Count
  return [pscustomobject]@{
    Finished = $finished; Unavailable = $unavail; NloStart = $nloStart; NloStop = $nloStop; Recover = $recover; StartTel = $startReqTel
  }
}

function Snapshot-PutLoc([string]$label) {
  $putPath = Join-Path $OutDir ("snap_{0}_put.txt" -f $label)
  $locPath = Join-Path $OutDir ("snap_{0}_loc.txt" -f $label)
  $putN = 0
  $locN = 0
  if ($env:SERVER_HOST) {
    $puts = ssh -o BatchMode=yes -o ConnectTimeout=12 $ssh "docker logs atmr-backend-1 --since 3m 2>&1 | grep 'PUT /api/v1/driver/me/location' | grep -v Darwin | tail -20" 2>&1
    $puts | Out-File $putPath -Encoding utf8
    $putN = @($puts | Where-Object { $_ -match 'PUT' }).Count
    $tpl = Join-Path $OutDir "snap_loc_template.py"
    if (-not (Test-Path $tpl)) { $tpl = "c:\Users\jasiq\atmr\docs\ops\_release_exec_p0d_2026-08-16\d5_cold_start\snap_loc_template.py" }
    $py = (Get-Content -Raw $tpl).Replace("__DRIVER_ID__", "$DriverId")
    $pyFile = Join-Path $OutDir ("snap_{0}_snap.py" -f $label)
    [IO.File]::WriteAllText($pyFile, ($py -replace "`r`n", "`n"))
    scp -o BatchMode=yes -o ConnectTimeout=12 $pyFile "${ssh}:/tmp/d5_cold_snap.py" 2>$null | Out-Null
    $locOut = ssh -o BatchMode=yes -o ConnectTimeout=20 $ssh "docker cp /tmp/d5_cold_snap.py atmr-backend-1:/tmp/d5_cold_snap.py && docker exec atmr-backend-1 python /tmp/d5_cold_snap.py" 2>&1
    $locOut | Out-File $locPath -Encoding utf8
    if ($locOut -match '(?m)^N (\d+)') { $locN = [int]$Matches[1] }
  } else {
    "NO_SSH" | Out-File $putPath -Encoding utf8
    "NO_SSH" | Out-File $locPath -Encoding utf8
  }
  return [pscustomobject]@{ PutN = $putN; LocN = $locN }
}

function Full-Snap([string]$label) {
  TLog "SNAP_BEGIN $label"
  $svc = Snapshot-Service $label
  $nat = Snapshot-NativeDelta $label (Get-Date)
  $pl = Snapshot-PutLoc $label
  $line = "SNAP $label fgCount=$($svc.FgCount) binds=$($svc.Binds) isFg=$($svc.IsForeground) startReq=$($svc.StartRequested) finished=$($nat.Finished) unavailable=$($nat.Unavailable) nlo_start=$($nat.NloStart) nlo_stop=$($nat.NloStop) recover=$($nat.Recover) startTel=$($nat.StartTel) PUT=$($pl.PutN) LOC=$($pl.LocN) lastAct=$($svc.LastActivity)"
  TLog $line
  return $svc
}

# --- install ---
if (-not $SkipInstall) {
  TLog "UNINSTALL $pkg"
  & $adb -s $d uninstall $pkg 2>&1 | Out-File (Join-Path $OutDir "uninstall.txt") -Encoding utf8
  TLog "INSTALL $Apk"
  & $adb -s $d install -r $Apk 2>&1 | Tee-Object (Join-Path $OutDir "install.txt")
  $id = & $adb -s $d shell dumpsys package $pkg 2>$null | Select-String "versionCode=|versionName=|flags=\[" | Select-Object -First 6
  TLog ("PKG " + (($id | ForEach-Object { $_.Line.Trim() }) -join " | "))
}

TLog "LAUNCH MainActivity - login driver $DriverId + mission required"
& $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null

# Wait until LocationTaskService is up (mission tracking)
TLog "WAIT_READY LocationTaskService max=${WaitReadySec}s"
$ready = $false
$deadline = (Get-Date).AddSeconds($WaitReadySec)
while ((Get-Date) -lt $deadline) {
  $probe = & $adb -s $d shell dumpsys activity services LocationTaskService 2>$null | Out-String
  if ($probe -match "ch\.liri\.operations/expo\.modules\.location\.services\.LocationTaskService" -and $probe -match "isForeground=true") {
    $ready = $true
    break
  }
  Start-Sleep 5
}
if (-not $ready) {
  TLog "NOT_READY - abort before cold force-stop (login/mission missing?)"
  exit 2
}
TLog "READY_PRE_COLD - taking pre-force snapshot then force-stop"
Full-Snap "PRE_FORCE" | Out-Null

# Cold start
TLog "FORCE_STOP"
& $adb -s $d shell am force-stop $pkg 2>$null | Out-Null
Start-Sleep 2
$ps = & $adb -s $d shell pidof $pkg 2>$null
TLog "PID_AFTER_STOP='$ps' (expect empty)"
if ($ps -and $ps.Trim().Length -gt 0) {
  TLog "WARN process still alive - second force-stop"
  & $adb -s $d shell am force-stop $pkg 2>$null | Out-Null
  Start-Sleep 2
  $ps = & $adb -s $d shell pidof $pkg 2>$null
  TLog "PID_AFTER_STOP2='$ps'"
}

& $adb -s $d logcat -c 2>$null | Out-Null
TLog "RELAUNCH cold"
& $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null

# Wait FGS again after cold (user may need to re-open mission)
$ColdReadySec = 900
TLog "WAIT_COLD_FGS max=${ColdReadySec}s (re-open mission if needed)"
$deadline2 = (Get-Date).AddSeconds($ColdReadySec)
$coldReady = $false
while ((Get-Date) -lt $deadline2) {
  $probe = & $adb -s $d shell dumpsys activity services LocationTaskService 2>$null | Out-String
  if ($probe -match "ch\.liri\.operations/expo\.modules\.location\.services\.LocationTaskService" -and $probe -match "isForeground=true" -and $probe -match "startRequested=true") {
    $coldReady = $true
    break
  }
  Start-Sleep 5
}
if (-not $coldReady) {
  TLog "COLD_FGS_NOT_UP within 180s - snap anyway as T0_FAIL"
  Full-Snap "T0_FAIL" | Out-Null
  TLog "END_FAIL"
  exit 3
}

$t0 = Get-Date
Full-Snap "T0" | Out-Null
Start-Sleep 30
Full-Snap "T30" | Out-Null
Start-Sleep 30
Full-Snap "T60" | Out-Null
Start-Sleep 60
Full-Snap "T120" | Out-Null

TLog "HOME 120s"
& $adb -s $d shell input keyevent KEYCODE_HOME 2>$null | Out-Null
Start-Sleep 120
Full-Snap "HOME120" | Out-Null

TLog "END_OK"
exit 0
