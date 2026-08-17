# Canary BG_FRESHNESS build 135 — HOME 120s + conflict=0
param(
  [string]$AdbSerial = "192.168.1.33:35129",
  [int]$DriverId = 20135,
  [int]$ExpectVersionCode = 135,
  [int]$FgWarmSec = 20,
  [int]$HomeSec = 120,
  [int]$PollSec = 15,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_p0e_bg_freshness_135_2026-08-17",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [string]$PyLocal = "c:\Users\jasiq\atmr\docs\ops\_p0e_bg_freshness_rca.py"
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline.txt"
$putCsv = Join-Path $OutDir "put_samples.csv"
"ts,phase,elapsed_s,put_15s,isFg,startReq,finished,conflict_cum" | Out-File $putCsv -Encoding utf8

function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) [BG135] $msg"
  try { [IO.File]::AppendAllText($timeline, $line + [Environment]::NewLine) } catch {}
  Write-Host $line
}

if (Test-Path $DeployEnv) {
  Get-Content $DeployEnv | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    if ($_ -match '^\s*(\w+)=(.*)$') {
      Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'"))
    }
  }
}
$ssh = "$($env:SERVER_USER)@$($env:SERVER_HOST)"
$adb = $AdbPath
$d = $AdbSerial
$pkg = "ch.liri.operations"

$state = (& $adb -s $d get-state 2>&1 | Out-String).Trim()
if ($state -ne "device") { throw "device offline: $state" }
if (-not $env:SERVER_HOST) { throw "SERVER_HOST missing" }

$pkgDump = & $adb -s $d shell dumpsys package $pkg 2>&1 | Out-String
$vc = if ($pkgDump -match "versionCode=(\d+)") { [int]$Matches[1] } else { -1 }
TLog "PREFLIGHT versionCode=$vc expect=$ExpectVersionCode"
if ($vc -ne $ExpectVersionCode) { TLog "ABORT wrong versionCode"; exit 2 }

function Get-Svc {
  $raw = & $adb -s $d shell dumpsys activity services LocationTaskService 2>$null | Out-String
  $isFg = if ($raw -match "isForeground=(true|false)") { $Matches[1] } else { "absent" }
  $startReq = if ($raw -match "startRequested=(true|false)") { $Matches[1] } else { "absent" }
  return [pscustomobject]@{ IsForeground = $isFg; StartRequested = $startReq }
}

function Get-Finished {
  $tmp = Join-Path $OutDir "_nat.txt"
  & $adb -s $d logcat -d -t 8000 2>$null |
    Select-String -Pattern "Finished task 'background-location-task'" |
    ForEach-Object { $_.Line } | Out-File $tmp -Encoding utf8
  return @(Select-String -Path $tmp -Pattern "Finished" -EA SilentlyContinue).Count
}

function Get-Put15 {
  $puts = ssh -o BatchMode=yes -o ConnectTimeout=10 $ssh "docker logs atmr-backend-1 --since 15s 2>&1 | grep 'PUT /api/v1/driver/me/location' | grep -v Darwin | wc -l" 2>&1
  if ("$puts" -match '(\d+)') { return [int]$Matches[1] }
  return -1
}

function Get-ConflictSince([string]$sinceIso) {
  # Count DLQ conflict lines after marker time (UTC-ish from docker logs)
  $out = ssh -o BatchMode=yes -o ConnectTimeout=15 $ssh "docker logs atmr-tracking-kafka-consumer-1 --since 4m 2>&1 | grep -c 'event_id_payload_conflict' || true" 2>&1
  if ("$out" -match '(\d+)') { return [int]$Matches[1] }
  return -1
}

function SampleLocal([string]$phase, [int]$elapsed, [int]$conflictCum) {
  $svc = Get-Svc
  $fin = Get-Finished
  $put = Get-Put15
  $ts = Get-Date -Format o
  "$ts,$phase,$elapsed,$put,$($svc.IsForeground),$($svc.StartRequested),$fin,$conflictCum" | Out-File $putCsv -Append -Encoding utf8
  TLog "SAMPLE phase=$phase t=+${elapsed}s put15=$put isFg=$($svc.IsForeground) startReq=$($svc.StartRequested) finished=$fin conflictCum=$conflictCum"
}

scp -o BatchMode=yes -o ConnectTimeout=10 $PyLocal "${ssh}:/tmp/p0e_bg_freshness_rca.py" 2>$null | Out-Null
ssh -o BatchMode=yes -o ConnectTimeout=15 $ssh "docker cp /tmp/p0e_bg_freshness_rca.py atmr-backend-1:/tmp/p0e_bg_freshness_rca.py" 2>$null | Out-Null

# Baseline conflict count before HOME
$conflictBefore = Get-ConflictSince ""
TLog "CONFLICT_BASELINE_4m=$conflictBefore"

TLog "FG_WARM ${FgWarmSec}s"
& $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null
& $adb -s $d logcat -c 2>$null | Out-Null
Start-Sleep 3
$ready = Get-Svc
TLog "READY isFg=$($ready.IsForeground) startReq=$($ready.StartRequested)"
if ($ready.IsForeground -ne "true") { TLog "ABORT no FGS"; exit 3 }

$tFg = Get-Date
SampleLocal "FG" 0 $conflictBefore
while (((Get-Date) - $tFg).TotalSeconds -lt $FgWarmSec) {
  $remain = [int]($FgWarmSec - ((Get-Date) - $tFg).TotalSeconds)
  if ($remain -le 0) { break }
  Start-Sleep ([Math]::Min($PollSec, $remain))
  SampleLocal "FG" ([int]((Get-Date) - $tFg).TotalSeconds) $conflictBefore
}

TLog "HOME press - remote timeline ${HomeSec}s"
& $adb -s $d shell input keyevent KEYCODE_HOME 2>$null | Out-Null
Start-Sleep 1

$remoteOut = Join-Path $OutDir "server_timeline.txt"
$remoteCmd = "docker exec -e P0E_DRIVER_ID=$DriverId -e P0E_BG_SEC=$HomeSec -e P0E_POLL_SEC=$PollSec -e P0E_PHASE=HOME atmr-backend-1 python /tmp/p0e_bg_freshness_rca.py 2>/dev/null"
$remoteJob = Start-Job -ScriptBlock {
  param($sshHost, $cmd, $out)
  ssh -o BatchMode=yes -o ConnectTimeout=30 $sshHost $cmd 2>&1 | Out-File $out -Encoding utf8
} -ArgumentList $ssh, $remoteCmd, $remoteOut

$t0 = Get-Date
$conflictHomeStart = Get-ConflictSince ""
SampleLocal "HOME" 0 $conflictHomeStart
while (((Get-Date) - $t0).TotalSeconds -lt $HomeSec) {
  Start-Sleep $PollSec
  $el = [int]((Get-Date) - $t0).TotalSeconds
  if ($el -gt $HomeSec) { $el = $HomeSec }
  $cNow = Get-ConflictSince ""
  SampleLocal "HOME" $el $cNow
}

TLog "WAIT remote job"
Wait-Job $remoteJob -Timeout ($HomeSec + 90) | Out-Null
Receive-Job $remoteJob | Out-Null
Remove-Job $remoteJob -Force -ErrorAction SilentlyContinue

$conflictAfter = Get-ConflictSince ""
$conflictDelta = if ($conflictHomeStart -ge 0 -and $conflictAfter -ge 0) { $conflictAfter - $conflictHomeStart } else { -1 }
TLog "CONFLICT_HOME_START=$conflictHomeStart AFTER=$conflictAfter DELTA=$conflictDelta"

TLog "REMOTE_OUT"
$hyp = "UNKNOWN"
$dleDelta = -1
$canonDelta = -1
$restLast = ""
if (Test-Path $remoteOut) {
  Get-Content $remoteOut | ForEach-Object {
    TLog ("SRV " + $_)
    if ($_ -match 'dle_delta_id=(-?\d+)') { $dleDelta = [int]$Matches[1] }
    if ($_ -match 'canon_delta_seq=(-?\d+)') { $canonDelta = [int]$Matches[1] }
    if ($_ -match "HYPOTHESIS (\S+)") { $hyp = $Matches[1] }
    if ($_ -match "rest_statuses=\[(.*)\]") { $restLast = $Matches[1] }
  }
} else {
  TLog "WARN no remote out"
}

# Also grab exact conflict lines during window
$conflictDump = Join-Path $OutDir "conflicts_home.txt"
ssh -o BatchMode=yes -o ConnectTimeout=20 $ssh "docker logs atmr-tracking-kafka-consumer-1 --since 3m 2>&1 | grep 'event_id_payload_conflict' | tail -n 50" 2>&1 | Out-File $conflictDump -Encoding utf8
$conflictLinesHome = @(Get-Content $conflictDump -EA SilentlyContinue | Where-Object { $_ -match 'event_id_payload_conflict' }).Count
TLog "CONFLICT_LINES_SINCE_3m=$conflictLinesHome"

$endSvc = Get-Svc
$fgsOk = ($endSvc.IsForeground -eq "true" -and $endSvc.StartRequested -eq "true")
$conflictOk = ($conflictLinesHome -eq 0)
$dleOk = ($dleDelta -gt 0)
$canonOk = ($canonDelta -gt 0)
$restOk = ($restLast -notmatch 'stale' -and $restLast -match 'live|recent')

TLog "GATE fgs=$fgsOk conflict0=$conflictOk dleAdv=$dleOk($dleDelta) canonAdv=$canonOk($canonDelta) restOk=$restOk hyp=$hyp"

if ($conflictOk -and $dleOk -and $canonOk -and $fgsOk -and $restOk) {
  TLog "VERDICT BG_FRESHNESS_FIX_CANARY_VALIDATED"
  exit 0
}
if (-not $conflictOk) {
  TLog "VERDICT BG_FRESHNESS_FIX_CANARY_FAIL conflict>0"
  exit 4
}
TLog "VERDICT BG_FRESHNESS_FIX_CANARY_FAIL other_gate"
exit 5
