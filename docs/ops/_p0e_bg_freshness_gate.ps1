# Mini-RCA: continuity temps reel apres HOME (build 133) - SEPARE Q1
param(
  [string]$AdbSerial = "192.168.1.33:34343",
  [int]$DriverId = 20135,
  [int]$FgWarmSec = 20,
  [int]$HomeSec = 120,
  [int]$PollSec = 15,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_p0e_bg_freshness_2026-08-17",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [string]$PyLocal = "c:\Users\jasiq\atmr\docs\ops\_p0e_bg_freshness_rca.py"
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline.txt"
$putCsv = Join-Path $OutDir "put_samples.csv"
"ts,phase,elapsed_s,put_15s,isFg,startReq,finished" | Out-File $putCsv -Encoding utf8

function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) [BG_FRESH] $msg"
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

if ((& $adb -s $d get-state 2>&1 | Out-String).Trim() -ne "device") { throw "device offline" }
if (-not $env:SERVER_HOST) { throw "SERVER_HOST missing" }

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

function SampleLocal([string]$phase, [int]$elapsed) {
  $svc = Get-Svc
  $fin = Get-Finished
  $put = Get-Put15
  $ts = Get-Date -Format o
  "$ts,$phase,$elapsed,$put,$($svc.IsForeground),$($svc.StartRequested),$fin" | Out-File $putCsv -Append -Encoding utf8
  TLog "PUT_SAMPLE phase=$phase t=+${elapsed}s put15=$put isFg=$($svc.IsForeground) startReq=$($svc.StartRequested) finished=$fin"
}

scp -o BatchMode=yes -o ConnectTimeout=10 $PyLocal "${ssh}:/tmp/p0e_bg_freshness_rca.py" 2>$null | Out-Null
ssh -o BatchMode=yes -o ConnectTimeout=15 $ssh "docker cp /tmp/p0e_bg_freshness_rca.py atmr-backend-1:/tmp/p0e_bg_freshness_rca.py" 2>$null | Out-Null

TLog "FG_WARM ${FgWarmSec}s"
& $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null
& $adb -s $d logcat -c 2>$null | Out-Null
Start-Sleep 3
$ready = Get-Svc
TLog "READY isFg=$($ready.IsForeground) startReq=$($ready.StartRequested)"
if ($ready.IsForeground -ne "true") { TLog "ABORT no FGS"; exit 3 }

$tFg = Get-Date
SampleLocal "FG" 0
while (((Get-Date) - $tFg).TotalSeconds -lt $FgWarmSec) {
  $remain = [int]($FgWarmSec - ((Get-Date) - $tFg).TotalSeconds)
  if ($remain -le 0) { break }
  Start-Sleep ([Math]::Min($PollSec, $remain))
  SampleLocal "FG" ([int]((Get-Date) - $tFg).TotalSeconds)
}

TLog "HOME press - start remote server timeline ${HomeSec}s"
& $adb -s $d shell input keyevent KEYCODE_HOME 2>$null | Out-Null
Start-Sleep 1

$remoteOut = Join-Path $OutDir "server_timeline.txt"
$remoteCmd = "docker exec -e P0E_DRIVER_ID=$DriverId -e P0E_BG_SEC=$HomeSec -e P0E_POLL_SEC=$PollSec -e P0E_PHASE=HOME atmr-backend-1 python /tmp/p0e_bg_freshness_rca.py 2>/dev/null"
$remoteJob = Start-Job -ScriptBlock {
  param($sshHost, $cmd, $out)
  ssh -o BatchMode=yes -o ConnectTimeout=30 $sshHost $cmd 2>&1 | Out-File $out -Encoding utf8
} -ArgumentList $ssh, $remoteCmd, $remoteOut

$t0 = Get-Date
SampleLocal "HOME" 0
while (((Get-Date) - $t0).TotalSeconds -lt $HomeSec) {
  Start-Sleep $PollSec
  $el = [int]((Get-Date) - $t0).TotalSeconds
  if ($el -gt $HomeSec) { $el = $HomeSec }
  SampleLocal "HOME" $el
}

TLog "WAIT remote job"
Wait-Job $remoteJob -Timeout ($HomeSec + 90) | Out-Null
Receive-Job $remoteJob | Out-Null
Remove-Job $remoteJob -Force -ErrorAction SilentlyContinue

TLog "REMOTE_OUT"
if (Test-Path $remoteOut) {
  Get-Content $remoteOut | ForEach-Object { TLog ("SRV " + $_) }
} else {
  TLog "WARN no remote out"
}

TLog "END_BG_FRESHNESS_CAPTURE"
exit 0
