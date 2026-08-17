# P0-E L1/L2 — release safety gate (build 133) — SÉPARÉ de Q1 RCA
# L1 HOME : GPS/FGS/PUT/LOC DOIVENT continuer (mission active)
# L2 force-stop : GPS/FGS/PUT/LOC DOIVENT s'arrêter (après fenêtre de vidage)
param(
  [string]$AdbSerial = "192.168.1.33:34343",
  [int]$DriverId = 20135,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_p0e_l12_lifecycle_2026-08-17",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [int]$ExpectVersionCode = 133,
  [int]$L1HomeSec = 150,
  [int]$L2WaitSec = 75,
  [int]$DrainSec = 15,
  [int]$PollSec = 30
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline_L12.txt"
$csv = Join-Path $OutDir "samples_L12.csv"
"ts,phase,elapsed_s,fgCount,isFg,startReq,finished,unregister,put_n,loc_n,dle_max_id,dle_max_seq,session" | Out-File $csv -Encoding utf8

function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) [L12] $msg"
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
if (-not $env:SERVER_HOST) { throw "SERVER_HOST missing (.local.deploy.env)" }

$pkgDump = & $adb -s $d shell dumpsys package $pkg 2>&1 | Out-String
$vc = if ($pkgDump -match "versionCode=(\d+)") { [int]$Matches[1] } else { -1 }
$vn = if ($pkgDump -match "versionName=([^\s]+)") { $Matches[1] } else { "?" }
TLog "PREFLIGHT versionName=$vn versionCode=$vc expect=$ExpectVersionCode"
if ($vc -ne $ExpectVersionCode) { TLog "ABORT wrong versionCode"; exit 2 }

function Get-SvcMetrics {
  $raw = & $adb -s $d shell dumpsys activity services LocationTaskService 2>$null | Out-String
  $fg = if ($raw -match "startForegroundCount=(\d+)") { [int]$Matches[1] } else { -1 }
  $isFg = if ($raw -match "isForeground=(true|false)") { $Matches[1] } else { "absent" }
  $startReq = if ($raw -match "startRequested=(true|false)") { $Matches[1] } else { "absent" }
  return [pscustomobject]@{ FgCount = $fg; IsForeground = $isFg; StartRequested = $startReq; RawLen = $raw.Length }
}

function Get-NativeCounts {
  $tmp = Join-Path $OutDir "_nat_tmp.txt"
  & $adb -s $d logcat -d -t 12000 2>$null |
    Select-String -Pattern "Unregistering task 'background-location-task'|Finished task 'background-location-task'|Registered task 'background-location-task'" |
    ForEach-Object { $_.Line } |
    Out-File $tmp -Encoding utf8
  $fin = @(Select-String -Path $tmp -Pattern "Finished task 'background-location-task'" -EA SilentlyContinue).Count
  $unreg = @(Select-String -Path $tmp -Pattern "Unregistering task 'background-location-task'" -EA SilentlyContinue).Count
  return [pscustomobject]@{ Finished = $fin; Unregister = $unreg }
}

function Get-PutSince([int]$sinceSec) {
  $puts = ssh -o BatchMode=yes -o ConnectTimeout=10 $ssh "docker logs atmr-backend-1 --since ${sinceSec}s 2>&1 | grep 'PUT /api/v1/driver/me/location' | grep -v Darwin | wc -l" 2>&1
  if ("$puts" -match '(\d+)') { return [int]$Matches[1] }
  return -1
}

function Get-DleSnap {
  $pyFile = Join-Path $OutDir "_dle_snap.py"
  $py = @"
from app import create_app
from sqlalchemy import text
app=create_app()
with app.app_context():
  from models import db
  active=db.session.execute(text('''
    SELECT tracking_session_id FROM tracking_sessions
    WHERE driver_id=$DriverId AND status='active'
    ORDER BY id DESC LIMIT 1
  ''')).scalar()
  row=db.session.execute(text('''
    SELECT COALESCE(MAX(id),0), COALESCE(MAX(sequence_id),0),
           COUNT(*) FILTER (WHERE created_at>=now()-interval '90 seconds')
    FROM driver_location_events WHERE driver_id=$DriverId
  ''')).first()
  print('MAX_ID', int(row[0]))
  print('MAX_SEQ', int(row[1]))
  print('N90', int(row[2]))
  print('SESS', active)
"@
  [IO.File]::WriteAllText($pyFile, ($py -replace "`r`n", "`n"))
  scp -o BatchMode=yes -o ConnectTimeout=10 $pyFile "${ssh}:/tmp/p0e_l12_dle.py" 2>$null | Out-Null
  $out = ssh -o BatchMode=yes -o ConnectTimeout=25 $ssh "docker cp /tmp/p0e_l12_dle.py atmr-backend-1:/tmp/p0e_l12_dle.py && docker exec atmr-backend-1 python /tmp/p0e_l12_dle.py 2>/dev/null" 2>&1 | Out-String
  $maxId = if ($out -match 'MAX_ID (\d+)') { [long]$Matches[1] } else { -1 }
  $maxSeq = if ($out -match 'MAX_SEQ (\d+)') { [long]$Matches[1] } else { -1 }
  $n90 = if ($out -match 'N90 (\d+)') { [int]$Matches[1] } else { -1 }
  $sess = if ($out -match 'SESS (trk_sess_\S+|None)') { $Matches[1] } else { "NA" }
  return [pscustomobject]@{ MaxId = $maxId; MaxSeq = $maxSeq; N90 = $n90; Sess = $sess; Raw = $out }
}

function Sample([string]$phase, [int]$elapsed, [int]$putWindowSec) {
  $svc = Get-SvcMetrics
  $nat = Get-NativeCounts
  $putN = Get-PutSince $putWindowSec
  $dle = Get-DleSnap
  $ts = Get-Date -Format o
  $row = "$ts,$phase,$elapsed,$($svc.FgCount),$($svc.IsForeground),$($svc.StartRequested),$($nat.Finished),$($nat.Unregister),$putN,$($dle.N90),$($dle.MaxId),$($dle.MaxSeq),$($dle.Sess)"
  [IO.File]::AppendAllText($csv, $row + [Environment]::NewLine)
  TLog "SAMPLE phase=$phase t=+${elapsed}s isFg=$($svc.IsForeground) startReq=$($svc.StartRequested) fgCount=$($svc.FgCount) finished=$($nat.Finished) unreg=$($nat.Unregister) PUT($putWindowSec)=${putN} LOC90=$($dle.N90) maxId=$($dle.MaxId) maxSeq=$($dle.MaxSeq) sess=$($dle.Sess)"
  return [pscustomobject]@{ Svc = $svc; Nat = $nat; PutN = $putN; Dle = $dle }
}

# --- READY ---
& $adb -s $d logcat -c 2>$null | Out-Null
& $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null
Start-Sleep 5
$ready = Sample "READY" 0 60
if ($ready.Svc.IsForeground -ne "true" -or $ready.Svc.StartRequested -ne "true") {
  TLog "ABORT FGS not alive before L1"
  exit 3
}
if ($ready.Dle.Sess -eq "NA" -or $ready.Dle.Sess -eq "None") {
  TLog "ABORT no active tracking session"
  exit 3
}
$session0 = $ready.Dle.Sess
$maxId0 = $ready.Dle.MaxId
TLog "ANCHOR sess=$session0 maxId=$maxId0"

# --- L1 HOME ---
TLog "L1_BEGIN HOME ${L1HomeSec}s (GPS MUST continue)"
& $adb -s $d shell input keyevent KEYCODE_HOME 2>$null | Out-Null
Start-Sleep 2
$t0 = Get-Date
$sL1 = Sample "L1_HOME" 0 60
$elapsed = 0
while ($elapsed -lt $L1HomeSec) {
  Start-Sleep $PollSec
  $elapsed = [int]((Get-Date) - $t0).TotalSeconds
  if ($elapsed -gt $L1HomeSec) { $elapsed = $L1HomeSec }
  $sL1 = Sample "L1_HOME" $elapsed 60
}
$maxId1 = $sL1.Dle.MaxId
$sess1 = $sL1.Dle.Sess
$fgs1 = $sL1.Svc.IsForeground
$put1 = $sL1.PutN
$l1Pass = ($fgs1 -eq "true") -and ($sL1.Svc.StartRequested -eq "true") -and ($maxId1 -gt $maxId0) -and ($sess1 -eq $session0) -and ($put1 -gt 0)
TLog "L1_VERDICT pass=$l1Pass fgs=$fgs1 put=$put1 deltaId=$($maxId1-$maxId0) sessStable=$($sess1 -eq $session0)"
if (-not $l1Pass) {
  TLog "VERDICT L12_FAIL (L1 BACKGROUND)"
  exit 4
}

# --- L2 FORCE-STOP ---
TLog "L2_BEGIN FORCE-STOP wait=${L2WaitSec}s drain=${DrainSec}s"
$preStop = Sample "L2_PRE" 0 30
$maxIdPre = $preStop.Dle.MaxId
$maxSeqPre = $preStop.Dle.MaxSeq
& $adb -s $d logcat -c 2>$null | Out-Null
& $adb -s $d shell am force-stop $pkg 2>$null | Out-Null
TLog "FORCE_STOP_ISSUED"
Start-Sleep $DrainSec
$afterDrain = Sample "L2_DRAIN" $DrainSec 30
# baseline after drain window
$maxIdBase = $afterDrain.Dle.MaxId
$maxSeqBase = $afterDrain.Dle.MaxSeq
$t2 = Get-Date
$elapsed = $DrainSec
$sL2 = $afterDrain
while ($elapsed -lt ($DrainSec + $L2WaitSec)) {
  Start-Sleep $PollSec
  $elapsed = $DrainSec + [int]((Get-Date) - $t2).TotalSeconds
  if ($elapsed -gt ($DrainSec + $L2WaitSec)) { $elapsed = $DrainSec + $L2WaitSec }
  $sL2 = Sample "L2_STOP" $elapsed 30
}
$fgs2 = $sL2.Svc.IsForeground
$start2 = $sL2.Svc.StartRequested
$put2 = $sL2.PutN
$deltaId = $sL2.Dle.MaxId - $maxIdBase
$deltaSeq = $sL2.Dle.MaxSeq - $maxSeqBase
$fin2 = $sL2.Nat.Finished
# FGS absent OR isForeground false; no new PUT; flat DLE after drain
$fgsGone = ($fgs2 -ne "true") -and ($start2 -ne "true")
$flat = ($deltaId -eq 0) -and ($put2 -eq 0)
$l2Pass = $fgsGone -and $flat -and ($fin2 -eq 0)
TLog "L2_VERDICT pass=$l2Pass fgsGone=$fgsGone isFg=$fgs2 startReq=$start2 put30=$put2 deltaIdAfterDrain=$deltaId finished=$fin2 (preStop maxId=$maxIdPre maxSeq=$maxSeqPre baseAfterDrain maxId=$maxIdBase)"

if ($l1Pass -and $l2Pass) {
  TLog "VERDICT L12_PASS"
  exit 0
}
TLog "VERDICT L12_FAIL (L2 FORCE-STOP)"
exit 5
