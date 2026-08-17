# Discriminant FLP système vs Expo LocationTask delivery — FG 75s
param(
  [string]$AdbSerial = "192.168.1.33:35129",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_p0e_flp_vs_expo_2026-08-17",
  [int]$WindowSec = 75,
  [int]$PollSec = 15
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline.txt"
$csv = Join-Path $OutDir "samples.csv"
"ts,elapsed_s,uptime_s,fused_et_s,fused_age_s,fused_lat,fused_lon,gps_et_s,gps_age_s,unavail_delta,finished_delta,dle_n,max_seq,canon_seq" |
  Out-File $csv -Encoding utf8

function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) [FLP_EXPO] $msg"
  [IO.File]::AppendAllText($timeline, $line + [Environment]::NewLine)
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

function Parse-EtSeconds([string]$raw) {
  # et=+11h40m25s535ms or et=+25s535ms
  if ($raw -notmatch 'et=\+([^\s\]]+)') { return $null }
  $et = $Matches[1]
  $h = 0; $m = 0; $s = 0; $ms = 0
  if ($et -match '(\d+)h') { $h = [int]$Matches[1] }
  if ($et -match '(\d+)m') { $m = [int]$Matches[1] }
  if ($et -match '(\d+)s') { $s = [int]$Matches[1] }
  if ($et -match '(\d+)ms') { $ms = [int]$Matches[1] }
  return ($h * 3600 + $m * 60 + $s + $ms / 1000.0)
}

function Get-UptimeSec {
  $u = (& $adb -s $d shell cat /proc/uptime 2>$null | Out-String).Trim()
  if ($u -match '^([\d\.]+)') { return [double]$Matches[1] }
  return -1
}

function Get-ProviderLast([string]$provider) {
  $dump = & $adb -s $d shell dumpsys location 2>$null | Out-String
  # Prefer block for "gps provider:" / "fused provider:" last location line
  $pat = "(?s)$provider provider:.*?last location=(Location\[[^\]]+\])"
  if ($dump -match $pat) {
    $loc = $Matches[1]
    $lat = $null; $lon = $null; $et = $null
    if ($loc -match 'Location\[\w+\s+(-?[\d\.]+),(-?[\d\.]+)') {
      $lat = $Matches[1]; $lon = $Matches[2]
    }
    $et = Parse-EtSeconds $loc
    return [pscustomobject]@{ Lat = $lat; Lon = $lon; EtSec = $et; Raw = $loc }
  }
  return $null
}

function Count-LogPatterns {
  $tmp = Join-Path $OutDir "_logcat_snip.txt"
  & $adb -s $d logcat -d -t 4000 2>$null | Out-File $tmp -Encoding utf8
  $unavail = @(Select-String -Path $tmp -Pattern 'Location unavailable for foreground-service task delivery' -EA SilentlyContinue).Count
  $fin = @(Select-String -Path $tmp -Pattern "Finished task 'background-location-task'" -EA SilentlyContinue).Count
  return [pscustomobject]@{ Unavail = $unavail; Finished = $fin }
}

function Get-ServerSnap {
  $out = ssh -o BatchMode=yes -o ConnectTimeout=20 $ssh "docker exec atmr-backend-1 python /tmp/_p0e_snap_dle_canon.py 2>/dev/null" 2>&1 | Out-String
  $sess = $null; $n = -1; $seq = -1; $cseq = -1
  if ($out -match 'SESS\s+(\S+)') { $sess = $Matches[1] }
  if ($out -match 'DLE\s+(\d+)\s+(\d+)') { $n = [int]$Matches[1]; $seq = [int]$Matches[2] }
  if ($out -match 'CANON\s+(\d+)') { $cseq = [int]$Matches[1] }
  return [pscustomobject]@{ Sess = $sess; DleN = $n; MaxSeq = $seq; CanonSeq = $cseq; Raw = $out }
}

TLog "START window=${WindowSec}s"
& $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null
& $adb -s $d logcat -c 2>$null | Out-Null
Start-Sleep 2
$focus = (& $adb -s $d shell dumpsys window 2>$null | Select-String 'mCurrentFocus' | Select-Object -First 1)
TLog "FOCUS $focus"

$t0 = Get-Date
$prevUnavail = 0
$prevFin = 0
$firstFusedEt = $null
$lastFusedEt = $null
$freshFixSeen = $false  # age < 30s at some sample
$etMoved = $false

while (((Get-Date) - $t0).TotalSeconds -lt $WindowSec) {
  $el = [int]((Get-Date) - $t0).TotalSeconds
  $uptime = Get-UptimeSec
  $fused = Get-ProviderLast "fused"
  $gps = Get-ProviderLast "gps"
  $logs = Count-LogPatterns
  $du = $logs.Unavail - $prevUnavail
  $df = $logs.Finished - $prevFin
  $prevUnavail = $logs.Unavail
  $prevFin = $logs.Finished

  $fusedAge = if ($fused -and $fused.EtSec -ne $null -and $uptime -gt 0) { [math]::Round($uptime - $fused.EtSec, 1) } else { -1 }
  $gpsAge = if ($gps -and $gps.EtSec -ne $null -and $uptime -gt 0) { [math]::Round($uptime - $gps.EtSec, 1) } else { -1 }

  if ($fused -and $fused.EtSec -ne $null) {
    if ($null -eq $firstFusedEt) { $firstFusedEt = $fused.EtSec }
    if ($null -ne $lastFusedEt -and [math]::Abs($fused.EtSec - $lastFusedEt) -gt 0.5) { $etMoved = $true }
    $lastFusedEt = $fused.EtSec
  }
  if ($fusedAge -ge 0 -and $fusedAge -lt 30) { $freshFixSeen = $true }

  $srv = Get-ServerSnap
  $line = "$(Get-Date -Format o),$el,$uptime,$($fused.EtSec),$fusedAge,$($fused.Lat),$($fused.Lon),$($gps.EtSec),$gpsAge,$du,$df,$($srv.DleN),$($srv.MaxSeq),$($srv.CanonSeq)"
  [IO.File]::AppendAllText($csv, $line + [Environment]::NewLine)

  TLog ("SAMPLE t=+{0}s uptime={1:n0}s fused_age={2}s lat={3} lon={4} gps_age={5}s unavail_d={6} finished_d={7} dle_n={8} max_seq={9} canon={10}" -f `
    $el, $uptime, $fusedAge, $fused.Lat, $fused.Lon, $gpsAge, $du, $df, $srv.DleN, $srv.MaxSeq, $srv.CanonSeq)

  $remain = $WindowSec - ((Get-Date) - $t0).TotalSeconds
  if ($remain -le 0) { break }
  Start-Sleep ([Math]::Min($PollSec, [Math]::Max(1, [int]$remain)))
}

# Final logcat extract
$logOut = Join-Path $OutDir "logcat_window.txt"
& $adb -s $d logcat -d -t 5000 2>$null |
  Select-String -Pattern 'Location unavailable|Finished task .background-location|LocationTask|TaskService' |
  ForEach-Object { $_.Line } |
  Out-File $logOut -Encoding utf8

$unavailTotal = @(Select-String -Path $logOut -Pattern 'Location unavailable' -EA SilentlyContinue).Count
$finTotal = @(Select-String -Path $logOut -Pattern 'Finished task' -EA SilentlyContinue).Count
$srvEnd = Get-ServerSnap

TLog "SUMMARY freshFixAgeLt30=$freshFixSeen fusedEtMoved=$etMoved firstEt=$firstFusedEt lastEt=$lastFusedEt"
TLog "SUMMARY unavail_lines=$unavailTotal finished_lines=$finTotal dle_n=$($srvEnd.DleN) max_seq=$($srvEnd.MaxSeq) sess=$($srvEnd.Sess)"

# Verdict CAS
$cas = "INCONCLUSIVE"
if ($freshFixSeen -and $unavailTotal -gt 0 -and $srvEnd.DleN -le 0) {
  $cas = "A_FLP_FRESH_EXPO_UNAVAILABLE"
} elseif ((-not $freshFixSeen) -and (-not $etMoved) -and $unavailTotal -gt 0) {
  $cas = "B_NO_FRESH_SYSTEM_FIX"
} elseif ($freshFixSeen -and $finTotal -eq 0 -and $srvEnd.DleN -le 0) {
  $cas = "C_EXPO_TASK_NO_JS"
} elseif ($freshFixSeen -and $finTotal -gt 0 -and $srvEnd.DleN -le 0) {
  $cas = "C_TASK_RUNS_NO_ENQUEUE"
}
TLog "VERDICT_CAS $cas"
Write-Host "VERDICT_CAS=$cas"
if ($cas -match '^A_') { exit 10 }
if ($cas -match '^B_') { exit 11 }
if ($cas -match '^C_') { exit 12 }
exit 13
