# FG OTA J1-J7 sur binaire 136 - correlation P8 / ATMR_JS_J (60-90s)
# Objectif : trouver le premier arret entre J1 et J7. Pas de build 137.
param(
  [string]$AdbSerial = "192.168.1.33:35129",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_p0e_ota_j1j7_136_fg_2026-08-17",
  [int]$WindowSec = 75,
  [int]$OtaWaitSec = 25,
  [int]$PollSec = 15
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline.txt"
$csv = Join-Path $OutDir "samples.csv"
"ts,elapsed_s,fused_age_s,p2b,p5acc,p5rej,p8js,j1,j2,j3,j4,j5,j6,j7,dle_n,max_seq" |
  Out-File $csv -Encoding utf8

function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) [OTA_J] $msg"
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

function Get-FusedAge {
  $dump = & $adb -s $d shell dumpsys location 2>$null | Out-String
  $uptime = Get-UptimeSec
  if ($dump -match '(?s)fused provider:.*?last location=(Location\[[^\]]+\])') {
    $et = Parse-EtSeconds $Matches[1]
    if ($et -ne $null -and $uptime -gt 0) {
      return [math]::Round($uptime - $et, 1)
    }
  }
  return -1
}

function Count-Tag([string]$path, [string]$pat) {
  return @(Select-String -Path $path -Pattern $pat -EA SilentlyContinue).Count
}

function Get-ServerSnap {
  if (-not $env:SERVER_HOST) {
    return [pscustomobject]@{ DleN = -1; MaxSeq = -1; Raw = "NO_SSH" }
  }
  $out = ssh -o BatchMode=yes -o ConnectTimeout=20 $ssh "docker exec atmr-backend-1 python /tmp/_p0e_snap_dle_canon.py 2>/dev/null" 2>&1 | Out-String
  $n = -1; $seq = -1
  if ($out -match 'DLE\s+(\d+)\s+(\d+)') { $n = [int]$Matches[1]; $seq = [int]$Matches[2] }
  return [pscustomobject]@{ DleN = $n; MaxSeq = $seq; Raw = $out }
}

# Preflight
$vc = (& $adb -s $d shell dumpsys package $pkg 2>$null | Out-String)
if ($vc -notmatch 'versionCode=136') { TLog "ABORT wrong versionCode (need 136)"; exit 2 }

if ($env:SERVER_HOST) {
  scp -o BatchMode=yes -o ConnectTimeout=10 "c:\Users\jasiq\atmr\docs\ops\_p0e_snap_dle_canon.py" "${ssh}:/tmp/_p0e_snap_dle_canon.py" 2>$null | Out-Null
  ssh -o BatchMode=yes -o ConnectTimeout=15 $ssh "docker cp /tmp/_p0e_snap_dle_canon.py atmr-backend-1:/tmp/_p0e_snap_dle_canon.py" 2>$null | Out-Null
}

TLog "START force-stop + reopen to load OTA (runtime 1.0.12 / production)"
& $adb -s $d shell am force-stop $pkg 2>$null | Out-Null
Start-Sleep 2
& $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null
TLog "OTA wait ${OtaWaitSec}s (checkAutomatically ON_LOAD)"
Start-Sleep $OtaWaitSec

# Second open after possible soft reload / update apply
& $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null
Start-Sleep 3

& $adb -s $d logcat -c 2>$null | Out-Null
Start-Sleep 1
TLog "LOGCAT CLEARED - FG window=${WindowSec}s"

$t0 = Get-Date
while (((Get-Date) - $t0).TotalSeconds -lt $WindowSec) {
  $el = [int]((Get-Date) - $t0).TotalSeconds
  $fusedAge = Get-FusedAge
  $snip = Join-Path $OutDir "_logcat_snip.txt"
  & $adb -s $d logcat -d -t 12000 2>$null | Out-File $snip -Encoding utf8

  $c = @{
    p2b = (Count-Tag $snip 'ATMR_LTC_P.*P2b onLocationResult size=[1-9]')
    p5acc = (Count-Tag $snip 'ATMR_LTC_P.*P5 filter.*accepted=true')
    p5rej = (Count-Tag $snip 'ATMR_LTC_P.*P5 filter.*accepted=false')
    p8js = (Count-Tag $snip 'ATMR_LTC_P.*P8 executeTask JS=true')
    j1 = (Count-Tag $snip 'ATMR_JS_J J1_TASK_ENTER')
    j2 = (Count-Tag $snip 'ATMR_JS_J J2_LOCATION_SELECTED')
    j3 = (Count-Tag $snip 'ATMR_JS_J J3_LOCATION_DECISION')
    j4 = (Count-Tag $snip 'ATMR_JS_J J4_TRACKING_CONTEXT')
    j5 = (Count-Tag $snip 'ATMR_JS_J J5_PAYLOAD_FROZEN')
    j6 = (Count-Tag $snip 'ATMR_JS_J J6_ENQUEUE_RESULT')
    j7 = (Count-Tag $snip 'ATMR_JS_J J7_FLUSH_RESULT')
  }
  $srv = Get-ServerSnap
  $line = "$(Get-Date -Format o),$el,$fusedAge,$($c.p2b),$($c.p5acc),$($c.p5rej),$($c.p8js),$($c.j1),$($c.j2),$($c.j3),$($c.j4),$($c.j5),$($c.j6),$($c.j7),$($srv.DleN),$($srv.MaxSeq)"
  [IO.File]::AppendAllText($csv, $line + [Environment]::NewLine)

  TLog ("SAMPLE t=+{0}s fused={1}s p5acc={2} p8js={3} J1={4} J2={5} J3={6} J4={7} J5={8} J6={9} J7={10} dle={11}" -f `
    $el, $fusedAge, $c.p5acc, $c.p8js, $c.j1, $c.j2, $c.j3, $c.j4, $c.j5, $c.j6, $c.j7, $srv.DleN)

  $remain = $WindowSec - ((Get-Date) - $t0).TotalSeconds
  if ($remain -le 0) { break }
  Start-Sleep ([Math]::Min($PollSec, [Math]::Max(1, [int]$remain)))
}

# Final extracts
$fullRaw = Join-Path $OutDir "logcat_raw_window.txt"
& $adb -s $d logcat -d -t 20000 2>$null | Out-File $fullRaw -Encoding utf8

$corr = Join-Path $OutDir "logcat_p8_j1j7.txt"
Select-String -Path $fullRaw -Pattern 'ATMR_LTC_P.*(P5 |P8 )|ATMR_JS_J|Finished task .background-location|ExpoUpdates|UpdatesController' -EA SilentlyContinue |
  ForEach-Object { $_.Line } |
  Out-File $corr -Encoding utf8

$jsOnly = Join-Path $OutDir "logcat_atmr_js_j.txt"
Select-String -Path $fullRaw -Pattern 'ATMR_JS_J' -EA SilentlyContinue |
  ForEach-Object { $_.Line } |
  Out-File $jsOnly -Encoding utf8

$ltcOnly = Join-Path $OutDir "logcat_atmr_ltc_p.txt"
Select-String -Path $fullRaw -Pattern 'ATMR_LTC_P' -EA SilentlyContinue |
  ForEach-Object { $_.Line } |
  Out-File $ltcOnly -Encoding utf8

$p8js = Count-Tag $corr 'P8 executeTask JS=true'
$j1 = Count-Tag $jsOnly 'J1_TASK_ENTER'
$j2 = Count-Tag $jsOnly 'J2_LOCATION_SELECTED'
$j3 = Count-Tag $jsOnly 'J3_LOCATION_DECISION'
$j4 = Count-Tag $jsOnly 'J4_TRACKING_CONTEXT'
$j5 = Count-Tag $jsOnly 'J5_PAYLOAD_FROZEN'
$j6 = Count-Tag $jsOnly 'J6_ENQUEUE_RESULT'
$j7 = Count-Tag $jsOnly 'J7_FLUSH_RESULT'
$j3rej = Count-Tag $jsOnly 'J3_LOCATION_DECISION.*accepted=false'
$j6ins = Count-Tag $jsOnly 'J6_ENQUEUE_RESULT.*inserted=true'
$j6null = Count-Tag $jsOnly 'J6_ENQUEUE_RESULT.*inserted=false'
$srv = Get-ServerSnap

# First-stop verdict
$firstStop = "INCONCLUSIVE"
$detail = ""
if ($p8js -eq 0) {
  $firstStop = "NO_P8"
  $detail = "pas de P8 JS=true dans la fenetre - natif/FG non livre"
} elseif ($j1 -eq 0) {
  $firstStop = "J1_ABSENT"
  $detail = "P8 OK J1 absent - entree handler JS / TaskManager"
} elseif ($j2 -eq 0) {
  $firstStop = "J2_ABSENT"
  $detail = "J1 OK J2 absent - extraction location"
} elseif ($j3rej -gt 0 -and $j4 -eq 0 -and $j5 -eq 0) {
  $firstStop = "J3_REJECT"
  $detail = "J2 OK J3 reject - gate contexte/eligibilite"
} elseif ($j4 -eq 0) {
  $firstStop = "J4_ABSENT"
  $detail = "J3 OK mais J4 absent - owner/mission/session"
} elseif ($j5 -eq 0) {
  $firstStop = "J5_ABSENT"
  $detail = "J4 OK J5 absent - construction/freeze payload"
} elseif ($j6 -eq 0 -or ($j6null -gt 0 -and $j6ins -eq 0)) {
  $firstStop = "J6_ENQUEUE_FAIL"
  $detail = "J5 OK J6 null/rejected - ledger SQLite / READY"
} elseif ($j7 -eq 0) {
  $firstStop = "J7_ABSENT"
  $detail = "J6 inserted OK J7 absent - flush"
} elseif ($srv.DleN -le 0) {
  $firstStop = "J7_OK_DLE0"
  $detail = "J7 ok mais DLE=0 - ingest serveur (hors A4b local)"
} else {
  $firstStop = "PASS_PIPELINE"
  $detail = "chaine P8->J7 + DLE>0"
}

TLog ("SUMMARY p8js={0} j1={1} j2={2} j3={3} j3rej={4} j4={5} j5={6} j6={7} j6ins={8} j6null={9} j7={10} dle={11}" -f `
  $p8js, $j1, $j2, $j3, $j3rej, $j4, $j5, $j6, $j6ins, $j6null, $j7, $srv.DleN)
TLog ("FIRST_STOP {0} :: {1}" -f $firstStop, $detail)
Write-Host ("FIRST_STOP={0}" -f $firstStop)
Write-Host ("DETAIL={0}" -f $detail)
exit 0
