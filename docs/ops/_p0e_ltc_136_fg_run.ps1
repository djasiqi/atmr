# FG instrumenté build 136 — ATMR_LTC_P + fused dumpsys (60–90s)
param(
  [string]$AdbSerial = "192.168.1.33:35129",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_p0e_ltc_136_fg_2026-08-17",
  [int]$WindowSec = 75,
  [int]$PollSec = 15
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline.txt"
$csv = Join-Path $OutDir "samples.csv"
"ts,elapsed_s,fused_age_s,fused_lat,fused_lon,p0,p2b,p2c,p5acc,p5rej,p6,p8js,p8empty,finished,unavail,dle_n,max_seq" |
  Out-File $csv -Encoding utf8

function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) [LTC136] $msg"
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

function Get-Fused {
  $dump = & $adb -s $d shell dumpsys location 2>$null | Out-String
  if ($dump -match '(?s)fused provider:.*?last location=(Location\[[^\]]+\])') {
    $loc = $Matches[1]
    $lat = $null; $lon = $null
    if ($loc -match 'Location\[\w+\s+(-?[\d\.]+),(-?[\d\.]+)') {
      $lat = $Matches[1]; $lon = $Matches[2]
    }
    return [pscustomobject]@{ Lat = $lat; Lon = $lon; EtSec = (Parse-EtSeconds $loc); Raw = $loc }
  }
  return $null
}

function Count-Tag([string]$path, [string]$pat) {
  return @(Select-String -Path $path -Pattern $pat -EA SilentlyContinue).Count
}

function Get-ServerSnap {
  $out = ssh -o BatchMode=yes -o ConnectTimeout=20 $ssh "docker exec atmr-backend-1 python /tmp/_p0e_snap_dle_canon.py 2>/dev/null" 2>&1 | Out-String
  $n = -1; $seq = -1
  if ($out -match 'DLE\s+(\d+)\s+(\d+)') { $n = [int]$Matches[1]; $seq = [int]$Matches[2] }
  return [pscustomobject]@{ DleN = $n; MaxSeq = $seq; Raw = $out }
}

# Ensure snap script present
scp -o BatchMode=yes -o ConnectTimeout=10 "c:\Users\jasiq\atmr\docs\ops\_p0e_snap_dle_canon.py" "${ssh}:/tmp/_p0e_snap_dle_canon.py" 2>$null | Out-Null
ssh -o BatchMode=yes -o ConnectTimeout=15 $ssh "docker cp /tmp/_p0e_snap_dle_canon.py atmr-backend-1:/tmp/_p0e_snap_dle_canon.py" 2>$null | Out-Null

$vc = (& $adb -s $d shell dumpsys package ch.liri.operations 2>$null | Out-String)
if ($vc -notmatch 'versionCode=136') { TLog "ABORT wrong version"; exit 2 }

TLog "START window=${WindowSec}s expect=136"
& $adb -s $d shell am start -n ch.liri.operations/.MainActivity 2>$null | Out-Null
Start-Sleep 2
& $adb -s $d logcat -c 2>$null | Out-Null
Start-Sleep 1

$t0 = Get-Date
$prev = @{
  p0 = 0; p2b = 0; p2c = 0; p5acc = 0; p5rej = 0; p6 = 0; p8js = 0; p8empty = 0; fin = 0; unavail = 0
}

while (((Get-Date) - $t0).TotalSeconds -lt $WindowSec) {
  $el = [int]((Get-Date) - $t0).TotalSeconds
  $uptime = Get-UptimeSec
  $fused = Get-Fused
  $fusedAge = if ($fused -and $fused.EtSec -ne $null -and $uptime -gt 0) {
    [math]::Round($uptime - $fused.EtSec, 1)
  } else { -1 }

  $snip = Join-Path $OutDir "_logcat_snip.txt"
  & $adb -s $d logcat -d -t 8000 2>$null | Out-File $snip -Encoding utf8

  $c = @{
    p0 = (Count-Tag $snip 'ATMR_LTC_P.*P0 ')
    p2b = (Count-Tag $snip 'ATMR_LTC_P.*P2b ')
    p2c = (Count-Tag $snip 'ATMR_LTC_P.*P2c ')
    p5acc = (Count-Tag $snip 'ATMR_LTC_P.*P5 filter.*accepted=true')
    p5rej = (Count-Tag $snip 'ATMR_LTC_P.*P5 filter.*accepted=false')
    p6 = (Count-Tag $snip 'ATMR_LTC_P.*P6 ')
    p8js = (Count-Tag $snip 'ATMR_LTC_P.*P8 executeTask JS=true')
    p8empty = (Count-Tag $snip 'ATMR_LTC_P.*P8 executeTask JS=false')
    fin = (Count-Tag $snip "Finished task 'background-location-task'")
    unavail = (Count-Tag $snip 'Location unavailable for foreground-service task delivery')
  }

  $srv = Get-ServerSnap
  $line = "$(Get-Date -Format o),$el,$fusedAge,$($fused.Lat),$($fused.Lon),$($c.p0),$($c.p2b),$($c.p2c),$($c.p5acc),$($c.p5rej),$($c.p6),$($c.p8js),$($c.p8empty),$($c.fin),$($c.unavail),$($srv.DleN),$($srv.MaxSeq)"
  [IO.File]::AppendAllText($csv, $line + [Environment]::NewLine)

  TLog ("SAMPLE t=+{0}s fused_age={1}s p2b={2}(+{3}) p2c={4}(+{5}) p5acc={6} p5rej={7} p8js={8} p8empty={9} finished={10} unavail={11} dle={12}" -f `
    $el, $fusedAge, $c.p2b, ($c.p2b - $prev.p2b), $c.p2c, ($c.p2c - $prev.p2c), `
    $c.p5acc, $c.p5rej, $c.p8js, $c.p8empty, $c.fin, $c.unavail, $srv.DleN)

  $prev = $c
  $remain = $WindowSec - ((Get-Date) - $t0).TotalSeconds
  if ($remain -le 0) { break }
  Start-Sleep ([Math]::Min($PollSec, [Math]::Max(1, [int]$remain)))
}

# Final extracts
$full = Join-Path $OutDir "logcat_atmr_ltc.txt"
& $adb -s $d logcat -d -t 15000 2>$null |
  Select-String -Pattern 'ATMR_LTC_P|Location unavailable|Finished task .background-location' |
  ForEach-Object { $_.Line } |
  Out-File $full -Encoding utf8

$p2b = Count-Tag $full 'P2b '
$p2c = Count-Tag $full 'P2c '
$p5acc = Count-Tag $full 'P5 filter.*accepted=true'
$p5rej = Count-Tag $full 'P5 filter.*accepted=false'
$p8js = Count-Tag $full 'P8 executeTask JS=true'
$p8empty = Count-Tag $full 'P8 executeTask JS=false'
$p2cFalse = Count-Tag $full 'P2c onLocationAvailability isLocationAvailable=false'
$p2bNonEmpty = Count-Tag $full 'P2b onLocationResult size=[1-9]'

$srv = Get-ServerSnap
TLog "SUMMARY p2b=$p2b p2bNonEmpty=$p2bNonEmpty p2c=$p2c p2cFalse=$p2cFalse p5acc=$p5acc p5rej=$p5rej p8js=$p8js p8empty=$p8empty dle=$($srv.DleN)"

# Verdict
$cas = "INCONCLUSIVE"
if ($p2bNonEmpty -eq 0 -and $p2cFalse -gt 0) {
  $cas = "A2prime_AVAIL_FALSE_NO_RESULT"
} elseif ($p2b -eq 0 -and $p2c -eq 0) {
  $cas = "A1_NO_CALLBACK"
} elseif ($p5rej -gt 0 -and $p5acc -eq 0 -and $p8js -eq 0) {
  $cas = "A3_TIMESTAMP_FILTER"
} elseif ($p5acc -gt 0 -and $p8js -eq 0) {
  $cas = "A4_ACCEPTED_NO_JS"
} elseif ($p8js -gt 0 -and $srv.DleN -le 0) {
  $cas = "A4b_JS_NO_DLE"
} elseif ($p8js -gt 0 -and $srv.DleN -gt 0) {
  $cas = "PASS_PIPELINE"
}
TLog "VERDICT_CAS $cas"
Write-Host "VERDICT_CAS=$cas"
exit 0
