# P0-D D3 — timeline native LocationTaskService (Prod 126)
# Prérequis: FGS healthy en TOP. Pas de whitelist.
param(
  [Parameter(Mandatory = $true)][string]$AdbSerial,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_release_exec_p0d_2026-08-16\d3_native",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [int]$WaitHealthySeconds = 120
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "timeline.txt"
function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) $msg"
  Add-Content -Path $timeline -Value $line -ErrorAction SilentlyContinue
  Write-Host $line
}

$adb = $AdbPath
$d = $AdbSerial

function Get-FgsCore {
  $svc = & $adb -s $d shell dumpsys activity services ch.liri.operations 2>$null | Out-String
  $lines = @($svc -split "`n")
  $block = @()
  $in = $false
  foreach ($ln in $lines) {
    if ($ln -match "LocationTaskService") { $in = $true }
    if ($in) {
      $block += $ln
      if ($ln -match "^\s*\* ServiceRecord\{" -and $block.Count -gt 3) { break }
      if ($block.Count -gt 80) { break }
    }
  }
  return @{ raw = ($block -join "`n"); all = $svc }
}

function Dump-Point([string]$label) {
  $t = Get-Date -Format o
  $core = Get-FgsCore
  $core.raw | Out-File (Join-Path $OutDir "svc_$label.txt") -Encoding utf8
  $hit = @($core.raw -split "`n" | Where-Object {
    $_ -match "LocationTaskService|isForeground=|startRequested=|getFgsAllowStart=|startForegroundCount=|foregroundId=|types=|executeNesting|createdFromFg|infoAllowStartForeground=|Bindings:|ConnectionRecord|destroyTime|restartTime|lastActivity"
  })
  $hit | Out-File (Join-Path $OutDir "fgs_$label.txt") -Encoding utf8

  $proc = & $adb -s $d shell "dumpsys activity processes | grep -A20 'ch.liri.operations'" 2>$null | Out-String
  $proc | Out-File (Join-Path $OutDir "proc_$label.txt") -Encoding utf8

  $pidApp = (& $adb -s $d shell pidof ch.liri.operations 2>$null | Out-String).Trim()
  "PID=$pidApp TIME=$t" | Out-File (Join-Path $OutDir "meta_$label.txt") -Encoding utf8

  $summary = (($hit | Where-Object { $_ -match "isForeground=|startRequested=|getFgsAllowStart=|startForegroundCount=" }) -join " || ")
  TLog "POINT_$label pid=$pidApp $summary"
}

function Capture-Am([string]$label) {
  $file = Join-Path $OutDir "am_$label.txt"
  & $adb -s $d logcat -d -t 8000 2>$null |
    Select-String -Pattern "LocationTaskConsumer|LocationTaskService|ActiveServices|ForegroundService|startForeground|stopForeground|stopSelf|Stopping service|Destroying service|Killing|onCreate|onStartCommand|onBind|onUnbind|onDestroy|onTaskRemoved|Background started FGS|FGS: Denied|Location unavailable|Could not find a location task|did not then call Service.startForeground|ch\.liri\.operations" |
    ForEach-Object { $_.Line } |
    Out-File $file -Encoding utf8
  TLog "AM_$label lines=$((@(Get-Content $file -EA SilentlyContinue)).Count)"
}

# --- recover healthy FGS ---
TLog "D3_START device=$d"
& $adb -s $d logcat -c 2>$null | Out-Null
& $adb -s $d shell am force-stop ch.liri.operations 2>$null | Out-Null
Start-Sleep 2
& $adb -s $d shell monkey -p ch.liri.operations -c android.intent.category.LAUNCHER 1 2>$null | Out-Null
Start-Sleep 5
& $adb -s $d shell am start -n ch.liri.operations/.MainActivity 2>$null | Out-Null

$deadline = (Get-Date).AddSeconds($WaitHealthySeconds)
$healthy = $false
while ((Get-Date) -lt $deadline) {
  Dump-Point "wait"
  $f = Get-Content (Join-Path $OutDir "fgs_wait.txt") -Raw -EA SilentlyContinue
  if ($f -match "isForeground=true" -and $f -match "startRequested=true" -and $f -match "getFgsAllowStart=PROC_STATE_TOP") {
    $healthy = $true
    TLog "HEALTHY_FGS"
    break
  }
  & $adb -s $d shell am start -n ch.liri.operations/.MainActivity 2>$null | Out-Null
  Start-Sleep 8
}
if (-not $healthy) {
  TLog "ABORT_NOT_HEALTHY"
  exit 2
}

& $adb -s $d logcat -c 2>$null | Out-Null
Dump-Point "T0"
Capture-Am "T0"

TLog "HOME"
& $adb -s $d shell input keyevent KEYCODE_HOME 2>$null | Out-Null

$prev = 0
foreach ($sec in @(2, 10, 15, 20, 25, 30)) {
  Start-Sleep -Seconds ($sec - $prev)
  $prev = $sec
  Dump-Point ("HOME_${sec}s")
  Capture-Am ("HOME_${sec}s")
}

# Full session extract
& $adb -s $d logcat -d -t 15000 2>$null |
  Select-String -Pattern "LocationTaskConsumer|LocationTaskService|ActiveServices|ForegroundService|startForeground|stopForeground|stopSelf|Stopping service|Destroying service|Killing|onDestroy|onUnbind|Background started FGS|Location unavailable|Could not find|did not then call" |
  ForEach-Object { $_.Line } |
  Out-File (Join-Path $OutDir "am_session.txt") -Encoding utf8

TLog "D3_END"
