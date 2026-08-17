# P0-D D3-C - chaine livraison Fused -> Consumer -> Task (Prod 126)
# Read-only. Pas de whitelist batterie.
# Prerequis: FGS healthy en TOP (mission chauffeur active).
param(
  [Parameter(Mandatory = $true)][string]$AdbSerial,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_release_exec_p0d_2026-08-16\d3c_delivery",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [int]$WaitHealthySeconds = 180,
  [int]$HomeSeconds = 180,
  [int]$IntervalSeconds = 10
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
$pkg = "ch.liri.operations"

function Get-FgsBlock {
  $svc = & $adb -s $d shell dumpsys activity services $pkg 2>$null | Out-String
  $lines = @($svc -split "`n")
  $block = @(); $in = $false
  foreach ($ln in $lines) {
    if ($ln -match "LocationTaskService") { $in = $true }
    if ($in) {
      $block += $ln
      if ($ln -match "^\s*\* ServiceRecord\{" -and $block.Count -gt 3) { break }
      if ($block.Count -gt 90) { break }
    }
  }
  return ($block -join "`n")
}

function Get-LocReg {
  $raw = & $adb -s $d shell dumpsys location 2>$null | Out-String
  $lines = @($raw -split "`n")
  $hits = @()
  for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match [regex]::Escape($pkg)) {
      $start = [Math]::Max(0, $i - 3)
      $end = [Math]::Min($lines.Count - 1, $i + 8)
      for ($j = $start; $j -le $end; $j++) { $hits += $lines[$j] }
      $hits += "---"
    }
  }
  $active = @($lines | Where-Object {
    $_ -match "WorkSource\{[^}]*$pkg" -and $_ -match "Request\["
  })
  $extra = @($lines | Where-Object {
    $_ -match "fused provider:|ProviderRequest\[|HIGH_ACCURACY|ListenerRegistration|CallbackListener|PendingIntent" -and
    ($_ -match $pkg -or $_ -match "fused provider:" -or $_ -match "ProviderRequest")
  } | Select-Object -First 80)
  return @{
    packageHits = ($hits -join "`n")
    activeReqs = ($active -join "`n")
    filtered = ($extra -join "`n")
    rawLen = $raw.Length
    activeCount = $active.Count
  }
}

function Dump-Point([string]$label) {
  $t = Get-Date -Format o
  $pidApp = (& $adb -s $d shell pidof $pkg 2>$null | Out-String).Trim()

  $fgs = Get-FgsBlock
  $fgs | Out-File (Join-Path $OutDir "svc_$label.txt") -Encoding utf8
  $fgsHit = @($fgs -split "`n" | Where-Object {
    $_ -match "LocationTaskService|isForeground=|startRequested=|getFgsAllowStart=|startForegroundCount=|foregroundId=|Bindings:|ConnectionRecord|createdFromFg|infoAllowStartForeground="
  })
  $fgsHit | Out-File (Join-Path $OutDir "fgs_$label.txt") -Encoding utf8

  $loc = Get-LocReg
  $loc.packageHits | Out-File (Join-Path $OutDir "locpkg_$label.txt") -Encoding utf8
  $loc.activeReqs | Out-File (Join-Path $OutDir "locactive_$label.txt") -Encoding utf8
  $loc.filtered | Out-File (Join-Path $OutDir "locfilt_$label.txt") -Encoding utf8

  $hasReg = ($loc.activeCount -gt 0) -or ($loc.packageHits -match "Request\[")
  $fgTrue = $fgs -match "isForeground=true"
  $srTrue = $fgs -match "startRequested=true"
  $allow = if ($fgs -match "getFgsAllowStart=(\S+)") { $Matches[1] } else { "?" }

  & $adb -s $d logcat -d -t 4000 2>$null |
    Select-String -Pattern "LocationTaskConsumer|LocationTaskService|TaskService|TaskManager|FusedLocationProvider|GmsLocationProvider|LocationManagerService|Location unavailable|Started location updates|onLocationResult|onLocationAvailability|executeTask|Background started FGS|FGS: Denied|Stopping service|Destroying service|stopSelf|stopForeground" |
    ForEach-Object { $_.Line } |
    Out-File (Join-Path $OutDir "am_$label.txt") -Encoding utf8

  $amFile = Join-Path $OutDir "am_$label.txt"
  $amLines = @(Get-Content $amFile -EA SilentlyContinue)
  $unavail = @($amLines | Where-Object { $_ -match "Location unavailable" }).Count
  $startedCb = @($amLines | Where-Object { $_ -match "Started location updates via LocationCallback" }).Count
  $lmsPkg = @($amLines | Where-Object { $_ -match "WorkSource\{[^}]*ch\.liri\.operations" }).Count

  "PID=$pidApp TIME=$t hasReg=$hasReg activeCount=$($loc.activeCount) fg=$fgTrue sr=$srTrue allow=$allow unavail=$unavail" |
    Out-File (Join-Path $OutDir "meta_$label.txt") -Encoding utf8

  TLog "POINT_$label pid=$pidApp fg=$fgTrue sr=$srTrue allow=$allow hasLocReg=$hasReg activeReqs=$($loc.activeCount) unavailHits=$unavail startedCb=$startedCb lmsPkg=$lmsPkg"
}

TLog "D3C_START device=$d home=${HomeSeconds}s interval=${IntervalSeconds}s"
& $adb -s $d logcat -c 2>$null | Out-Null
& $adb -s $d shell input keyevent KEYCODE_WAKEUP 2>$null | Out-Null
& $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null

$deadline = (Get-Date).AddSeconds($WaitHealthySeconds)
$healthy = $false
while ((Get-Date) -lt $deadline) {
  Dump-Point "wait"
  $f = Get-Content (Join-Path $OutDir "fgs_wait.txt") -Raw -EA SilentlyContinue
  if ($f -match "isForeground=true" -and $f -match "startRequested=true") {
    $healthy = $true
    TLog "HEALTHY_FGS"
    break
  }
  & $adb -s $d shell am start -n "$pkg/.MainActivity" 2>$null | Out-Null
  Start-Sleep 8
}
if (-not $healthy) {
  TLog "ABORT_NOT_HEALTHY - login chauffeur + mission IN_PROGRESS requis"
  exit 2
}

& $adb -s $d logcat -c 2>$null | Out-Null
Dump-Point "T0"

TLog "HOME"
& $adb -s $d shell input keyevent KEYCODE_HOME 2>$null | Out-Null

$elapsed = 0
while ($elapsed -lt $HomeSeconds) {
  Start-Sleep -Seconds $IntervalSeconds
  $elapsed += $IntervalSeconds
  Dump-Point ("HOME_{0}s" -f $elapsed)
}

& $adb -s $d logcat -d -t 20000 2>$null |
  Select-String -Pattern "LocationTaskConsumer|LocationTaskService|TaskService|FusedLocation|GmsLocation|LocationManagerService|Location unavailable|Started location updates|onLocationAvailability|Background started FGS|FGS: Denied|Stopping service|Destroying service|WorkSource.*ch\.liri" |
  ForEach-Object { $_.Line } |
  Out-File (Join-Path $OutDir "am_session.txt") -Encoding utf8

TLog "D3C_END"
