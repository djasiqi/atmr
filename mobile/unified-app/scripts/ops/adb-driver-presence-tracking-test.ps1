# Test ADB suivi GPS chauffeur — présence 07h-19h (sans mission active requise).
# Usage:
#   .\scripts\ops\adb-driver-presence-tracking-test.ps1
#   .\scripts\ops\adb-driver-presence-tracking-test.ps1 -DriverId 7135 -FgSeconds 45 -BgSeconds 120
param(
    [string]$DriverId = $env:DRIVER_ID,
    [string]$Package = "ch.liri.operations",
    [int]$FgSeconds = 45,
    [int]$BgSeconds = 90
)

$ErrorActionPreference = "Continue"
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$repoRoot = Split-Path $root -Parent
$logDir = Join-Path $repoRoot "tmp"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "adb_presence_tracking_$stamp.txt"

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

function RedisLoc($id) {
    if (-not $id) { return $null }
    try {
        $raw = docker exec atmr-redis-1 redis-cli HGETALL "driver:${id}:loc" 2>&1
        if ($LASTEXITCODE -ne 0) { return $null }
        $map = @{}
        $key = $null
        foreach ($line in ($raw -split "`n")) {
            $t = $line.Trim()
            if (-not $t) { continue }
            if ($null -eq $key) { $key = $t; continue }
            $map[$key] = $t
            $key = $null
        }
        return $map
    } catch { return $null }
}

function FormatLoc($map) {
    if (-not $map -or $map.Count -eq 0) { return "(vide)" }
    $lat = $map["lat"]
    $lon = $map["lon"]
    $ts = $map["ts"]
    $bg = $map["is_background"]
    $mode = $map["location_mode"]
    $acc = $map["accuracy"]
    return "ts=$ts lat=$lat lon=$lon bg=$bg mode=$mode acc=$acc"
}

$devices = adb devices 2>&1 | Select-String "device$"
if (-not $devices) {
    Write-Host "FAIL: aucun device ADB" -ForegroundColor Red
    exit 2
}

Log "=== Test presence tracking (FG ${FgSeconds}s + BG ${BgSeconds}s) ==="
Log "Log: $logFile"

$hour = (Get-Date).Hour
$inWindow = $hour -ge 7 -and $hour -lt 19
Log "Heure locale PC: $(Get-Date -Format 'HH:mm') — fenêtre 07-19h: $(if ($inWindow) { 'OUI' } else { 'NON' })"

$dump = adb shell dumpsys package $Package 2>&1 | Out-String
$fgLoc = $dump -match "ACCESS_FINE_LOCATION.*granted=true"
$bgLoc = $dump -match "ACCESS_BACKGROUND_LOCATION.*granted=true"
Log "Permission FINE_LOCATION: $(if ($fgLoc) { 'OK' } else { 'MANQUANTE' })"
Log "Permission BACKGROUND_LOCATION: $(if ($bgLoc) { 'OK' } else { 'MANQUANTE' })"

if (-not $DriverId) {
    $keys = docker exec atmr-redis-1 redis-cli KEYS "driver:*:loc" 2>&1
    $match = $keys | Select-String -Pattern "driver:(\d+):loc" | Select-Object -First 1
    if ($match) { $DriverId = $match.Matches.Groups[1].Value }
}
Log "DriverId cible Redis: $DriverId"
Log "Redis AVANT: $(FormatLoc (RedisLoc $DriverId))"

adb logcat -c 2>$null
$logcatJob = Start-Job -ScriptBlock {
    param($out)
    adb logcat -v time ReactNativeJS:V ExpoLocation:V LocationTask:V chromium:V *:S 2>&1 | Tee-Object -FilePath $out
} -ArgumentList (Join-Path $logDir "adb_presence_logcat_$stamp.txt")

Log "Foreground: lancement $Package..."
adb shell am start -n "$Package/.MainActivity" -a android.intent.action.MAIN 2>&1 | Out-Null
Start-Sleep -Seconds $FgSeconds
Log "Redis FG (+${FgSeconds}s): $(FormatLoc (RedisLoc $DriverId))"

Log "Background: KEYCODE_HOME..."
adb shell input keyevent KEYCODE_HOME 2>&1 | Out-Null
Start-Sleep -Seconds $BgSeconds
Log "Redis BG (+${BgSeconds}s): $(FormatLoc (RedisLoc $DriverId))"

# Foreground service location
$fgs = adb shell dumpsys activity services 2>&1 | Select-String -Pattern "ch.liri|location|Location" | Select-Object -First 15
if ($fgs) {
    Log "--- Services (extrait) ---"
    $fgs | ForEach-Object { Log $_.Line.Trim() }
}

Stop-Job $logcatJob -ErrorAction SilentlyContinue
Remove-Job $logcatJob -Force -ErrorAction SilentlyContinue

$catFile = Join-Path $logDir "adb_presence_logcat_$stamp.txt"
if (Test-Path $catFile) {
    $hits = Get-Content $catFile | Select-String -Pattern "tracking|presence|location|ingest|queue|FGS|background" -CaseSensitive:$false
    Log "--- Logcat hits ($($hits.Count) lignes) ---"
    $hits | Select-Object -Last 40 | ForEach-Object { Log $_.Line }
}

Log "=== FIN ==="
