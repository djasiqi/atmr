# Gate 0A-1 : vérifie ACCESS_BACKGROUND_LOCATION sur ch.liri.operations
# 0A-2 / 0A-3 : à valider sur device (QA panel + build EXPO_PUBLIC_ENABLE_BG_LOCATION=1)

$ErrorActionPreference = "Stop"
$package = "ch.liri.operations"

$adb = Get-Command adb -ErrorAction SilentlyContinue
if (-not $adb) {
  Write-Host "FAIL: adb introuvable dans PATH"
  exit 2
}

$devices = adb devices | Select-String "device$"
if (-not $devices) {
  Write-Host "SKIP: aucun appareil Android connecté (brancher S23 + USB debugging)"
  exit 3
}

Write-Host "=== Gate 0A-1 : dumpsys package $package ==="
$dump = adb shell dumpsys package $package 2>&1 | Out-String
$bgLine = $dump -split "`n" | Where-Object { $_ -match "ACCESS_BACKGROUND_LOCATION" } | Select-Object -First 3
if ($bgLine) {
  $bgLine | ForEach-Object { Write-Host $_ }
} else {
  Write-Host "WARN: ligne ACCESS_BACKGROUND_LOCATION non trouvée dans dumpsys"
}

if ($dump -match "ACCESS_BACKGROUND_LOCATION.*granted=true") {
  Write-Host "PASS: ACCESS_BACKGROUND_LOCATION granted=true"
  exit 0
}

Write-Host "FAIL: ACCESS_BACKGROUND_LOCATION non accordée (granted=true absent)"
Write-Host "Action: Paramètres > Apps > Lirie > Autorisations > Position > Toujours autoriser"
exit 1
