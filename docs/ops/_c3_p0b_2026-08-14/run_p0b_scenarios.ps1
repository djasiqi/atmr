# Canary P0-B targeted — B1/B2/B3/B7 (+ hooks B4-B6)
$ErrorActionPreference = "Continue"
$adb = "C:\Users\jasiq\AppData\Local\Android\Sdk\platform-tools\adb.exe"
$d = "100.81.106.54:39179"
$outDir = "c:\Users\jasiq\atmr\docs\ops\_c3_p0b_2026-08-14"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Ensure-Reverse {
  & $adb -s $d reverse tcp:8081 tcp:8081 | Out-Null
  & $adb -s $d reverse tcp:15100 tcp:15100 | Out-Null
}

function Open-App {
  Ensure-Reverse
  $url = "lirie://expo-development-client/?url=" + [uri]::EscapeDataString("http://127.0.0.1:8081")
  & $adb -s $d shell am start -a android.intent.action.VIEW -d "$url" ch.liri.operations 2>$null | Out-Null
  Start-Sleep -Seconds 3
}

function Cold-Start-App {
  Ensure-Reverse
  & $adb -s $d shell am force-stop ch.liri.operations | Out-Null
  Start-Sleep -Seconds 2
  $url = "lirie://expo-development-client/?url=" + [uri]::EscapeDataString("http://127.0.0.1:8081")
  & $adb -s $d shell am start -a android.intent.action.VIEW -d "$url" ch.liri.operations 2>$null | Out-Null
  Start-Sleep -Seconds 18
}

function Capture-Logcat([string]$label) {
  $pidApp = (& $adb -s $d shell pidof ch.liri.operations).Trim()
  $file = Join-Path $outDir "logcat_$label.txt"
  if ($pidApp) {
    & $adb -s $d logcat -d --pid=$pidApp -t 1500 *:S ReactNativeJS:I | Out-File $file -Encoding utf8
  } else {
    "NO_PID" | Out-File $file -Encoding utf8
  }
  $authSkip = @(Select-String -Path $file -Pattern "auth_not_usable" -ErrorAction SilentlyContinue).Count
  $invoked = @(Select-String -Path $file -Pattern "task_invoked" -ErrorAction SilentlyContinue).Count
  $skipped = @(Select-String -Path $file -Pattern "task\.skipped" -ErrorAction SilentlyContinue).Count
  $ensure = @(Select-String -Path $file -Pattern "ensure_headless" -ErrorAction SilentlyContinue).Count
  $session = @(Select-String -Path $file -Pattern "ensure_headless.*(SESSION_AVAILABLE)" -ErrorAction SilentlyContinue).Count
  $temp = @(Select-String -Path $file -Pattern "AUTH_TEMPORARILY_UNAVAILABLE" -ErrorAction SilentlyContinue).Count
  $unavail = @(Select-String -Path $file -Pattern "TRACKING_IDENTITY_UNAVAILABLE" -ErrorAction SilentlyContinue).Count
  $published = @(Select-String -Path $file -Pattern "presence\.published" -ErrorAction SilentlyContinue).Count
  $hydrated = @(Select-String -Path $file -Pattern "presence\.hydrated" -ErrorAction SilentlyContinue).Count
  $cleared = @(Select-String -Path $file -Pattern "presence\.cleared" -ErrorAction SilentlyContinue).Count
  $errFg = @(Select-String -Path $file -Pattern "ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED" -ErrorAction SilentlyContinue).Count
  $startFail = @(Select-String -Path $file -Pattern "start_failed" -ErrorAction SilentlyContinue).Count
  $msg = "METRO_$label auth_not_usable=$authSkip invoked=$invoked skipped=$skipped ensure=$ensure session_ensure=$session temp=$temp unavail=$unavail published=$published hydrated=$hydrated cleared=$cleared err_fg=$errFg start_fail=$startFail pid=$pidApp"
  $msg | Tee-Object -FilePath (Join-Path $outDir "summary_$label.txt")
  Write-Host $msg
}

function Snap-Db([string]$label) {
  $py = @"
from app import create_app
from datetime import datetime, timezone, timedelta
app = create_app()
with app.app_context():
    from models import db
    from sqlalchemy import text
    since = datetime.now(timezone.utc) - timedelta(minutes=12)
    print('LABEL', '$label')
    print('NOW', datetime.now().astimezone().isoformat())
    health = list(db.session.execute(text('''
        SELECT recorded_at, app_state, fgs_running, native_task_running,
               last_fix_age_seconds, constraint_reason, native_start_error
        FROM driver_device_health_events
        WHERE driver_id = 19 AND recorded_at >= :since
        ORDER BY recorded_at DESC LIMIT 20
    '''), {'since': since}).mappings())
    print('HEALTH_N', len(health))
    fgs_t = sum(1 for h in health if h.get('fgs_running') is True)
    print('FGS_TRUE_RATIO', f'{fgs_t}/{len(health)}')
    print('NATIVE_ERR_N', sum(1 for h in health if h.get('native_start_error')))
    for h in health[:6]:
        print('H', h['recorded_at'], 'app=', h['app_state'], 'fgs=', h['fgs_running'], 'nat=', h['native_task_running'],
              'fix=', h['last_fix_age_seconds'], 'cr=', h['constraint_reason'], 'err=', (h['native_start_error'] or '')[:80])
    locs = list(db.session.execute(text('''
        SELECT created_at, mission_id, capture_id
        FROM driver_location_events
        WHERE driver_id = 19 AND created_at >= :since
        ORDER BY created_at DESC LIMIT 12
    '''), {'since': since}).fetchall())
    print('LOC_N', len(locs))
    for r in locs[:5]:
        print('LOC', r[0], 'mission=', r[1], 'cap=', r[2])
"@
  $tmp = Join-Path $env:TEMP "snap_$label.py"
  Set-Content -Path $tmp -Value $py -Encoding UTF8
  docker cp $tmp "atmrstg-backend-1:/tmp/snap_$label.py" | Out-Null
  $raw = docker exec atmrstg-backend-1 python "/tmp/snap_$label.py" 2>&1
  $clean = $raw | Where-Object { $_ -match '^(LABEL|NOW|HEALTH|FGS|NATIVE|H |LOC)' }
  $clean | Tee-Object -FilePath (Join-Path $outDir "snap_$label.txt")
}

Write-Host "==== B1 Cold start / session restored ===="
& $adb -s $d logcat -c | Out-Null
Cold-Start-App
Start-Sleep -Seconds 20
Capture-Logcat "B1_cold"
Snap-Db "B1_cold"

Write-Host "==== B2 Background / headless task ===="
& $adb -s $d logcat -c | Out-Null
Open-App
Start-Sleep -Seconds 5
& $adb -s $d shell input keyevent KEYCODE_HOME | Out-Null
Start-Sleep -Seconds 45
Capture-Logcat "B2_bg"
Snap-Db "B2_bg"
Open-App

Write-Host "==== B3 Refresh window (observe TEMP then SESSION) ===="
& $adb -s $d logcat -c | Out-Null
Open-App
Start-Sleep -Seconds 35
Capture-Logcat "B3_refresh"
Snap-Db "B3_refresh"

Write-Host "==== B7 Runtime JS recreate (force-stop = memory wipe) ===="
& $adb -s $d logcat -c | Out-Null
Cold-Start-App
Start-Sleep -Seconds 25
# Then background to force headless after memory wipe
& $adb -s $d shell input keyevent KEYCODE_HOME | Out-Null
Start-Sleep -Seconds 35
Capture-Logcat "B7_runtime"
Snap-Db "B7_runtime"
Open-App

Write-Host "==== DONE automated B1/B2/B3/B7 ===="
