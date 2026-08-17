# Reprise canary A+B — pré-check + T12 stabilisation 5 min
$ErrorActionPreference = "Continue"
$adb = "C:\Users\jasiq\AppData\Local\Android\Sdk\platform-tools\adb.exe"
$d = "100.81.106.54:40175"
$outDir = "c:\Users\jasiq\atmr\docs\ops\_c3_ab_2026-08-14"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$timeline = Join-Path $outDir "timeline.txt"
function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) $msg"
  Add-Content -Path $timeline -Value $line
  Write-Host $line
}

function Ensure-Reverse {
  & $adb -s $d reverse tcp:8081 tcp:8081 | Out-Null
  & $adb -s $d reverse tcp:15100 tcp:15100 | Out-Null
}

function Ensure-App-Fg {
  Ensure-Reverse
  $url = "lirie://expo-development-client/?url=" + [uri]::EscapeDataString("http://127.0.0.1:8081")
  & $adb -s $d shell input keyevent KEYCODE_WAKEUP 2>$null | Out-Null
  & $adb -s $d shell wm dismiss-keyguard 2>$null | Out-Null
  & $adb -s $d shell am start -a android.intent.action.VIEW -d "$url" ch.liri.operations 2>$null | Out-Null
  Start-Sleep -Seconds 2
}

function Capture-Logcat([string]$label) {
  $pidApp = (& $adb -s $d shell pidof ch.liri.operations).Trim()
  $file = Join-Path $outDir "logcat_$label.txt"
  if ($pidApp) {
    & $adb -s $d logcat -d --pid=$pidApp -t 2000 *:S ReactNativeJS:I | Out-File $file -Encoding utf8
  } else {
    "NO_PID" | Out-File $file -Encoding utf8
  }
  $authSkip = @(Select-String -Path $file -Pattern "auth_not_usable" -ErrorAction SilentlyContinue).Count
  $invoked = @(Select-String -Path $file -Pattern "task_invoked" -ErrorAction SilentlyContinue).Count
  $skipped = @(Select-String -Path $file -Pattern "task\.skipped" -ErrorAction SilentlyContinue).Count
  $errFg = @(Select-String -Path $file -Pattern "ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED" -ErrorAction SilentlyContinue).Count
  $startFail = @(Select-String -Path $file -Pattern "start_failed" -ErrorAction SilentlyContinue).Count
  $both = @(Select-String -Path $file -Pattern 'start_in_flight.: 1' -ErrorAction SilentlyContinue | Where-Object { $_.Line -match 'stop_in_flight.: 1' }).Count
  $starts = @(Select-String -Path $file -Pattern "start_requested|nlo_start" -ErrorAction SilentlyContinue).Count
  $stops = @(Select-String -Path $file -Pattern "stop_requested|nlo_stop" -ErrorAction SilentlyContinue).Count
  $d19 = @(Select-String -Path $file -Pattern "driver_id: 19|driver:19[^0-9]|presence\.published.*19" -ErrorAction SilentlyContinue).Count
  $d20 = @(Select-String -Path $file -Pattern "driver_id: 20|driver:20[^0-9]|presence\.published.*20" -ErrorAction SilentlyContinue).Count
  $ensure = @(Select-String -Path $file -Pattern "ensure_headless" -ErrorAction SilentlyContinue).Count
  $session = @(Select-String -Path $file -Pattern "SESSION_AVAILABLE" -ErrorAction SilentlyContinue).Count
  $msg = "SIG_$label auth_not_usable=$authSkip invoked=$invoked skipped=$skipped concurrent_both=$both err_fg=$errFg start_fail=$startFail starts=$starts stops=$stops ensure=$ensure session=$session d19_hits=$d19 d20_hits=$d20 pid=$pidApp"
  $msg | Tee-Object -FilePath (Join-Path $outDir "summary_$label.txt")
  TLog $msg
}

function Snap-Db([string]$label) {
  $py = @"
from app import create_app
from datetime import datetime, timezone, timedelta
app = create_app()
with app.app_context():
    from models import db
    from sqlalchemy import text
    since = datetime.now(timezone.utc) - timedelta(minutes=15)
    print('LABEL', '$label')
    print('NOW', datetime.now().astimezone().isoformat())
    for did in (20, 19):
        health = list(db.session.execute(text('''
            SELECT recorded_at, app_state, fgs_running, native_task_running,
                   last_fix_age_seconds, native_last_fix_age_seconds, constraint_reason,
                   native_start_error, trigger_reason
            FROM driver_device_health_events
            WHERE driver_id = :did AND recorded_at >= :since
            ORDER BY recorded_at DESC LIMIT 20
        '''), {'did': did, 'since': since}).mappings())
        fgs_t = sum(1 for h in health if h.get('fgs_running') is True)
        nat_t = sum(1 for h in health if h.get('native_task_running') is True)
        errs = [h for h in health if h.get('native_start_error')]
        print(f'HEALTH_{did}_N', len(health))
        print(f'FGS_{did}', f'{fgs_t}/{len(health)}')
        print(f'NATIVE_{did}', f'{nat_t}/{len(health)}')
        print(f'NATIVE_ERR_{did}', len(errs))
        for h in health[:5]:
            print(f'H{did}', h['recorded_at'], 'app=', h['app_state'], 'fgs=', h['fgs_running'], 'nat=', h['native_task_running'],
                  'fix=', h['last_fix_age_seconds'], 'nfix=', h['native_last_fix_age_seconds'],
                  'cr=', h['constraint_reason'], 'err=', (h['native_start_error'] or '')[:100])
        locs = list(db.session.execute(text('''
            SELECT created_at, mission_id, capture_id
            FROM driver_location_events
            WHERE driver_id = :did AND created_at >= :since
            ORDER BY created_at DESC LIMIT 12
        '''), {'did': did, 'since': since}).fetchall())
        print(f'LOC_{did}_N', len(locs))
        for r in locs[:4]:
            print(f'LOC{did}', r[0], 'mission=', r[1], 'cap=', r[2])
    b = db.session.execute(text('SELECT id, status, driver_id FROM booking WHERE id=26')).fetchone()
    print('BOOK26', b)
"@
  $tmp = Join-Path $env:TEMP "snap_ab_$label.py"
  Set-Content -Path $tmp -Value $py -Encoding UTF8
  docker cp $tmp "atmrstg-backend-1:/tmp/snap_ab_$label.py" | Out-Null
  $raw = docker exec atmrstg-backend-1 python "/tmp/snap_ab_$label.py" 2>&1
  $clean = $raw | Where-Object { $_ -match '^(LABEL|NOW|HEALTH|FGS|NATIVE|H20|H19|LOC|BOOK)' }
  $clean | Tee-Object -FilePath (Join-Path $outDir "snap_$label.txt")
}

function Capture-All([string]$label) {
  Capture-Logcat $label
  Snap-Db $label
}

TLog "RESUME_PRECHECK_START device=$d"
Ensure-Reverse
Ensure-App-Fg
Start-Sleep -Seconds 25
Capture-All "resume_precheck"

TLog "RESUME_T12_STABILIZE_5MIN_START"
Ensure-App-Fg
Start-Sleep -Seconds 300
Capture-All "resume_T12_stabilize"

TLog "RESUME_T12_DONE"
Write-Host "OK resume T12 complete"
