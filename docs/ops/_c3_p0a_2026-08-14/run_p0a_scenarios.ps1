# Canary P0-A stress runner — S23 via ADB wireless
$ErrorActionPreference = "Continue"
$adb = "C:\Users\jasiq\AppData\Local\Android\Sdk\platform-tools\adb.exe"
$d = "192.168.1.102:46419"
$outDir = "c:\Users\jasiq\atmr\docs\ops\_c3_p0a_2026-08-14"
$metroLog = "C:\Users\jasiq\.cursor\projects\c-Users-jasiq-atmr\terminals\150900.txt"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Ensure-Reverse {
  & $adb -s $d reverse tcp:8081 tcp:8081 | Out-Null
  & $adb -s $d reverse tcp:15100 tcp:15100 | Out-Null
}

function Ensure-App-Fg {
  & $adb -s $d shell am start -n ch.liri.operations/.MainActivity 2>$null | Out-Null
  Start-Sleep -Seconds 2
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
        SELECT recorded_at, app_state, tracking_active, fgs_running, native_task_running,
               last_fix_age_seconds, native_last_fix_age_seconds, constraint_reason,
               native_start_error, trigger_reason, release_sha
        FROM driver_device_health_events
        WHERE driver_id = 19 AND recorded_at >= :since
        ORDER BY recorded_at DESC LIMIT 25
    '''), {'since': since}).mappings())
    print('HEALTH_N', len(health))
    fgs_t = sum(1 for h in health if h.get('fgs_running') is True)
    nat_t = sum(1 for h in health if h.get('native_task_running') is True)
    print('FGS_TRUE_RATIO', f'{fgs_t}/{len(health)}')
    print('NATIVE_TRUE_RATIO', f'{nat_t}/{len(health)}')
    errs = [h for h in health if h.get('native_start_error')]
    print('NATIVE_ERR_N', len(errs))
    for h in health[:10]:
        err = (h['native_start_error'] or '')[:140]
        print('H', h['recorded_at'], 'app=', h['app_state'], 'fgs=', h['fgs_running'], 'nat=', h['native_task_running'],
              'fix=', h['last_fix_age_seconds'], 'nfix=', h['native_last_fix_age_seconds'],
              'cr=', h['constraint_reason'], 'trig=', h['trigger_reason'], 'err=', err)
    for h in errs[:5]:
        print('ERR', h['recorded_at'], (h['native_start_error'] or '')[:200])
    locs = list(db.session.execute(text('''
        SELECT created_at, mission_id, capture_id
        FROM driver_location_events
        WHERE driver_id = 19 AND created_at >= :since
        ORDER BY created_at DESC LIMIT 20
    '''), {'since': since}).fetchall())
    print('LOC_N', len(locs))
    for r in locs[:6]:
        print('LOC', r[0], 'mission=', r[1], 'cap=', r[2])
"@
  $tmp = Join-Path $env:TEMP "snap_$label.py"
  Set-Content -Path $tmp -Value $py -Encoding UTF8
  docker cp $tmp "atmrstg-backend-1:/tmp/snap_$label.py" | Out-Null
  $raw = docker exec atmrstg-backend-1 python "/tmp/snap_$label.py" 2>&1
  $clean = $raw | Where-Object { $_ -notmatch 'DeprecationWarning|Eventlet|Variables recommand|GATEWAY_|SAFERPAY|OpenTelemetry|SOCKET.IO|Message queue|Niveau de log|Security' }
  $clean | Tee-Object -FilePath (Join-Path $outDir "snap_$label.txt")
}

function Capture-Metro([string]$label) {
  $lines = Get-Content $metroLog -Tail 500 -ErrorAction SilentlyContinue
  $nlo = @($lines | Select-String -Pattern "start_requested|stop_requested|start_failed|stop_failed|start_in_flight|stop_in_flight|ERR_FOREGROUND|nlo_start|nlo_stop|fgs_recover|anti_zombie|NativeTaskInactive")
  $nlo | Out-File (Join-Path $outDir "metro_$label.txt") -Encoding utf8
  $concurrent = 0
  foreach ($m in $nlo) {
    if ($m.Line -match 'start_in_flight["'': ]+1' -and $m.Line -match 'stop_in_flight["'': ]+1') { $concurrent++ }
  }
  # Also detect JSON-ish "start_in_flight": 1 with nearby stop in same object
  $both = @($nlo | Where-Object { $_.Line -match 'start_in_flight.: 1' -and $_.Line -match 'stop_in_flight.: 1' }).Count
  $errFg = @($nlo | Where-Object { $_.Line -match 'ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED' }).Count
  $fails = @($nlo | Where-Object { $_.Line -match 'start_failed|stop_failed|NativeTaskInactive' }).Count
  $starts = @($nlo | Where-Object { $_.Line -match 'start_requested' }).Count
  $stops = @($nlo | Where-Object { $_.Line -match 'stop_requested' }).Count
  $msg = "METRO_$label starts=$starts stops=$stops concurrent_both=$both err_fg=$errFg fails=$fails nlo_hits=$($nlo.Count)"
  $msg | Tee-Object -FilePath (Join-Path $outDir "summary_$label.txt")
  Write-Host $msg
}

function Scenario-Home([int]$n = 15) {
  Write-Host "==== HOME x$n ===="
  for ($i = 1; $i -le $n; $i++) {
    & $adb -s $d shell input keyevent KEYCODE_HOME | Out-Null
    Start-Sleep -Milliseconds 400
    Ensure-App-Fg
    Start-Sleep -Milliseconds 400
  }
  Start-Sleep -Seconds 8
  Capture-Metro "T2_home"
  Snap-Db "T2_home"
}

function Scenario-Shade([int]$n = 20) {
  Write-Host "==== SHADE x$n ===="
  Ensure-App-Fg
  for ($i = 1; $i -le $n; $i++) {
    # Expand notification shade
    & $adb -s $d shell cmd statusbar expand-notifications 2>$null | Out-Null
    Start-Sleep -Milliseconds 250
    & $adb -s $d shell cmd statusbar collapse 2>$null | Out-Null
    Start-Sleep -Milliseconds 250
  }
  Start-Sleep -Seconds 8
  Capture-Metro "T3_shade"
  Snap-Db "T3_shade"
}

function Scenario-HomeApp([int]$n = 12) {
  Write-Host "==== HOME<->APP x$n ===="
  for ($i = 1; $i -le $n; $i++) {
    & $adb -s $d shell input keyevent KEYCODE_HOME | Out-Null
    Start-Sleep -Milliseconds 350
    Ensure-App-Fg
    Start-Sleep -Milliseconds 350
  }
  Start-Sleep -Seconds 8
  Capture-Metro "T_home_app"
  Snap-Db "T_home_app"
}

function Scenario-LockUnlock([int]$cycles = 5) {
  Write-Host "==== LOCK/UNLOCK x$cycles ===="
  Ensure-App-Fg
  for ($i = 1; $i -le $cycles; $i++) {
    & $adb -s $d shell input keyevent KEYCODE_SLEEP | Out-Null
    Start-Sleep -Seconds 3
    & $adb -s $d shell input keyevent KEYCODE_WAKEUP | Out-Null
    Start-Sleep -Milliseconds 500
    # Swipe up to unlock (approx) — may need PIN; try dismiss keyguard
    & $adb -s $d shell wm dismiss-keyguard 2>$null | Out-Null
    & $adb -s $d shell input keyevent KEYCODE_MENU 2>$null | Out-Null
    Start-Sleep -Seconds 1
    Ensure-App-Fg
    Start-Sleep -Seconds 2
  }
  Start-Sleep -Seconds 10
  Capture-Metro "T6_lock"
  Snap-Db "T6_lock"
}

function Scenario-Oscillation([int]$cycles = 12) {
  Write-Host "==== AppState OSC x$cycles ===="
  for ($i = 1; $i -le $cycles; $i++) {
    & $adb -s $d shell input keyevent KEYCODE_HOME | Out-Null
    Start-Sleep -Milliseconds 200
    & $adb -s $d shell cmd statusbar expand-notifications 2>$null | Out-Null
    Start-Sleep -Milliseconds 150
    & $adb -s $d shell cmd statusbar collapse 2>$null | Out-Null
    Start-Sleep -Milliseconds 150
    Ensure-App-Fg
    Start-Sleep -Milliseconds 200
  }
  Start-Sleep -Seconds 10
  Capture-Metro "T9_osc"
  Snap-Db "T9_osc"
}

function Scenario-AntiZombie {
  Write-Host "==== ANTI-ZOMBIE wait 3min BG ===="
  Ensure-App-Fg
  & $adb -s $d shell input keyevent KEYCODE_HOME | Out-Null
  Start-Sleep -Seconds 180
  Ensure-App-Fg
  Start-Sleep -Seconds 15
  Capture-Metro "T10_zombie"
  Snap-Db "T10_zombie"
}

function Scenario-Stabilize5min {
  Write-Host "==== STABILIZE 5min FG ===="
  Ensure-App-Fg
  Start-Sleep -Seconds 300
  Capture-Metro "T12_stabilize"
  Snap-Db "T12_stabilize"
}

Ensure-Reverse
Write-Host "==== PRECHECK ===="
Snap-Db "P0_precheck"
Capture-Metro "P0_precheck"

Scenario-Home 15
Scenario-Shade 25
Scenario-HomeApp 12
Scenario-LockUnlock 5
Scenario-Oscillation 12
Scenario-AntiZombie
Scenario-Stabilize5min

Write-Host "==== DONE ===="
Get-ChildItem $outDir | Select-Object Name,Length | Format-Table
