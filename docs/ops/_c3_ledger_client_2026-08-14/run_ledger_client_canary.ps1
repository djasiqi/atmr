# Canary C-LEDGER-CLIENT isolé — SERVER inchangé
# Device: S23 SM-S911B / driver 19 / mission 26
$ErrorActionPreference = "Continue"
$adb = "C:\Users\jasiq\AppData\Local\Android\Sdk\platform-tools\adb.exe"
$d = "RFCW20QC53W"
$outDir = "c:\Users\jasiq\atmr\docs\ops\_c3_ledger_client_2026-08-14"
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
  & $adb -s $d shell am start -a android.intent.action.VIEW -d "$url" ch.liri.operations 2>$null | Out-Null
  Start-Sleep -Seconds 2
}

function Capture-Logcat([string]$label) {
  $pidApp = (& $adb -s $d shell pidof ch.liri.operations).Trim()
  $file = Join-Path $outDir "logcat_$label.txt"
  if ($pidApp) {
    & $adb -s $d logcat -d --pid=$pidApp -t 4000 *:S ReactNativeJS:I | Out-File $file -Encoding utf8
  } else {
    "NO_PID" | Out-File $file -Encoding utf8
  }
  $patterns = @{
    enqueue_blocked = "enqueue_blocked"
    readiness = "tracking\.session\.readiness"
    register_failed = "register_failed"
    register_deferred = "register_deferred"
    register_ok_ready = 'readiness.: .READY'
    enqueued = "tracking\.queue\.enqueued"
    quarantined = "ledger_invalid_quarantined"
    auth_not_usable = "auth_not_usable"
    err_fg = "ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED"
    nlo_start = "nlo_start"
    nlo_stop = "nlo_stop"
    session_gen_null_enqueue = "session_generation.: null"
  }
  $counts = @()
  foreach ($k in $patterns.Keys) {
    $n = @(Select-String -Path $file -Pattern $patterns[$k] -ErrorAction SilentlyContinue).Count
    $counts += "$k=$n"
  }
  $msg = "SIG_$label pid=$pidApp " + ($counts -join " ")
  $msg | Tee-Object -FilePath (Join-Path $outDir "summary_$label.txt")
  TLog $msg
}

function Pull-Db([string]$label) {
  $b64 = Join-Path $outDir "${label}.b64"
  $db = Join-Path $outDir "${label}.db"
  $walb64 = Join-Path $outDir "${label}_wal.b64"
  $wal = Join-Path $outDir "${label}.db-wal"
  $rawDb = & $adb -s $d exec-out "run-as ch.liri.operations sh -c 'base64 files/SQLite/driver_tracking_queue_v5.db'" 2>$null
  if (-not $rawDb) {
    "PULL_DB_FAIL $label" | Tee-Object -FilePath (Join-Path $outDir "sqlite_$label.txt")
    return
  }
  Set-Content -Path $b64 -Value $rawDb -Encoding ASCII
  [IO.File]::WriteAllBytes($db, [Convert]::FromBase64String(($rawDb -join '' -replace '\s','')))
  $rawWal = & $adb -s $d exec-out "run-as ch.liri.operations sh -c 'base64 files/SQLite/driver_tracking_queue_v5.db-wal'" 2>$null
  if ($rawWal) {
    Set-Content -Path $walb64 -Value $rawWal -Encoding ASCII
    try {
      [IO.File]::WriteAllBytes($wal, [Convert]::FromBase64String(($rawWal -join '' -replace '\s','')))
    } catch {}
  }
  $ro = Join-Path $outDir "${label}_ro.db"
  python -c @"
import sqlite3, shutil, os
src=r'$db'
dst=r'$ro'
shutil.copy2(src, dst)
wal=r'$wal'
if os.path.exists(wal) and os.path.getsize(wal)>0:
  shutil.copy2(wal, dst+'-wal')
con=sqlite3.connect(dst)
try:
  con.execute('PRAGMA wal_checkpoint(FULL)')
except Exception as e:
  print('CHECKPOINT_ERR', e)
con.close()
con=sqlite3.connect(dst)
print('LABEL', '$label')
an=con.execute(\"SELECT count(*) FROM tracking_queue WHERE session_generation IS NULL AND state NOT IN ('persisted','tombstone','rejected')\").fetchone()[0]
ao=con.execute(\"SELECT count(*) FROM tracking_queue WHERE session_generation IS NOT NULL AND state NOT IN ('persisted','tombstone','rejected')\").fetchone()[0]
rn=con.execute(\"SELECT count(*) FROM tracking_queue WHERE session_generation IS NULL AND state='rejected'\").fetchone()[0]
print('ACTIVE_NULL', an)
print('ACTIVE_OK', ao)
print('REJECTED_NULL', rn)
for r in con.execute(\"SELECT location_event_id, tracking_session_id, session_generation, sequence_id, state, queued_at FROM tracking_queue WHERE state NOT IN ('persisted','tombstone','rejected') ORDER BY queued_at DESC LIMIT 8\"):
  print('ACTIVE', r)
for r in con.execute(\"SELECT location_event_id, session_generation, sequence_id, state, queued_at FROM tracking_queue WHERE session_generation IS NOT NULL ORDER BY queued_at DESC LIMIT 5\"):
  print('RECENT_OK', r)
con.close()
"@ | Tee-Object -FilePath (Join-Path $outDir "sqlite_$label.txt")
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
    b = db.session.execute(text('SELECT id, status, driver_id FROM booking WHERE id=26')).fetchone()
    print('BOOK26', b)
    locs = list(db.session.execute(text('''
        SELECT created_at, mission_id, location_event_id, session_generation
        FROM driver_location_events
        WHERE driver_id = 19 AND created_at >= :since
        ORDER BY created_at DESC LIMIT 15
    '''), {'since': since}).mappings())
    print('LOC19_N', len(locs))
    for r in locs[:8]:
        print('LOC19', r['created_at'], 'm=', r['mission_id'], 'gen=', r.get('session_generation'), 'id=', str(r.get('location_event_id'))[:24])
    health = list(db.session.execute(text('''
        SELECT recorded_at, app_state, fgs_running, native_task_running,
               last_fix_age_seconds, constraint_reason
        FROM driver_device_health_events
        WHERE driver_id = 19 AND recorded_at >= :since
        ORDER BY recorded_at DESC LIMIT 8
    '''), {'since': since}).mappings())
    fgs = sum(1 for h in health if h.get('fgs_running') is True)
    print('HEALTH_N', len(health), 'FGS', f'{fgs}/{len(health)}')
    for h in health[:4]:
        print('H19', h['recorded_at'], 'app=', h['app_state'], 'fgs=', h['fgs_running'], 'nat=', h['native_task_running'], 'fix=', h['last_fix_age_seconds'], 'cr=', h['constraint_reason'])
"@
  $tmp = Join-Path $env:TEMP "snap_cl_$label.py"
  Set-Content -Path $tmp -Value $py -Encoding UTF8
  docker cp $tmp "atmrstg-backend-1:/tmp/snap_cl_$label.py" | Out-Null
  $raw = docker exec atmrstg-backend-1 python "/tmp/snap_cl_$label.py" 2>&1
  $clean = $raw | Where-Object { $_ -match '^(LABEL|NOW|BOOK|LOC|HEALTH|H19|FGS)' }
  $clean | Tee-Object -FilePath (Join-Path $outDir "snap_$label.txt")
}

function Capture-All([string]$label) {
  Capture-Logcat $label
  Pull-Db $label
  Snap-Db $label
}

function Set-Network([string]$mode) {
  # mode: off | on
  if ($mode -eq "off") {
    & $adb -s $d shell svc wifi disable 2>$null | Out-Null
    & $adb -s $d shell svc data disable 2>$null | Out-Null
  } else {
    & $adb -s $d shell svc wifi enable 2>$null | Out-Null
    & $adb -s $d shell svc data enable 2>$null | Out-Null
  }
}

# ===== RUN =====
# USB ADB : les coupures wifi/data ne doivent PAS casser adb
Set-Network "on"
Ensure-Reverse
& $adb -s $d logcat -c | Out-Null

TLog "PRECHECK_START USB=$d"
Ensure-App-Fg
Start-Sleep -Seconds 8
Capture-All "P0_precheck"

# --- Reload bundle (CLIENT patch) ---
TLog "RELOAD_BUNDLE_START"
& $adb -s $d shell am force-stop ch.liri.operations
Start-Sleep -Seconds 3
Ensure-App-Fg
# Attendre Metro bundle + login/session restore
Start-Sleep -Seconds 45
Capture-All "C1_after_reload"

# C1 — démarrage normal FG 60s
TLog "C1_START"
Ensure-App-Fg
Start-Sleep -Seconds 60
Capture-All "C1_steady"

# C2 — fenêtre REGISTERING : cold start rapide (capture dès relaunch)
TLog "C2_START"
& $adb -s $d logcat -c | Out-Null
& $adb -s $d shell am force-stop ch.liri.operations
Start-Sleep -Seconds 2
Ensure-App-Fg
Start-Sleep -Seconds 8
Capture-Logcat "C2_registering_window"
Pull-Db "C2_registering_window"
Start-Sleep -Seconds 35
Capture-All "C2_after_ready"

# C5 check early (poison quarantine on load/flush) — already in Pull-Db outputs

# C3 — register failure via offline
TLog "C3_START"
& $adb -s $d logcat -c | Out-Null
Set-Network "off"
# Force new session attempt while offline: force-stop + relaunch
& $adb -s $d shell am force-stop ch.liri.operations
Start-Sleep -Seconds 2
Ensure-App-Fg
Start-Sleep -Seconds 40
Capture-All "C3_offline_failed"
# Restore network
Set-Network "on"
# Re-ensure reverse after wifi
Start-Sleep -Seconds 8
Ensure-Reverse
Ensure-App-Fg
Start-Sleep -Seconds 50
Capture-All "C3_after_recover"

# C6 — offline mid-flight (sans force-stop) puis recover
TLog "C6_START"
& $adb -s $d logcat -c | Out-Null
Ensure-App-Fg
Start-Sleep -Seconds 15
Set-Network "off"
Start-Sleep -Seconds 35
Capture-All "C6_offline"
Set-Network "on"
Start-Sleep -Seconds 8
Ensure-Reverse
Ensure-App-Fg
Start-Sleep -Seconds 45
Capture-All "C6_online_resume"

# C4 — rotation : force-stop + relaunch crée nouvelle session (observe pas d'ancienne mid-rotate)
TLog "C4_START"
& $adb -s $d logcat -c | Out-Null
$beforeDb = Join-Path $outDir "C4_before_sessions.txt"
Pull-Db "C4_before"
Ensure-App-Fg
Start-Sleep -Seconds 20
# Force expire via AsyncStorage if present — fallback force-stop rotate
& $adb -s $d shell am force-stop ch.liri.operations
Start-Sleep -Seconds 2
Ensure-App-Fg
Start-Sleep -Seconds 12
Capture-Logcat "C4_mid_rotate"
Pull-Db "C4_mid_rotate"
Start-Sleep -Seconds 40
Capture-All "C4_after_ready"

# Final gate
TLog "FINAL_GATE"
Ensure-App-Fg
Start-Sleep -Seconds 30
Capture-All "FINAL"

TLog "DONE_LEDGER_CLIENT_CANARY"
Write-Host "OUT=$outDir"
