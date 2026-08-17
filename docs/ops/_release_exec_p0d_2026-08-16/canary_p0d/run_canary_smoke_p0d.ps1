# Canary P0-D smoke — build 126 inchangé, backend patché
param(
  [string]$AdbSerial = "RFCW20QC53W",
  [int]$DriverId = 20135,
  [string]$OutDir = "c:\Users\jasiq\atmr\docs\ops\_release_exec_p0d_2026-08-16\canary_p0d",
  [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$DeployEnv = "c:\Users\jasiq\atmr\.local.deploy.env",
  [int]$FgSeconds = 120,
  [int]$BgSeconds = 120,
  [int]$LockSeconds = 60
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$timeline = Join-Path $OutDir "smoke_timeline.txt"
function TLog([string]$msg) {
  $line = "$(Get-Date -Format o) $msg"
  Add-Content -Path $timeline -Value $line
  Write-Host $line
}

Get-Content $DeployEnv | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  if ($_ -match '^\s*export\s+(\w+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'")) }
  elseif ($_ -match '^\s*(\w+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value ($Matches[2].Trim('"').Trim("'")) }
}
$sshTarget = "$($env:SERVER_USER)@$($env:SERVER_HOST)"
$adb = $AdbPath
$d = $AdbSerial
$st = (& $adb -s $d get-state 2>&1 | Out-String).Trim()
if ($st -ne "device") { throw "Device $d state='$st'" }

$pkg = (& $adb -s $d shell dumpsys package ch.liri.operations 2>$null | Select-String -Pattern "versionName=|versionCode=" | Select-Object -First 4) -join " "
TLog "SMOKE_P0D_START driver=$DriverId device=$d $pkg"

# Wait API healthy
for ($i=0; $i -lt 20; $i++) {
  $h = ssh -o BatchMode=yes $sshTarget "docker inspect atmr-backend-1 --format '{{.State.Health.Status}}'" 2>&1
  TLog "API_HEALTH $h"
  if ("$h".Trim() -eq "healthy") { break }
  Start-Sleep 3
}

$probePy = @'
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
import sys, json
label = sys.argv[1]
did = int(sys.argv[2])
app = create_app(); app.app_context().push()
from models import db
now = datetime.now(timezone.utc)
since = now - timedelta(minutes=20)
cut = now  # caller passes phases; we use window
# LOC last 20m
locs = list(db.session.execute(text("""
  SELECT created_at, recorded_at, location_event_id, sequence_id, tracking_session_id,
         session_generation, event_payload_hash
  FROM driver_location_events
  WHERE driver_id=:did AND created_at>=:since
  ORDER BY created_at DESC LIMIT 40
"""), {"did": did, "since": since}).mappings())
print("LABEL", label)
print("NOW", now.isoformat())
print("LOC_N", len(locs))
for r in locs[:15]:
    print("LOC", r["created_at"], "rec=", r["recorded_at"], "seq=", r["sequence_id"],
          "eid=", (r["location_event_id"] or "")[:28], "sid=", (r["tracking_session_id"] or "")[:24])
# ingest max seq
ing = db.session.execute(text("""
  SELECT MAX(sequence_id) AS mx, COUNT(*) AS n,
         MAX(recorded_at) AS last_rec
  FROM tracking_ingest_events
  WHERE driver_id=:did AND received_at>=:since
"""), {"did": did, "since": since}).mappings().first()
print("INGEST", dict(ing) if ing else None)
active = list(db.session.execute(text("""
  SELECT id, status::text FROM booking
  WHERE driver_id=:did AND status::text IN ('ASSIGNED','ACCEPTED','EN_ROUTE','IN_PROGRESS')
  ORDER BY id DESC LIMIT 5
"""), {"did": did}).fetchall())
print("ACTIVE", [(a[0], a[1]) for a in active])
# idempotency marker: same eid count in window
dup = db.session.execute(text("""
  SELECT location_event_id, COUNT(*) AS c
  FROM driver_location_events
  WHERE driver_id=:did AND created_at>=:since
  GROUP BY location_event_id HAVING COUNT(*)>1
  LIMIT 5
"""), {"did": did, "since": since}).fetchall()
print("MULTI_ROW_SAME_EID", [(d[0], d[1]) for d in dup])
'@
$probeLocal = Join-Path $OutDir "canary_probe.py"
[System.IO.File]::WriteAllText($probeLocal, ($probePy -replace "`r`n","`n"))
scp -o BatchMode=yes $probeLocal "${sshTarget}:/tmp/canary_probe.py" | Out-Null
ssh -o BatchMode=yes $sshTarget "docker cp /tmp/canary_probe.py atmr-backend-1:/tmp/canary_probe.py" | Out-Null

$dlqPy = @'
import json, os, time
from kafka import KafkaConsumer, TopicPartition
from collections import Counter
BOOT = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092")
TOPIC = "driver.location.dlq.v2"
DID = 20135
consumer = KafkaConsumer(
    bootstrap_servers=[x.strip() for x in BOOT.split(",") if x.strip()],
    enable_auto_commit=False,
    auto_offset_reset="earliest",
    consumer_timeout_ms=12000,
    value_deserializer=lambda b: json.loads(b.decode("utf-8")) if b else None,
)
parts = consumer.partitions_for_topic(TOPIC) or set()
tps = [TopicPartition(TOPIC, p) for p in sorted(parts)]
consumer.assign(tps)
end = consumer.end_offsets(tps)
begin = consumer.beginning_offsets(tps)
for tp in tps:
    consumer.seek(tp, max(begin[tp], end[tp] - 80))
scanned = 0
conflicts = 0
types = Counter()
recent = []
cutoff_ms = int(time.time() * 1000) - 25 * 60 * 1000
for msg in consumer:
    scanned += 1
    val = msg.value
    if not isinstance(val, dict):
        continue
    om = val.get("original_message") or {}
    if om.get("driver_id") not in (DID, str(DID)):
        continue
    et = str(val.get("error_type") or "")
    types[et] += 1
    ts = val.get("timestamp") or 0
    if "payload_conflict" in et:
        conflicts += 1
        if isinstance(ts, int) and ts >= cutoff_ms:
            pl = om.get("payload") if isinstance(om.get("payload"), dict) else {}
            recent.append({
                "dlq_offset": msg.offset,
                "error_type": et,
                "eid": om.get("location_event_id") or pl.get("location_event_id"),
                "seq": pl.get("sequence_id"),
                "recorded_at": pl.get("recorded_at"),
                "original_offset": val.get("original_offset"),
            })
print(json.dumps({"scanned": scanned, "types": dict(types), "conflicts_all_window": conflicts, "conflicts_recent": recent[:20]}, indent=2))
'@
$dlqLocal = Join-Path $OutDir "canary_dlq_probe.py"
[System.IO.File]::WriteAllText($dlqLocal, ($dlqPy -replace "`r`n","`n"))
scp -o BatchMode=yes $dlqLocal "${sshTarget}:/tmp/canary_dlq_probe.py" | Out-Null
ssh -o BatchMode=yes $sshTarget "docker cp /tmp/canary_dlq_probe.py atmr-tracking-kafka-consumer-1:/tmp/canary_dlq_probe.py" | Out-Null

function Capture-Logcat([string]$label) {
  $pidApp = (& $adb -s $d shell pidof ch.liri.operations 2>$null | Out-String).Trim()
  $file = Join-Path $OutDir "logcat_$label.txt"
  if ($pidApp) {
    & $adb -s $d logcat -d --pid=$pidApp -t 8000 *:S ReactNativeJS:I Expo:V TaskManager:V | Out-File $file -Encoding utf8
  } else {
    "NO_PID" | Out-File $file -Encoding utf8
  }
  $finished = @(Select-String -Path $file -Pattern "Finished.*background-location-task|TaskService.*Finished|background-location-task" -EA SilentlyContinue).Count
  $authSkip = @(Select-String -Path $file -Pattern "auth_not_usable" -EA SilentlyContinue).Count
  $nativeErr = @(Select-String -Path $file -Pattern "native_start_error" -EA SilentlyContinue).Count
  $both = @(Select-String -Path $file -Pattern 'start_in_flight.: 1' -EA SilentlyContinue | Where-Object { $_.Line -match 'stop_in_flight.: 1' }).Count
  $genNull = @(Select-String -Path $file -Pattern "generation.: null|generation=null" -EA SilentlyContinue).Count
  $msg = "SIG_$label finishedish=$finished auth_not_usable=$authSkip native_start_error=$nativeErr overlap=$both gen_null=$genNull pid=$pidApp"
  $msg | Tee-Object -FilePath (Join-Path $OutDir "summary_$label.txt")
  TLog $msg
}

function Snap-Prod([string]$label) {
  $out = ssh -o BatchMode=yes $sshTarget "docker exec atmr-backend-1 python /tmp/canary_probe.py $label $DriverId" 2>&1
  $out | Out-File (Join-Path $OutDir "snap_$label.txt") -Encoding utf8
  $useful = @($out | Where-Object { $_ -match '^(LABEL|NOW|LOC|INGEST|ACTIVE|MULTI)' })
  TLog ("SNAP_$label " + (($useful | Select-Object -First 16) -join " | "))
}

function Snap-Dlq([string]$label) {
  $out = ssh -o BatchMode=yes $sshTarget "docker exec atmr-tracking-kafka-consumer-1 python /tmp/canary_dlq_probe.py" 2>&1
  $out | Out-File (Join-Path $OutDir "dlq_$label.txt") -Encoding utf8
  TLog ("DLQ_$label " + (($out | Select-Object -Last 8) -join " "))
}

function Ensure-Foreground {
  & $adb -s $d shell am start -n ch.liri.operations/.MainActivity 2>$null | Out-Null
  Start-Sleep 2
}
function Go-Home {
  & $adb -s $d shell input keyevent KEYCODE_HOME 2>$null | Out-Null
  Start-Sleep 1
}
function Lock-Device {
  & $adb -s $d shell input keyevent KEYCODE_SLEEP 2>$null | Out-Null
  Start-Sleep 1
}
function Unlock-Device {
  & $adb -s $d shell input keyevent KEYCODE_WAKEUP 2>$null | Out-Null
  Start-Sleep 1
  & $adb -s $d shell input keyevent 82 2>$null | Out-Null
  Start-Sleep 1
  & $adb -s $d shell input swipe 540 2000 540 400 250 2>$null | Out-Null
  Start-Sleep 2
}

& $adb -s $d logcat -c 2>$null
Ensure-Foreground
Capture-Logcat "PRE"
Snap-Prod "PRE"
Snap-Dlq "PRE"

TLog "PHASE_FG ${FgSeconds}s"
Ensure-Foreground
Start-Sleep $FgSeconds
Capture-Logcat "FG"
Snap-Prod "FG"

TLog "PHASE_HOME ${BgSeconds}s"
Go-Home
Start-Sleep $BgSeconds
Capture-Logcat "HOME"
Snap-Prod "HOME"
Snap-Dlq "HOME"

TLog "PHASE_LOCK ${LockSeconds}s"
Ensure-Foreground
Start-Sleep 5
Lock-Device
Start-Sleep $LockSeconds
Unlock-Device
Ensure-Foreground
Start-Sleep 40
Capture-Logcat "LOCK"
Snap-Prod "LOCK"
Snap-Dlq "LOCK"

TLog "SMOKE_P0D_END"
