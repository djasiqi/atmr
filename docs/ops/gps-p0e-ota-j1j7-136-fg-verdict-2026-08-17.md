# P0-E — Run FG OTA J1→J7 sur 136 — verdict premier arrêt

```text
device                 = 192.168.1.33:35129 (SM-S911B)
versionCode            = 136
OTA J1→J7              = chargée (force-stop + reopen)
fenêtre                = ~75 s FG (21:44:42 → 21:46:11 local)
artefacts              = docs/ops/_p0e_ota_j1j7_136_fg_2026-08-17/
```

## Question du run

```text
P8 executeTask JS=true
        ↓
où est le premier arrêt entre J1 et J7 ?
```

## Compteurs

| Signal | N |
|--------|---|
| P8 JS=true | **2** |
| J1 / J2 / J3 / J4 / J5 / J6 / J7 | **2 / 2 / 2 / 2 / 2 / 2 / 2** |
| J3 rejected | **0** |
| J6 inserted=true | **2** |
| J6 inserted=false | **0** |
| DLE session | **0** |

## Corrélation même location (cycle 1)

```text
P5  time=1786995926659  lat=46.2115991 lon=6.1262035  accepted=true
P8  JS=true bundles=1   (même tick 21:45:28.940)
J1  locations_count=1
J2  recorded_at=2026-08-17T19:45:26.659Z  (= P5 time)  age_ms=2287
J3  accepted=true reason=pass_gates
J4  missionId=38243 IN_PROGRESS lease=driver_active owner_gen=trk-msxn6uli-f8pjm9tmdu
J5  event_id=trk_1786995928946_mpvjz3jr  seq=69  recorded_at=…26.659Z  inserted=true
J6  inserted=true reason=ok  (même event_id)
J7  sent=3 queue_depth=296 backend_acked=0
    last_event_id=trk_1786990013757_28orbghr   ← ANCIEN (≠ J5)
```

## Corrélation même location (cycle 2)

```text
P5  time=1786995945755  lat=46.2115982 lon=6.1261955
J2  recorded_at=2026-08-17T19:45:45.755Z
J5  event_id=trk_1786995948251_21nxmq3b  seq=71  inserted=true
J6  inserted=true
J7  sent=3 queue_depth=292 backend_acked=0
    last_event_id=trk_1786990127345_f85vufrv   ← ANCIEN (≠ J5)
```

## Grille (arrêt)

```text
P8 ✅
J1 ✅  (pas absent)
J2 ✅  (pas rejet/absent)
J3 ✅  (pass_gates)
J4 ✅  (owner/mission/session OK — 38243, driver_active)
J5 ✅  (payload frozen + event_id neuf)
J6 ✅  (inserted=true ×2)
J7 présent ⚠️  mais PAS « PUT nouveau ACK » :
             sent>0 sur FILE ANCIENNE
             backend_acked=0
             last_event_id ≠ event_id J5
             queue_depth ≈ 290+
DLE=0
```

## Verdict figé (sans élargir le RCA)

```text
A4b JS HANDLER → ENQUEUE     = CLOSED ✅
  (P8→J1→J6 : nouvelle location native freeze + insert SQLite)

PREMIER ARRÊT OBSERVÉ        = J7 FLUSH / TRANSMISSION ★
  - flush tourne (sent=3)
  - n’ACK pas (backend_acked=0)
  - n’avance pas jusqu’aux event_id frais (last_ack = vieux eids)
  - backlog local ~290

INGEST SERVEUR (DLE)         = PAS ENCORE le next RCA
  (grille : seulement si J7 PUT nouveau ✅ + DLE=0)
```

```text
FIRST_STOP = J7_FLUSH_NO_ACK_NEW
(= pas J1–J6 ; pas encore « redescendre serveur »)
```

## Hors scope (inchangé)

Build 137 · HOME · Q1 · modif serveur · PLAY
