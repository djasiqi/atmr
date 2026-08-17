# P0-E — GPS confirmation / map freshness

```text
OUVERT              = 2026-08-17
STATUT              = RCA OPEN ★
D5 / RC132          = FROZEN (hors scope — ne pas modifier)
PLAY ROLLOUT 132    = HOLD ⛔
GPS UI freshness    = CE CHANTIER
```

## Gel D5 (ne plus toucher)

```text
D5 / RC132
RCA                 = CLOSED ✅
CANARY              = VALIDATED ✅
RELEASE REVIEW      = PASS ✅
RC132               = FROZEN ✅
PLAY ROLLOUT        = HOLD ⛔
```

Interdit dans ce train : patch D5 ownership, re-run C1–C4, commit sur `d5-rc-final`, mélanger fix UI au SHA S.

Si un fix **mobile** est requis → **nouveau build** (ex. 133), canary UI séparé.  
Si fix **backend / WS / dashboard** uniquement → OK sans toucher RC132.

---

## Symptômes (départ)

```text
APP CHAUFFEUR
"GPS connecté · Non confirmé"     → Q1 MOBILE (ACK / confirmation)

DASHBOARD / MAP
"Aucun GPS récent"                → Q2 WEB (freshness / présence)

ALORS QUE (pipeline bas)
Finished / PUT / LOC / FGS        = continue / alive (observé smoke RC132)
```

Ne pas supposer Q1 ≡ Q2.

---

## Chaîne à tracer

```text
GPS mobile
  → queue / PUT
  → backend ingest
  → persistance LOC (PG)
  → écriture Redis loc:canonical   ★
  → ACK / confirmation mobile      ★ Q1
  → snapshot GET /me/drivers/live  ★ Q2
  → WebSocket fanout
  → dashboard / marker
```

---

## LOC témoin mission 38243 (serveur)

Source : `driver_location_events` prod, probe 2026-08-17.

```text
WITNESS_ID           = 5846
driver_id            = 20135
mission_id           = 38243
tracking_session_id  = trk_sess_1786965149557_7lkzgzna
session_generation   = 1675
sequence_id (seq)    = 28
location_event_id    = trk_1786965505020_ff3d7bps
recorded_at          = 2026-08-17T11:18:25.532340+00:00
received_at          = (colonne absente en PG — proxy = created_at ingest)
persisted_at         = created_at = 2026-08-17T11:18:26.550024+00:00
persist_lag_s        ≈ 1.02
location_mode        = mission_live
source               = http
lat/lon              = 46.2115669 / 6.1262394
```

Driver row alignée :

```text
last_position_update = 2026-08-17T11:18:25.532340+00:00  (= recorded_at)
latitude/longitude   = mêmes coords
booking 38243        = IN_PROGRESS / driver_id=20135
```

### Fraîcheur backend (référence code)

| Couche | Timestamp utilisé | Seuils |
|--------|-------------------|--------|
| `resolve_location_freshness_timestamp` | **recorded_at** > received_at > ts | — |
| `compute_location_status` (`mission_live`) | âge de ce timestamp | live≤20 / recent≤90 / stale≤300 / sinon offline |
| Fallback DB (Redis vide) | `Driver.last_position_update` | **`last_known`** (pas live/recent) |

À l’instant du persist (âge ≈ 1 s) : statut théorique = **live**.  
Probe ~11:31Z (âge ≈ 804–866 s) : sans Redis → **offline** sur âge pur ; via projection REST → **`last_known`**.

---

## Sentry ED / EC — hors incident

```text
PYTHON-FLASK-ED  = PROBE-INDUCED / HORS INCIDENT ✅
  → /tmp/_p0e_probe_freshness.py SELECT driver.updated_at (colonne absente)
  → https://lirie.sentry.io/issues/PYTHON-FLASK-ED  (resolved 2026-08-17)

PYTHON-FLASK-EC  = PROBE-INDUCED / HORS INCIDENT ✅
  → /tmp/_rc132_loc_probe.py ST_Y(location::geometry) (colonne absente)
  → https://lirie.sentry.io/issues/PYTHON-FLASK-EC  (resolved 2026-08-17)

production app defect = NON
P0-E causal           = NON
```

Probes suivants : **schema-aware** uniquement (`_p0e_put_discriminant.py`).

---

## État figé — Q2 localisé, branche exacte OPEN

```text
Q2 ROOT FAMILY      = INGEST → CANONICAL REDIS ★
exact branch        = OPEN   (acceptation / promotion / expiry — non tranché)
fanout              = HORS CAUSE ✅
TTL expiry current  = plausible / non causalement suffisant
  (canonical+last_raw vides + LOC PG ~29 min + TTL 1200s
   = compatible expiration ; ≠ preuve « Redis jamais alimenté »)
Q1                  = OPEN

NEXT = 1 PUT LIVE → last_raw + canonical immédiatement
```

---

## Discriminant T0 — un PUT live (Cas A/B/C/D)

Dès reprise tracking driver **20135** :

```text
T0 = premier PUT live

immédiatement après :
1. Redis last_raw     — existe ? accept_status ? reason ? seq/session/mission ?
2. Redis loc:canonical — existe ? recorded_at ? seq ? TTL ?
3. PostgreSQL         — nouvelle driver_location_event ? last_position_update avance ?
```

| Cas | Signature | Root Q2 |
|-----|-----------|---------|
| **A** | last_raw présent ; status = observability_only / ignored / stale ; canonical absent | politique ingest / promotion ★ |
| **B** | accept_status = accepted(_canonical) ; canonical absent **immédiatement** | writer / `promote_location_candidate` ★ |
| **C** | accepted ; canonical présent puis disparaît | TTL / delete / overwrite ★ |
| **D** | PUT reçu ; last_raw **aussi** absent | plus haut : endpoint / mode ingest / chemin Redis |

Script prêt (schema-aware) : `docs/ops/_p0e_t0_put_capture.py`

```text
# sur serveur, après scp :
docker cp /tmp/_p0e_t0_put_capture.py atmr-backend-1:/tmp/_p0e_t0_put_capture.py
docker exec atmr-backend-1 python /tmp/_p0e_t0_put_capture.py
```

Pont Q1 (sans fusion) : si `accept_status ≠ accepted` → peut expliquer à la fois « Non confirmé » mobile **et** absence de promotion canonical.

### Probe 11:47Z — **non décisif** (pré-T0)

```text
canonical / legacy / last_raw = EMPTY (ttl=-2)
LATEST_PG = 5846 @ 11:18:26Z  (age ≈1756 s > TTL 1200)
→ compatible TTL expiry ; A/B/C/D non tranché
```

Device adb offline au last check — fenêtre T0 à rouvrir.

---

## Discriminant Q1 (MOBILE) — ouvert, séparé

Label `GPS connecté · Non confirmé` = `formatBridgeSyncLabel` (ACK rejected/stale/ignored corrélé seq+event).

```text
Q1 = OPEN ★
Pont possible via accept_status du même PUT (Cas A) — pas de fusion avant preuve
```

---

## T0 CAPTURE — 2026-08-17 ~11:54:56Z ★

Contexte adb : device `192.168.1.33:34343` reconnecté ; FGS alive ; versionCode **132**.  
Sortie : `docs/ops/_p0e_t0_capture_NOW.txt`

```text
VERDICT case=D ★★  (T0 + T0+5s)
PG live ✅  Redis last_raw/canonical/legacy ABSENTS ❌
```

---

## Attribution D1/D2 — Q2 EXACT ROOT ★

→ `gps-p0e-case-d-attribution-2026-08-17.md`  
→ fix plan : `gps-p0e-q2-pg-first-fix-plan-2026-08-17.md`

```text
Q2 RCA              = CLOSED ✅
PREFERRED FIX       = PG-FIRST CANONICAL ★
PROD FLAG           = HOLD ⛔
FLAG-ONLY           = NO-GO (prod image sans location_candidate / _maybe_promote)
BACKEND CANARY      = NEXT ★
gates module local  = ALL_GATES_PASS ✅
Q1                  = OPEN (après restore canonical)
RC132               = FROZEN ✅
```
