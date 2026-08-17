# P0-E — Session tracking témoin propre (20135)

```text
GO                = cold restart tracking témoin uniquement
PG_FIRST          = reste false ⛔
PHASE 2           = PAS ENCORE
GLOBAL ENABLE     = NO-GO
RC132             = FROZEN
```

## Objectif

```text
ancienne session / replay pollué
        ↓
cold restart tracking (device)
        ↓
nouveau tracking_session_id
nouveaux location_event_id / capture_id
seq propre
        ↓
DLE id > 5903 continue
DLQ event_id_payload_conflict = 0
```

Pas de purge manuelle du ledger **serveur**.  
Reset ledger **SQLite local** autorisé uniquement pour sortir des items poison (même identité, payload muté) — voir incident distinct.

## Contrainte RC132 prod

```text
versionCode=132  non-debuggable
run-as / rm SQLite  = IMPOSSIBLE sur ce build
beginNewTrackingSession = n’altère PAS les items déjà en file
```

Donc le reset ledger via `run-as` (canary debug C3) **n’est pas disponible**.  
Option réelle sur le témoin prod :

### A — Force-stop + relaunch (léger, d’abord)

1. Baseline serveur (`DLE max`, `OLD_SESSION`).
2. `adb shell am force-stop ch.liri.operations` (coupe le spam DLQ).
3. Relancer l’app ; au boot : watermark sur `ingested_non_persisted`.
4. Si les poison eid (84/85) sont bien `ingested_non_persisted`, le watermark peut les passer `persisted` (déjà en PG).
5. **Démarrer un nouveau cycle tracking** (mission) → nouveaux eid.
6. Gate `_p0e_gate_clean_session.py`.

Si après relaunch le DLQ reprend avec les **mêmes** eid → option B.

### B — Clear storage app (cycle client propre, recommandé si A échoue)

```text
adb shell pm clear ch.liri.operations
→ re-login chauffeur 20135
→ démarrer tracking mission
→ gate
```

C’est un reset **client** (pas purge ledger serveur). Re-login obligatoire.

## Gate PASS (tous)

```text
PG_FIRST=false
OUTBOX=true
consumer healthy
tracking_session_id ≠ trk_sess_1786965149557_7lkzgzna
location_event_id uniques
capture_id uniques
seq monotone
event_id_payload_conflict (fenêtre 5m) = 0
DLE id > 5903 et continue
Driver.last_position avance
```

## Résultat exécution 2026-08-17 (~12:46–12:51Z)

### Actions device

```text
force-stop ch.liri.operations (RC132 / SM-S911B)
→ relaunch monkey LAUNCHER
→ pid 3204 → 27093
```

`run-as` / rm SQLite **impossible** (build non-debuggable). Pas de `pm clear` (évite re-login tant que le flux avance).

### Gate (session courante)

| Critère | Résultat |
|---------|----------|
| PG_FIRST=false | ✅ |
| OUTBOX=true | ✅ |
| consumer/backend healthy | ✅ |
| `tracking_session_id` nouveau | ✅ `trk_sess_1786966963875_1tbcieoy` (≠ old `…7lkzgzna`) |
| gen | ✅ 1677 |
| DLE id > 5903 | ✅ (ex. 5996+) |
| eid / capture uniques (fenêtre) | ✅ |
| seq monotone (session courante) | ✅ |
| Driver.last_position avance | ✅ |
| `event_id_payload_conflict` = 0 | ⛔ encore >0 |

### Nuance DLQ (ne bloque plus le persist)

Échantillon DLQ `trk_1786968610824_lpfualcp` seq=86 **déjà présent en PG** (id 5995).  
→ conflit = **retry mobile post-persist** avec payload muté (incident distinct), pas un gel DLE.

```text
async LOC → PG        = REPRIS ✅
DLQ absolu = 0        = NON (incident mobile toujours actif)
DLE continue          = OUI ✅
```

### Verdict pré-Phase 2

```text
SESSION PROPRE TÉMOIN     = PASS (identité / PG / seq)
GATE STRICT DLQ=0         = FAIL (retry post-persist)
RE-GO PHASE 2             = HOLD jusqu’à décision :
  (a) assouplir le gate DLQ = « pas de conflit bloquant le PG »
  (b) ou attendre correctif mobile (hors RC132)
```

Scripts : `_p0e_gate_clean_session.sh` / `_p0e_gate_clean_session.py` / `_p0e_dlq_vs_pg.py`
