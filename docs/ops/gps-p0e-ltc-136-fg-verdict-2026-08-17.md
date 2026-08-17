# P0-E — Build 136 FG instrumenté — verdict LTC P0→P8

## Install

```text
versionCode              = 136 ✅
versionName              = 1.0.12
install                  = Success (sideload)
device                   = 192.168.1.33:35129
```

## Run FG ~75 s (21:13–21:15 local)

Artefacts : `docs/ops/_p0e_ltc_136_fg_2026-08-17/`

### Chaîne temporelle (smoking gun réel)

```text
21:14:17.643  P2c isLocationAvailable=true
21:14:17.643  P2b onLocationResult size=1 lat=46.21158 lon=6.12625 time=…055
21:14:17.643  P5 accepted=true  (sLastTimestamp avancé)
21:14:17.643  P6 directFGS bundles.size=1
21:14:17.643  P8 executeTask JS=true bundles.size=1
21:14:18.325  Finished background-location-task
… (×3 cycles similaires)
```

En parallèle : `P2c isLocationAvailable=false` oscille souvent → log `Location unavailable…` **mais ce n’est plus bloquant** : des `P2b` non vides passent quand availability repasse à true.

### Compteurs fenêtre

| Signal | N |
|--------|---|
| P2b non-vide | 3 |
| P2c (dont false) | 14 (7× false) |
| P5 accepted / rejected | **3 / 0** |
| P8 JS=true / empty | **3 / 0** |
| Finished | ≥2 |
| DLE session active | **0** |
| PUT ~3 m | 75 (retries ≠ preuve) |
| promote | 0 |

fused_age observé ~6–11 s (pas toujours <6 s, mais FLP vivant + coords qui bougent légèrement).

## Verdict sous-couche

```text
A1  (pas de callback)              = EXCLU ✅
A3  (filtre sLastTimestamp)        = EXCLU ✅  (rejected=0)
A2' (availability=false only)      = PARTIEL — flaps, mais P2b livre quand true
A4  (accepté sans JS)              = EXCLU ✅  (P8 JS=true)

NATIVE LTC → JS                   = PASS ★★★
  P2b → P5 → P6 → P8 execute OK

POST-JS / DLE                     = FAIL ★★★
  session active …jfaf7k6t DLE=0
  canonical absent
  0 promote
```

Classification script : `A4b_JS_NO_DLE`  
(= locations livrées au Task JS, **aucune** DLE persistée).

```text
EXACT NATIVE SUB-LAYER (LTC)
= plus le goulot pour « 0 capture » sur ce run 136

NEXT ROOT FAMILY
= JS task handler → enqueue / session lease / flush
  OU ingest serveur (PUT retries sans nouvel event_id)
```

Note : `hostPaused=true` pendant tout le run FG (MainActivity) — signal secondaire lifecycle Expo ; n’a pas empêché P8.

## Statut figé

```text
BUILD 136 DIAG           = INSTALLÉ + RUN FG FAIT ✅
IMMUTABILITY 135         = SOUTENU (hors scope ce run)
HOME #3                  = HOLD
Q1 / UX / PLAY / SERVER  = HOLD / HOLD / HOLD / inchangé

P-TECH LTC native        = LIVRE AU JS ✅
P-TECH post-JS → DLE     = OPEN ★★★
```

## ✅ Implémenté

- Install APK 136
- Run FG instrumenté 75 s
- Verdict documenté (LTC OK → rupture après P8)
