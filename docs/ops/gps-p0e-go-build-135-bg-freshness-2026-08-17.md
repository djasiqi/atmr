# P0-E — GO build 135 production-apk (canary BG_FRESHNESS immutability)

## Statut figé

```text
ROOT BG_FRESHNESS            = CLOSED ✅
FIX IMMUTABILITY             = IMPLEMENTED ✅
SERVER IDEMPOTENCE           = INCHANGÉ ✅
TESTS                        = PASS ✅

versionCode                  = 135 (app.json)
version / runtimeVersion     = 1.0.12
EAS PROFILE                  = production-apk
usage                        = INTERNAL canary BG_FRESHNESS ONLY
QA panel                     = ON (EXPO_PUBLIC_TRACKING_QA_PANEL=1)

EAS BUILD                    = SUBMITTED ✅
  https://expo.dev/accounts/drinjasiqi/projects/operations-app/builds/a06b58dc-9f61-4598-927b-6c8c710f9d99
  id = a06b58dc-9f61-4598-927b-6c8c710f9d99

Q1 / build 134               = HOLD ⛔
PLAY                         = HOLD ⛔
ACK BEHAVIOR PATCH           = HOLD ⛔
```

## Objectif du build

Valider **uniquement** :

```text
1 event_id = 1 capture_id = 1 payload immuable
= retries strictement identiques
→ event_id_payload_conflict = 0 sous HOME 120 s
→ DLE / canonical / REST / 1/8 restent frais
```

## Commande

```bash
cd mobile/unified-app
eas build --platform android --profile production-apk --non-interactive --no-wait
```

## Canary (après FINISHED + install)

```text
1. Installer versionCode 135
2. Mission IN_PROGRESS
3. Session active stable
4. FG baseline quelques secondes
5. HOME 120 s
6. Tracking normal
7. Observer mobile + consumer + PG + Redis + REST/map
```

### Gate strict

| Critère | Attendu |
|---------|---------|
| `event_id_payload_conflict` | **0** (requis) |
| DLE | continue |
| canonical seq | continue |
| canonical TTL | renouvelé ~1200 |
| REST | live/recent |
| frontend | reste 1/8 |
| FGS | alive |
| Finished | continue |
| PUT 202 | continue |
| session | stable |
| consumer / outbox | healthy |
| traceback | 0 |

### Smoking gun

```text
HOME
→ retries éventuels
→ même event_id = même payload
→ conflict = 0
→ DLE + canonical avancent
→ frontend ne tombe pas à 0/8
```

Si un retry est capturé : deepEqual wire (`recorded_at`, `sent_at`, lat/lon) sous le même eid.

### Si FAIL

```text
STOP
→ FIRST conflicting eid
→ comparer payload PG original vs retry
→ pas de patch serveur / frontend
```

### Si PASS

```text
BG_FRESHNESS FIX CANARY = VALIDATED ✅
→ alors seulement reprendre Q1 ACK
```

## Script capture recommandé

Réutiliser / adapter :

- `docs/ops/_p0e_bg_freshness_gate.ps1` + `_p0e_bg_freshness_rca.py`
- + grep consumer `event_id_payload_conflict` sur la fenêtre HOME

## ✅ Implémenté

- `app.json` versionCode **135**
- Doc GO + gate canary
- EAS build soumis : https://expo.dev/accounts/drinjasiqi/projects/operations-app/builds/a06b58dc-9f61-4598-927b-6c8c710f9d99
