# P0-E Q3 canary 133 — verdict SESSION STABILITY

## Fenêtre

```text
GATE_START_UTC = 2026-08-17T14:44:40Z
GATE_END_UTC   = 2026-08-17T14:46:07Z
STABLE_SEC     = 75
```

## Verdict serveur

```text
VERDICT STABLE_Q3_PASS ✅

ANCHOR = trk_sess_1786977672739_0rzte5pe (gen 1698)
PG_FIRST = false ✅
OUTBOX = true ✅
backend = healthy ✅
consumer = running ✅
outbox publisher = running ✅

rotation active          = 0 ✅
POST /tracking/sessions  = 0 (Traefik fenêtre) ✅
location 409             = 0 ✅
DLE nouvelles            = 5 (seq 21→25) sur X uniquement ✅
eid/capture uniques      = 5/5 ✅
DLE superseded nouvelles = 0 ✅
```

## Device / smoking gun reconnect

```text
logcat reconnect_resync { rotated:false } = NON OBSERVÉ
(filtre ReactNativeJS + dump buffer : aucun [driver-telemetry] reconnect_resync)
```

Le gate serveur prouve l’absence de rotate / POST sessions pendant 75 s avec DLE qui avancent.  
La preuve télémétrie explicite des 2 reconnects n’apparaît pas dans logcat (release peut ne pas remonter `console.info`, ou reconnects non déclenchés / non capturés).

## Statut figé

```text
Q3 PATCH CANARY (serveur) = VALIDATED ✅  STABLE_Q3_PASS
Q3 smoking-gun logcat     = INCONCLUSIVE (pas de ligne reconnect_resync)

PG_FIRST                  = OFF ✅
P5-B                      = HOLD jusqu'à GO explicite
PLAY                      = HOLD

NEXT optionnel
  = 1 fenêtre courte avec 2 Wi-Fi off/on + logcat -v time '*:S' ReactNativeJS:V
    pour capturer reconnect_resync
  OU accepter VALIDATED serveur et GO re-GO P5-B
```

Artefacts : `docs/ops/_p0e_q3_canary_133/`
