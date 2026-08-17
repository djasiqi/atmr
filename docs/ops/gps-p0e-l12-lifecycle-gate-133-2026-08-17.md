# P0-E — Gate L1/L2 lifecycle (build 133) — release safety SÉPARÉ de Q1

## Statut figé

```text
BUILD                        = 133 (1.0.12)
DEVICE                       = 192.168.1.33:34343 SM-S911B
DRIVER                       = 20135
SESSION ANCHOR               = trk_sess_1786979474208_5whqvvm6

L1 BACKGROUND (HOME 150s)    = PASS ✅
L2 FORCE-STOP (drain+75s)    = PASS ✅
VERDICT                      = L12_PASS ✅

Q1 RCA                       = non contaminé (gate séparé)
ACK patch                    = HOLD ⛔
PLAY                         = HOLD ⛔
Recents swipe                = NON TESTÉ (séparé, optionnel)
```

## Règle produit rappelée

```text
A. HOME / arrière-plan + mission IN_PROGRESS
   → GPS DOIT continuer (FGS / Finished / PUT / LOC)

B. Swipe Recents
   → test séparé (FGS peut survivre = normal produit)

C. VRAI force-stop (am force-stop / Paramètres)
   → GPS DOIT cesser ; pas de redémarrage autonome
```

## L1 — BACKGROUND (HOME 150 s)

| Critère | Observé |
|---------|---------|
| FGS alive | `isFg=true` `startReq=true` toute la fenêtre ✅ |
| Finished continue | 1 → 7 ✅ |
| PUT continue | PUT(60) = 51→9 (toujours >0) ✅ |
| LOC avance | maxId 6412 → 6413 (+1) ✅ |
| Session stable | même `…5whqvvm6` ✅ |

**Note** : avance DLE mince (+1) et `N90` tombé à 0 en fin de fenêtre alors que les PUT continuent — possible lag async / idempotence / conflits payload. **Pas un FAIL L1** (FGS+PUT+session OK). Hors scope Q1 ; surveiller si Play.

## L2 — FORCE-STOP

```text
adb shell am force-stop ch.liri.operations
drain 15s + observe 75s
```

| Critère | Observé |
|---------|---------|
| FGS absent | `isFg=absent` `startReq=absent` ✅ |
| Finished nouveau | 0 ✅ |
| PUT(30) | 0 ✅ |
| DLE après drain | deltaId = 0 (plat) ✅ |

Session PG peut rester `active` côté serveur (attendu : force-stop ne clos pas la session serveur). Critère = **pas de nouveaux PUT/LOC**, pas nouvelle prod GPS.

## Artefacts

- Script : `docs/ops/_p0e_l12_lifecycle_gate.ps1`
- Timeline : `docs/ops/_p0e_l12_lifecycle_2026-08-17/timeline_L12.txt`
- Samples : `docs/ops/_p0e_l12_lifecycle_2026-08-17/samples_L12.csv`

## NEXT

```text
L12_PASS → aucun patch lifecycle
→ build 134 QA Q1 déjà lancé (EAS)
→ install 134 après FINISHED
→ capture T3 → PG X → T7
```

EAS 134 (lancé en parallèle, usage interne QA only) :

https://expo.dev/accounts/drinjasiqi/projects/operations-app/builds/1028700a-39f9-4f87-9917-2a347e7db457

## ✅ Implémenté

- Gate L1 HOME + L2 force-stop exécuté sur **133**
- Verdict **L12_PASS** documenté (séparé Q1)
