# P0-E — Discriminant FLP système vs Expo delivery (FG) — CAS A

## Cadre figé (confirmé)

```text
BG_FRESHNESS                 = plus la couche racine
P-TECH                       = PRODUCTION/DELIVERY LOCATION (même FG) ★★★
IMMUTABILITY 135             = SOUTENU ✅ (conflict=0)
HOME #3                      = ABANDONNÉ / HOLD ⛔
Q1 / PATCH UX / SERVER       = HOLD / HOLD / inchangé
```

## Question posée

> Pendant `Location unavailable`, Android possède-t-il un fix frais ?

## Run FG ciblé ~75 s (2026-08-17 ~20:23 local)

App ramenée ; focus parfois NotificationShade (pas idéal), mais **FGS + providers actifs**.  
Artefacts : `docs/ops/_p0e_flp_vs_expo_2026-08-17/`

| t | fused_age | coords | unavail Δ | Finished Δ | DLE |
|---|-----------|--------|-----------|------------|-----|
| +0 s | **4,5 s** | 46.211593, 6.126211 | 0 | 0 | 0 |
| +27 s | **1,7 s** | 46.211602, 6.126221 | 3 | 2 | 0 |
| +55 s | **5,5 s** | 46.211599, 6.126214 | 2 | 1 | 0 |

```text
fused et avance          = 42126.5 → 42180.5 ✅ (nouveaux fixes système)
freshFixAgeLt30          = True ✅
Location unavailable     = 4 lignes ★
Finished task            = 3 (TaskService se réveille)
DLE / max_seq / canon    = 0 / 0 / absent
session active           = trk_sess_1786991006079_9m3wr4ud (encore une rotation — secondaire)
```

## Smoking gun (aligné)

```text
FLP / GPS provider     = fix frais (âge 1–6 s) qui se met à jour
Expo                   = Location unavailable for foreground-service task delivery
TaskService            = Finished présent (wake) mais
JS / ingest            = aucun nouvel event_id / DLE=0
```

## Verdict

```text
CAS A  = A_FLP_FRESH_EXPO_UNAVAILABLE ★★★

Android FLP produit des fixes frais ✅
Expo LocationTaskConsumer signale unavailable ★
aucune capture exploitable côté pipeline ATMR ❌
```

Nuance : ce n’est pas exactement « aucun Finished » — le task **finit**, mais sans transformer le fix système en payload Location / enqueue. Famille racine reste **FLP → EXPO DELIVERY** (pas « le GPS Android est mort », pas Redis/PG/P5-B).

```text
CAS B (provider mort)     = EXCLU ✅ (fixes frais + et qui avance)
CAS C pur (task jamais)   = PARTIEL — Finished existe, mais pas d’enqueue
```

## Session rotations (secondaire)

```text
…ypmkdr5z → …ojihpgbv → …9m3wr4ud
```

À attribuer (post-Q3) plus tard ; **pas la cause** des 0 captures (chaque session active reste à 0 DLE).

## Formulation exacte (confirmée)

```text
Android / fused system = frais ✅
Expo task pipeline      = pas de nouvelle location exploitable ❌
Finished ≠ location JS  ✅
```

Nuance build ATMR : le log `Location unavailable…` vient de **`onLocationAvailability` (LocationCallback FGS patch)**, pas du PendingIntent stock SDK54.

→ Instrumentation P1–P8 : `docs/ops/gps-p0e-ltc-instrumentation-p1-p8-2026-08-17.md`  
→ Next : build interne diag + 1 run FG (`ATMR_LTC_P`) — **pas EAS sans GO**

## ✅ Implémenté

- Script discriminant : `docs/ops/_p0e_flp_vs_expo_fg.ps1`
- Run FG exécuté ; **CAS A** documenté
