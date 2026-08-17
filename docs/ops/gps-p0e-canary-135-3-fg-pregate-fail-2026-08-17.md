# P0-E — Canary 135 #3 — FG PRE-GATE #1 FAIL → HOME non lancé

## Statut

```text
FG PRE-GATE #1 (90s)         = FAIL ❌
HOME #3                      = NON LANCÉ ⛔ (règle GO)
conflict (3m)                = 0 ✅
session                      = trk_sess_1786985556979_ypmkdr5z (stable)
baseline_seq                 = 19
new_event_id                 = 0
seq_delta                    = 0
canonical_delta              = 0
REST                         = offline (age ~1000s)
recorded_at last             = 2026-08-17T16:57:46Z (figé)
IMMUTABILITY 135             = toujours SUPPORTED (conflict=0)
BG_FRESHNESS E2E             = pas testé (#3)
SERVER                       = NE PAS TOUCHER ✅
PLAY                         = HOLD ⛔
```

## Lecture

Le pré-gate durci a fait son travail : **pas de nouvelles captures FG** → pas de HOME.  
`PUT=continue` n’a pas été utilisé comme critère de PASS.

Cause alignée #2 : pipeline flush/FGS peut vivre, mais **aucun nouvel event_id / recorded_at récent**.

## Next opérateur

```text
1. App au premier plan (déjà MainActivity)
2. Déplacer physiquement le téléphone (dizaines/centaines de m)
   pour éviter filtres Fused too close / too fast
3. Répondre « GPS prêt » / « retry FG »
4. Agent rejoue FG PRE-GATE (≥3 new eid + REST live)
5. PASS seulement → HOME 120 s #3
```

## Artefacts

- Plan : `docs/ops/gps-p0e-canary-135-3-plan-2026-08-17.md`
- Script : `docs/ops/_p0e_fg_pregate_135_3.py`
- Out : `docs/ops/_p0e_bg_freshness_135_3_2026-08-17/fg_pregate.txt`

## ✅ Implémenté

- Pré-gate FG durci exécuté (FAIL)
- HOME #3 bloqué volontairement
- conflict=0 noté (fix 135 non remis en cause)
