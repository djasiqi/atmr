# Canary 135 #3 — FG PRE-GATE durci puis HOME (conditionnel)

## Règle

```text
HOME #3 UNIQUEMENT SI FG PRE-GATE PASS
PUT=continue n'est PAS un signal de PASS
```

## FG PRE-GATE (requis)

```text
nouveaux fixes / event_id     ≥ 3
seq                           avance (baseline+1…)
DLE + canonical               avancent
recorded_at                   récent
REST                          live|recent
session                       stable
conflict                      0
idéalement                    déplacement physique téléphone
```

## HOME PASS (si FG OK)

```text
nouveaux event_id/capture_id  continuent
recorded_at                   avance
DLE + canonical.seq           avancent
TTL                           se renouvelle
REST                          live|recent
conflict                      0
FGS                           alive
```

## Verdicts attendus

```text
PASS #3 → IMMUTABILITY 135 VALIDATED + BG_FRESHNESS E2E VALIDATED
FAIL FG → STOP, pas de HOME (GPS FG pas vivant)
FAIL HOME après FG PASS + Location unavailable → ticket distinct BG location delivery
conflict=0 reste = fix 135 tient
```

## ✅ Implémenté

- Script : `docs/ops/_p0e_fg_pregate_135_3.py`
- FG PRE-GATE #1 (90s) : **FAIL** — `new_eid=0`, seq/canon figés à 19, REST offline  
  → détail : `gps-p0e-canary-135-3-fg-pregate-fail-2026-08-17.md`
- HOME #3 : **non lancé** (règle GO)
