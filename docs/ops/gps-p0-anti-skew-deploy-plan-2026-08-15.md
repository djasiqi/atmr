# Plan anti-skew déploiement P0 — ÉBAUCHE (non figée)

```text
STATUT     = DRAFT / DEFERRED
CONDITION  = dry-run cherry-pick G0 VERT d’abord
REF FAIL   = gps-p0-cherry-pick-dry-run-2026-08-15.md
DATE       = 2026-08-15
```

Tant que le composite `927640a0` + 5 commits P0 n’est pas prouvé propre, **ne pas** figer ce plan comme procédure de release.

## Intention (après TIP `R` unique)

```text
release TIP = R

1. Build R une seule fois
2. API / celery / ws       → image sha-R
3. Vérifications initiales
4. consumer + outbox       → recreate avec image sha-R
5. fanout / dlq            → également image sha-R
                              MAIS état HOLD conservé
6. p0-hold.yml             → reste appliqué
7. fanout ENABLED=false    → confirmé runtime
```

**Capital** : aligner l’image ≠ lever le HOLD.

## Topologie cible post-release (HOLD inchangé)

```text
API              sha-R   Up
celery           sha-R   Up
ws               sha-R   Up
consumer         sha-R   Up
outbox           sha-R   Up

fanout-1         sha-R   Created/stopped
fanout-2         sha-R   Created/stopped
dlq              sha-R   état prévu par HOLD

TRACKING_PROCESSED_FANOUT_ENABLED=false
dans le runtime concerné
```

## Gates liés

```text
G0 composition   = ROUGE aujourd’hui → plan non exécutable
G2 skew          = déjà expliqué ; ce plan le résorbe côté images seulement
DEPLOY           = NO-GO jusqu’à G0–G5 + GO explicite
```

```text
✅ **Implémenté** : ébauche écrite, explicitement non figée après dry-run FAIL.
**Reste à faire** : figer + jouer le plan seulement après composite cherry-pick propre + TIP `R`.
```
