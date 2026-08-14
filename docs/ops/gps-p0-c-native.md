# P0-C-NATIVE — CLOSED / REQUALIFIED

```text
TICKET                     = P0-C-NATIVE
STATUT                     = CLOSED / REQUALIFIED
CAUSE GPS                  = NON (faux diagnostic causal)
PATCH                      = NO-GO
DIAGNOSTIC                 = gps-p0-c-native-diagnostic-2026-08-14.md
PARENT                     = gps-p0-c-loc-stale-after-pause.md
```

## Verdict

Le natif **continuait** à produire des positions fraîches après 18:13 (enqueue + GNSS ts).  
L’apparence « GPS mort » venait du **HOL ledger** + persistence bloquée + métriques health trompeuses.

```text
N1 / N2 / N3   EXCLUDED
N4             CONFIRMED
```

## Observabilité (futur sujet séparé)

`native_last_fix_age` / `last_fix_age` ≠ âge du vrai `Location.timestamp`.  
Ne plus les utiliser comme preuve directe de fraîcheur GNSS.

## Implémentation

✅ **Implémenté** : ticket fermé requalifié ; plus de piste causale GPS sur P0-C.
