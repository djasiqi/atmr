# P0-C-NATIVE — Diagnostic read-only post-18:13 (N1–N4)

```text
GO                         = READ-ONLY post-18:13 only — FAIT
PATCH                      = NO-GO
VERDICT                    = N4 CONFIRMED ; P0-C-NATIVE = CLOSED / REQUALIFIED
HORS PÉRIMÈTRE             = 18:09–18:10 (GNSS OK, C-LEDGER)
                           generation=null (C-LEDGER)
```

## Freeze statut

```text
P0-A / P0-B / C3          CLOSED / PASS ✅
P0-C-LEDGER-CLIENT        CLOSED / PASS ✅
P0-C-LEDGER-SERVER        CLOSED / PASS ✅
P0-C-NATIVE               CLOSED / REQUALIFIED
N1 / N2 / N3              EXCLUDED
N4                        CONFIRMED
OBSERVABILITY             DESIGN READY — gps-p0-c-observability-design.md
PATCH OBSERVABILITY       NO-GO
```

---

## Health post-18:13 (FGS up, âges qui montent)

Après reset ~18:13:36 (`nfix=0`, `fix=0`) :

| Heure | FGS | fix (last_fix_age) | nfix | cr |
|-------|-----|---------------------|------|-----|
| 18:13:36 | true | 0 | 0 | — |
| 18:14:41 | true | 65 | 55 | — |
| 18:15:42 | true | 126 | 116 | — |
| 18:16:42 | true | 186 | 177 | — |
| 18:18:43 | true | 307 | 298 | fix_stale |
| 18:21:45 | true | 489 | 479 | fix_stale |

Définition métriques (code) :

```text
native_last_fix_age  = âge(lastTaskInvokedAt)     // invoke task BG, PAS âge GNSS
last_fix_age         = âge(lastWatchAtMs)         // dernier callback watchPositionAsync
```

→ Un `nfix` qui monte ≠ « aucun Location object » ; ça dit surtout que le **task FGS / watch** ne se rappellent plus.

---

## Queue post-18:13 — GNSS encore produit et enqueued

Dès 18:13:00+, SQLite montre des enqueue **continus** avec :

- `timestamp` GNSS qui **avance** (ex. 16:13 → 16:22 Z)
- lat/lng qui **bougent légèrement** (~46.21159–46.21165)
- `session_generation` **numérique** (665+) — sessions register OK
- `state=non_ingested` (bloqués en HOL derrière les 3 items C-LEDGER)

Exemples :

```text
18:14:02  gnss=16:14:01.371Z  lat=46.2116015  gen=669  seq=1  non_ingested
18:15:02  gnss=16:15:00.952Z  lat=46.2116123  gen=675  seq=1  non_ingested
18:16:03  gnss=16:16:02.160Z  lat=46.2116303  gen=679  seq=1  non_ingested
18:18:00  gnss=16:17:58.973Z  lat=46.2116078  gen=687  seq=1  non_ingested
18:20:01  gnss=16:19:59.947Z  lat=46.2116188  gen=694  seq=1  non_ingested
```

Écart enqueue↔GNSS typiquement **~1–2 s** → pas un recycle d’ancre 17:20.

Nuance : quelques réutilisations de même `timestamp` sur 1–2 enqueue consécutifs (ex. 16:13:35.132) — recycle court possible, **pas** le pattern dominant.

---

## Classification N1–N4

| Cas | Verdict | Preuve |
|-----|---------|--------|
| **N1** aucun callback Location après 18:13 | **EXCLUDED** | enqueue + GNSS ts frais continus |
| **N2** callbacks mais timestamp GNSS qui vieillit seul | **EXCLUDED** (primaire) | timestamps GNSS avancent avec le mur |
| **N3** nouveaux fixes rejetés avant enqueue | **EXCLUDED** | présents en queue `non_ingested` |
| **N4** nouveaux fixes enqueued | **CONFIRMED** | SQLite post-18:13 |

### Conséquence

```text
P0-C-NATIVE = CLOSED / REQUALIFIED

Ce n’était pas un GPS mort sous FGS.
Les fixes frais continuaient ; HOL ledger + métriques health
donnaient l’illusion d’un stale GNSS.
```

---

## Séparation stricte

| Ne pas confondre | |
|------------------|---|
| `generation=null` | C-LEDGER only |
| 18:09–18:10 GNSS | hors C-NATIVE |
| `nfix`↑ health | ≠ absence de Location ; métrique task/watch |
| 0 row PG | largement ledger HOL, pas N1 |

---

## Implémentation

✅ **Implémenté** : diagnostic N1–N4 post-18:13 ; **N4 CONFIRMED** ; requalification C-NATIVE ; RCA LEDGER figé à part.  
**Reste à faire** : optionnel — divergence task/watch vs enqueue ; **PATCH NO-GO**.
