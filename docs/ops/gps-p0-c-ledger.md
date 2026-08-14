# P0-C-LEDGER — Queue items stuck in duplicate ↔ ledger_ids_missing without persisted row

```text
TICKET                     = P0-C-LEDGER
PARENT                     = P0-C
STATUT                     = CLOSED / PASS (CLIENT + SERVER canaries)
                           OBSERVABILITY = DESIGN ONLY / PATCH NO-GO
PATCH CLIENT               = CLOSED / PASS ✅
PATCH SERVER               = CLOSED / PASS ✅ (Option B + canary S1–S6)
FUTURS SUJETS              = OBSERVABILITY design READY — gps-p0-c-observability-design.md
DESIGN CLIENT              = gps-p0-c-ledger-client-design.md
DESIGN SERVER              = gps-p0-c-ledger-server-design.md
DESIGN OBSERVABILITY       = gps-p0-c-observability-design.md
CANARY CLIENT              = gps-c3-ledger-client-canary-2026-08-14.md
CANARY SERVER              = gps-c3-ledger-server-canary-2026-08-14.md
                           (voir gps-p0-c-loc-stale-after-pause.md)
INDÉPENDANCE               = ne pas merger avec P0-C-NATIVE
                           ne pas fusionner CLIENT + SERVER dans un futur patch
                           ne pas rouvrir ledger pour OBSERVABILITY
PREUVE INITIALE            = gps-p0-c-diagnostic-2026-08-14.md
DIAGNOSTIC CHAÎNE          = gps-p0-c-ledger-diagnostic-2026-08-14.md
DIAGNOSTIC BODIES          = gps-p0-c-ledger-bodies-2026-08-14.md
DIAGNOSTIC READINESS       = gps-p0-c-ledger-readiness-2026-08-14.md
OBJETS                     = 3 queue_item_id (18:09:52 / 18:10:10 / 18:10:29)
SESSION PIVOT              = trk_sess_1786722711491_2h9hb2ps
```

## Freeze (figé)

```text
session_id   = présent
sequence     = présent monotone (53/54/55 ; session 1→55)
generation   = NULL dès l'enqueue

→ ledger_ids_missing → claim non libéré → duplicate_unproven → boucle

GPS 18:09–18:10 = vrais nouveaux fixes (≠ C-NATIVE)
```

Cause technique immédiate :

> `createLocalTrackingSession()` rend une session utilisable par l'enqueue avant `generation` register.

## Split conceptuel (futurs designs séparés)

```text
P0-C-LEDGER-CLIENT
  session utilisable avant generation
  + session locale active alors que register n'a pas abouti (B)

P0-C-LEDGER-SERVER
  claim non libéré sur ledger_ids_missing
  + duplicate_unproven avant preuve de persistence
```

## Readiness session pivot (A/B/C)

| Scénario | Verdict |
|----------|---------|
| **A** race readiness (enqueue non gated) | **DESIGN CONFIRMED** (seq1 à T0+1ms) |
| **B** register absent/échoué, session reste active | **CONFIRMED** (0 row PG ; gen null 18+ min ; watermark 403) |
| **C** register OK, generation non réinjectée | **EXCLUDED** |

Détail : [gps-p0-c-ledger-readiness-2026-08-14.md](gps-p0-c-ledger-readiness-2026-08-14.md).

## Objets

| queue_item_id | Création | PG | gen | seq |
|---------------|----------|-----|-----|-----|
| `trk_1786723792342_u8w2gqur` | 18:09:52 | 0 | **null** | 53 |
| `trk_1786723810647_11n415gl` | 18:10:10 | 0 | **null** | 54 |
| `trk_1786723829101_tdhsi20c` | 18:10:29 | 0 | **null** | 55 |

## Implémentation

✅ **Implémenté** : cycle claim ; bodies ; L1 ; freeze cause amont ; split CLIENT/SERVER ; readiness A/B/C ; **CLIENT CLOSED** ; **SERVER CLOSED (canary S1–S6 PASS)** ; design OBSERVABILITY READY.  
**Reste à faire** : patch OBSERVABILITY (GO explicite) — ledger figé.
