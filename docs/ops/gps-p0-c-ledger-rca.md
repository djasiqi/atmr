# P0-C-LEDGER — RCA causal (fermé pour diagnostic)

```text
STATUT DIAGNOSTIC CAUSAL   = CLOSED
P0-C-NATIVE                = CLOSED / REQUALIFIED (N4 — pas une cause GPS)
PATCH                      = NO-GO (preuves conservées : claims Redis, queue, watermark 403)
```

## RCA figé

```text
P0-C-LEDGER-CLIENT = CONFIRMED

createLocalTrackingSession()
→ session immédiatement considérée exploitable
→ enqueue dès T0+1 ms
→ generation = null
→ 55/55 événements invalides pour le ledger
→ register serveur jamais abouti (PG tracking_sessions = 0)
→ session locale pourtant maintenue active
```

```text
P0-C-LEDGER-SERVER = CONFIRMED

event incomplet
→ claim Redis acquis
→ ledger_ids_missing
→ claim non libéré
→ retry
→ duplicate_event_id_unproven
→ release
→ reclaim
→ boucle
```

## Chaîne bout-en-bout

```text
GNSS frais → enqueue → items generation=NULL → HOL
→ claim/duplicate/ledger_ids_missing → queue bloquée
→ nouveaux fixes non_ingested → 0 LOC PG → illusion GPS stale
```

## Futurs designs (séparés, après GO)

```text
C-LEDGER-CLIENT   ✅ CLOSED / PASS — gps-c3-ledger-client-canary-2026-08-14.md
C-LEDGER-SERVER   ✅ CLOSED / PASS — gps-c3-ledger-server-canary-2026-08-14.md
OBSERVABILITY     ✅ DESIGN READY — gps-p0-c-observability-design.md
                  → PATCH = NO-GO
```

Preuve cible ledger (atteinte) : un item invalide **ne peut plus** bloquer les positions fraîches derrière lui.

**Prochain GO** : patch OBSERVABILITY (après GO explicite) — ne pas rouvrir le ledger.

Docs : `gps-p0-c-ledger.md`, `gps-p0-c-observability-design.md`, `gps-p0-c-loc-stale-after-pause.md`.
