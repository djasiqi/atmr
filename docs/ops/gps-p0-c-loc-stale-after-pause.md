# P0-C — Clôture diagnostic causal

```text
DIAGNOSTIC CAUSAL          = CLOSED
P0-C GLOBAL                = CLOSED / PASS ✅
DATE                       = 2026-08-15 (freeze global Genève)
FREEZE                     = gps-p0-global-freeze-2026-08-15.md
```

## Freeze officiel

```text
P0-A                    CLOSED / PASS ✅
P0-B                    CLOSED / PASS ✅
C3 GLOBAL               CLOSED / PASS ✅

C-LEDGER-CLIENT         CLOSED / PASS ✅
C-LEDGER-SERVER         CLOSED / PASS ✅
OBSERVABILITY           CLOSED / PASS ✅

P0-C causal             CLOSED ✅
P0-C-NATIVE             CLOSED / REQUALIFIED
P0-C GLOBAL             CLOSED / PASS ✅

OBSERVABILITY DESIGN         = READY ✅ — gps-p0-c-observability-design.md
OBSERVABILITY IMPLEMENTATION = IMPLÉMENTÉ ✅
OBSERVABILITY CANARY         = PASS ✅ — gps-p0-c-observability-canary-2026-08-15.md
TRACKING FUNCTIONAL PATCH    = NO-GO
LEDGER PATCH                 = CLOSED
```

### Règle de freeze

> **Ne plus modifier P0-A, P0-B, C-LEDGER ou l’observabilité sans nouvelle preuve de régression.**

```text
NOUVEL INCIDENT GPS
→ nouvelle branche RCA
→ ne pas rouvrir automatiquement A / B / P0-C
```

Phase suivante = **release / deployment control** (GO explicite) — voir [gps-p0-global-freeze-2026-08-15.md](gps-p0-global-freeze-2026-08-15.md).

Preuves Redis claims / queue SQLite / watermark 403 **conservées** (ne pas purger).

Nuances canary SERVER (hors RCA prod) :

- Migration `capture_id` = prérequis environnement canary
- `401 Token revoked` = harness-only (`token_version`)

---

## Chaîne causale (figée)

```text
GNSS continue de produire des fixes frais
↓
queue locale continue d’enqueue
↓
certains items historiques generation=NULL
↓
HOL ledger
↓
claim Redis / duplicate_unproven / ledger_ids_missing
↓
progression queue bloquée
↓
nouveaux fixes restent non_ingested
↓
PostgreSQL ne reçoit plus de nouveaux LOC
↓
dashboard / health donnent l’impression d’un GPS stale
```

**Le problème principal n’est plus GPS**, mais **ledger / queue / persistence avec HOL blocking** — **fermé** (CLIENT + SERVER canaries PASS).

---

## P0-C-NATIVE — requalification

Faux diagnostic causal. Post-18:13 : fixes frais encore enqueued (N4).  
Anciens `nfix` / `last_fix_age` health ≈ `lastTaskInvokedAt` / `lastWatchAtMs` — **pas** l’âge du vrai `Location.timestamp`.  
→ lacune d’**observabilité**, pas cause d’incident.

Docs : [gps-p0-c-native.md](gps-p0-c-native.md) · [diag N1–N4](gps-p0-c-native-diagnostic-2026-08-14.md)

---

## P0-C-LEDGER — CLOSED

| Branche | Statut |
|---------|--------|
| **CLIENT** | CLOSED / PASS — [canary](gps-c3-ledger-client-canary-2026-08-14.md) |
| **SERVER** | CLOSED / PASS — [canary](gps-c3-ledger-server-canary-2026-08-14.md) |

Docs : [gps-p0-c-ledger-rca.md](gps-p0-c-ledger-rca.md) · [ledger](gps-p0-c-ledger.md)

---

## Branches

```text
C-LEDGER-CLIENT
BRANCHE          = CLOSED / PASS ✅

C-LEDGER-SERVER
BRANCHE          = CLOSED / PASS ✅
CANARY           = PASS ✅ — gps-c3-ledger-server-canary-2026-08-14.md

OBSERVABILITY
DESIGN           = READY ✅ — gps-p0-c-observability-design.md
IMPLEMENTATION   = IMPLÉMENTÉ ✅ (mesure / classification — pas de patch tracking)
CANARY           = PASS ✅ — gps-p0-c-observability-canary-2026-08-15.md
BRANCHE          = CLOSED / PASS ✅
```

## Sujets restants

```text
1. C-LEDGER-CLIENT   ✅ CLOSED / PASS
2. C-LEDGER-SERVER   ✅ CLOSED / PASS
3. OBSERVABILITY     ✅ CLOSED / PASS
   → location_fix_age_seconds = now - Location.timestamp
   → task_invoke_age_seconds ≠ fraîcheur GNSS (compat native_last_fix_age)
   → historique P0-C classé PIPELINE (jamais GNSS / fix_stale)
```

Invariant ledger (fermé) :

> Un item invalide ou non enregistré ne doit jamais pouvoir bloquer les positions fraîches suivantes.

Invariant observabilité (fermé) :

> Aucune alerte « GPS stale » ne doit conclure à une panne GNSS si elle ne repose pas sur le timestamp du dernier vrai `Location`.

**Prochain GO** : freeze commits + snapshot prod lecture seule pour passer **G0–G5 VERT** — voir [gps-p0-release-readiness-2026-08-15.md](gps-p0-release-readiness-2026-08-15.md). **PROD DEPLOY = NO-GO**.

---

## Implémentation

✅ **Implémenté** : diagnostic causal P0-C terminé ; NATIVE requalifié/fermé ; **CLIENT + SERVER CLOSED/PASS** ; **design + impl + canary OBSERVABILITY PASS** ; **P0-C GLOBAL CLOSED / PASS** ; freeze global [gps-p0-global-freeze-2026-08-15.md](gps-p0-global-freeze-2026-08-15.md).  
**Reste à faire** : rien diagnostic P0 ; GO release/deployment control si déploiement.
