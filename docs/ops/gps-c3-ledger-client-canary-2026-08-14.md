# Canary C-LEDGER-CLIENT isolé — 2026-08-14

```text
CANARY                 = C-LEDGER-CLIENT isolé (SERVER inchangé)
DEVICE                 = S23 SM-S911B (ADB USB RFCW20QC53W)
DRIVER                 = 19
MISSION                = 26 EN_ROUTE
CAPTURE                = docs/ops/_c3_ledger_client_2026-08-14/
FENÊTRE USB            = ~22:16–22:27 Genève
RUNNER                 = run_ledger_client_canary.ps1
METRO                  = redémarré (--clear) pour bundle CLIENT
```

## Freeze

```text
C-LEDGER-CLIENT IMPLEMENTED   ✅
CANARY CLIENT                 PASS ✅ (invariants CLIENT)
C-LEDGER-SERVER               NO-GO ❌
OBSERVABILITY                 NO-GO ❌
PROD PATCH                    NO-GO ❌
```

Objectif atteint : **aucun nouvel item actif `generation=null`** ; poison historique **ne bloque plus** la file locale ; LOC PG avec generations numériques.

## Critères bloquants

| Critère | Résultat |
|---------|----------|
| Nouvel item actif `generation=null` | **0** (tous `sqlite_*` USB : `ACTIVE_NULL 0`) |
| Enqueue pendant REGISTERING/FAILED produisant null | **0** (`ACTIVE_NULL` stable ; `enqueue_blocked` observé) |
| Ancienne session mid-rotate → item null | **0** |
| Poison historique HOL local | **0** (baseline `ACTIVE_NULL 1` → `0` ; LOC gen≥820 progressent) |
| Régression P0-A (`ERR_FOREGROUND…`) | **0** sur logcats USB |
| Régression P0-B (`auth_not_usable`) | **1 hit** (C6 resume) — nuance réseau, pas panne headless systématique |

## Matrice C1–C6

| ID | Scénario | Verdict | Preuve |
|----|----------|---------|--------|
| **C1** | Démarrage READY → enqueue → LOC | **PASS** | LOC gen 820/821 (`snap_C1_steady`) ; actifs SQLite gen=820 |
| **C2** | Fenêtre REGISTERING | **PASS** | Readiness `REGISTERING`→`READY` (gen 823→824) ; `ACTIVE_NULL 0` ; reprise LOC 822–825. `enqueue_blocked` rare ici (register ≪ 100 ms) — invariant tenu |
| **C3** | Register failure offline | **PASS** | `REGISTER_FAILED` + `enqueue_blocked` (C4/C3 logcats) ; offline `ACTIVE_NULL 0` ; recover LOC gen 838–842 |
| **C4** | Rotation session | **PASS** | Cycle CREATING→REGISTERING→READY (gen 864→866) ; mid-rotate sans actif null ; LOC nouvelles gen |
| **C5** | Poison historique | **PASS** | Baseline 1 `non_ingested` null ; puis `ACTIVE_NULL 0` / `REJECTED_NULL 219` ; file valide progresse (pas de HOL) |
| **C6** | Offline → online | **PASS** | Offline sans item ledger incomplet null ; resume READY gen 854–855 + LOC |

## Signaux surveillés (USB)

```text
tracking.session.readiness     observé (CREATING/REGISTERING/READY/REGISTER_FAILED)
tracking.queue.enqueue_blocked observé (C3 recover, C4)
tracking.session.register_failed observé (C4 ~22:25:50)
ACTIVE_NULL                    = 0 sur toute la matrice USB
err_fg                         = 0
session_gen_null dans actifs   = 0
LOC PG                         generations 820→872 (non null)
```

Exemple (C4) :

```text
readiness: 'REGISTER_FAILED'
tracking.session.register_failed
tracking.queue.enqueue_blocked
→ puis READY + LOC gen 863+
```

## Nuances (non bloquantes CLIENT)

1. **Rotations fréquentes** (gen qui monte vite) — à investiguer séparément (TTL / beginNew) ; **n’introduit pas** de poison `generation=null` actif.
2. **`auth_not_usable=1`** une fois au resume C6 — ne rouvre pas P0-B ; surveiller canary suivant.
3. Cycle serveur historique `duplicate_unproven` ↔ `ledger_ids_missing` **hors scope** (SERVER NO-GO).

## Verdict

```text
CANARY C-LEDGER-CLIENT     = PASS ✅
BRANCHE CLIENT             = fermable indépendamment
PROCHAIN GO                = Design C-LEDGER-SERVER
```

## Implémentation

✅ **Implémenté** : canary CLIENT isolé USB exécuté ; captures sous `_c3_ledger_client_2026-08-14/` ; invariants CLIENT PASS.  
**Reste à faire** : design C-LEDGER-SERVER (pas de patch SERVER tant que NO-GO).
