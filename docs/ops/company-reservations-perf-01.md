# COMPANY-RESERVATIONS-PERF-01

**Statut** : ✅ **CLOSED / PASS** — chemin principal (liste `Toutes`, sans recherche, sans jour unique).

Le `OR/EXISTS` unique forçait un seq scan ~40k lignes + JIT ~350–380 ms. Le chemin propriétaire indexé + `UNION ALL` de 4 branches disjointes contourne ça, sans changer qui voit quoi.

```text
COMPANY-RESERVATIONS-PERF-01

ROOT CAUSE
OR/EXISTS visibility predicate
→ seq scan ~40k rows
→ JIT ~350–380 ms

FIX
Owner indexed fast path
+
4 disjoint visibility branches
+
UNION ALL preserving historical visibility semantics

RESULTS (Emmenez-moi, 40 546 bookings)

List server          582 ms → 86 ms
List HTTP            846 ms → 295 ms
List SQL fetch       526 ms → 37 ms
Stats                768 ms → 66 ms

EXPLAIN
OR/EXISTS + LIMIT 25     Seq scan 40 k + JIT 384 ms → 398 ms
Owner company_id = 1     Index scan 301, top-N      → 0,7 ms
UNION ALL 4 branches     Index owner + vides        → 5 ms
COUNT stats en OR        Seq scan + JIT 351 ms      → 360 ms

Indexes              EXISTING / REUSED
  ix_booking_company_scheduled
  ix_booking_executing_company_id
  ix_booking_transfers_owner_company_status
  ix_dispatch_offer_company_id
Migration            NONE

Visibility parity    TESTED
Top-k                 TESTED
Routes                TESTED
Union disjointness    TESTED
Open-offer NULL exec  TESTED (NOT SQL three-valued)

AUTH
/auth/me duplicate   mitigated by single-flight
Single cache          PASS

PREFETCH
inflight skip         PASS
fresh-cache skip      PASS
same staleTime        PASS

STATUS = CLOSED / PASS
```

## Fichiers

- `backend/services/companies/booking_visibility.py` — 4 prédicats disjoints = union de l’OR historique
- `backend/routes/companies.py` — liste / stats sans recherche ni jour : `id IN (UNION ALL)` + top-k
- `frontend/src/utils/authMeSingleFlight.js`
- `frontend/src/utils/companyReservationsPrefetch.js`

## Garde-fou régression

Les quatre branches `UNION ALL` doivent rester **disjointes**. Un chevauchement futur produirait des doublons silencieux.

```text
REGRESSION GUARD

legacy_visibility_ids
==
union_visibility_ids

AND

COUNT(union rows)
==
COUNT(DISTINCT union booking_id)
```

✅ **Implémenté** : `backend/tests/services/test_booking_visibility.py` (équivalence OR, COUNT = DISTINCT, branches pairwise disjointes, y compris course propriétaire + transfert). Complément dans `backend/tests/routes/test_companies.py`.

Le garde-fou a aussi révélé un écart réel : `NOT (executing_company_id = company)` est NULL si l’exécutant est vide, donc les offres ouvertes disparaissaient du `UNION`. Corrigé avec `IS DISTINCT FROM` (`_not_executor`).

## Tickets hors T1 (ouverts)

```text
RESERVATIONS-DATE-FILTER-PERF

Current:
date filter keeps historical OR/EXISTS path

Reason:
linked-return visibility semantics are different

Action:
profile separately with EXPLAIN ANALYZE
before attempting UNION rewrite

STATUS = BACKLOG / MEASURE FIRST
```

```text
RESERVATIONS-COLD-AUTH-BOOTSTRAP

Current:
hard reload without warm session may still feel slow

Reason:
/reservations HTTP is now ~295 ms; remaining cold cost is
likely auth/shell bootstrap, not the reservations list

Action:
profile cold /auth/me + shell independently

STATUS = BACKLOG / SEPARATE TICKET
```

```text
RESERVATIONS MAIN LIST PERF = PASS
VISIBILITY UNION            = PASS
AUTH SINGLE-FLIGHT          = PASS
DATE FILTER PERF            = SEPARATE TICKET
COLD AUTH BOOTSTRAP         = SEPARATE TICKET
```
