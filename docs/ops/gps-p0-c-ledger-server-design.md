# P0-C — Design : C-LEDGER-SERVER (claim lifecycle + sémantique duplicate)

```text
TICKET                     = P0-C-LEDGER-SERVER
PHASE                      = CLOSED
STATUT                     = CLOSED / PASS ✅ — canary S1–S6
RCA                        = CONFIRMED (gps-p0-c-ledger-rca.md)
PARENT                     = gps-p0-c-loc-stale-after-pause.md
CLIENT                     = CLOSED / PASS ✅ (gps-c3-ledger-client-canary-2026-08-14.md)
CANARY SERVER              = PASS ✅ (gps-c3-ledger-server-canary-2026-08-14.md)
INDÉPENDANCE               = PR séparée de CLIENT et OBSERVABILITY
OBSERVABILITY              = DESIGN READY — gps-p0-c-observability-design.md (PATCH NO-GO)
```

Documents liés :

- [gps-p0-c-ledger-rca.md](gps-p0-c-ledger-rca.md)
- [gps-p0-c-ledger.md](gps-p0-c-ledger.md)
- [gps-p0-c-ledger-bodies-2026-08-14.md](gps-p0-c-ledger-bodies-2026-08-14.md)
- [gps-c3-ledger-client-canary-2026-08-14.md](gps-c3-ledger-client-canary-2026-08-14.md)
- Code : `backend/services/geolocation/driver_location_dedup.py`, `backend/routes/driver.py` (~2042–2106, ~2467–2540), `backend/services/tracking/sync_ledger_ack.py`

---

## Objectif

Corriger deux défauts serveur **indépendants** déjà prouvés, sans casser l’idempotence des événements **réellement persistés**.

### S1 — Claim lifecycle

```text
claim Redis acquis (SET NX)
→ validation / ledger échoue AVANT persistence prouvée
→ le claim DOIT être release
```

Invariant :

> Aucun chemin d’erreur **pré-persistence** ne peut abandonner un claim `SET NX`.

### S2 — Duplicate semantics

```text
SET NX échoue
≠
preuve que l’événement est persisté
```

`duplicate` doit être distingué de :

| Terme | Signification |
|-------|----------------|
| `duplicate_persisted` | Preuve ledger/PG (ou cache idempotent `persisted_sync`) |
| `duplicate_unproven` | Claim présent / collision, **aucune** preuve de persistence |
| `claim_in_flight` | Claim récent, autre requête probablement en cours |
| `stale_claim` | Claim orphelin (TTL restant mais pas de preuve / owner mort) |

Invariant :

> `duplicate` ne peut **jamais** signifier `persisted` sans preuve ledger/PG (ou cache durable équivalent).

### Non-objectifs (cette PR SERVER)

- Gate CLIENT / readiness session (déjà CLOSED)
- Health `last_location_fix_at` / fix_age → **OBSERVABILITY**
- Purge manuelle Redis / queue de production
- Changer le TTL claim sauf besoin explicite documenté

### Anti-patterns refusés

```text
# A — claim orphelin (bug actuel sur ids_missing)
claim OK → ledger_ids_missing → return 200 ingested_non_persisted
→ SANS release_location_event_id

# B — « fix » naïf qui casse l’idempotence
SET NX fail → DEL claim toujours
→ double write possible sur retry d’un event déjà persisté

# C — ACK final sur unproven
ack_status=duplicate + ok=true traité comme « déjà traité »
→ client stoppe le retry alors que PG n’a pas la row
```

---

## Diagnostic figé (rappel)

```text
event incomplet (ex. generation=null)
→ claim Redis acquis
→ sync_ledger_ack → ids_missing / ledger_ids_missing
→ driver.py : 200 ingested_non_persisted SANS release   ← trou S1
→ retry
→ SET NX fail → duplicate_event_id
→ pas de cache persisted_sync
→ release + accept_reason=duplicate_event_id_unproven
→ mais ack_status reste "duplicate"                       ← trou S2 (sémantique client)
→ reclaim possible → boucle / HOL côté file
```

Chemins qui **release déjà** (à conserver) :

- `conflict_409` après claim
- `ledger_failed_503` / unproven post-persist attempt
- `db_persist_failed` (503)
- branche `duplicate_event_id` **sans** preuve `persisted_sync` (release + unproven)

Chemin **manquante** (à corriger) :

- `ledger_ack.kind == "ids_missing"` (~2467–2499 `driver.py`) — **pas de release**

---

## Invariants serveur (figés avant implémentation)

```text
1. claim acquis + validation échoue
   → release garanti

2. claim acquis + persistence échoue
   → release ou état retryable explicite (jamais poison permanent)

3. "duplicate" ≠ "persisted"
   sans preuve ledger/PG (ou cache idempotent persisted_sync)

4. duplicate_unproven
   → ne doit pas ACKer comme succès final « déjà traité »

5. retry d’un événement valide
   → doit pouvoir progresser après disparition d’un claim orphelin

6. un événement invalide
   → ne peut pas créer de HOL durable côté serveur (claim + ACK)

7. aucune suppression d’une vraie protection
   d’idempotence pour les événements déjà persistés
```

Règle de robustesse :

> Le correctif SERVER doit être testé avec le CLIENT corrigé, **et** rester correct face à un **vieux client** qui envoie encore `generation=null`. Le serveur ne dépend jamais du fait que tous les clients sont à jour.

---

## State machine (claim / verify)

```text
UNCLAIMED
   ↓ SET NX OK
CLAIMED
   ├─ invalid ids / validation KO ──→ RELEASED + REJECTED (ou 4xx/200 non-final explicite)
   ├─ persistence fail ─────────────→ RELEASED + RETRYABLE
   └─ persistence OK ───────────────→ PERSISTED (+ cache idempotent persisted_sync)

SET NX fail
   ↓
VERIFY
   ├─ preuve PG/ledger ou cache persisted_sync → DUPLICATE_PERSISTED (ACK final OK)
   ├─ claim récent (in-flight) ───────────────→ CLAIM_IN_FLIGHT / RETRY (pas final)
   └─ aucune preuve ──────────────────────────→ DUPLICATE_UNPROVEN
                                                 ≠ success final
                                                 → release orphelin si stale, sinon retry
```

### Matrice décisions (cible)

| Situation | Claim | Preuve PG/ledger | Réponse cible | Release ? |
|-----------|-------|------------------|---------------|-----------|
| Premier passage, IDs incomplets | acquis | non | `ids_missing` / non-final | **OUI** |
| Premier passage, persist OK | acquis | oui | `persisted_sync` | NON (claim = preuve court-terme) |
| Premier passage, persist KO | acquis | non | 503 retryable | **OUI** |
| Retry, row déjà en PG | SET NX fail | oui | `duplicate_persisted` | NON |
| Retry, claim orphelin, pas de row | SET NX fail | non | `duplicate_unproven` retryable | **OUI** (stale) |
| Retry concurrent in-flight | SET NX fail | non | `claim_in_flight` retryable | NON (sauf TTL/stale policy) |

### Cas à ne pas casser

```text
request A persistée correctement
→ retry identique
→ déduplication efficace
→ pas de deuxième row métier
```

Donc : **ne jamais** `DEL claim` systématique sur tout `SET NX` fail. Uniquement après **VERIFY** négatif (unproven / stale), ou sur chemins d’erreur pré-persistence du **holder** du claim.

---

## Contrats ACK / accept_reason (HTTP)

Séparer nettement le vocabulaire exposé au client :

| `accept_reason` (cible) | `ack_status` (cible) | Final ? | Client doit |
|-------------------------|----------------------|---------|-------------|
| `duplicate_persisted` | `duplicate` + `durability=persisted_sync` | oui | retirer / tombstone local |
| `duplicate_event_id_unproven` | **pas** succès final (ex. `retry` / `ingested_non_persisted` + `retryable=true`) | non | retry |
| `claim_in_flight` | retryable | non | backoff court |
| `ledger_ids_missing` | non-final + `retryable` selon politique | non* | ne pas traiter comme persisted ; invalide → drop/quarantine côté CLIENT déjà |

\* Pour IDs structurellement invalides (`generation=null`), le SERVER release le claim et peut renvoyer un rejet **non retryable** (`invalid_ledger_ids`) afin d’éviter une boucle infinie même avec un vieux client — sans jamais laisser un claim orphelin.

Recommandation design (choix d’implémentation à trancher en PR) :

```text
Option A (minimale) :
  ids_missing → release + même corps qu’aujourd’hui mais retryable=true explicite
  + corriger ack duplicate_unproven pour ne plus ressembler à un succès final

Option B (préférée) :
  ids_missing structurel → 422/409 invalid_ledger_ids, retryable=false, release
  ids_missing transitoire (si jamais) → 503 + release
  duplicate_unproven → ack_status dédié / retryable=true, pas "duplicate" final
```

**Option B préférée** pour T7 (anciens poison) : un event invalide ne boucle plus indéfiniment.

---

## Points d’ancrage code (impl future)

1. **`driver.py` branche `ids_missing`**  
   Ajouter `release_location_event_id` (symétrique conflict/503).  
   Ajuster `accept_reason` / `retryable` / status HTTP selon Option B.

2. **`should_skip_location_ingest` / branche `duplicate_event_id`**  
   Renommer / clarifier le VERIFY :
   - preuve `persisted_sync` → `duplicate_persisted`
   - sinon → `duplicate_unproven` **sans** ACK final ; release stale claim  
   Optionnel : distinguer `claim_in_flight` via TTL restant / age du claim (GET TTL Redis) si disponible sans sur-ingénierie.

3. **`sync_ledger_ack.py`**  
   Conserver `ids_missing` ; le **caller** reste responsable du release (ownership claim au niveau route). Documenter l’invariant dans le docstring.

4. **Idempotence cache**  
   Ne stocker `persisted_sync` **que** si ledger/PG prouvé. Ne jamais promouvoir unproven en cache durable.

5. **Tests** (T1–T7 ci-dessous) avant canary.

---

## Critères de validation (design → tests)

```text
T1 ledger_ids_missing
   → claim absent après traitement

T2 persistence exception
   → claim ne devient pas poison permanent

T3 duplicate + row persistée
   → duplicate_persisted / pas de double write

T4 duplicate + aucune row
   → jamais ACK final « déjà traité »

T5 retries concurrents
   → une seule persistence

T6 stale/orphan claim
   → récupération déterministe (release + retry OK)

T7 anciens poison events (generation=null)
   → ne peuvent plus boucler indéfiniment (reject non-retryable + release)
```

Canary SERVER (après GO impl) : CLIENT déjà corrigé **et** un harness « vieux client » (payload `session_generation=null`) pour T1/T7.

---

## Critères PASS design → implémentation

```text
PASS SERVER si :
- tout chemin pré-persistence après SET NX appelle release (ou documente exception PERSISTED)
- duplicate_unproven ≠ succès final
- duplicate_persisted conserve l’idempotence (pas de 2e row)
- T1–T7 couverts par tests
- vieux client generation=null ne laisse pas de claim orphelin

FAIL si :
- ids_missing sans release
- DEL claim sur tout SET NX fail
- ack_status=duplicate final sans preuve PG
```

---

## Indépendance / ordre

| Sujet | Relation |
|-------|----------|
| C-LEDGER-CLIENT | **CLOSED / PASS** — ne pas rouvrir ; SERVER reste robuste sans lui |
| OBSERVABILITY | Après ; ne pas coupler |
| P0-A / P0-B | Ne pas rouvrir |

---

## Décisions

```text
DESIGN SERVER              = READY (ce document)
PATCH SERVER               = ✅ IMPLÉMENTÉ (Option B)
TESTS T1–T7                = PASS (test_ledger_server_claim_lifecycle_p0c.py)
CANARY SERVER              = PASS ✅ (gps-c3-ledger-server-canary-2026-08-14.md)
BRANCHE SERVER             = CLOSED / PASS ✅
C-LEDGER-CLIENT            = CLOSED / PASS ✅
OBSERVABILITY              = DESIGN READY — gps-p0-c-observability-design.md (PATCH NO-GO)
PROCHAINE ÉTAPE            = ne plus toucher le ledger ; OBSERVABILITY si GO patch
```

---

## Implémentation

✅ **Implémenté** : design C-LEDGER-SERVER (S1 claim lifecycle, S2 duplicate semantics, state machine, invariants 1–7, critères T1–T7, ancrage code, robustesse vieux clients).

✅ **Implémenté** : patch runtime Option B —
- `driver.py` : `ids_missing` → `422 invalid_ledger_ids` + `retryable=false` + `release` ; VERIFY `duplicate_persisted` / `claim_in_flight` / `duplicate_event_id_unproven` (`ack_status` non final pour unproven/in_flight) ; release sur exception pré-persistence.
- `driver_location_dedup.py` : `release` instrumenté (`lifecycle=acquired|released`), TTL claim, classify, release post-`duplicate_proximity`.
- `sync_ledger_ack.py` : docstring ownership claim (caller = release).
- Tests : `backend/tests/services/test_ledger_server_claim_lifecycle_p0c.py` (T1–T7 + assert Redis final).
- Canary : `gps-c3-ledger-server-canary-2026-08-14.md` — S1–S6 PASS.

**Reste à faire** : rien côté SERVER (CLOSED). OBSERVABILITY = design séparé.
