# P0-C-LEDGER — Diagnostic read-only #2 (chaîne des 3 queue_item_id)

```text
GO                         = DIAGNOSTIC READ-ONLY (exécuté)
PATCH                      = NO-GO
PARENT                     = gps-p0-c-ledger.md
OBJETS                     = trk_1786723792342_u8w2gqur
                             trk_1786723810647_11n415gl
                             trk_1786723829101_tdhsi20c
CODE                       = lecture seule (pas de patch)
```

## Freeze

```text
P0-A / P0-B / C3     CLOSED / PASS
P0-C-NATIVE          CONFIRMED post-18:13 (18:09–18:10 = GNSS OK, hors NATIVE)
P0-C-LEDGER          CONFIRMED
  CLIENT             session avant generation + register absent (B)
  SERVER             claim non libéré sur ledger_ids_missing
C-SEQUENCING         CONTRIBUTING / PARTIAL
C4                   EXCLUDED
PATCH                NO-GO
```

---

## Chaîne suivie

```text
queue_item_id  (= location_event_id côté HTTP)
→ claim Redis  atmr:driver_location:event:{driver}:{sha256}
→ UpdateDriverLocationUseCase / should_skip_location_ingest
→ (si nouveau) projection live + extract_sync_ledger_ids(body)
→ try_commit_sync_ledger_ack  OU  ids_missing court-circuit
→ construction ACK
→ (souvent) pas de store_idempotent_response
```

Fichiers :

- `backend/services/geolocation/driver_location_dedup.py` — claim / release
- `backend/routes/driver.py` — branche `duplicate_event_id` / `ids_missing`
- `backend/services/tracking/sync_ledger_ack.py` — `extract_sync_ledger_ids`, `try_commit_sync_ledger_ack`

---

## Les cinq questions

### 1. Où le système décide-t-il `duplicate` ?

Dans `should_skip_location_ingest` → `claim_location_event_id` (Redis `SET NX`) :

- claim **échoue** → `dedup_reason = duplicate_event_id`
- puis `driver.py` (~2042–2106) :
  - si cache idempotent `durability == persisted_sync` → ACK `duplicate` **prouvé**
  - sinon → **`duplicate_event_id_unproven`** + `release_location_event_id` + `retryable: true`

Les ACK canary observés = **`duplicate_event_id_unproven`** (pas le chemin « déjà en base »).

### 2. Quelle row / clé est considérée comme existante ?

**Pas une row PG.**  
Clé Redis :

```text
atmr:driver_location:event:{driver_id}:{sha256(location_event_id)[:32]} = "1"
TTL défaut = 600 s (DRIVER_LOCATION_EVENT_ID_TTL_SEC)
```

Lecture live (2026-08-14 ~diagnostic, read-only) :

| location_event_id | claim Redis | TTL |
|-------------------|-------------|-----|
| `trk_1786723792342_u8w2gqur` | **présent** `b'1'` | ~599 s |
| `trk_1786723810647_11n415gl` | **présent** | ~599 s |
| `trk_1786723829101_tdhsi20c` | **présent** | ~600 s |
| ancre `trk_1786720810924_cmh86hnn` | absent | -2 |

→ Le « duplicate » pointe un **marqueur Redis orphelin**, pas `driver_location_events`.

### 3. Pourquoi `ledger_ids_missing` apparaît-il ensuite ?

Après un claim **réussi** (post-release), le handler atteint `try_commit_sync_ledger_ack` seulement si `projection_ok`.  
Sinon / ou dans `try_commit_sync_ledger_ack` :

```text
si tracking_session_id manquant
   OU session_generation is None
   OU sequence_id is None
   OU location_event_id manquant
→ SyncLedgerAckResult(kind="ids_missing", reason="ledger_ids_missing")
```

Branche `driver.py` (~2467–2499) :

- HTTP **200**, `ack_status=ingested_non_persisted`
- **`release_location_event_id` NON appelé**
- `store_idempotent_response` NON appelé (réservé à `durable_ok`)
- `db_persisted: True` ici = **projection live OK**, **pas** row ledger/`driver_location_events`

Donc : IDs ledger incomplets dans le body (ou absents à l’extraction) → ACK non durable **tout en laissant le claim Redis**.

### 4. Une transaction DB a-t-elle été rollback ?

Pour le chemin **`ids_missing`** : **aucune** `persist_location_event_with_outbox` n’est tentée → **pas de TX ledger** → pas de rollback ledger.  
Cohérent avec **0 row PG** pour les 3 IDs.

(Les chemins `conflict_409` / commit KO font bien `rollback` + `release` — **non** ceux observés dans les ACK canary.)

### 5. L’ACK `duplicate` est-il produit avant vérification de la persistence ?

**Oui.** Ordre effectif :

```text
1) claim Redis déjà posé ?
2) si oui → chercher cache idempotent persisted_sync
3) si cache absent → ACK duplicate_event_id_unproven (+ release)
   → JAMAIS de re-vérification PG métier à ce stade
```

La persistence métier n’est consultée **que** via le cache idempotent `persisted_sync` (écrit uniquement après `durable_ok`).  
Sans ce cache → « duplicate » **unproven** même si PG est vide.

---

## Cycle reproductible (explique l’alternance)

```text
A. claim NX OK
   → ids_missing (pas de release)
   → ACK ingested_non_persisted / ledger_ids_missing
   → claim RESTE

B. retry
   → claim NX KO
   → pas de proven persisted_sync
   → release + ACK duplicate_event_id_unproven

C. retry
   → retour en A
```

Correspond aux ~50/50 `duplicate` ↔ `ingested_non_persisted` **par** `queue_item_id`, et aux claims Redis encore frais (TTL ~600) **sans** row PG.

Cas le plus inquiétant (formulé par le ticket) :

```text
dedupe marker créé
→ persistence n’écrit pas la row métier
→ marker reste (ids_missing)
→ retry = duplicate unproven
→ release + reclaim
→ re-ids_missing
→ item impossible à faire progresser
```

**Confirmé au niveau code + observation Redis/PG/ACK.**  
Cause racine du *pourquoi* les IDs ledger manquent dans le body des 3 items = encore à préciser (payload flush / `session_generation` / envelope) — piste **contributive** C-SEQUENCING, pas encore bug autonome.

---

## Ce que C-LEDGER n’est pas

| Non-cause | Preuve |
|-----------|--------|
| Replay identité ancre 17:20:10 | IDs ≠ ancre (diag #1) |
| « duplicate = déjà en PG » | 0 row + `unproven` + pas de `persisted_sync` |
| Bug FGS / A | FGS up ; hors scope |
| auth / B | auth OK |

---

## Lacunes acceptées (read-only suivant, sans patch)

1. Capturer **un** body HTTP exact des 3 items (`tracking_session_id`, `session_generation`, `sequence_id`).
2. Confirmer si `ids_missing` vient du body queue ou d’un strip à l’ingress.
3. Mesurer si le client retire l’item de la queue sur `ingested_non_persisted` (HOL).

---

## Implémentation

✅ **Implémenté** : réponses aux 5 questions ; cycle claim↔ids_missing↔unproven documenté ; preuve Redis live claims présents / PG 0.  
**Reste à faire** : lacunes payload body (optionnel) ; **PATCH NO-GO**.
