# P0-D Design — D4-SERVER-A/B/C/D (idempotence HTTP retries)

```text
P0-D DIAGNOSTIC          = CLOSED ✅
D4-B CAUSAL              = CONFIRMED ✅
HASH PROVENANCE          = CLOSED ✅

P0-D DESIGN              = READY ✅
P0-D IMPLEMENTATION DETAIL = FIGÉ ✅
P0-D PATCH SERVER        = DONE ✅
P0-D CANARY              = NO-GO
GENERAL DISTRIBUTION     = NO-GO
```

Date : 2026-08-16  
Contexte causal : `D4_EVENT_ID_PAYLOAD_COMPARE.md`, `D4_HASH_PROVENANCE.md`

---

## Problème à résoudre (deux contraintes simultanées)

```text
1. Nouveaux retries idempotents
   → même événement métier = même identité canonique

2. Événements déjà persistés reconnus
   → ne pas invalider les hashes legacy (_payload_hash prod)
```

Le bug n’est **pas seulement** `recorded_at = now` : un changement brutal d’algo/hash sous le même `location_event_id` créerait des **faux** `event_id_payload_conflict`.

---

## D4-SERVER-A — timestamp métier stable

Pour `PUT /driver/me/location` :

```text
recorded_at = timestamp du Location fourni par le client
              (champ client `timestamp` / équivalent fix)
≠ heure d’arrivée HTTP
≠ sent_at
≠ now() serveur
```

Effet attendu :

```text
event X retry #1 @ 13:57
event X retry #2 @ 13:58
event X retry #3 @ 13:59

recorded_at métier = identique
```

Notes d’implémentation (hors scope patch) :

- Mapper explicitement `timestamp` → `recorded_at` à l’ingress si `recorded_at` absent.
- Ne pas écraser un `recorded_at` client déjà présent.
- `sent_at` = métadonnée transport (`now` OK).

---

## D4-SERVER-B — identité canonique métier

Le duplicate check **ne doit pas** dépendre uniquement d’un hash opaque dont l’algorithme peut évoluer.

Identité stable proposée :

```text
driver_id
location_event_id

tracking_session_id
session_generation
sequence_id

recorded_at          # timestamp du fix Location
latitude
longitude
accuracy             # accuracy_m
speed                # speed_mps
heading
```

Hors identité (transport / runtime) :

```text
HTTP arrival time / received_at_ms
sent_at
retry timestamp
consumer timestamp
outbox processing time
trace_id
is_background          # sauf décision produit contraire documentée
```

### Règle `capture_id` (obligatoire)

> **`capture_id` ne doit pas entrer silencieusement dans l’identité idempotente d’un événement historique si les anciennes versions du serveur ne l’utilisaient pas.**

Sinon le déploiement d’une version tip (hash outbox + `capture_id`) recrée exactement le faux conflit que ce design élimine.

Pour les **nouveaux** événements uniquement : si on décide d’inclure `capture_id` plus tard, cela exige un **nouveau** `payload_schema_version` (ou flag algo) + chemin D4-SERVER-D — pas un ajout silencieux sous `tracking-event-payload-v1`.

---

## D4-SERVER-C — sémantique duplicate

```text
event_id inconnu
→ persistence
→ store identité canonique + hash (algo documenté)

event_id connu
+ identité métier identique
→ duplicate_persisted
→ succès idempotent
→ 0 nouvelle row LOC/ingest
→ 0 DLQ

event_id connu
+ identité métier réellement différente
→ event_id_payload_conflict
→ DLQ / alerte
```

Le conflict **reste utile** : on ne le supprime pas ; on le réserve aux vrais collisions d’identité métier.

---

## D4-SERVER-D — compatibilité legacy

Constat provenance : deux algos sous le label `tracking-event-payload-v1` ; prod Kafka outbox = `_payload_hash` (JSON dict **sans** `capture_id`).

Lorsqu’un `event_id` existe déjà :

```text
nouveau hash == hash stocké
→ duplicate confirmé directement

sinon
→ NE PAS déclarer immédiatement conflict
→ charger / comparer l’identité métier persistée (D4-SERVER-B)

identité métier équivalente
→ legacy duplicate compatible
→ duplicate_persisted

identité différente
→ vrai event_id_payload_conflict
```

Invariant :

```text
HASH mismatch  ≠  automatiquement  PAYLOAD métier mismatch
```

Cela protège les rows déjà écrites avec l’ancien `_payload_hash`.

---

## Ordre de conception recommandé (avant code)

1. Spécifier le mapping ingress A (`timestamp` → `recorded_at`).
2. Spécifier la fonction `business_identity_equal(stored_row, incoming)` (tolérances float / canon temps).
3. Spécifier la séquence de décision C+D dans `persist_location_event_with_outbox` (et tout autre chemin qui DLQ sur conflict).
4. Décider si le hash **stocké pour les nouveaux events** reste `_payload_hash` ou bascule F-02 scaled **avec nouveau schema_version**.
5. Verrouiller la règle `capture_id` (hors identité legacy).
6. Implémenter tests D4-T1..T8 → seulement ensuite GO patch.

---

## Matrice de tests (bloquante avant patch)

| Id | Scénario | Attendu |
|----|----------|---------|
| **D4-T1** | même `event_id`, même fix, 6 retries espacés | 1 persistence ; 5 `duplicate_persisted` ; 0 DLQ |
| **D4-T2** | même `event_id` ; `recorded_at` HTTP différent ; même timestamp Location | duplicate |
| **D4-T3** | même `event_id` ; hash legacy différent ; identité métier identique | `duplicate_persisted` |
| **D4-T4** | même `event_id` ; lat/lng réellement différents | `event_id_payload_conflict` |
| **D4-T5** | ancien event PG (`_payload_hash` prod) ; retry sous nouveau code | pas de faux conflict |
| **D4-T6** | `capture_id` absent sur legacy | compatibilité (pas de faux conflict) |
| **D4-T7** | nouvel événement | hash/identity déterministes |
| **D4-T8** | retry après HOME/BG (repro incident) | PUT 202 → Kafka → consumer → 0 DLQ → LOC PG continue |

---

## Hors scope de ce design

- Patch runtime / deploy
- Whitelist batterie / `deviceidle`
- Correction FGS `DENIED` (phénomène secondaire documenté)
- Unification forcée F-02 scaled sur tout le pipeline sans D4-SERVER-D

---

## Prochain GO utile

```text
GO IMPLEMENTATION DETAIL = FAIT → D4_IMPLEMENTATION_DETAIL.md
GO PATCH                 = encore NO-GO
  (signatures / normalisations / insertion figés ;
   attendre GO explicite + T1–T8)
```
