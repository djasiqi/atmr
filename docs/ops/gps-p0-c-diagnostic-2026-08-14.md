# P0-C — Diagnostic read-only (corrélation 17:20:10 → 18:12–18:20)

```text
GO                         = DIAGNOSTIC READ-ONLY (exécuté)
PATCH P0-C                 = NO-GO (inchangé)
P0-A / P0-B / C3           = CLOSED / PASS (inchangé)
FENÊTRE                    = ancre 17:20:10 → reprise 18:12 → obs. ~18:20
CAPTURE                    = docs/ops/_c3_ab_2026-08-14/
ARTEFACTS                  = p0c_acks_resume.csv, p0c_seq_context_resume.csv
```

## Freeze (inchangé)

```text
P0-A / P0-B / CANARY A+B / FULL C3 / C3 GLOBAL = CLOSED / PASS ✅
P0-C                                           = OPEN / DIAGNOSTIC
PATCH P0-C                                     = NO-GO
```

---

## 1. Ancre — dernier vrai LOC persisté

| Champ | Valeur |
|-------|--------|
| `created_at` / HTTP persist | **2026-08-14 17:20:10.670+02** |
| `recorded_at` (fix) | **2026-08-14 17:20:10.642+02** |
| `location_event_id` | `trk_1786720810924_cmh86hnn` |
| `tracking_session_id` | `trk_sess_1786720806086_c879gquj` |
| `session_generation` | 652 |
| `sequence_id` | 1 |
| lat / lng / acc | 46.1901524 / 6.1445484 / 3.0 m |
| mission / source | 26 / `http` |
| `LOC_AFTER_ANCHOR` (PG) | **0** |

Décodage ID : `1786720810924` → **17:20:10.924** local (cohérent).

---

## 2. Première vérification (prioritaire)

Les ACK reprise `duplicate` + `ingested_non_persisted` concernent-ils **le même `session_id + sequence` que 17:20:10** ?

### Réponse : **NON**

| | Ancre 17:20:10 | ACK reprise (tête de queue) |
|--|----------------|----------------------------|
| event / queue_item | `trk_1786720810924_cmh86hnn` | `trk_1786723792342_u8w2gqur` |
| | | `trk_1786723810647_11n415gl` |
| | | `trk_1786723829101_tdhsi20c` |
| session | `trk_sess_1786720806086_c879gquj` | **absente** des lignes ACK ; ≠ ancre |
| présence PG des 3 IDs | — | **n=0** chacun |
| présence ancre dans logcat reprise | — | **0 hit** |

Décodage des 3 `queue_item_id` (ms embarqué) :

| ID | Création locale (depuis ID) |
|----|------------------------------|
| `…3792342…` | **18:09:52** |
| `…3810647…` | **18:10:10** |
| `…3829101…` | **18:10:29** |

→ Ce ne sont **pas** des rejeux d’identité du point 17:20:10.  
→ **C4 (replay exact ancre session/seq/event) = EXCLUDED.**

---

## 3. Portrait des ACK reprise

```text
ACK_TOTAL (resume_precheck + resume_T12) = 151
  duplicate                 = 78   reason=duplicate_event_id_unproven
  ingested_non_persisted    = 73   reason=ledger_ids_missing
UNIQUE_QUEUE_ITEMS          = 3   (toujours les mêmes)
```

| queue_item_id | n | duplicate | non_persisted |
|---------------|---|-----------|---------------|
| `trk_1786723792342_u8w2gqur` | 51 | 26 | 25 |
| `trk_1786723810647_11n415gl` | 50 | 26 | 24 |
| `trk_1786723829101_tdhsi20c` | 50 | 26 | 24 |

Observation clé : **le même `queue_item_id` alterne** entre `duplicate` et `ingested_non_persisted` selon le retry — ce n’est pas deux populations d’événements distincts.

`oldest_item_age_ms` à 18:15:43 ≈ **351300** (~5,9 min) → cohérent avec création **18:09:52**, pas avec 17:20.

HTTP 18:xx **récent** + items datés **18:09–18:10** = flush répété d’une tête de file bloquée (pas un PUT « neuf » portant l’event 17:20:10).

`ledger_ids_missing` (code backend) = IDs ledger incomplets (`tracking_session_id` / `session_generation` / `sequence_id` / `location_event_id`) → ACK `ingested_non_persisted` sans persist durable ledger. Les 3 IDs **n’apparaissent pas** dans `driver_location_events`.

---

## 4. Runtime / GNSS vs queue (18:08–18:20)

Health (extrait) :

| Heure | FGS | nfix | lecture |
|-------|-----|------|---------|
| 18:10:49 | true | **0** | fix natif **frais** (fenêtre de création des 3 items) |
| 18:11:14 | false | — | dip FGS (reprise / deep-link) |
| 18:13:36+ | true | 0→2 puis **55→298** | nfix **monotone croissant** |
| 18:18:43 | true | 298 | `fix_stale` |

Règle respectée : `FGS=true` ≠ nouveaux fixes.

Pendant T12 reprise, **5 `tracking.queue.enqueued`** avec **5 sessions nouvelles**, toujours `sequence_id=1`, `queue_depth` 30→34 :

```text
18:16:03  trk_sess_1786724158138_…  seq=1  depth=30
18:16:29  trk_sess_1786724188460_…  seq=1  depth=31
18:16:58  trk_sess_1786724203657_…  seq=1  depth=32
18:17:32  trk_sess_1786724248886_…  seq=1  depth=33
18:18:00  trk_sess_1786724279279_…  seq=1  depth=34
```

Les flush ACK ne portent **que** les 3 items 18:09–18:10 → **head-of-line blocking** : les nouveaux enqueue n’obtiennent pas d’ACK dans la capture.

Watermark poll répété sur session stale `trk_sess_1786722711491_2h9hb2ps` → **HTTP 403**.

Presence reprise : `driver:19:company:unknown`, `session_generation_id: 0` (logcat precheck 18:12:49).

---

## 5. HTTP emission vs GNSS fix timestamp

| Question | Statut |
|----------|--------|
| PUT HTTP à 18:12+ portant **event_id / session ancre 17:20:10** | **NON observé** (IDs différents) |
| PUT HTTP à 18:12+ portant **fix GNSS horodaté 17:20:10** (lat/lng/ts) | **INCONCLUSIF** — télémétrie ACK/enqueue **sans** lat/lng/`recorded_at` |
| Distinction HTTP ts vs fix ts | **partiellement OK** via âge queue + IDs ; **pas** de payload body dans logcat |

---

## 6. Tableau de décision C1–C4

```text
C1 Native/GNSS              CONFIRMED (fenêtre post-18:13)
                            — nfix/last_fix_age croissent sous FGS=true
                            — nuance : à ~18:10 nfix=0 (fixes frais possibles
                              pour les 3 queue_item créés alors)

C2 Client sequencing        CONFIRMED (partiel)
                            — churn de tracking_session_id + sequence_id=1
                              à chaque enqueue 18:16–18:18
                            — PAS « nouvelle position + ancienne sequence ancre »
                            — nouveaux enqueue non ACKés (bloqués derrière HOL)

C3 Backend persistence      CONFIRMED
                            — ingested_non_persisted + ledger_ids_missing
                            — 3 event_id absents de driver_location_events
                            — alternance duplicate_event_id_unproven sur les
                              MÊMES event_id (retry non durable)

C4 Replay ancien fix 17:20  EXCLUDED (identité session/seq/event)
                            — INCONCLUSIF au niveau coords GNSS exactes
                              (pas de lat/lng dans télémétrie reprise)
```

### Synthèse en une phrase

Ce n’est **pas** un rejeu d’identité du LOC 17:20:10 ; c’est une **tête de file ~18:09–18:10** retried en boucle (`duplicate` ↔ `ledger_ids_missing`), pendant que le **natif devient stale après 18:13** et que le client **continue d’enqueuer sous de nouvelles sessions** sans faire avancer la persistence PG.

---

## 7. Lacunes acceptées (pas de patch)

Pour un prochain diagnostic read-only **plus fin** (toujours NO-GO patch) :

1. Logger / capturer sur **une** tentative : `recorded_at` GNSS + lat/lng + `session_generation` + body ACK complet.
2. Corréler un PUT gateway body (si access log enrichi) aux 3 `queue_item_id`.
3. Clarifier pourquoi `duplicate` et `ledger_ids_missing` alternent pour le **même** `location_event_id`.

---

## Implémentation

✅ **Implémenté** : corrélation read-only ancre ↔ ACK ↔ runtime ; CSV `p0c_acks_resume.csv` / `p0c_seq_context_resume.csv` ; tableau C1–C4.  
**Reste à faire** : rien de patch ; éventuel diagnostic payload GNSS (lat/lng/ts) si GO observation complémentaire — **PATCH reste NO-GO**.
