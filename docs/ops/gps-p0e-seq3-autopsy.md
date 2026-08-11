# Autopsie seq=3 — watermark session (Phase 2)

**Date** : 2026-08-11  
**Mode** : 100 % read-only (aucune écriture SQL)  
**Image alignée** : consumer + outbox + backend = `390076efc61ca71332c749a67aff1e6fc7c2d626`

## Session figée

| Champ | Valeur |
| --- | --- |
| `driver_id` | `3` |
| `tracking_session_id` | `trk_sess_1786447105637_ivp6dqaq` |
| `session_generation` | `1204` |

## Watermark PG

| Champ | Valeur |
| --- | --- |
| `contiguous_persisted_through` | **2** |
| `max_seen_sequence` | **23** |
| `first_seen_at` | 2026-08-11 11:18:25 UTC |
| `last_seen_at` | 2026-08-11 11:47:42 UTC |
| `closed_at` | NULL |

## Chronologie ledger / DLE / outbox

- Séquences présentes : **1, 2, 4…23** (22 rows) — source `http` partout.
- **`sequence_id = 3` : absent** de `tracking_ingest_events`, `driver_location_events`, et de l’outbox joinée.
- Trous : uniquement **3** entre 1 et max.
- Gaps enregistrés dans `tracking_sequence_gaps` : dès l’arrivée de seq=4 (11:20:41 UTC), puis gaps `3→N` non résolus à chaque avancée de `max_seen` — comportement attendu du watermark contigu.

Fenêtre critique :

| Seq | `location_event_id` | `recorded_at` (UTC) |
| ---: | --- | --- |
| 2 | `trk_1786447160034_o4ll0444` | 11:19:20 |
| **3** | **ABSENT** | — |
| 4 | `trk_1786447241129_ajuo648t` | 11:20:41 |

## Access logs

Aucun PUT identifiable portant explicitement `sequence_id=3` / cet `location_event_id` manquant. Les logs d’access ne conservent pas le body ; on ne peut pas prouver un 401/403/429/202 dédié à seq=3. Des `200` ~672/673 (okhttp) coexistent avec des réponses plus courtes (393/421/636) et des `403` iOS (`CFNetwork`) sur d’autres fenêtres — non attribuables à seq=3 sans corrélation body.

## Classification

**`NON DÉTERMINABLE CÔTÉ SERVEUR`**

Preuves serveur robustes :

```text
seq 3 absent du ledger
+ absent DLE
+ absent outbox
+ pas de requête HTTP identifiable pour cet event
+ max_seen=23, contiguous=2, gaps 3→N non résolus
```

Indiscernable sans traces mobile : jamais capturé / capturé non transmis / encore SQLite client.

Ce n’est **pas** une session bloquée : les seq ≥4 ont été ACK-persistées individuellement (DLE + outbox publiée) alors que le watermark contigu reste à 2 — cohérent avec invariant #3 ops (tombstone mobile sur ACK individuel, pas sur `contiguous_persisted_through`).

## Invariant #8

**NON PROUVÉ** — trou seq=3 documenté mais cause exacte non établie côté serveur.

## Script

[`scripts/ops-p0e-kafka-align-phase2.sh`](../../scripts/ops-p0e-kafka-align-phase2.sh)
