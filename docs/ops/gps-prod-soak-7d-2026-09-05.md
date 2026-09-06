# GPS PROD SOAK — 7 DAYS

**Date d’extraction** : 2026-09-05 (soir, Europe/Zurich)  
**Fenêtre terrain** : 2026-08-29 21:26 UTC → 2026-09-04 17:19 UTC  
**Sources** : Prometheus historique (rétention 30 j, process démarré le 2026-08-10) + journal `driver_location_events`  
**Code GPS** : non modifié (gel volontaire)  
**Clôture officielle** : 2026-09-05 — soak **clos**, moteur **CLOSED / FROZEN**. Ne pas rouvrir le diagnostic.

## Verdict

```text
GPS PROD SOAK 7D
STATUS = AMBER
DIAGNOSTIC = COMPLETE
ENGINE = CLOSED / FROZEN

BACKEND                GREEN
REDIS / FANOUT         GREEN
p99                    EXPLAINED — FIFO iOS
429                    NOT FIFO CAUSE
RATE LIMIT             UNCHANGED
FREQUENCY              UNCHANGED
KAFKA                  UNCHANGED

SEPARATE WORKSTREAMS
7514  → MOBILE CONTINUITY
39067 → BOOKING / ASSIGNMENT LIFECYCLE

NEXT
GPS MOBILE BATTERY / ENERGY OPTIMIZATION
```

---

| Question | Réponse |
| -------- | ------- |
| `GPS PROD 7 DAYS = PASS` | **NON** — verdict **AMBER**, **diagnostic figé** |
| Diagnostic moteur GPS | **COMPLETE** — plus d’enquête pipeline / p99 / 429 |
| Hotfix GPS maintenant | **NON** |
| Architecture / state machine / fraîcheur métier | **GREEN** (HTTP → Redis → fanout) |
| p99 LIVE 117 s | **EXPLAINED** — FIFO iOS, capture ≳ upload utile ; backend 0 ms |
| 429 → FIFO | **NON** — bord de fenêtre courte 30/10 s ; clé par `driver_id` |
| Monitoring reconcilié | **AMBER / P1** — hors moteur GPS |

```text
GPS PROD 7 DAYS — DIAGNOSTIC FIGÉ

PIPELINE BACKEND           GREEN
REDIS CANONICAL            GREEN
SOCKET.IO FANOUT           GREEN

LIVE p99                   EXPLAINED
CAUSE                      iOS FIFO / capture ≳ upload utile
BACKEND LATENCY            NO
429 CAUSE FIFO             NO

HTTP 429                   EDGE SHORT WINDOW
RATE-LIMIT KEY             PER DRIVER
RATE-LIMIT CHANGE          NO

GPS HOTFIX                 NO
GPS CODE                   FROZEN
NO FREQUENCY CHANGE
NO KAFKA CHANGE
```

Hors moteur GPS (ne justifient **aucune** modification de cadence, Redis, fanout, Kafka, rate limiter) :

```text
7514   → MOBILE CONTINUITY / FGS / OS LIFECYCLE
39067  → BOOKING + ASSIGNMENT + MOBILE STATE CLEANUP
         0 faux mission_live après COMPLETED
```

Prochain chantier : [`gps-mobile-battery-baseline-2026-09-05.md`](gps-mobile-battery-baseline-2026-09-05.md).

Le déploiement de l’après-midi a bien reset les compteurs backend (`driver_location_received_total` ≈ 1 870 depuis restart). Les chiffres ci-dessous viennent des **séries Prometheus 7 j** et de Postgres, pas des `/metrics` instantanés.

---

## Scorecard

```text
Drivers observed                 7
Active mission hours (span j/ch) 154.8 h
Missions DLE distinctes          109
Bookings COMPLETED / RETURN      81 / 57

GPS points received (Prom)       122 971
  dont mission_live HTTP         82 621
  dont presence HTTP             40 350
Persisted DLE (Postgres)         96 587
Accepted canonical (Prom)        97 596
Redis canonical writes           97 596
Fanout driver_location_update    122 962
Fanout driver_live_state_update  97 596
Observability-only               25 365
Dedup event_id / proximity       16 792 / 1 070

Kafka persist 7d                 0   (voie non utilisée cette semaine)
Kafka fanout emit 7d             0
Kafka lag actuel                 0
Kafka E2E (21 j, historique)     p50 635 ms · p95 1.36 s · p99 1.90 s

Freshness LIVE (DLE created−recorded)
  p50                            2.1 s
  p95                            103 s
  p99                            117 s     ← runbook cible p99 < 60 s
  share < 60 s                   79.8 %
  share < 5 s                    54.0 %

Freshness PRESENCE (DLE)         p50 0.32 s · p95 301 s · p99 503 s
Canonical update latency HTTP    p50 0.43 s · p95 114 s · p99 300 s (bucket cap)
  mission_live                   p50 1.25 s · p99 118 s

Invariant violations             0
GPS 429 (PUT driver_location)    1 859
Auth 429 (refresh_token)         11 349   (hors GPS)
WS batch rate-limited            0
Fleet frozen incidents           0
Stale LIVE (preuve carte)        non mesurable côté backend seul

Offline / trous intra-mission    57 (>2 min) · 48 (>5 min) · 34 (>10 min)
Worst intra-mission              279 min  DRIVER-7514  mission 39067
Trous nuit/week-end exclus       oui (filtre même mission_id)

OSRM availability                99.90 %  (165 025 req, 165 erreurs match)
Google fallback                  série absente (0 observé)
```

---

## 1. Chaîne réelle cette semaine

La voie qui a porté le terrain n’est **pas** Kafka persist/fanout.

```text
Téléphone
  → PUT HTTP
  → backend ingest / LocationService
  → Redis canonical
  → Socket.IO fanout (backend)
  → carte
```

Preuves :

- `tracking_kafka_persist_total` : **0** sur 7 j, 3 j, 1 j et 1 h. Le compteur du consumer vaut 4 892 depuis le 17 août — tout le volume est **antérieur** à la fenêtre.
- `tracking_fanout_emit_total` : **0**. Le fanout utile est `driver_location_fanout_events_total` (job `atmr-backend`).
- Kafka lag partitions `raw.v2` / `processed.v2` : **0** (consumer vivant, file vide).
- Transport `kafka` sur `driver_location_received_total` : **0**.

Donc comparer `received` à `tracking_kafka_persist_total` cette semaine **ne mesure pas** la santé GPS. La persistance à certifier est :

- `driver_location_processed_total{accepted_canonical}` ≈ Redis canonical ≈ DLE.

Ratio canonical / received ≈ **79 %**. Le solde est surtout `accepted_observability_only` (présence ou points non canoniques), pas une perte Kafka.

`tracking_pipeline_divergence_total{kind=canonical_without_ledger}` ≈ **1 009** (~1 % des canonical). Écart ledger, pas une carte figée.

---

## 2. Activité chauffeur par chauffeur

| driver_id | points | LIVE | PRESENCE | missions | premier fix | dernier fix |
| --------- | -----: | ---: | -------: | -------: | ----------- | ----------- |
| 4 | 35 918 | 24 847 | 11 071 | 33 | 30 août 06:06Z | 4 sept 17:19Z |
| 7755 | 33 632 | 24 358 | 9 274 | 24 | 30 août 14:08Z | 4 sept 16:15Z |
| 23345 | 19 082 | 10 470 | 8 612 | 12 | 1 sept 11:06Z | 4 sept 11:31Z |
| 7514 | 4 284 | 622 | 3 662 | 11 | 29 août 21:26Z | 4 sept 12:49Z |
| 20135 | 1 272 | 0 | 1 272 | 0 | 31 août 21:00Z | 31 août 22:04Z |
| 16150 | 1 244 | 1 207 | 37 | 4 | 31 août 08:56Z | 31 août 18:46Z |
| 3 | 1 155 | 839 | 316 | 30 | 29 août 21:39Z | 4 sept 14:19Z |

Volume journalier (timezone Zurich) :

| Jour | points | dont LIVE | chauffeurs |
| ---- | -----: | --------: | ---------: |
| 29 août | 106 | 0 | 2 |
| 30 août | 3 706 | 2 050 | 4 |
| 31 août | 14 086 | 12 426 | 6 |
| 1 sept | 15 252 | 8 864 | 6 |
| 2 sept | 28 957 | 15 802 | 5 |
| 3 sept | 20 972 | 16 542 | 5 |
| 4 sept | 13 508 | 6 659 | 5 |
| 5 sept | 0 terrain | — | week-end |

Pas d’activité GPS le samedi 5 septembre au moment de l’audit — cohérent avec le calendrier, pas une panne flotte.

---

## 3. Fraîcheur

L’histogramme runbook `driver_tracking_position_freshness_seconds` est **vide**. Il n’est observé que dans `ingest_persist.py` (voie Kafka). Cette voie n’a rien persisté sur 7 j → **on ne peut pas certifier le p99 officiel**.

Proxys utilisés (même sémantique `recorded_at → accept/persist`) :

1. `created_at - recorded_at` sur DLE (96 587 points).
2. `driver_location_canonical_update_latency_seconds` (97 595 observations HTTP).

Les deux donnent le même ordre de grandeur LIVE : **p50 1–2 s, p99 ≈ 117–118 s**.

Interprétation :

- Un chauffeur immobile / une file locale qui flush après tunnel n’est **pas** une panne. Le plafond LIVE à ~121 s colle à une fenêtre `too_old_for_mode`.
- Le contrat runbook **p99 < 60 s** n’est **pas** tenu sur LIVE (79,8 % des points LIVE sont < 60 s ; la queue fait le reste).
- PRESENCE p95/p99 longs (5–10 min) sont attendus : cadence large + flush, **pas** un incident LIVE.

**Stale displayed as LIVE** : non prouvable depuis le backend seul. Le modèle OFF/BLOCKED/PRESENCE/LIVE + âge `recorded_at` est en place ; il manque une métrique carte (âge affiché vs mode).

---

## 4. Incidents invisibles

### A. Trous puis reprise

Les trous bruts > 2 min sur tout le LIVE (126, dont worst 40 h) sont **presque tous nuit / week-end**. Ils ne comptent pas.

Sur **la même `mission_id`** :

| Seuil | N |
| ----- | -: |
| > 2 min | 57 |
| > 5 min | 48 |
| > 10 min | 34 |
| worst | 279 min — DRIVER-7514 / mission 39067 (2 sept 14:48 → 19:28 UTC) |

Autres worst intra-mission : 16150/38922 (167 min), 4/38923 (110 min), 7514/38914 (104 min).

Ce n’est **pas** encore un bug GPS démontré : attente patient, multi-leg, pause, ou vrai dropout mobile. C’est la liste à recouper (FGS, réseau, queue, heartbeat) **sans** changer le code.

DRIVER le plus bruité en trous intra-mission : **7514** (12 trous > 2 min, worst 279 min) — aussi le profil le plus PRESENCE-heavy.

### B. Doubles / sequence

- `location_event_id` dupliqués : **0**
- `sequence_id` qui recule dans une session : **1 841** (à surveiller owner/lease ; pas de preuve de double carte)
- Dedup proximity : 1 070 (attendu)

### C. LIVE alors que bloqué

Pas de signal Prometheus d’invariant (INV-*) sur 7 j. Pas d’alerte `TrackingFleetFrozen`.

---

## 5. 429

Cible soak « GPS 429 = 0 » : **non tenue**.

| Endpoint | 429 / 7 j |
| -------- | --------: |
| `auth_refresh_token` POST | 11 349 |
| **`driver_driver_location` PUT** | **1 859** |
| `geocode_geocode_address` GET | 92 |
| WS `driver_location_batch` | 0 |

~1,5 % des PUT location sont rejetés. **Causalité tranchée** : les 429 ne créent pas la FIFO ([`gps-429-fifo-causality-2026-09-03.md`](gps-429-fifo-causality-2026-09-03.md)). Filet en bord de fenêtre courte (31 succès / 10 s pour une limite 30). **Ne pas augmenter les plafonds.**

---

## 6. Monitoring — P1 réel

Confirmé sur le serveur :

- `MONITORING_BASIC_AUTH_USERS` **absent** de `.env.production` et `.env.production.local` → le compose monitoring est **ignoré** au deploy.
- Prometheus / Grafana / Alertmanager : **up depuis le 10 août**, non recréés aujourd’hui.
- Fichiers `monitoring/` copiés le 5 sept 14:37 (bind-mount). Hash host = hash conteneur (même fichiers). `lastConfigTime` Prometheus = **2026-08-10** : pas de reload process.
- Exporters **down** : `postgres-exporter`, `redis-exporter`, `node-exporter`, `cadvisor`, `blackbox-exporter`, cible `tracking-processed-fanout`.
- Alertes **fantômes** firing : `PostgreSQLDown`, `RedisDown` (Postgres/Redis sont healthy). `TrackingDeviceHealthPipelineSilent` firing samedi soir — attendu sans chauffeur, mais l’alerte est trop naïve le week-end.
- **Skew images** (hors GPS code, ops) :
  - backend / ws-service : `sha-c8e0109` (5 sept)
  - `tracking-kafka-consumer` : `sha-d5694d8` (**17 août**)
  - `tracking-outbox-publisher` : `sha-286737a` (**15 août**)

Conséquence : on peut certifier les métriques backend + le journal DLE. On **ne peut pas** garantir que dashboards/rules en mémoire = commit déployé aujourd’hui, ni faire confiance aux alertes infra.

OpenTelemetry « non installé » : dette, **non bloquant**, ne pas toucher pendant le soak.

---

## 7. Ce que je ne ferais pas maintenant

Gel **figé** après diagnostic :

- pas de hotfix GPS ;
- pas d’augmentation des plafonds 30/10 s ni 120/60 s ;
- pas de changement de fréquences, dedup, leases, PRESENCE/LIVE, Redis, Kafka, fallback Maps ;
- pas de chantier batterie **dans ce soak** — baseline extraite à part.

---

## 8. Critères PASS (inchangés) et clôture diagnostic

```text
GPS PROD 7 DAYS = PASS uniquement si :

LIVE freshness p99 < 60 s
PUT driver_location 429 = 0 ou niveau résiduel explicitement justifié
0 trou inexpliqué > 10 min pendant mission
0 invariant
0 fleet frozen
0 faux LIVE après mission terminée
0 faux LIVE sur fix stale
monitoring reconcilié et fiable
```

```text
GPS PROD SOAK 7D
STATUS = AMBER, DIAGNOSTIC COMPLETE

NO GPS HOTFIX
NO RATE LIMIT CHANGE
NO FREQUENCY CHANGE
NO KAFKA CHANGE

OPEN SEPARATELY
- 7514 mobile continuity
- 39067 lifecycle zombie
- monitoring P1

NEXT WORKSTREAM
GPS MOBILE BATTERY / ENERGY OPTIMIZATION
```

Preuves figées :

- Autopsie 7514 / 39067 : [`gps-autopsy-7514-39067-2026-09-02.md`](gps-autopsy-7514-39067-2026-09-02.md)
- p99 DRIVER-4 / 7755 : [`gps-p99-drivers-4-7755-2026-09-05.md`](gps-p99-drivers-4-7755-2026-09-05.md)
- 429 ↔ FIFO : [`gps-429-fifo-causality-2026-09-03.md`](gps-429-fifo-causality-2026-09-03.md)
- Baseline batterie : [`gps-mobile-battery-baseline-2026-09-05.md`](gps-mobile-battery-baseline-2026-09-05.md)

- 7514/39067 = silence mobile + zombie `mission_live` — **hors moteur GPS**.
- p99 117 s = **file mobile iOS** `recorded_at → received_at` (backend 0 ms).
- 429 = bord de fenêtre courte ; **ne causent pas** la FIFO ; clé par `driver_id`.
