# Sprint 2 — Design flotte / tracking (PAS DE CODE)

**Statut** : document de conception — implémentation **bloquée** jusqu'au STOP GATE P2 ([`gps-tracking-pipeline.md`](./gps-tracking-pipeline.md)) + revue S2.1.

---

## S2.1 — Relecture Redis au fanout Socket.IO

### Règle fallback Kafka obligatoire

```text
Redis MISS + payload Kafka processed présent → emit dégradé depuis Kafka (OBLIGATOIRE)
→ tracking_fanout_redis_read_total{result=miss}
```

### Limitation race `mission_status` (L2)

Redis canonical = lat/lon + presence + device_health. **`mission_status` n'est PAS dans Redis.**

| Option | Description | Avantages | Inconvénients |
|---|---|---|---|
| **A — DB live** (recommandée) | `mission_status` depuis DB à chaque fanout | Simple | Race REST/Socket ~100–500 ms sur transition course |
| **B — Redis snapshot** | Écrire `mission_status` dans canonical + TTL | Alignement REST/Socket | Couplage booking ↔ tracking |
| **C — Kafka processed snapshot** | Propager statut dans payload processed | Cohérence au moment ingest | Ne résout pas changement post-persist |

**Sémantique option A (N3)** :

```text
S2.1 + option A = divergence POSITION corrigée via relecture Redis
                 ≠ divergence MISSION_STATUS (race DB acceptée, documentée)
```

### Test synthétique REST vs Socket (O3)

1. Script Python/Node : `GET /companies/me/drivers/live` puis subscribe Socket.IO 30 s
2. Comparer par `driver_id` : `lat`, `lon`, `status`, `mission_status`
3. Tolérance position : Δ < 1 m, âge < 5 s
4. Hors scope option A : `mission_status` — divergences < 1 s sur transition acceptées

### Métriques post-déploiement S2.1

- `tracking_fanout_redis_read_total{result=hit|miss|error}`
- `tracking_fanout_redis_read_latency_seconds`
- `tracking_fanout_build_latency_seconds`

Fichiers cibles : [`processed_fanout_consumer.py`](../../backend/services/tracking/processed_fanout_consumer.py).

---

## S2.2 — Batch overlay Socket web

### Règles buffer (P3)

```text
1. shouldAcceptRealtimeEvent(event_id) AVANT insertion Map
2. Clé = driver_id
3. Valeur = payload avec canonicalRealtimeTimeMs MAX (dernier gagne)
4. Réseau désordonné : timestamp canonique, pas ordre réception
5. driver_live_state_update → flush immédiat (hors batch)
```

### Asymétrie T_web / T_mobile (D2)

| Plateforme | T proposé | Justification |
|---|---|---|
| Web | **250 ms** | Compromis latence / batch (si STOP GATE ev/s > 20) |
| Mobile | **500 ms inchangé** | [`gpsFlushScheduler.ts`](../../mobile/unified-app/src/features/company/realtime/gpsFlushScheduler.ts) |

T_web **finalisé après STOP GATE**.

### Mini-audit gpsFlushScheduler (N4)

Confirmer règles 1–5 ou documenter écarts : dedup `event_id`, latest by timestamp, flush immédiat `driver_live_state_update`.

Fichiers web cibles : overlay TanStack + `useCompanyDriversLiveOverlay`.

---

## S2.3 — Clustering mobile réactif au zoom

Seuil proposé : re-cluster si `latitudeDelta` change > **10 %** (pinch zoom).

Conditionnel UX — décision après STOP GATE 50/100 drivers.

Fichiers : [`OperationalFleetMap.tsx`](../../mobile/unified-app/src/features/company/components/maps/OperationalFleetMap.tsx), [`fleetMapLogic.ts`](../../mobile/unified-app/src/features/company/components/maps/fleetMapLogic.ts).

---

## Matrice GO/NO-GO post-STOP GATE

| Item | Pronostic | Décision |
|---|---|---|
| S2.1 Redis fanout | Presque certainement GO (position) | |
| S2.2 Batch overlay | Dépend ev/s + queue_size | |
| S2.3 Zoom cluster | Dépend UX 50/100 | |

---

## Hors scope Sprint 2

- Harmonisation resync 60 s / 120 s web/mobile
- ETA Web (décision produit)
- Implémentation L2 mission_status option B/C sans arbitrage produit
