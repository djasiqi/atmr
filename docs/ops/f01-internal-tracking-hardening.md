# F-01 — Durcissement ingestion GPS interne

**Statut** : Plan verrouillé implémenté — clôture du scénario P0 « appelant public anonyme ».

> **Suite F-02** : perte silencieuse post-acceptation ws →
> [`f02-internal-tracking-durability.md`](f02-internal-tracking-durability.md)
> (ACK = commit PostgreSQL ; Kafka hors frontière ACK).

## Objectif

Empêcher l’accès anonyme / public à `POST /api/internal/tracking/ingest` et garantir un chemin fail-closed ws-service → backend.

## Risque résiduel accepté

Un **ws-service compromis** qui possède `INTERNAL_SERVICE_TOKEN` peut forger des événements pour n’importe quel `driver_id` du tenant résolu.  
Frontière de confiance = ws-service (JWT/socket). Pas de mTLS ni d’assertion signée session dans ce lot.

## Audience (exacte)

- `INTERNAL_SERVICE_AUDIENCE` = `ws-service` (défaut obligatoire en prod).
- Header `X-Internal-Service` **obligatoire**, match exact via `hmac.compare_digest`.
- Clé rate-limit = constante normalisée `ws-service` (jamais la valeur brute client).

## Flag d’arrêt

`INTERNAL_TRACKING_INGEST_ENABLED=false` → réponse `503 ingest_disabled` **avant** Redis.  
Le ws-service traite comme `5xx` (requeue + backoff), dans la limite du buffer / spool.

## Idempotence Redis (accélérateur ; autorité = PG depuis F-02)

- Clé : `tracking:ingest:{driver_id}:{sha256(location_event_id)}`
- `pending:{nonce}` (TTL **60 s**) → commit PG → Lua `done` (TTL **24 h**) si nonce match
- Échec persist → Lua DEL si nonce match
- `done` / `pending` : indices ; **pas de 503 contention Redis seule** (F-02)
- Redis KO → continuer vers PostgreSQL

## Isolation réseau / Traefik

- Pas de port hôte `5000` sur le backend.
- Labels Compose + [`traefik/dynamic.yml`](../../traefik/dynamic.yml) :
  `Host(\`api.lirie.ch\`) && !PathPrefix(\`/api/internal\`)`
- Smoke public **strictement 404** (un 401 = isolation cassée).

## Ordre de déploiement (obligatoire)

1. Injecter `INTERNAL_SERVICE_TOKEN` (CURRENT) ; `NEXT` optionnel.
2. `docker compose -f docker-compose.production.yml --env-file .env.production config`
3. Fermer port 5000 + corriger Traefik (labels + `dynamic.yml`) ; reload Traefik.
4. Smoke public → **404** ; vérifier absence d’écoute `:5000`.
5. Déployer **ws-service** (nouveau buffer / circuit / headers).
6. Déployer **backend** fail-closed.
7. Smoke interne (réseau Docker) : 401 sans token, 200 avec token.
8. Re-smoke public → **404**.
9. Vérifier Redis (pending/done), métriques ws (retry / circuit) ; ledger F-02 si activé.

## Rotation dual-token (9 étapes)

1. Backend : `CURRENT=ancien`, `NEXT=nouveau`
2. Recréer **uniquement** backend
3. Vérifier les deux secrets acceptés
4. Ws : `CURRENT=nouveau`
5. Recréer **uniquement** ws-service
6. Vérifier ingestion
7. Backend : `CURRENT=nouveau`, `NEXT=` (vide)
8. Recréer backend
9. Vérifier ancien secret → **401**

## Rollback (anti fail-open)

- **Interdit** de republier une image fail-open.
- Conserver port fermé + exclusion Traefik.
- Option : `INTERNAL_TRACKING_INGEST_ENABLED=false` + drain buffer ws.

## Checklist GO

- [ ] Secret ≥ 32 caractères dans coffre + `.env.production`
- [ ] `traefik/dynamic.yml` déployé / rechargé
- [ ] `curl -X POST https://api.lirie.ch/api/internal/tracking/ingest` → **404**
- [ ] `curl -X POST https://api.lirie.ch/api/internal/tracking/ingest/` → **404**
- [ ] Smoke interne avec token → 2xx / ingest
- [ ] Tests : `docker compose … exec -T backend python -m pytest tests/security/test_internal_tracking_f01.py -q`
- [ ] Suite F-02 : [`f02-internal-tracking-durability.md`](f02-internal-tracking-durability.md) (capacité duale + tests §17)

## Checklist ROLLBACK

- [ ] Pas de rollback vers image pré-F-01 fail-open
- [ ] Traefik exclusion conservée
- [ ] Flag `INTERNAL_TRACKING_INGEST_ENABLED=false` si besoin
