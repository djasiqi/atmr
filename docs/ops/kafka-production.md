# Runbook — Kafka en production (ATMR / LIRIE)

Contrat déploiement :

- **`scripts/deploy-production.sh`** : stack applicative **uniquement** — ne doit **jamais** activer `--profile kafka` ni fusionner `docker-compose.kafka*.yml`.
- **`INIT_TOPICS=1 scripts/deploy-kafka-production.sh`** : **seul chemin officiel** pour déployer brokers + raccord `atmr-network` + consumers profile `kafka`, avec preflight et validations post-deploy.
- **Kafka OFF** : les 4 flags à `false` (ou absents) dans `.env.production`, **aucun** consumer Kafka actif.
- **Kafka ON** : brokers healthy, DNS `kafka-broker-*` depuis `atmr-network`, topics créés, consumers du profile `kafka` en `running`.

Référence variables : `env.kafka.production.example` à la racine du dépôt.

## Mode Kafka OFF (nominal)

1. Dans `.env.production` :

   - `KAFKA_ENABLED=false`
   - `TRACKING_INGEST_ASYNC_ENABLED=false`
   - `TRACKING_PROCESSED_FANOUT_ENABLED=false`
   - `WS_KAFKA_CONSUMER_ENABLED=false`

2. Déploiement applicatif : CI/CD habituel ou `scripts/deploy-production.sh` (sans Kafka).

3. Validation :

   ```bash
   cd /srv/atmr
   scripts/check-kafka-production.sh off
   ```

   Attendu : exit `0` — flags non « true », aucun consumer Kafka actif, backend `healthy`.

> **Ne pas** utiliser `scripts/check-kafka-production.sh preflight-on` pour valider Kafka OFF : ce mode exige les **4 flags à true**.

## Mode Kafka ON (bascule officielle)

### Prérequis

- Fichiers présents sous `/srv/atmr` :  
  `docker-compose.production.yml`, `docker-compose.kafka.yml`, `docker-compose.kafka.atmr-network.yml`
- `.env.production` : les **4** flags Kafka à `true` (cohérence stricte).
- Réseau Docker `atmr-network` (créé automatiquement au deploy Kafka si `ATMR_AUTO_CREATE_NETWORK=1`, défaut dans `deploy-kafka-production.sh`).

### Première montée (topics à créer)

```bash
cd /srv/atmr
scripts/check-kafka-production.sh preflight-on    # optionnel (dry-run)
INIT_TOPICS=1 scripts/deploy-kafka-production.sh
scripts/check-kafka-production.sh on              # validation finale
```

### Déploiements suivants (topics déjà présents)

```bash
cd /srv/atmr
scripts/deploy-kafka-production.sh
scripts/check-kafka-production.sh on
```

### Comportement attendu

- Preflight : 4 flags, 3 YAML, `atmr-network`, résolution Compose (profile `kafka`).
- Post-deploy : brokers `healthy` (via `docker inspect`), DNS, `kafka-broker-api-versions`, liste des topics, consumers `tracking-kafka-consumer`, `tracking-processed-fanout`, `kafka-dlq-consumer` en `running`.

### Garde-fou `FORCE=1`

- Contourne **uniquement** le preflight (flags / fichiers / réseau / résolution) — **réservé bootstrap initial**.
- **Ne contourne jamais** le post-deploy (brokers, DNS, topics, consumers) : échec → exit `3`.

## Critères de succès

| État | Commande | Exit attendu |
|------|-----------|--------------|
| OFF | `scripts/check-kafka-production.sh off` | `0` |
| ON | `scripts/deploy-kafka-production.sh` puis `scripts/check-kafka-production.sh on` | `0` / `0` |

Codes `deploy-kafka-production.sh` : `0` OK, `2` preflight refusé, `3` post-deploy refusé.

## Rollback Kafka ON → OFF

1. Arrêter la stack Kafka (sans `down` de toute la prod si possible) :

   ```bash
   docker compose -f docker-compose.production.yml \
     -f docker-compose.kafka.yml \
     -f docker-compose.kafka.atmr-network.yml \
     --profile kafka down
   ```

2. Repasser les 4 flags à `false` dans `.env.production`.

3. Redémarrer les services qui lisent ces flags :

   ```bash
   docker compose -f docker-compose.production.yml restart backend ws-service
   ```

4. Valider :

   ```bash
   scripts/check-kafka-production.sh off
   ```

## Erreurs fréquentes

1. **`NoBrokersAvailable` (clients Python / ws-service)**  
   Les brokers ne sont pas joignables depuis l’app — souvent **pas de raccord `atmr-network`** sur les brokers. Le script officiel impose `docker-compose.kafka.atmr-network.yml`. Diagnostic : `scripts/check-kafka-production.sh on`.

2. **DNS : `kafka-broker-1` introuvable depuis le backend**  
   Même cause : merge Compose sans `docker-compose.kafka.atmr-network.yml`. Les checks utilisent `getent` depuis `backend` ou depuis `atmr-kafka-broker-1` (sans pull d’image externe).

3. **Oubli de `docker-compose.kafka.atmr-network.yml`**  
   Refusé en preflight (`kafka_check_compose_files`). Un `FORCE=1` ne doit pas masquer un post-deploy : `kafka_check_dns_from_atmr_network` échouera (exit `3`).

4. **Flags incohérents (mix true/false)**  
   Preflight bloque avec le détail des 4 variables. Diagnostic : `scripts/check-kafka-production.sh preflight-on` (avec les 4 à `true` **avant** correction, ou corriger `.env` puis relancer).

5. **Consumer en `Restarting`**  
   Vérifier image backend à jour (sortie propre si `KAFKA_ENABLED=false`). `docker compose ... ps` doit afficher `Up`, pas `Restarting`.

## Consumer notifications (`kafka-notifications`)

Le profile Compose **`kafka-notifications`** (`kafka-consumer` dans `docker-compose.kafka.yml`) est **hors contrat** du déploiement officiel documenté ici : seul le profile **`kafka`** (3 consumers tracking/DLQ) est validé par `deploy-kafka-production.sh` et `check-kafka-production.sh on`. Le mode `off` détecte toutefois un `kafka-consumer` encore actif.

## CI manuel (P1)

Workflow dédié : `.github/workflows/deploy-kafka.yml` (`workflow_dispatch`, confirmation `kafka-on-production`, `dry_run` par défaut). Ne passe **jamais** `FORCE=1`.
