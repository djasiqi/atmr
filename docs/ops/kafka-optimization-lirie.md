# Runbook — Optimisation Kafka LIRIE (mono-serveur)

Kafka **conservé** pour le pipeline GPS temps réel. Objectif : dimensionner correctement un serveur Hetzner 16 Go sans supprimer le contrat métier async.

Références : [kafka-production.md](kafka-production.md), `env.kafka.production.example`, `scripts/kafka-topics.contract.json`.

## Ordre d'exécution

```text
P0  Swap 4 Go (hôte)                    ✅ fait
P1  Phase 1 — 3 brokers + topics v2    ⏳ topics v2 restant
OBS 14 jours — RAM, lag, latence P95  ← en cours (fin 2026-07-02)
    Décision Go / No-Go Phase 2       (NO-GO par défaut si RAM confortable)
P2  Phase 2 — mono-broker             ⏸️ seulement si pression mémoire revient
```

**Principe** : une optimisation qui dégrade la réactivité carte (< 2 s perçu) est un échec, même avec 4 Go RAM récupérés.

**Objectif LIRIE** : tracking GPS fluide avec une RAM raisonnable — pas « moins de RAM » pour moins de RAM.

---

## P0 — Swap hôte (immédiat, hors dépôt)

Sur `atmr-prod-fsn1`, **avant** toute migration topics :

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
swapon --show
```

Vérification : `Swap:` > 0 dans `free -h`.

✅ **Implémenté** (2026-06-19) : swap 4 Go actif sur `atmr-prod-fsn1` — risque OOM initial fortement réduit.

---

## Phase 1 — Stabilisation (3 brokers)

### Changements dépôt (déployer sur serveur)

- `docker-compose.kafka.yml` : broker-3 RF aligné, heap 1G, limites 2G, threads harmonisés, kafka-ui en profile `kafka-ui`
- Scripts topics paramétrables (`KAFKA_DEFAULT_PARTITIONS=6`, `KAFKA_CREATE_INACTIVE_TOPICS=false`)
- Topics v2 via variables d'environnement

### Variables d'environnement (Phase 1 topics v2)

**Source dans le dépôt** (une des deux) :


| Fichier                                         | Rôle                                                                              |
| ----------------------------------------------- | --------------------------------------------------------------------------------- |
| `scripts/env.production.defaults.fragment`      | Défaut prod — appendu par `deploy-production.sh`                                  |
| `scripts/env.production.local.kafka-v2.example` | Modèle à copier dans `/srv/atmr/.env.production.local` (surcharge non versionnée) |


```env
KAFKA_COMPOSE_FILE=docker-compose.kafka.yml
KAFKA_DEFAULT_PARTITIONS=6
KAFKA_DLQ_PARTITIONS=3
KAFKA_CREATE_INACTIVE_TOPICS=false
KAFKA_TOPIC_DRIVER_LOCATION_RAW=driver.location.raw.v2
KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED=driver.location.processed.v2
KAFKA_TOPIC_DRIVER_LOCATION_DLQ=driver.location.dlq.v2
```

Référence complète : `env.kafka.production.example`.

### Déploiement migration v2

Après mise à jour du fragment (CI / `deploy-production.sh`) **ou** copie du bloc dans `.env.production.local` :

```bash
cd /srv/atmr
INIT_TOPICS=1 scripts/deploy-kafka-production.sh
docker compose -f docker-compose.production.yml restart backend ws-service
scripts/check-kafka-production.sh on
scripts/check-kafka-tracking-pipeline.sh
```

Ne pas éditer `.env.production` à la main sur le serveur — il est régénéré par `deploy-production.sh` (CI + fragment + `.env.production.local`).

### Rollback Phase 1

Remettre les noms de topics **sans** `.v2`, redémarrer consumers + backend + ws-service. Les anciens topics restent intacts.

### Suppression anciens topics

**Après 14 jours** d'observation OK seulement :

```bash
docker exec atmr-kafka-broker-1 kafka-topics --bootstrap-server localhost:9092 --delete --topic driver.location.raw
# idem processed, dlq, notifications.* inactifs si présents
```

---

## Observation 14 jours — KPI


| KPI                  | Seuil            | Commande / PromQL                                                                            |
| -------------------- | ---------------- | -------------------------------------------------------------------------------------------- |
| Latence GPS pipeline | P95 < 2 s        | `histogram_quantile(0.95, sum(rate(tracking_kafka_e2e_latency_seconds_bucket[5m])) by (le))` |
| Lag consumers        | 0 heures ouvrées | `kafka-consumer-groups.sh --describe`                                                        |
| Publish errors       | 0 / 24 h         | `tracking_kafka_publish_errors_total`                                                        |
| RAM stack Kafka      | delta vs J0      | `docker stats --no-stream`                                                                   |
| Mémoire hôte         | MemAvailable     | `free -h`                                                                                    |


Baseline J0 : capturer `free -h`, `docker stats`, lag, P50/P95 latence **avant** bascule v2.

**Période** : J0 = 2026-06-18 → fin **2026-07-02** (14 jours). Décision Phase 2 **uniquement** à cette date, sur métriques réelles.

### Journal d'observation (à compléter 1×/jour)


| Date       | MemAvailable | Swap used  | Lag max | P95 GPS (s) | Notes                                 |
| ---------- | ------------ | ---------- | ------- | ----------- | ------------------------------------- |
| 2026-06-18 | ~1,6 Gi      | 0          | 0       | —           | J0 — urgence mémoire                  |
| 2026-06-19 | ~6,8 Gi      | 4 Go actif | 0       | —           | P0+P1 compose — gain principal obtenu |
| …          |              |            |         |             |                                       |


---

## Audit production — baseline J0 (hôte P0)

**Serveur** : `atmr-prod-fsn1`  
**Date mesures** : 2026-06-18 (post-déploiement Kafka ON)  
**Contexte** : début observation 14 jours — **J0**

### Mesures hôte


| Métrique                       | Valeur observée                               |
| ------------------------------ | --------------------------------------------- |
| RAM totale                     | ~16 Gi                                        |
| MemAvailable                   | ~~1,6 Gi (~~10 % — tendu)                     |
| `free` instantané (vmstat 5 s) | ~290–310 Mi stable                            |
| Inactive(anon)                 | ~12,9 Gi — empreinte JVM/containers dominante |
| Swap                           | **0** — `swpd=0`, `si/so=0`                   |
| CPU user (vmstat t1–t4)        | 73–77 % (pic instantané)                      |
| CPU idle (même fenêtre)        | 5–8 %                                         |
| Run queue `r`                  | 7–11                                          |
| I/O disque `bi/bo`             | ~130–196 K blocks/s                           |
| Disque `/`                     | 46 G / 150 G (32 %), 98 G libres              |


### Synthèse P0


| Critère               | Statut    | Commentaire                                                  |
| --------------------- | --------- | ------------------------------------------------------------ |
| RAM disponible        | ⚠️ Tendu  | ~1,6 Gi available — marge faible pour Kafka + stack ATMR     |
| Swap                  | 🔴 Absent | Risque OOM killer sans tampon ; pas de dégradation gracieuse |
| Pression mémoire anon | ⚠️ Élevée | ~12,9 Gi inactive(anon) — JVM/containers                     |
| CPU (instantané)      | ⚠️ Pic    | 73–77 % user — à corréler avec lag/consumers (étapes P1–P8)  |
| I/O disque            | ⚠️ Actif  | ~130–196 K blocks/s — cohérent Kafka, à surveiller           |
| Espace disque `/`     | ✅ OK      | 32 % utilisé                                                 |


**Verdict P0** : hôte fonctionnel mais sous **contrainte mémoire significative**. L'absence de swap et la faible marge available constituent le **principal risque infra** avant baseline métier. Le disque n'est pas un goulot.

### Recommandations post-audit P0 (classées)


| Priorité | Action                                                                                  | Statut J0                                                      |
| -------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| P0       | Documenter marge RAM comme baseline J0                                                  | ✅ Ce document                                                  |
| P0       | **Configurer swap 4 Go** (section [P0 — Swap hôte](#p0--swap-hôte-immédiat-hors-dépôt)) | ✅ **2026-06-19**                                               |
| P0       | Surveiller OOM killer (lecture seule J0–J14)                                            | Voir commandes ci-dessous                                      |
| P0       | Ne pas lancer Phase 2 mono-broker avant 14 jours                                        | STOP GATE actif                                                |
| P1       | Corréler Inactive(anon) avec `docker stats` / heaps JVM (audit P3)                      | ✅ [Audit P1–P8](#audit-production--p1p8-2026-06-18-ssh-live)   |
| P1       | Recréer conteneurs Kafka pour appliquer compose optimisé                                | ✅ **2026-06-19** — heap 1G, RF=2, limites 2G/512M              |
| P1       | Arrêter kafka-ui (profile optionnel)                                                    | ✅ **2026-06-19** — ~486 Mi libérés                             |
| P1       | Basculer topics v2                                                                      | 🔴 Activer via fragment ou `.env.production.local` puis deploy |
| P2       | Alerting `MemAvailable` < 1 Gi, run queue `r` > 8                                       | `LowMemoryAvailable` en place                                  |


### Surveillance OOM (lecture seule, quotidien J0–J14)

```bash
# Événements OOM récents (kernel)
sudo dmesg -T 2>/dev/null | grep -iE 'oom|killed process|out of memory' | tail -20

# Journal kernel (systemd)
sudo journalctl -k --since "24 hours ago" 2>/dev/null | grep -iE 'oom|killed process' | tail -20

# Snapshot mémoire rapide (à noter 1×/jour)
free -h
swapon --show
```

### Prochaines étapes audit (P1–P8)

- `scripts/check-kafka-production.sh on` — 2026-06-18 ✅
- `scripts/check-kafka-tracking-pipeline.sh` — 2026-06-19 ✅ (smoke OK, latence P95 en attente)
- Variables `.env.production` — 2026-06-18 ✅
- `docker stats` — 2026-06-18 ✅
- `kafka-topics --describe` — 2026-06-18 ✅ (36 partitions legacy)
- `kafka-consumer-groups --describe` — 2026-06-18 ✅ (lag=0)
- Métriques latence GPS P95 (Prometheus/Grafana)
- Cohérence broker-3 RF / heap — 2026-06-18 ⚠️ runtime ≠ compose

**Fin observation J0 + 14 jours** : `DATE_FIN = J0 + 14j` → rapport Go/No-Go Phase 2.

---

## Audit production — P1–P8 (2026-06-18, SSH live)

**Commande** : `scripts/check-kafka-production.sh on` → **tous [OK]** (4 flags, compose, réseau, 3 brokers healthy, DNS, API, 5 topics attendus, 3 consumers running).

### Configuration `.env.production` effective


| Variable                       | Valeur prod                                                      |
| ------------------------------ | ---------------------------------------------------------------- |
| 4 flags Kafka                  | tous `true`                                                      |
| `KAFKA_COMPOSE_FILE`           | `docker-compose.kafka.yml`                                       |
| `KAFKA_BOOTSTRAP_SERVERS`      | 3 brokers                                                        |
| `KAFKA_DEFAULT_PARTITIONS`     | `6` (script init — **topics existants encore 36**)               |
| `KAFKA_CREATE_INACTIVE_TOPICS` | `false`                                                          |
| Topics v2                      | **commentés** — noms legacy actifs (`driver.location.raw`, etc.) |


### RAM conteneurs (`docker stats`)


| Conteneur             | RAM      | Limite     | CPU % (instantané) |
| --------------------- | -------- | ---------- | ------------------ |
| kafka-broker-1        | 1,95 Gi  | illimitée* | 309 % (pic)        |
| kafka-broker-2        | 1,55 Gi  | illimitée* | 4 %                |
| kafka-broker-3        | 1,56 Gi  | illimitée* | 13 %               |
| zookeeper ×3          | ~1,27 Gi | illimitée* | < 1 %              |
| kafka-ui              | 486 Mi   | illimitée* | 0,3 %              |
| tracking consumers ×3 | ~76 Mi   | cgroup OK  | < 5 %              |
| backend               | 1,96 Gi  | 3 Gi       | 0,2 %              |
| ws-service            | 66 Mi    | illimitée* | 1,4 %              |


 Limite affichée `15,24 GiB` = pas de cgroup mémoire sur brokers/ZK/UI (conteneurs **non recréés** depuis 2 semaines).

**Stack Kafka+ZK+UI ≈ 6,8 Gi** + backend ≈ **8,8 Gi** JVM lourde.

### Topics cluster (état réel)


| Topic                       | Partitions | RF  | Notes                                                  |
| --------------------------- | ---------- | --- | ------------------------------------------------------ |
| `driver.location.raw`       | **36**     | 2   | 1 message (p16), lag=0                                 |
| `driver.location.processed` | **36**     | 2   | fanout + ws actifs                                     |
| `driver.location.dlq`       | 36         | 2   |                                                        |
| `notifications.dlq`         | 36         | 2   |                                                        |
| `atmr.ops.smoke`            | 1          | 2   |                                                        |
| Topics legacy inactifs      | 36         | 2   | push/sms/email, mission, dispatch… **encore présents** |


**Phase 1 topics v2** : non basculée (vars commentées). `KAFKA_DEFAULT_PARTITIONS=6` n'affecte que les **nouveaux** topics.

### Consumer groups


| Groupe                            | Lag                 | État                            |
| --------------------------------- | ------------------- | ------------------------------- |
| `tracking-ingest-consumer-group`  | **0**               | 1 msg raw (p16), consumer actif |
| `tracking-processed-fanout-group` | **0** / offsets `-` | 2 membres, topic processed      |
| `ws-service-shared`               | **0**               | aiokafka actif                  |
| `kafka-dlq-consumer-group`        | non détaillé        | running                         |


### Cohérence compose (écart dépôt ↔ runtime)


| Élément                    | Fichier sur serveur           | Conteneur running (2 sem.)     |
| -------------------------- | ----------------------------- | ------------------------------ |
| `KAFKA_HEAP_OPTS` 1G       | présent dans compose (3×)     | **absent** — recreate requis   |
| broker-3 RF=2 via env      | partiellement dans compose    | **RF=3 hardcodé** encore actif |
| kafka-ui profile optionnel | à vérifier au prochain deploy | **actif** (486 Mi)             |
| Limites mémoire 2G brokers | dans compose récent           | **non appliquées**             |


### Hôte J0 (re-mesure SSH)


| Métrique     | Valeur                                    |
| ------------ | ----------------------------------------- |
| MemAvailable | **~2,0 Gi** (légèrement mieux que 1,6 Gi) |
| Swap         | **0**                                     |
| Disque `/`   | 51 G / 150 G (35 %)                       |
| vmstat CPU   | 29–52 % user (non soutenu à 77 %)         |


### Synthèse P1–P8

```
┌─────────────────────────────────────────────────────────────┐
│  AUDIT KAFKA PROD — 2026-06-18 — atmr-prod-fsn1             │
├─────────────────────────────────────────────────────────────┤
│  ✅  check-kafka-production.sh on                           │
│  ✅  Pipeline GPS : lag=0, consumers healthy                │
│  ✅  4 flags Kafka cohérents                                │
│  ⚠️  Topics legacy 36 partitions (v2 non basculée)          │
│  ⚠️  Compose optimisé sur disque, conteneurs non recréés    │
│  ⚠️  broker-3 RF=3 encore actif en runtime                  │
│  ⚠️  kafka-ui actif (~486 Mi)                               │
│  🔴  Swap absent (sudo requis)                              │
│  🔴  Brokers sans limite cgroup / heap en runtime            │
└─────────────────────────────────────────────────────────────┘
```

### Actions recommandées (ordre)


| Priorité | Action                                            | Comment exécuter                                                                 |
| -------- | ------------------------------------------------- | -------------------------------------------------------------------------------- |
| **P0**   | Swap 4 Go                                         | ✅ **2026-06-19**                                                                 |
| **P1**   | Recréer stack Kafka (heap, RF, limites, kafka-ui) | ✅ **2026-06-19**                                                                 |
| **P1**   | Basculer topics v2                                | Fragment ou `.env.production.local` → `INIT_TOPICS=1 deploy-kafka-production.sh` |
| **P1**   | Observation 14 jours + latence P95                | Journal + Grafana — fin **2026-07-02**                                           |
| **—**    | Phase 2 mono-broker                               | ⏸️ **NO-GO** par défaut — Go seulement si MemAvailable moyen < 3 Go              |


**J0 observation** : démarrée le **2026-06-18**. Fin prévue **2026-07-02**.

### Déploiement Phase 1 compose (2026-06-19)

Actions exécutées sur serveur (sans sudo) :

1. **Arrêt kafka-ui** — conteneur supprimé (~486 Mi libérés).
2. **Recréation brokers** — heap `-Xms1G -Xmx1G`, `mem_limit: 2G`, RF=2 sur les 3 brokers.
3. **Recréation zookeeper-2/3** — `mem_limit: 512M` (ZK-1 recréé avec broker-1).
4. **Script** `check-kafka-tracking-pipeline.sh` déployé (CRLF corrigé sur serveur).

**Résultats post-déploiement** :


| Métrique                       | Avant                      | Après                          |
| ------------------------------ | -------------------------- | ------------------------------ |
| MemAvailable hôte              | ~2,0 Gi                    | **~6,8 Gi**                    |
| Stack Kafka+ZK                 | ~6,8 Gi                    | **~1,9 Gi**                    |
| kafka-ui                       | 486 Mi actif               | **arrêté**                     |
| Brokers heap/limites           | illimité, ~1,5–2 Gi chacun | **1G heap, 2G cgroup**         |
| `check-kafka-production.sh on` | OK                         | **OK (3/3 healthy)**           |
| Topics                         | 36 partitions legacy       | **inchangé** (v2 non basculée) |


**Incident mineur** : brokers 2/3 `NodeExists` ZK au premier recreate (résolu par redémarrage groupé).

**Encore à faire (Phase 1)** : appliquer fragment topics v2 sur serveur, mesures latence P95 Prometheus, journal d'observation J0→J14.

### État actuel post-intervention (2026-06-19)

Le contexte J0 (urgence mémoire) **n'est plus d'actualité** :


| Indicateur     | J0 (2026-06-18) | Actuel (2026-06-19)     |
| -------------- | --------------- | ----------------------- |
| MemAvailable   | ~1,6–2,0 Gi     | **~6,8 Gi**             |
| Swap           | 0               | **4 Go actif**          |
| Stack Kafka+ZK | ~6,8 Gi         | **~1,9 Gi**             |
| kafka-ui       | 486 Mi actif    | **arrêté**              |
| Pipeline GPS   | lag=0           | **lag=0**               |
| Risque OOM     | 🔴 élevé        | 🟢 **fortement réduit** |


**Verdict opérationnel** : l'urgence mémoire est **résolue**. La suite = terminer Phase 1 (topics v2) + observation 14 jours. Phase 2 n'est **pas** la prochaine étape par défaut.

---


| Critère              | Go Phase 2                                                                                                     | No-Go Phase 2 (défaut prudent)                          |
| -------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Durée observation    | ≥ 14 jours                                                                                                     | < 14 jours                                              |
| Latence P95 GPS      | < 2 s, régression < 20 % vs J0                                                                                 | > 2 s ou régression                                     |
| Lag / erreurs        | Stables (0)                                                                                                    | Récurrents                                              |
| **Pression mémoire** | MemAvailable **moyen < 3 Go** sur 14 jours **OU** alertes `LowMemoryAvailable` récurrentes **OU** OOM observés | MemAvailable **5–7 Go** stable, pas de pression mémoire |
| Équipe               | Fenêtre maintenance validée                                                                                    | Indisponible                                            |


**Règle de décision** (post-gain Phase 1) :

```text
Si après 14 jours :
  MemAvailable : 5–7 Go
  Lag : 0
  P95 GPS < 2 s
→ Ne pas migrer vers mono-broker.
  Le problème RAM est résolu ; le risque de migration > bénéfice attendu.
```

**No-Go** (cas le plus probable après 2026-06-19) : **rester sur 3 brokers Phase 1**, documenter la baseline confortable, réévaluer dans 30 jours **seulement** si la pression mémoire revient.

Rapport Go/No-Go : documenter dans ticket interne avec captures Grafana + `docker stats` + moyenne MemAvailable sur 14 jours.

---

## Phase 2 — Mono-broker (optionnel, si Go validé)

> ⏸️ **NO-GO par défaut au 2026-06-19** — les gains Phase 1 + swap ont déjà résolu l'urgence mémoire. Phase 2 n'est pertinente que si la **pression mémoire revient** pendant l'observation (voir critères Go/No-Go ci-dessus).

### Prérequis STOP GATE

- Swap 4 Go actif — **✅ 2026-06-19**
- Phase 1 complète (topics v2 + pipeline T1–T6)
- Observation 14 jours terminée
- **MemAvailable moyen < 3 Go** sur la période **OU** pression mémoire récurrente documentée
- Rapport Go documenté (justifiant le bénéfice > risque)
- Tests T1–T13 exécutés (`scripts/check-kafka-tracking-pipeline.sh`)
- Backup volumes Kafka
- Heap Phase 2 = **1G** (pas 768M sans mesure JMX 7 jours)

### Variables d'environnement Phase 2

Surcharger via `scripts/env.production.defaults.fragment` ou `.env.production.local` (ne pas éditer `.env.production` à la main) :

```env
KAFKA_COMPOSE_FILE=docker-compose.kafka.single.yml
KAFKA_BOOTSTRAP_SERVERS=kafka-broker-1:29092
KAFKA_BROKER_REPLICATION_FACTOR=1
KAFKA_TOPIC_REPLICATION_FACTOR=1
```

### Migration (fenêtre ~30–45 min)

```bash
cd /srv/atmr
scripts/check-kafka-production.sh on
docker compose -f docker-compose.production.yml \
  -f docker-compose.kafka.yml \
  -f docker-compose.kafka.atmr-network.yml \
  --profile kafka stop tracking-kafka-consumer tracking-processed-fanout kafka-dlq-consumer

docker compose -f docker-compose.production.yml \
  -f docker-compose.kafka.yml \
  -f docker-compose.kafka.atmr-network.yml \
  down

# Mettre à jour .env.production (bootstrap + RF=1 + KAFKA_COMPOSE_FILE)

INIT_TOPICS=1 KAFKA_COMPOSE_FILE=docker-compose.kafka.single.yml scripts/deploy-kafka-production.sh

docker exec atmr-kafka-broker-1 kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 --all-groups \
  --reset-offsets --to-latest --execute

docker compose -f docker-compose.production.yml restart backend ws-service
scripts/check-kafka-production.sh on
scripts/check-kafka-tracking-pipeline.sh
```

### Rollback Phase 2

```bash
docker compose -f docker-compose.production.yml \
  -f docker-compose.kafka.single.yml \
  -f docker-compose.kafka.atmr-network.single.yml \
  --profile kafka down

# Restaurer KAFKA_COMPOSE_FILE=docker-compose.kafka.yml et bootstrap 3 brokers
# (via fragment ou .env.production.local)
INIT_TOPICS=0 scripts/deploy-kafka-production.sh
```

### SPOF accepté

- Restart broker = interruption tracking async (batch 503, chemin 202)
- **PUT unitaire** : fallback sync conservé (ne pas retirer)
- Monitoring : alertes `Phase4TrackingLagCritical`, `Phase4TrackingKafkaE2eLatencyHigh`, `WsServiceKafkaConsumerDown`

---

## Étude KRaft (optionnel, non bloquant)

PoC local : `docker-compose.kafka.kraft.yml` — 1 nœud broker+controller, **sans Zookeeper** (~400 Mo gain estimé).

```bash
docker compose -f docker-compose.yml -f docker-compose.kafka.kraft.yml up -d
INIT_TOPICS=1 BOOTSTRAP_SERVERS=kafka-broker-1:29092 ./scripts/kafka-init-topics.sh
```

**Non validé prod** tant que healthchecks, deploy script et consumers n'ont pas été testés sur 7 jours en dev. Phase 2 par défaut = `docker-compose.kafka.single.yml` (ZK + 1 broker).

---

## Checklist tests T1–T13

Exécuter via `scripts/check-kafka-tracking-pipeline.sh` (automatisé partiel) + tests manuels carte (T4, T5, T13).


| ID  | Test                    | Critère                      |
| --- | ----------------------- | ---------------------------- |
| T1  | PUT /driver/me/location | HTTP 202                     |
| T2  | POST /locations/batch   | HTTP 202                     |
| T3  | ingest_consumer         | lag=0                        |
| T4  | fanout Socket.IO        | événement carte              |
| T5  | ws-service              | position WS                  |
| T6  | DLQ                     | JSONL si message invalide    |
| T7  | Kafka down + PUT        | 200 sync                     |
| T8  | Kafka down + batch      | 503                          |
| T9  | Restart broker          | reconnect < 30 s             |
| T10 | RAM Phase 1             | delta documenté              |
| T11 | RAM Phase 2             | stack < 3,5 Go               |
| T12 | stop broker             | PUT OK, batch 503            |
| T13 | Latence                 | P95 < 2 s, 10 envois manuels |


---

## Monitoring

- Dashboard Grafana : `driver-location-pipeline` — panel « Latence Kafka E2E raw→processed »
- Alertes : `monitoring/prometheus/rules/infrastructure_alerts.yml`
  - `LowMemoryAvailable` (< 1,5 Go)
  - `Phase4TrackingKafkaE2eLatencyHigh` (P95 > 2 s)
  - `Phase4TrackingLagCritical` (topic raw*)
- JMX brokers : ports 9101–9103 exposés ; mesure heap via `docker exec atmr-kafka-broker-1 jcmd 1 VM.flags` ou UI si `KAFKA_UI_ENABLED=1`

---

## STOP GATE — Phase 1 (fin 14 jours)

- Swap actif — **✅ 2026-06-19**
- Baseline RAM J0 documentée — [Audit production — baseline J0](#audit-production--baseline-j0-hôte-p0)
- Compose optimisé déployé (heap, limites, kafka-ui arrêté) — **✅ 2026-06-19**
- Topics v2 + pipeline OK (T1–T6)
- 14 jours écoulés (fin **2026-07-02**)
- P95 latence < 2 s
- Lag=0, publish errors=0
- Journal MemAvailable complété sur 14 jours
- Anciens topics conservés jusqu'à validation
- Rapport Go/No-Go rédigé

**Décision Phase 1** : ✅ **GO** — terminer topics v2 + observation.

## STOP GATE — Phase 2

- Pression mémoire documentée (MemAvailable moyen < 3 Go **OU** alertes récurrentes)
- Go documenté avec justification bénéfice > risque
- T12 + T13 staging/prod
- Post-migration GPS OK
- P95 < 2 s J+7
- RAM stack < 3,5 Go

**Décision Phase 2 au 2026-06-19** : ⏸️ **NO-GO** — gains Phase 1 suffisants ; réévaluer le **2026-07-02** uniquement si métriques mémoire le justifient.

---

✅ **Implémenté** (2026-06-18) : fichiers compose/scripts/env/monitoring ; runbook P0→P2 LIRIE.

✅ **Baseline J0 hôte P0** (2026-06-18) : audit mémoire/swap/CPU/I/O documenté.

✅ **Intervention P0+P1** (2026-06-19) : swap 4 Go, compose Kafka optimisé, kafka-ui arrêté — MemAvailable ~6,8 Gi, stack Kafka+ZK ~1,9 Gi.

✅ **Audit P1–P8 live SSH** (2026-06-18) : Kafka ON validé, lag=0 — voir sections audit.

✅ **Verdict opérationnel** (2026-06-19) : plan mature, Phase 1 GO, Phase 2 NO-GO par défaut — observation 14 jours avant toute décision mono-broker.