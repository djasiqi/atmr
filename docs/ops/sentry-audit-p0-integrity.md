# Audit Sentry P0 — IntegrityError auth/GPS (août 2026)

## Résumé exécutif

| Issue | Décision | Preuve |
| --- | --- | --- |
| [PYTHON-FLASK-DZ](https://lirie.sentry.io/issues/PYTHON-FLASK-DZ) | **Resolved** | Fix `77f1d716` inclus dans Build & Deploy #528 (`f3e42b94`). Event unique du 2026-08-09 21:22 UTC **avant** le commit du fix (~22:17 UTC). |
| [PYTHON-FLASK-DE](https://lirie.sentry.io/issues/PYTHON-FLASK-DE) | **Laisser ouvert** | Correctif Git `a72c408a` présent dans `main`, mais déploiement du **consumer Kafka** non prouvé (deploy backend ≠ Kafka). |
| [PYTHON-FLASK-DF](https://lirie.sentry.io/issues/PYTHON-FLASK-DF) | **Laisser ouvert** | Cascade fail-stop de DE — même critère runtime. |

**Ne jamais** filtrer globalement les `IntegrityError` dans `before_send` Sentry.

## Correctifs livrés dans le dépôt (cette passe)

1. **Resolve DZ** (Sentry).
2. **Traçabilité release** :
   - [`scripts/deploy-production.sh`](../../scripts/deploy-production.sh) écrit `GIT_SHA` + `SENTRY_RELEASE` dans `.env.production`.
   - [`.github/workflows/deploy.yml`](../../.github/workflows/deploy.yml) exporte `SOURCE_SHA` → `GIT_SHA` / `SENTRY_RELEASE` avant deploy.
   - Consumer Kafka : surcharge `SOURCE_SHA` dans [`scripts/ops-tracking-p0-recreate-ingest.sh`](../../scripts/ops-tracking-p0-recreate-ingest.sh) + [`.github/workflows/deploy-kafka-p0.yml`](../../.github/workflows/deploy-kafka-p0.yml) + env explicite sur `tracking-kafka-consumer` dans [`docker-compose.production.yml`](../../docker-compose.production.yml).
3. **Test concurrence** : `tests/routes/test_session_resume.py::test_session_resume_concurrent_same_idempotency_key` (`@pytest.mark.integration`, 2 `test_client`, `Barrier`, reload DB).

## Critère de clôture DE/DF (à exécuter sur le serveur prod)

```bash
# 1) OCI revision du consumer actif
docker compose -f docker-compose.production.yml --profile kafka ps -q tracking-kafka-consumer \
| xargs -r docker inspect \
  --format '{{.Name}} image={{.Config.Image}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}'

# 2) Ancestry Git (pas une comparaison numérique de SHA)
git merge-base --is-ancestor \
  a72c408ad12826c8930f700f0ce29ea97ec65a83 \
  "$SOURCE_SHA"
```

Resolve DE + DF seulement si :

```text
consumer actif
+ OCI revision == SOURCE_SHA
+ a72c408a ancêtre de SOURCE_SHA
+ 0 nouvel événement DE/DF
```

Sinon : redéployer via `deploy-kafka-p0` / `ops-tracking-p0-recreate-ingest.sh`, ou diagnostiquer un consumer arrêté / pipeline désactivé.

Note : `a72c408a` **est** ancêtre de `f3e42b94` (main actuel) — prouvé en local via `git merge-base --is-ancestor`. Cela ne remplace pas l’inspection runtime du consumer.

## Classement des Issues unresolved restantes (python-flask)

### P0 — GPS / déploiement Kafka

| Issue | Action |
| --- | --- |
| DE / DF | Contrôle OCI + ancestry puis Resolve **ou** redeploy Kafka P0 |

### P1 — infra Kafka / realtime

| Issue | Lecture | Action |
| --- | --- | --- |
| DV / DW / DX | DNS/bootstrap Kafka (~4j) | Vérifier cluster / filtres `before_send` ; pas d’idempotence |
| E0 / E1 | rebalance / heartbeat ws-service (~15h) | Bruit consumer ; confirmer filtres existants |

### P1 — métier / scripts one-shot

| Issue | Lecture | Action |
| --- | --- | --- |
| DN | QR-Bill adresse débitrice (CP manquant) | Validation adresse côté facturation |
| AZ / CE / CJ / BM / DR / DT / DS / DP / DM / DH / CK | Souvent `__main__` / scripts / schéma | Corriger scripts ou ignore si hors API live |
| DG | `from __future__` mal placé (invoice clinique) | Fix import order (déjà suivi séparément si besoin) |
| DQ | N+1 billing opportunities | Perf facturation |
| DY | table `invoice_line` absente | Migrations / env script |
| DD | colonne `message.audio_url` | Migration manquante ou code legacy |
| 8Y / 8W | rotation secrets échouée | Ops Vault |

### P2 — hygiène

| Issue | Lecture | Action |
| --- | --- | --- |
| BV | session Socket.IO invalide | Bruit attendu / filtre |
| DJ / DK | 403 offers company | Authz attendue ou UX client |
| CT | JWT expiré institution | Bruit auth |

## Observabilité

Après le prochain **Build & Deploy**, les événements Sentry Flask/Celery doivent porter `release=<sha40>`.

Après le prochain **deploy-kafka-p0**, les erreurs du consumer doivent porter le `SOURCE_SHA` du recreate (pas forcément celui du dernier deploy backend).

✅ **Implémenté** : Resolve DZ ; `GIT_SHA`/`SENTRY_RELEASE` backend + surcharge Kafka ; test concurrence session-resume ; doc de triage. DE/DF volontairement non résolus sans preuve OCI prod.
