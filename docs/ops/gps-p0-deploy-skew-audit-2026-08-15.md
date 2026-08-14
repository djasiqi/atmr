# Audit read-only — skew images prod & chaîne deploy (2026-08-15)

```text
STATUT                     = AUDIT FAIT ✅ (aucune mutation)
BRANCHE RELEASE            = toujours NO-GO
ALEMBIC / DEPLOY / PURGE   = NO-GO
BASE SNAPSHOT              = gps-p0-prod-snapshot-2026-08-15.md
```

---

## A. Trois générations Git — ce qu’elles sont

| SHA (12) | Date (local commit) | Message | Relation à `927640a0` |
|----------|---------------------|---------|------------------------|
| **`390076efc61c`** | 2026-08-11 12:34 +0200 | `style: ruff format … P0-E / session` | **ancêtre** de `927640a0` (6 commits derrière) |
| **`16fd3e52418d`** | 2026-08-11 21:40 +0200 | `test(driver): aligner bridge sur fenêtre P0-F TIME` | **ancêtre** (1 commit derrière `927640a0`) |
| **`927640a0995a`** | 2026-08-11 22:58 +0200 | `fix(tracking): sérialiser register_tracking_session…` | tip API / `.env` `GIT_SHA` |

```text
390076ef ──► … ──► 16fd3e52 ──► 927640a0
   ↑ consumer+outbox Up          ↑ API/celery/ws Up
                    ↑ fanout/dlq Created (HOLD)
```

Ce n’est **pas** trois forks divergents : c’est une **ligne de commits** avec des **services pinnés à des points différents** de la même histoire.

### Contenu tracking entre `390076ef` et `927640a0`

Diff tracking notable :

- `session_registry.py` (sérialisation register — le tip `927640a0`)
- `presence_window.py`, `time_contract.py`, `location_event_id.py`
- Commits intermédiaires P0-F UI / docs réalignement Kafka

→ L’API tip a des correctifs session que le consumer `390076ef` **n’a pas**.

---

## B. Chaîne GitHub build/deploy — pourquoi le skew

### Deux pipelines distincts (volontaire)

| Pipeline | Workflow | Ce qu’il déploie | Kafka consumers ? |
|----------|----------|------------------|-------------------|
| **Build & Deploy** | `.github/workflows/deploy.yml` → `deploy-production.sh` | API, celery, flower, ws, stack « non-kafka » | **NON** — contrat explicite |
| **Deploy Kafka P0** | `.github/workflows/deploy-kafka-p0.yml` → `ops-tracking-p0-recreate-ingest.sh` | **ingest** (`tracking-kafka-consumer`) sur `SOURCE_SHA` + digest | **OUI** — autorité HOLD |

Preuve contrat API (`scripts/deploy-production.sh`) :

```text
Ce script ne doit JAMAIS inclure --profile kafka.
Ce script ne doit JAMAIS fusionner -f docker-compose.kafka*.yml.
```

Donc un deploy API à `927640a0` **ne met pas à jour** consumer/outbox/fanout.  
Le réalignement Kafka est un **workflow séparé** (`Deploy Kafka P0`), avec confirm `tracking-p0-recreate`, compose :

```text
-f docker-compose.production.yml
-f docker-compose.kafka.yml
-f docker-compose.kafka.atmr-network.yml
-f docker-compose.kafka.p0-hold.yml
```

### Chronologie ops cohérente avec le snapshot

1. **2026-08-11 matin** — Réalignement P0-E documenté : consumer = outbox = backend = **`390076ef`** ([gps-p0e-kafka-align-execution-2026-08-11.md](gps-p0e-kafka-align-execution-2026-08-11.md)). Fanout/DLQ **stopped (HOLD)**.
2. **Plus tard le même jour** — Build & Deploy (et/ou déploiements) avancent API/ws/celery vers **`16fd3e52`** puis **`927640a0`** **sans** recreate Kafka P0.
3. **Résultat observé** — API tip `927640a0` ; ingest reste **`390076ef`** (dernier recreate P0 réussi) ; fanout/dlq images **`16fd3e52`** leftover + **non running**.

```text
CONCLUSION B
→ Pas un pin « aléatoire » de trois tags indépendants.
→ Architecture HOLD : API et ingest ont des autorités de deploy séparées.
→ Le skew actuel = API a bougé ; recreate Kafka P0 n’a PAS suivi jusqu’à 927640a0.
→ Risque produit : correctifs session (927640a0) absents du consumer (390076ef).
```

---

## C. Fanout `flag=true` mais conteneurs `Created` — **prouvé intentionnel**

### Triple preuve

1. **Override compose HOLD** — `docker-compose.kafka.p0-hold.yml` :

```yaml
tracking-processed-fanout:
  environment:
    TRACKING_PROCESSED_FANOUT_ENABLED: "false"
```

2. **Runtime inspect** (read-only, 2026-08-15) :

```text
fanout Status=created
Image=…:sha-16fd3e52418d
TRACKING_PROCESSED_FANOUT_ENABLED=false   # effectif dans le conteneur
```

3. **Script recreate ingest** — `ops-tracking-p0-recreate-ingest.sh` :

- assert config : `tracking-processed-fanout.TRACKING_PROCESSED_FANOUT_ENABLED=false`
- `compose stop tracking-processed-fanout kafka-dlq-consumer`
- recreate **uniquement** `tracking-kafka-consumer`
- fail si fanout/dlq encore `running` après

Docs pipeline : pendant validation / HOLD, **ne pas activer** fanout — autorité `processed.v2 → ws-service` ([gps-tracking-pipeline.md](gps-tracking-pipeline.md)).

### Pourquoi `.env.production` dit `TRACKING_PROCESSED_FANOUT_ENABLED=true` ?

Le flag **global** reste `true` (cohérence des 4 flags Kafka clients), mais le **HOLD override** force `false` **sur le service fanout** et les conteneurs sont **stoppés / non démarrés**.  
Ce n’est **pas** une contradiction runtime : priorité `p0-hold.yml` > interpolations env (commentaire du fichier).

```text
CONCLUSION C
→ Fanout Created + ENABLED=false = HOLD P0 volontaire, pas un accident de flag.
→ Ne pas « réparer » en up fanout sans GO ops explicite (hors scope release P0 GPS).
```

---

## D. Migration `25ce766952e2` vs packs P0 cherry-pick

| Question | Réponse |
|----------|---------|
| Prod `alembic current` | `9b6638784019` |
| `25ce766952e2` en prod | **ABSENT** |
| `capture_id` dans tree `927640a0` | **ABSENT** (models/migrations) |
| Les 5 commits P0 **modifient-ils** migration/models `capture_id` ? | **NON** (patch SERVER/OBS/A/B/CLIENT sans ces fichiers) |
| `git grep capture_id` sur tip firewall | Oui — **parce que** l’historique branche contient déjà `e14cfbea` ; **≠** contenu du cherry-pick |

```text
RECOMMANDATION G1 (sous réserve dry-run cherry-pick)
→ Ne PAS embarquer 25ce766952e2 dans la release GPS P0 isolée
→ P0 cherry-pick depuis 927640a0 ne dépend pas de capture_id
→ Confirmer au dry-run : aucun conflit n’introduit models/migration capture_id
→ ALEMBIC PROD reste NO-GO pour cette release sauf preuve contraire au dry-run
```

---

## E. Implications pour la release P0

```text
prod-current-SHA (API)     = 927640a0995a…   ← base branche release
ingest réellement actif      = 390076ef…       ← à traiter dans le plan deploy
fanout                       = HOLD (ne pas up)

AVANT release/gps-p0-2026-08-15
→ documenter dans le plan deploy :
   1) deploy API (927640a0 + cherry-picks) via Build & Deploy
   2) recreate Kafka P0 ingest sur LE MÊME SHA tip release (digest)
   3) fanout reste HOLD sauf GO séparé
→ sinon on recrée le skew API≠consumer
```

```text
BRANCHE RELEASE              = toujours NO-GO (attendre GO après lecture de cet audit)
TAG / ALEMBIC / DEPLOY       = NO-GO
```

---

## Gates mis à jour

| Gate | Statut | Note |
|------|--------|------|
| G0 commits P0 | ✅ | |
| G0 release TIP | ⏳ | Base = `927640a0` ; pas encore créée |
| G1 prod current | ✅ | `9b6638784019` |
| G1 migration release | **✅ orienté** | P0 isolé → **exclure** `25ce766952e2` (confirmer dry-run) |
| G2 snapshot | ✅ | |
| G2 image alignment | **✅ expliqué** | HOLD + dual pipeline ; skew = API avancée sans recreate P0 |
| G3 N/N-1 | ⏳ | |
| G4 rollback | ❌ | `previous-release.json` absent |
| G5 baseline | ✅ partielle | |

```text
PROD DEPLOY = NO-GO ❌
```

## Implémentation

✅ **Implémenté** : audit A/B/C (SHAs, workflows, HOLD fanout prouvé par inspect) ; orientation G1 sans `capture_id` pour cherry-pick P0 ; plan deploy anti-skew documenté.  
**Reste à faire** : GO explicite dry-run cherry-pick / création `release/gps-p0-2026-08-15` ; fabriquer stratégie `previous-release.json` (G4).
