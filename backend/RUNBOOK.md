# Runbook — Gestion des catastrophes

Ce document décrit les procédures d'astreinte pour gérer les scénarios de catastrophe.

## Table des matières

1. [OSRM Down](#osrm-down)
2. [DB Read-Only](#db-read-only)
3. [Pic de charge](#pic-de-charge)
4. [Réseau flaky](#réseau-flaky)
5. [Killswitch](#killswitch)
6. [Healthchecks Kubernetes](#healthchecks-kubernetes)

---

## OSRM Down

### Symptômes

- Erreurs 502/503 sur les appels OSRM
- Timeouts sur les calculs de distance/temps
- Métriques `osrm_availability` = 0%

### Actions immédiates (0-5 min)

1. **Vérifier l'état du service**

   ```bash
   docker-compose ps osrm
   docker-compose logs osrm --tail 50
   ```

2. **Vérifier la connectivité**

   ```bash
   curl http://osrm:5000/health
   ```

3. **Si OSRM est down**
   - Le système doit utiliser le cache OSRM
   - Les nouvelles requêtes peuvent échouer gracieusement
   - **Ne PAS redémarrer immédiatement** (attendre diagnostic)

### Actions de récupération (5-15 min)

1. **Redémarrer OSRM**

   ```bash
   docker-compose restart osrm
   ```

2. **Vérifier que le service remonte**

   ```bash
   # Attendre 30s
   docker-compose logs osrm -f
   ```

3. **Vérifier que les requêtes reprennent**
   ```bash
   # Vérifier les métriques dans Flower ou logs
   curl http://api:5000/api/health
   ```

### RTO (Recovery Time Objective)

- **Objectif**: ≤ 30 secondes après restauration OSRM
- **Acceptable**: ≤ 2 minutes

---

## DB Read-Only

### Symptômes

- Erreurs SQL "read-only" dans les logs
- Écritures échouent (POST/PUT/PATCH)
- Lectures fonctionnent normalement

### Actions immédiates (0-5 min)

1. **Vérifier l'état de la DB**

   ```bash
   docker-compose exec postgres psql -U postgres -d atmr -c "SHOW transaction_read_only;"
   ```

2. **Vérifier les logs**

   ```bash
   docker-compose logs postgres --tail 50
   ```

3. **Si DB est en read-only**
   - Les lectures continuent (système partiellement opérationnel)
   - Les écritures sont rejetées (erreur HTTP 503)
   - **Activer le mode maintenance** si nécessaire (voir Killswitch)

### Actions de récupération (5-15 min)

1. **Passer la DB en read-write**

   ```bash
   docker-compose exec postgres psql -U postgres -d atmr -c "ALTER DATABASE atmr SET default_transaction_read_only = off;"
   ```

2. **Vérifier que les écritures reprennent**
   ```bash
   # Tester un endpoint POST/PUT
   curl -X POST http://api:5000/api/test-endpoint
   ```

### RTO

- **Objectif**: ≤ 5 minutes
- **Acceptable**: ≤ 10 minutes

---

## Pic de charge

### Symptômes

- Latence élevée (P95 > 5s)
- Taux d'erreur > 5%
- CPU/Memory proches de 100%
- Files d'attente Celery qui s'accumulent

### Actions immédiates (0-5 min)

1. **Vérifier les métriques**

   ```bash
   # Vérifier les ressources
   docker stats

   # Vérifier les logs d'erreur
   docker-compose logs api --tail 100 | grep ERROR
   ```

2. **Activer le rate limiting** (si disponible)

   ```bash
   # Via variable d'environnement ou config
   export ENABLE_RATE_LIMITING=true
   docker-compose restart api
   ```

3. **Scaler les services** (si possible)
   ```bash
   docker-compose up -d --scale celery-worker=5
   ```

### Actions de récupération (5-30 min)

1. **Identifier la source du pic**

   - Vérifier les logs pour patterns
   - Vérifier les métriques d'utilisation

2. **Optimiser**

   - Augmenter les ressources (CPU/Memory)
   - Ajouter des workers Celery
   - Activer le cache si disponible

3. **Si le pic persiste**
   - Activer le mode dégradé (voir Killswitch)
   - Réduire la fonctionnalité non-critique

### RTO

- **Objectif**: Système reste opérationnel avec ≥ 95% de succès
- **Acceptable**: ≥ 90% de succès, latence P95 < 10s

---

## Réseau flaky

### Symptômes

- Timeouts intermittents
- Erreurs de connexion
- Latence variable
- Perte de paquets

### Actions immédiates (0-5 min)

1. **Vérifier la connectivité**

   ```bash
   # Ping test
   ping -c 10 osrm
   ping -c 10 postgres
   ping -c 10 redis
   ```

2. **Vérifier les métriques réseau**

   ```bash
   # Vérifier packet loss
   # Vérifier latence dans les logs
   docker-compose logs api | grep -i timeout
   ```

3. **Vérifier les retries**
   - Les services doivent retry automatiquement
   - Vérifier que les backoffs exponentiels sont actifs

### Actions de récupération (5-30 min)

1. **Si réseau local flaky**
   - Vérifier les liens Docker
   - Redémarrer les réseaux si nécessaire
2. **Si réseau externe flaky**
   - Vérifier les providers (OSRM externe, APIs tierces)
   - Activer les timeouts plus longs si nécessaire
   - Activer le cache si disponible

### RTO

- **Objectif**: Pas de perte de données, retries automatiques
- **Acceptable**: Dégradation gracieuse, quelques erreurs temporaires

---

## Killswitch

Le killswitch permet d'activer rapidement un mode de maintenance ou dégradé.

### Activation du killswitch

1. **Via variable d'environnement**

   ```bash
   export MAINTENANCE_MODE=true
   docker-compose restart api
   ```

2. **Via ChatOps** (voir `chatops/killswitch.py`)
   ```bash
   python -m chatops.killswitch enable --reason "OSRM down, activating maintenance"
   ```

### Mode maintenance

En mode maintenance:

- Toutes les requêtes API retournent HTTP 503
- Message: "Service en maintenance - Merci de réessayer plus tard"
- Les tâches Celery continuent (si possible)
- Les logs sont conservés

### Mode dégradé

En mode dégradé:

- Fonctionnalités non-critiques désactivées
- Rate limiting agressif
- Cache prioritaire
- Logs simplifiés

### Désactivation du killswitch

```bash
export MAINTENANCE_MODE=false
docker-compose restart api
```

---

## Healthchecks Kubernetes

### Endpoints disponibles

L'application expose deux endpoints distincts pour Kubernetes :

1. **`/health`** - Liveness probe (simple)

   - Retourne : `{"status": "ok"}`
   - Status code : `200`
   - Usage : Vérifier que le processus est vivant
   - Ne vérifie **pas** les dépendances

2. **`/ready`** - Readiness probe (dépendances critiques)
   - Retourne : `{"status": "ready", "checks": {"database": "ok", "redis": "ok"}}`
   - Status code : `200` si prêt, `503` si non prêt
   - Usage : Vérifier que le pod peut recevoir du trafic
   - Vérifie : Database + Redis

### Configuration Kubernetes recommandée

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
    - name: api
      livenessProbe:
        httpGet:
          path: /health
          port: 5000
        initialDelaySeconds: 30
        periodSeconds: 10
        timeoutSeconds: 5
        failureThreshold: 3

      readinessProbe:
        httpGet:
          path: /ready
          port: 5000
        initialDelaySeconds: 10
        periodSeconds: 5
        timeoutSeconds: 3
        failureThreshold: 2
        successThreshold: 1
```

### Comportement

- **`/health`** : Toujours retourne `200` si le processus Flask est actif
- **`/ready`** : Retourne `503` si :
  - La base de données n'est pas accessible
  - Redis n'est pas configuré ou inaccessible

### Dépannage

**Problème : Pod en état `NotReady`**

```bash
# Vérifier manuellement
curl http://localhost:5000/ready

# Voir les checks détaillés
curl http://localhost:5000/health/detailed
```

**Solutions communes :**

- Si `database: error` → Vérifier connexion DB
- Si `redis: not_configured` → Vérifier variable `REDIS_URL`
- Si `redis: error` → Vérifier que Redis est accessible

### Notes

- `/health` reste simple pour éviter les redémarrages inutiles
- `/ready` est strict pour éviter le trafic vers un pod non fonctionnel
- `/health/detailed` reste disponible pour diagnostic mais n'est pas utilisé par K8s

---

## Plan de backout

### Vue d'ensemble

Le plan de backout permet de revenir rapidement à une version fonctionnelle en cas de problème critique après un déploiement ou une migration.

**RTO (Recovery Time Objective) :** < 5 minutes  
**RPO (Recovery Point Objective) :** 0 (pas de perte de données)

### ⚠️ Conditions de déclenchement

Déclencher un backout si :

- ❌ Erreurs 5xx en masse (> 10% des requêtes)
- ❌ Base de données corrompue après migration
- ❌ Service inaccessible
- ❌ Violation de SLO critique
- ❌ Incident de sécurité détecté

**Ne PAS faire de backout si :**

- ⚠️ Problème localisé à quelques utilisateurs
- ⚠️ Erreurs transitoires (< 1% des requêtes)
- ⚠️ Pas de confirmation de cause racine

### 1. Rollback migration Alembic

#### A. Identifier la migration problématique

```bash
# Voir la version actuelle de la DB
cd backend
flask db current

# Voir l'historique complet des migrations
flask db history

# Voir les détails d'une migration spécifique
flask db history --verbose | grep <revision_id>
```

#### B. Rollback d'une migration

```bash
# Rollback d'une seule migration (la dernière)
cd backend
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
export FLASK_CONFIG="production"
export SECRET_KEY="your-secret"
export JWT_SECRET_KEY="your-jwt-secret"

# Downgrade
flask db downgrade -1

# Vérifier l'état
flask db current
```

#### C. Rollback vers version spécifique

```bash
# Lister toutes les révisions disponibles
flask db history | head -20

# Rollback vers une révision spécifique
flask db downgrade <revision_id>

# Exemple: rollback vers la révision avant le dernier changement
flask db downgrade abc123def456
```

#### D. Rollback d'urgence (multiples migrations)

```bash
# Rollback de 3 migrations en une fois
flask db downgrade -3

# ⚠️ ATTENTION: Vérifier l'intégrité des données après un rollback multiple
```

#### E. Vérifier l'état après rollback

```bash
# 1. Vérifier version DB
flask db current

# 2. Vérifier tables existantes
psql -U atmr -d atmr -c "\dt"

# 3. Vérifier santé de l'API
curl http://localhost:5000/health
curl http://localhost:5000/ready

# 4. Vérifier logs d'erreur
docker-compose logs api --tail 100 | grep -i error

# 5. Tester une requête critique
curl -H "Authorization: Bearer $TOKEN" http://localhost:5000/api/bookings
```

### 2. Rollback déploiement Docker

#### A. Identifier version précédente

```bash
# Lister toutes les images disponibles
docker images | grep atmr-backend

# Lister les tags disponibles
docker images atmr-backend --format "table {{.Tag}}\t{{.CreatedAt}}"

# Voir l'historique Git pour identifier la version précédente
cd /path/to/atmr
git log --oneline -10
git tag -l "v*" | tail -5
```

#### B. Rollback image Docker (docker-compose)

```bash
# 1. Arrêter l'application actuelle
docker-compose down

# 2. Identifier et taguer l'image précédente
docker tag atmr-backend:<previous-tag> atmr-backend:latest
# OU récupérer depuis registry
docker pull registry.example.com/atmr-backend:v1.2.3
docker tag registry.example.com/atmr-backend:v1.2.3 atmr-backend:latest

# 3. Redémarrer avec l'image précédente
docker-compose up -d

# 4. Vérifier la santé
docker-compose ps
curl http://localhost:5000/health
curl http://localhost:5000/ready
```

#### C. Rollback image Docker (Kubernetes)

```bash
# 1. Identifier l'image précédente
kubectl get deployments -n atmr -o jsonpath='{.items[*].spec.template.spec.containers[*].image}'

# 2. Rollback vers version précédente
kubectl set image deployment/atmr-api api=registry.example.com/atmr-backend:v1.2.3 -n atmr

# 3. Ou utiliser rollout history
kubectl rollout history deployment/atmr-api -n atmr
kubectl rollout undo deployment/atmr-api -n atmr

# 4. Surveiller le rollback
kubectl rollout status deployment/atmr-api -n atmr
kubectl get pods -n atmr -w
```

#### D. Vérifier santé après rollback

```bash
# Healthchecks
curl http://localhost:5000/health
curl http://localhost:5000/ready

# Métriques Prometheus
curl http://localhost:5000/prometheus/metrics-http | grep http_request_errors_total

# Logs récents
docker-compose logs api --tail 100
# ou
kubectl logs -f deployment/atmr-api -n atmr --tail 100

# Tester requêtes critiques
curl -X GET http://localhost:5000/api/bookings \
  -H "Authorization: Bearer $JWT_TOKEN"

# Vérifier base de données accessible
docker-compose exec api flask db current
```

### 3. Rollback code (Git)

```bash
# 1. Identifier le commit problématique
git log --oneline -10

# 2. Créer une branche de hotfix depuis le commit précédent
git checkout -b hotfix/rollback-$(date +%Y%m%d)
git reset --hard <previous-commit-hash>

# 3. Push et merge emergency
git push origin hotfix/rollback-$(date +%Y%m%d)
# Merge via PR d'urgence ou directement:
git checkout main
git merge hotfix/rollback-$(date +%Y%m%d) --no-ff

# 4. Redéployer
docker-compose build --no-cache
docker-compose up -d
```

### 4. Procédures de test de backout

#### Test mensuel (Recommandé)

**Fréquence:** Le premier mardi de chaque mois  
**Environnement:** Staging uniquement

#### Test rollback migration

```bash
# 1. Appliquer une migration de test
flask db upgrade head

# 2. Vérifier l'état
flask db current

# 3. Rollback
flask db downgrade -1

# 4. Mesurer le temps (objectif: < 30 secondes)
time flask db downgrade -1

# 5. Vérifier intégrité
flask db current
psql -U atmr -d atmr -c "SELECT COUNT(*) FROM information_schema.tables;"
```

#### Test rollback déploiement

```bash
# 1. Déployer version de test
docker-compose up -d

# 2. Vérifier fonctionnement
curl http://localhost:5000/health

# 3. Rollback (mesurer le temps)
START=$(date +%s)
docker-compose down
docker-compose up -d
END=$(date +%s)
DURATION=$((END - START))
echo "Temps de rollback: ${DURATION}s (objectif: < 5 min)"

# 4. Vérifier santé
curl http://localhost:5000/ready
```

#### Checklist de validation post-rollback

- [ ] ✅ API répond sur `/health` (200)
- [ ] ✅ API répond sur `/ready` (200)
- [ ] ✅ Base de données accessible
- [ ] ✅ Redis accessible
- [ ] ✅ Aucune erreur 5xx dans les logs
- [ ] ✅ Tests de smoke passent
- [ ] ✅ Métriques Prometheus normales
- [ ] ✅ Pas de perte de données critiques

### 5. Communication post-rollback

Après un rollback en production :

1. **Notification immédiate :**

   - Alert Slack #incidents
   - Page on-call engineer
   - Notifier équipe backend

2. **Post-mortem (dans les 24h) :**

   - Documenter cause du problème
   - Analyser pourquoi le rollback était nécessaire
   - Identifier améliorations préventives
   - Mettre à jour ce RUNBOOK si nécessaire

3. **Mesures correctives :**
   - Corriger le problème dans la version suivante
   - Améliorer tests/validation avant déploiement
   - Documenter changements requis

### 6. RTO (Recovery Time Objective) et métriques

**Objectifs :**

- **Rollback migration :** < 30 secondes
- **Rollback déploiement :** < 5 minutes
- **Communication incident :** < 2 minutes

**Métriques à surveiller après rollback :**

```bash
# Taux d'erreur
curl http://localhost:5000/prometheus/metrics-http | grep http_requests_total

# Latence
curl http://localhost:5000/prometheus/metrics-http | grep http_request_duration_seconds

# Disponibilité endpoints critiques
curl http://localhost:5000/health
curl http://localhost:5000/api/bookings
```

### 7. Contacts d'urgence

En cas de besoin d'aide pour un rollback :

- **Lead Backend :** [Contact]
- **Lead DevOps/SRE :** [Contact]
- **On-call engineer :** PagerDuty
- **Base de données :** [Contact DBA]

---

## Sauvegarde et restauration de base de données

### Vue d'ensemble

Les sauvegardes PostgreSQL sont essentielles pour garantir la récupération en cas de perte de données.

**RPO (Recovery Point Objective) :** < 15 minutes  
**RTO (Recovery Time Objective) :** < 30 minutes  
**Fréquence de backup :** Toutes les heures en production

### Scripts disponibles

- **`scripts/backup_db.sh`** : Créer un backup de la base de données
- **`scripts/restore_db.sh`** : Restaurer depuis un backup
- **`scripts/test_backup_restore.sh`** : Tester le processus complet

### 1. Créer un backup

#### Utilisation basique

```bash
# Backup dans le répertoire par défaut (./backups)
./scripts/backup_db.sh

# Backup dans un répertoire spécifique
./scripts/backup_db.sh /path/to/backups
```

#### Fonctionnalités

- ✅ Crée deux formats : `.dump` (custom, rapide) et `.sql` (texte, lisible)
- ✅ Timestamp automatique dans le nom de fichier
- ✅ Liens symboliques `latest.dump` et `latest.sql`
- ✅ Compatible Docker Compose et installation locale
- ✅ Affiche la taille du backup

#### Exemple de sortie

```
🔄 Backup base de données PostgreSQL...
   Database: atmr
   Host: postgres:5432
   Mode: Docker Compose

✅ Backup créé avec succès!
   📦 Format custom: backups/atmr_backup_20250127_143022.dump (45M)
   📄 Format SQL: backups/atmr_backup_20250127_143022.sql (67M)
   🔗 Liens: backups/latest.dump, backups/latest.sql
```

### 2. Restaurer depuis un backup

#### Utilisation

```bash
# Restauration interactive (demande confirmation)
./scripts/restore_db.sh backups/atmr_backup_20250127_143022.dump

# Restauration forcée (sans confirmation)
./scripts/restore_db.sh backups/latest.dump --force

# Utiliser le dernier backup
./scripts/restore_db.sh backups/latest.dump --force
```

#### ⚠️ ATTENTION

La restauration **écrase complètement** la base de données actuelle. Assurez-vous de :

1. ✅ Avoir un backup récent avant restauration
2. ✅ Vérifier que le fichier de backup est valide
3. ✅ Avoir testé la procédure en staging

#### Formats supportés

- **`.dump`** : Format custom PostgreSQL (recommandé, plus rapide)
- **`.sql`** : Format SQL texte (plus lisible, plus lent)

Le script détecte automatiquement le format.

### 3. Tests de backup/restore

#### Test automatique

```bash
# Lancer le test complet
./scripts/test_backup_restore.sh
```

Le script effectue :

1. ✅ Création d'un backup
2. ✅ Ajout de données de test
3. ✅ Restauration depuis le backup
4. ✅ Vérification que les données de test ont été supprimées
5. ✅ Calcul des métriques RTO/RPO

#### Exemple de sortie

```
==========================================
🧪 TEST BACKUP/RESTORE PostgreSQL
==========================================

📦 Étape 1/4: Création du backup...
✅ Backup créé: backups/atmr_backup_20250127_143022.dump (12s)

📝 Étape 2/4: Création de données de test...
✅ Données de test créées (timestamp: 1706368222)

🔄 Étape 3/4: Restauration depuis le backup...
✅ Restauration terminée (15s)

🔍 Étape 4/4: Vérification de l'intégrité...
✅ Test réussi: données restaurées correctement
   📊 Tables restaurées: 42

==========================================
✅ TEST BACKUP/RESTORE RÉUSSI
==========================================

📊 Métriques:
   ⏱️  Temps de backup: 12s
   ⏱️  Temps de restauration: 15s
   ⏱️  Temps total: 27s

🎯 Objectifs:
   RTO (Restore Time Objective): 15s (objectif: < 30 min ✅)
   RPO (Recovery Point Objective): ~12s (objectif: < 15 min ✅)
```

### 4. Backup automatisé (Production)

#### Crontab (recommandé)

```bash
# Backup toutes les heures
0 * * * * /path/to/atmr/scripts/backup_db.sh /var/backups/atmr

# Backup quotidien à 2h du matin
0 2 * * * /path/to/atmr/scripts/backup_db.sh /var/backups/atmr/daily
```

#### Backup vers stockage distant

```bash
# Exemple: Backup + upload vers S3
./scripts/backup_db.sh /tmp/backups
aws s3 cp /tmp/backups/atmr_backup_*.dump s3://atmr-backups/ --recursive

# Nettoyer les anciens backups locaux (garder 7 jours)
find /tmp/backups -name "atmr_backup_*.dump" -mtime +7 -delete
```

### 5. Vérifications post-restauration

Après une restauration, vérifier :

```bash
# 1. Santé de l'API
curl http://localhost:5000/health
curl http://localhost:5000/ready

# 2. Nombre de tables
docker-compose exec postgres psql -U atmr -d atmr -c "\dt" | wc -l

# 3. Vérifier quelques données critiques
docker-compose exec postgres psql -U atmr -d atmr -c "SELECT COUNT(*) FROM company;"
docker-compose exec postgres psql -U atmr -d atmr -c "SELECT COUNT(*) FROM booking;"

# 4. Vérifier logs d'erreur
docker-compose logs api --tail 100 | grep -i error

# 5. Tester une requête API
curl -H "Authorization: Bearer $TOKEN" http://localhost:5000/api/bookings
```

### 6. Rétention des backups

**Recommandations :**

- **Backups horaires :** Garder 24 heures (24 backups)
- **Backups quotidiens :** Garder 7 jours (7 backups)
- **Backups hebdomadaires :** Garder 4 semaines (4 backups)
- **Backups mensuels :** Garder 12 mois (12 backups)

**Script de nettoyage :**

```bash
#!/bin/bash
# Nettoyer backups > 7 jours
BACKUP_DIR="/var/backups/atmr"
find "$BACKUP_DIR" -name "atmr_backup_*.dump" -mtime +7 -delete
find "$BACKUP_DIR" -name "atmr_backup_*.sql" -mtime +7 -delete
```

### 7. Tests mensuels de restauration

**Fréquence :** Le premier mercredi de chaque mois  
**Environnement :** Staging uniquement

```bash
# 1. Lancer le test
./scripts/test_backup_restore.sh

# 2. Documenter les résultats
# - Temps de backup (objectif: < 5 min)
# - Temps de restauration (objectif: < 30 min)
# - Taille du backup
# - Intégrité des données vérifiée

# 3. Si test échoue, investigation immédiate
```

### 8. Troubleshooting

**Problème : Backup échoue**

```bash
# Vérifier connexion PostgreSQL
docker-compose exec postgres pg_isready -U atmr

# Vérifier espace disque
df -h

# Vérifier permissions
ls -la backups/
```

**Problème : Restauration échoue**

```bash
# Vérifier format du backup
file backups/atmr_backup_*.dump

# Vérifier intégrité
pg_restore --list backups/atmr_backup_*.dump | head -20

# Vérifier espace disque disponible
df -h
```

**Problème : Données manquantes après restauration**

1. Vérifier que le bon backup a été utilisé
2. Vérifier la date/heure du backup
3. Vérifier les logs de restauration pour erreurs
4. Tester avec un autre backup si disponible

---

## Contacts

- **On-call engineer**: Voir rotation dans PagerDuty/OpsGenie
- **Lead DevOps**: [Contact]
- **CTO**: [Contact]

## Escalade

1. **Niveau 1** (0-15 min): On-call engineer
2. **Niveau 2** (15-30 min): Lead DevOps
3. **Niveau 3** (30+ min): CTO

---

_Dernière mise à jour: 2025-01-28_
