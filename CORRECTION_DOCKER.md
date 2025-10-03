# Correction des problèmes Docker - ATMR

## Problèmes identifiés et corrigés

### 1. **Erreur de configuration de base de données (atmr-api)**
**Erreur originale:**
```
RuntimeError: Either 'SQLALCHEMY_DATABASE_URI' or 'SQLALCHEMY_BINDS' must be set.
```

**Cause:** 
- Les variables d'environnement nécessaires (`DATABASE_URL`, `SECRET_KEY`, `JWT_SECRET_KEY`) n'étaient pas définies dans `docker-compose.yml`
- Le fichier `.env` n'est pas automatiquement chargé dans les conteneurs Docker

**Solution appliquée:**
- Ajout de toutes les variables d'environnement nécessaires dans `docker-compose.yml` pour tous les services
- Ajout d'un volume partagé `db-data` pour persister la base de données SQLite

### 2. **Erreur de contexte d'application (atmr-celery-worker)**
**Erreur originale:**
```
RuntimeError: Working outside of application context.
```

**Cause:**
- Les workers Celery n'avaient pas accès au contexte Flask lors de l'exécution des tâches
- La classe `FlaskTask` n'était pas correctement appliquée aux workers

**Solution appliquée:**
- Modification de `celery_app.py` pour créer automatiquement une instance Flask pour les workers
- Implémentation d'une classe `ContextTask` qui garantit que toutes les tâches s'exécutent dans le contexte Flask
- Définition de `celery.Task = ContextTask` au niveau global pour tous les workers

## Fichiers modifiés

### 1. `docker-compose.yml`
**Modifications:**
- Ajout des variables d'environnement manquantes pour tous les services :
  - `DATABASE_URL=sqlite:////app/production.db`
  - `SECRET_KEY`
  - `JWT_SECRET_KEY`
  - `MAIL_SERVER`, `MAIL_PORT`, `MAIL_USERNAME`, `MAIL_PASSWORD`, etc.
- Ajout d'un volume `db-data` pour persister la base de données
- Ajout de dépendances entre services (`celery-worker` et `celery-beat` dépendent maintenant de `api`)

### 2. `backend/celery_app.py`
**Modifications:**
- Ajout d'une fonction `get_flask_app()` qui crée l'application Flask pour les workers
- Création d'une classe `ContextTask` qui enveloppe toutes les tâches dans le contexte Flask
- Définition de `celery.Task = ContextTask` au niveau global
- Amélioration de la gestion du contexte Flask

## Instructions de déploiement

### Étape 1 : Arrêter les conteneurs existants
```bash
cd /path/to/atmr
docker-compose down
```

### Étape 2 : Nettoyer les volumes (optionnel, si vous voulez repartir de zéro)
```bash
docker-compose down -v
```

### Étape 3 : Reconstruire les images
```bash
docker-compose build --no-cache
```

### Étape 4 : Démarrer les services
```bash
docker-compose up -d
```

### Étape 5 : Vérifier les logs
```bash
# Vérifier tous les services
docker-compose logs -f

# Ou vérifier un service spécifique
docker-compose logs -f api
docker-compose logs -f celery-worker
docker-compose logs -f celery-beat
```

### Étape 6 : Initialiser la base de données (si nécessaire)
Si c'est la première fois ou si vous avez nettoyé les volumes :
```bash
# Accéder au conteneur API
docker-compose exec api bash

# Initialiser la base de données
flask db upgrade

# Créer un utilisateur admin (si nécessaire)
python add_admin.py

# Quitter le conteneur
exit
```

## Vérification du bon fonctionnement

### 1. Vérifier que l'API répond
```bash
curl http://localhost:5000/health
```

### 2. Vérifier Flower (monitoring Celery)
Ouvrir dans le navigateur : http://localhost:5555

### 3. Vérifier les logs des services
```bash
# API
docker-compose logs api | tail -20

# Celery Worker
docker-compose logs celery-worker | tail -20

# Celery Beat
docker-compose logs celery-beat | tail -20

# Redis
docker-compose logs redis | tail -20

# OSRM
docker-compose logs osrm | tail -20
```

## Services et ports

| Service | Port | Description |
|---------|------|-------------|
| API | 5000 | API Flask principale |
| Flower | 5555 | Interface de monitoring Celery |
| Redis | 6379 | Broker de messages et cache |
| OSRM | 5000 (interne) | Moteur de routage |

## Dépannage

### Si l'API ne démarre toujours pas
1. Vérifier que toutes les variables d'environnement sont correctement définies :
```bash
docker-compose exec api env | grep -E "DATABASE_URL|SECRET_KEY|JWT_SECRET_KEY"
```

2. Vérifier les permissions sur le volume de la base de données :
```bash
docker-compose exec api ls -la /app/production.db
```

### Si Celery ne fonctionne pas
1. Vérifier que Redis est accessible :
```bash
docker-compose exec celery-worker redis-cli -h redis ping
```

2. Vérifier que l'application Flask se charge correctement :
```bash
docker-compose exec celery-worker python -c "from app import create_app; app = create_app('production'); print('OK')"
```

### Si les tâches Celery échouent avec "Working outside of application context"
Cela ne devrait plus arriver avec les corrections apportées. Si le problème persiste :
1. Vérifier que `celery_app.py` a bien été modifié
2. Reconstruire l'image : `docker-compose build --no-cache celery-worker`
3. Redémarrer le service : `docker-compose restart celery-worker`

## Notes importantes

### Base de données
- La base de données SQLite est maintenant stockée dans un volume Docker persistant (`db-data`)
- Le fichier de base de données est accessible à `/app/production.db` dans les conteneurs
- Pour une utilisation en production réelle, il est recommandé d'utiliser PostgreSQL ou MySQL au lieu de SQLite

### Sécurité
- Les secrets (SECRET_KEY, JWT_SECRET_KEY, MAIL_PASSWORD) sont actuellement en clair dans `docker-compose.yml`
- Pour la production, utilisez Docker secrets ou des variables d'environnement externes
- Ne commitez jamais `docker-compose.yml` avec des secrets en production

### Performance
- SQLite n'est pas recommandé pour la production avec plusieurs workers
- Pour la production, migrez vers PostgreSQL :
  ```yaml
  environment:
    - DATABASE_URL=postgresql://user:password@postgres:5432/atmr
  ```

## Prochaines étapes recommandées

1. **Migration vers PostgreSQL** (pour la production)
2. **Utilisation de Docker secrets** pour les informations sensibles
3. **Configuration d'un reverse proxy** (Nginx) pour l'API
4. **Mise en place de backups** automatiques de la base de données
5. **Configuration de monitoring** (Prometheus, Grafana)
6. **Mise en place de logs centralisés** (ELK Stack)

## Support

Si vous rencontrez d'autres problèmes, vérifiez :
1. Les logs détaillés : `docker-compose logs -f`
2. L'état des conteneurs : `docker-compose ps`
3. Les ressources système : `docker stats`