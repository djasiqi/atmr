# Guide de démarrage rapide - ATMR

## 🚀 Démarrage en 3 étapes

### Étape 1 : Récupérer les dernières modifications
```bash
cd /path/to/atmr
git pull origin main
```

### Étape 2 : Déployer avec le script automatique
```bash
./deploy.sh
```

### Étape 3 : Vérifier que tout fonctionne
```bash
# Vérifier l'état des services
docker-compose ps

# Tester l'API
curl http://localhost:5000/health

# Ouvrir Flower dans le navigateur
# http://localhost:5555
```

## ✅ Ce qui a été corrigé

### Problème 1 : API ne démarre pas
**Erreur :** `RuntimeError: Either 'SQLALCHEMY_DATABASE_URI' or 'SQLALCHEMY_BINDS' must be set.`

**✓ Résolu :** Toutes les variables d'environnement nécessaires sont maintenant définies dans `docker-compose.yml`

### Problème 2 : Celery Worker échoue
**Erreur :** `RuntimeError: Working outside of application context.`

**✓ Résolu :** Le contexte Flask est maintenant correctement initialisé pour tous les workers Celery

## 📋 Commandes utiles

### Voir les logs en temps réel
```bash
# Tous les services
docker-compose logs -f

# Un service spécifique
docker-compose logs -f api
docker-compose logs -f celery-worker
docker-compose logs -f celery-beat
```

### Redémarrer un service
```bash
docker-compose restart api
docker-compose restart celery-worker
```

### Arrêter tous les services
```bash
docker-compose down
```

### Reconstruire et redémarrer
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Accéder à un conteneur
```bash
# Accéder au conteneur API
docker-compose exec api bash

# Accéder au conteneur Celery Worker
docker-compose exec celery-worker bash
```

### Initialiser/Migrer la base de données
```bash
# Accéder au conteneur API
docker-compose exec api bash

# Appliquer les migrations
flask db upgrade

# Créer un utilisateur admin (si nécessaire)
python add_admin.py

# Quitter
exit
```

## 🌐 URLs des services

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:5000 | API Flask principale |
| Flower | http://localhost:5555 | Monitoring Celery |

## 🔧 Dépannage rapide

### L'API ne répond pas
```bash
# Vérifier les logs
docker-compose logs api

# Vérifier que le conteneur tourne
docker-compose ps api

# Redémarrer l'API
docker-compose restart api
```

### Celery ne traite pas les tâches
```bash
# Vérifier les logs du worker
docker-compose logs celery-worker

# Vérifier que Redis est accessible
docker-compose exec celery-worker redis-cli -h redis ping

# Redémarrer le worker
docker-compose restart celery-worker
```

### Redis ne fonctionne pas
```bash
# Vérifier les logs
docker-compose logs redis

# Redémarrer Redis
docker-compose restart redis
```

### Tout nettoyer et recommencer
```bash
# Arrêter et supprimer tous les conteneurs et volumes
docker-compose down -v

# Reconstruire et redémarrer
docker-compose build --no-cache
docker-compose up -d

# Réinitialiser la base de données
docker-compose exec api flask db upgrade
```

## 📚 Documentation complète

Pour plus de détails, consultez :
- **CORRECTION_DOCKER.md** : Documentation complète des corrections
- **RESUME_CORRECTIONS.md** : Résumé des changements effectués
- **backend/.env.example** : Template de configuration

## ⚠️ Notes importantes

### Sécurité
- Les secrets dans `docker-compose.yml` sont pour le développement uniquement
- Pour la production, utilisez des variables d'environnement externes ou Docker secrets
- Ne commitez jamais `backend/.env` avec vos vrais secrets

### Performance
- SQLite est utilisé par défaut (développement)
- Pour la production, migrez vers PostgreSQL
- Ajustez le nombre de workers Celery selon vos besoins

### Base de données
- La base de données est persistée dans un volume Docker (`db-data`)
- Pour réinitialiser la base de données : `docker-compose down -v`
- Pensez à faire des backups réguliers en production

## 🆘 Besoin d'aide ?

Si vous rencontrez des problèmes :

1. Consultez les logs : `docker-compose logs -f`
2. Vérifiez l'état : `docker-compose ps`
3. Consultez la documentation complète dans `CORRECTION_DOCKER.md`
4. Vérifiez que toutes les variables d'environnement sont correctes

## 🎉 C'est tout !

Votre application ATMR devrait maintenant fonctionner correctement. Tous les services sont opérationnels et les erreurs précédentes sont résolues.