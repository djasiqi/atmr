# Guide de déploiement sur le serveur

## Étapes pour déployer la dernière version

### 1. Se connecter au serveur

```bash
ssh utilisateur@adresse-serveur
```

### 2. Aller dans le répertoire du projet

```bash
cd /chemin/vers/atmr
```

### 3. Récupérer les dernières modifications depuis Git

```bash
git pull origin main
```

### 4. Reconstruire les images Docker (si build local)

```bash
# Pour docker-compose de production
docker-compose -f docker-compose.production.yml build --no-cache

# OU si vous utilisez le docker-compose standard
docker-compose build --no-cache
```

### 5. Redémarrer les services avec les nouvelles images

```bash
# Pour docker-compose de production
docker-compose -f docker-compose.production.yml up -d --force-recreate

# OU si vous utilisez le docker-compose standard
docker-compose up -d --force-recreate
```

### 6. Vérifier l'état des services

```bash
docker-compose -f docker-compose.production.yml ps
# OU
docker-compose ps
```

### 7. Vérifier les logs (optionnel)

```bash
# Logs de l'API
docker-compose -f docker-compose.production.yml logs -f api

# Logs de tous les services
docker-compose -f docker-compose.production.yml logs -f
```

## Commandes rapides (tout en une fois)

```bash
cd /chemin/vers/atmr && \
git pull origin main && \
docker-compose -f docker-compose.production.yml build --no-cache && \
docker-compose -f docker-compose.production.yml up -d --force-recreate && \
docker-compose -f docker-compose.production.yml ps
```

## Si vous utilisez un registry Docker (Docker Hub, etc.)

Si vos images sont poussées sur un registry, vous pouvez simplement :

```bash
cd /chemin/vers/atmr && \
git pull origin main && \
docker-compose -f docker-compose.production.yml pull && \
docker-compose -f docker-compose.production.yml up -d --force-recreate
```

## Notes importantes

- `--force-recreate` : Force la recréation des conteneurs même si la configuration n'a pas changé
- `--no-cache` : Reconstruit les images sans utiliser le cache (pour être sûr d'avoir la dernière version)
- `-d` : Démarre les conteneurs en mode détaché (en arrière-plan)
- `-f docker-compose.production.yml` : Utilise le fichier de configuration de production
