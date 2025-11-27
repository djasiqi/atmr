# Guide de sécurisation Redis

Ce guide vous aide à sécuriser Redis après une alerte de sécurité du BSI (Bundesamt für Sicherheit in der Informationstechnik).

## Problème identifié

Redis est accessible depuis Internet sans authentification, ce qui constitue un risque de sécurité critique.

## Solution en 3 couches

### 1. Ne plus exposer Redis sur Internet

Redis ne doit **jamais** être exposé publiquement. Il doit être accessible uniquement depuis le réseau Docker interne.

### 2. Activer l'authentification Redis

Redis doit exiger un mot de passe pour toutes les connexions.

### 3. Vérifier que tout fonctionne

S'assurer que le backend et Celery continuent de fonctionner avec l'authentification activée.

## Vérifications sur le serveur

### Étape 1: Vérifier les ports exposés

```bash
sudo ss -tulpen | grep 6379
```

**Résultat attendu** : Aucune ligne (ou seulement des IP Docker internes comme `172.xx.xx.xx:6379`)

**Si vous voyez** `0.0.0.0:6379` ou `[::]:6379` → Redis est exposé sur Internet ❌

### Étape 2: Vérifier les processus Redis natifs

```bash
sudo ps aux | grep redis | grep -v grep
```

**Résultat attendu** : Aucun processus (ou seulement des processus Docker)

**Si vous voyez** un `redis-server` natif → Il faut soit l'arrêter, soit le reconfigurer

### Étape 3: Vérifier les conteneurs Redis Docker

```bash
cd /srv/atmr
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}" | grep redis
```

**Résultat attendu** :

```
atmr-redis    redis:7-alpine    6379/tcp
```

**⚠️ IMPORTANT** : Il ne doit **PAS** y avoir de mapping du type `0.0.0.0:6379->6379/tcp`

**Si vous voyez** un mapping `0.0.0.0:6379` → C'est celui-là que le BSI a détecté ❌

### Étape 4: Vérifier la configuration dans docker-compose.production.yml

Le service Redis doit avoir :

```yaml
redis:
  image: redis:7-alpine
  container_name: atmr-redis
  command:
    - sh
    - -c
    - |
      redis-server \
        --requirepass "${REDIS_PASSWORD}" \
        --appendonly yes \
        --maxmemory 256mb \
        --maxmemory-policy allkeys-lru \
        --bind 0.0.0.0 \
        --protected-mode yes
  environment:
    REDIS_PASSWORD: ${REDIS_PASSWORD}
  volumes:
    - redis_data:/data
  # ⚠️ IMPORTANT : en production, NE PAS exposer le port sur l'hôte
  # pas de: ports:
  #   - "0.0.0.0:6379:6379"
  expose:
    - "6379" # ✅ Réseau Docker interne uniquement
  networks:
    - atmr-network
```

**Points critiques** :

- ✅ `--requirepass "${REDIS_PASSWORD}"` → Redis refuse toute commande sans AUTH
- ✅ `expose:` au lieu de `ports:` → Redis n'est accessible que depuis le réseau Docker
- ❌ **AUCUN** bloc `ports:` sur ce service en prod

### Étape 5: Vérifier l'authentification Redis

```bash
# Depuis le conteneur backend
docker exec -it atmr-backend redis-cli -h redis -p 6379 -a "${REDIS_PASSWORD}" INFO server | grep requirepass
```

**Résultat attendu** : `requirepass` non vide

**Tester sans mot de passe** (devrait échouer) :

```bash
docker exec -it atmr-backend redis-cli -h redis -p 6379 INFO server
```

**Résultat attendu** : Erreur `NOAUTH Authentication required`

### Étape 6: Tester depuis l'extérieur

Depuis votre machine locale (ou un autre serveur Internet) :

```bash
redis-cli -h 138.201.155.201 -p 6379
```

**Résultat attendu** :

- Connexion échoue → ✅ Redis n'est pas accessible depuis Internet
- Connexion réussit → ❌ Redis est accessible depuis Internet (SÉCURITÉ CRITIQUE)

## Actions correctives

### Si Redis est exposé publiquement

1. **Arrêter les conteneurs problématiques** :

```bash
cd /srv/atmr
docker compose -f docker-compose.production.yml down redis
```

2. **Vérifier docker-compose.production.yml** :

   - S'assurer qu'il n'y a pas de `ports:` pour Redis
   - S'assurer qu'il y a `expose: - "6379"`

3. **Redémarrer avec la nouvelle configuration** :

```bash
docker compose -f docker-compose.production.yml up -d redis
```

4. **Vérifier** :

```bash
sudo ss -tulpen | grep 6379
# Ne doit rien retourner (ou seulement des IP Docker internes)
```

### Si un processus Redis natif tourne

1. **Arrêter le service** :

```bash
sudo systemctl stop redis
# ou
sudo systemctl stop redis-server
```

2. **Désactiver au démarrage** :

```bash
sudo systemctl disable redis
# ou
sudo systemctl disable redis-server
```

3. **Vérifier** :

```bash
sudo ps aux | grep redis | grep -v grep
# Ne doit rien retourner
```

### Ajouter un firewall (recommandé)

#### Avec ufw (si installé)

```bash
sudo ufw deny 6379/tcp
sudo ufw status
```

#### Via Hetzner Cloud Firewall

Dans le panneau Hetzner Cloud :

1. Aller dans **Firewalls**
2. Créer ou modifier une règle
3. **Bloquer** le port 6379 en entrée
4. Autoriser seulement :
   - 22 (SSH)
   - 80 (HTTP)
   - 443 (HTTPS)

## Vérification finale

Après les corrections, exécutez le script de diagnostic :

```bash
cd /srv/atmr
bash scripts/secure_redis.sh
```

**Résultat attendu** :

- ✅ Aucun port 6379 exposé au niveau host
- ✅ Aucun processus Redis natif
- ✅ Conteneurs Redis n'exposent pas le port publiquement
- ✅ Redis nécessite un mot de passe
- ✅ Redis refuse les connexions sans mot de passe

## Configuration actuelle (déjà sécurisée)

Le fichier `docker-compose.production.yml` est déjà configuré correctement :

- ✅ `--requirepass "${REDIS_PASSWORD}"` activé
- ✅ `expose:` au lieu de `ports:` (réseau Docker interne uniquement)
- ✅ `--protected-mode yes` activé

**Si vous avez encore des problèmes**, c'est probablement dû à :

1. Un ancien conteneur `backend-redis-1` qui expose encore le port
2. Un processus Redis natif qui tourne en dehors de Docker

## Script de diagnostic

Un script `scripts/secure_redis.sh` est disponible pour automatiser toutes ces vérifications.

```bash
cd /srv/atmr
bash scripts/secure_redis.sh
```

## Support

En cas de problème persistant :

1. Vérifier tous les conteneurs Redis : `docker ps -a | grep redis`
2. Vérifier tous les docker-compose.yml : `find /srv -name "docker-compose*.yml" -exec grep -l "redis" {} \;`
3. Vérifier les services systemd : `systemctl list-units | grep redis`
