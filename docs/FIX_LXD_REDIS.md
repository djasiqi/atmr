# Correction du Redis LXD exposé publiquement

## Problème identifié

Un processus Redis LXD écoute sur `*:6379`, ce qui le rend accessible depuis Internet sans authentification. C'est probablement celui que le BSI a détecté.

```
lxd      1982440  1.3  0.1  59984 21860 ?        Ssl  Nov19 152:08 redis-server *:6379
```

## Solution immédiate

### Option 1 : Arrêter le processus directement (rapide)

```bash
# Identifier le PID
sudo ps aux | grep "redis-server \*:6379" | grep -v grep

# Arrêter le processus
sudo kill <PID>
```

⚠️ **Note** : Cette méthode est temporaire. Le processus peut redémarrer automatiquement.

### Option 2 : Si c'est un conteneur LXD

```bash
# Lister les conteneurs LXD
lxc list

# Arrêter le conteneur qui contient Redis
lxc stop <container-name>

# Désactiver le démarrage automatique
lxc config set <container-name> boot.autostart false
```

### Option 3 : Si c'est un service systemd

```bash
# Arrêter le service
sudo systemctl stop redis
# ou
sudo systemctl stop redis-server

# Désactiver le démarrage automatique
sudo systemctl disable redis
# ou
sudo systemctl disable redis-server
```

### Option 4 : Si c'est un snap

```bash
# Arrêter le snap
sudo snap stop redis

# Désactiver le snap
sudo snap disable redis
```

## Vérification

### 1. Vérifier que le processus est arrêté

```bash
sudo ps aux | grep redis-server | grep -v grep | grep -v docker
```

**Résultat attendu** : Aucune ligne

### 2. Vérifier que le port n'est plus exposé

```bash
sudo ss -tulpen | grep 6379
```

**Résultat attendu** : Aucune ligne (ou seulement des IP Docker internes)

### 3. Tester depuis l'extérieur

Depuis votre machine locale :

```bash
redis-cli -h 138.201.155.201 -p 6379
```

**Résultat attendu** : Connexion échoue (timeout ou connexion refusée)

## Identification de la source

Pour identifier d'où vient ce processus Redis :

```bash
# Identifier le PID
PID=$(ps aux | grep "redis-server \*:6379" | grep -v grep | awk '{print $2}')

# Voir la commande complète
sudo cat /proc/$PID/cmdline | tr '\0' ' '

# Voir les fichiers ouverts
sudo lsof -p $PID | head -20

# Voir les variables d'environnement
sudo cat /proc/$PID/environ | tr '\0' '\n' | grep -i redis
```

## Prévention

### 1. Configurer le firewall Hetzner

Dans le panneau Hetzner Cloud :

- Bloquer le port 6379 en entrée
- Autoriser seulement : 22 (SSH), 80 (HTTP), 443 (HTTPS)

### 2. Vérifier régulièrement

```bash
# Script de vérification
sudo ss -tulpen | grep 6379
sudo ps aux | grep redis-server | grep -v grep | grep -v docker
```

## Scripts disponibles

- `scripts/secure_redis.sh` : Diagnostic complet
- `scripts/fix_redis_security.sh` : Correction automatique
- `scripts/fix_lxd_redis.sh` : Correction spécifique Redis LXD
