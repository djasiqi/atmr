# ✅ 3.5: Configuration Reverse Proxy (nginx/Traefik)

Configuration pour reverse proxy avec nginx ou Traefik pour l'application ATMR.

## Choix : nginx vs Traefik

### nginx

- ✅ **Recommandé pour production** : Mature, stable, performances excellentes
- ✅ Configuration explicite et contrôlée
- ✅ Caching avancé configurable
- ✅ Rate limiting intégré
- ⚠️ Nécessite redémarrage pour changements de config

### Traefik

- ✅ Auto-découverte des services Docker (pas besoin de redémarrer)
- ✅ Dashboard web pour monitoring
- ✅ Gestion automatique des certificats SSL (Let's Encrypt)
- ⚠️ Moins de contrôle fin sur le caching
- ⚠️ Plus récent, écosystème plus petit

## Configuration nginx

### Utilisation

```bash
# Démarrer avec nginx
docker-compose -f docker-compose.yml -f nginx/docker-compose.nginx.yml up -d
```

### Fonctionnalités configurées

- ✅ **Timeouts** : 60s client, 120s proxy
- ✅ **Body size limit** : 50MB (configurable via `client_max_body_size`)
- ✅ **Rate limiting** :
  - API générale : 100 req/s (burst 20)
  - Auth endpoints : 10 req/s (burst 5)
- ✅ **Caching** :
  - Fichiers statiques (`/static/`, `/uploads/`) : 1h-24h
  - API GET requests : 5 minutes
  - Cache zones configurées avec TTL adaptés
- ✅ **Gzip compression** : Activée pour text/JSON/CSS/JS
- ✅ **WebSocket support** : Configuré pour Socket.IO (`/socket.io/`)
- ✅ **Headers sécurité** : X-Frame-Options, X-Content-Type-Options, etc.

### Fichiers

- `nginx.conf` : Configuration principale nginx
- `docker-compose.nginx.yml` : Service Docker Compose pour nginx

## Configuration Traefik

### Utilisation

```bash
# Démarrer avec Traefik
docker-compose -f docker-compose.yml -f nginx/docker-compose.traefik.yml up -d
```

### Fonctionnalités configurées

- ✅ **Timeouts** : 60s read/write, 180s idle
- ✅ **Auto-découverte** : Services Docker détectés automatiquement
- ✅ **Métriques Prometheus** : Exposées pour monitoring
- ⚠️ **Rate limiting** : Nécessite Redis (optionnel)

### Fichiers

- `traefik.yml` : Configuration Traefik
- `docker-compose.traefik.yml` : Service Docker Compose pour Traefik

## Labels Docker pour Traefik

Pour que Traefik route automatiquement vers un service, ajouter ces labels :

```yaml
services:
  api:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=Host(`api.example.com`)"
      - "traefik.http.routers.api.entrypoints=web"
      - "traefik.http.services.api.loadbalancer.server.port=5000"
```

## Configuration avancée

### Variables d'environnement nginx

Modifier `nginx.conf` pour ajuster :

- `client_max_body_size` : Limite taille upload (défaut: 50MB)
- `proxy_read_timeout` : Timeout lecture réponse backend (défaut: 120s)
- `keepalive_timeout` : Timeout connexions keep-alive (défaut: 65s)

### Rate limiting

Ajuster dans `nginx.conf` :

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
```

### Caching

Ajuster TTL dans `nginx.conf` :

```nginx
proxy_cache_valid 200 302 5m;  # Cache succès 5 minutes
proxy_cache_valid 404 1m;      # Cache erreurs 404 1 minute
```

## HTTPS/SSL

### nginx (manuel)

1. Obtenir certificats SSL (Let's Encrypt via certbot)
2. Modifier `nginx.conf` pour activer SSL sur port 443
3. Redémarrer nginx

### Traefik (automatique)

1. Configurer Let's Encrypt dans `traefik.yml` :

```yaml
certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@example.com
      storage: /letsencrypt/acme.json
      httpChallenge:
        entryPoint: web
```

2. Ajouter labels aux services :

```yaml
labels:
  - "traefik.http.routers.api.tls.certresolver=letsencrypt"
  - "traefik.http.routers.api.tls=true"
```

## Monitoring

### nginx

- Logs : `/var/log/nginx/access.log` et `/var/log/nginx/error.log`
- Stats : Utiliser module `nginx_status` ou exporter Prometheus

### Traefik

- Dashboard : http://localhost:8080 (à protéger en production)
- Métriques Prometheus : http://localhost:8080/metrics

## Troubleshooting

### Problème : Timeouts 504 Gateway Timeout

**Solution** : Augmenter `proxy_read_timeout` dans nginx.conf :

```nginx
proxy_read_timeout 300s;  # 5 minutes
```

### Problème : Body too large

**Solution** : Augmenter `client_max_body_size` :

```nginx
client_max_body_size 100M;
```

### Problème : Rate limiting trop strict

**Solution** : Ajuster zones dans nginx.conf :

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=200r/s;
limit_req zone=api_limit burst=50 nodelay;
```

## Production

⚠️ **Important pour production** :

1. ✅ Désactiver dashboard Traefik ou le protéger avec authentification
2. ✅ Configurer HTTPS/SSL (obligatoire pour données sensibles)
3. ✅ Limiter accès aux métriques (Prometheus, nginx stats)
4. ✅ Configurer logs rotation
5. ✅ Ajuster rate limiting selon charge attendue
6. ✅ Monitorer utilisation cache et ajuster TTL si nécessaire
