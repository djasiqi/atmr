# Améliorations Redis & Rate Limits (C2)

**Date**: 8 janvier 2026  
**Contexte**: Suite aux problèmes de cache Redis lors des tests Locust  
**Statut**: ✅ IMPLÉMENTÉ

---

## 🎯 Objectif

Améliorer la gestion des rate limits Redis pour éviter les problèmes de cache persistant et faciliter la maintenance en production.

---

## 🔧 Implémentations

### 1️⃣ **Endpoint Admin pour Flush Rate Limits**

**Fichier**: `backend/routes/admin.py` (nouveau)

#### Endpoints créés :

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/admin/rate-limit/flush` | POST | Flush tous les compteurs de rate limit |
| `/api/admin/rate-limit/stats` | GET | Statistiques détaillées sur les rate limits |
| `/api/admin/redis/info` | GET | Informations sur l'instance Redis |
| `/api/admin/rate-limit/config` | GET | Configuration actuelle des rate limits |

#### Utilisation :

```bash
# Flush tous les rate limits (nécessite token JWT admin)
curl -X POST http://localhost:5000/api/admin/rate-limit/flush \
  -H "Authorization: Bearer <admin_token>"

# Obtenir les statistiques
curl -X GET http://localhost:5000/api/admin/rate-limit/stats \
  -H "Authorization: Bearer <admin_token>"
```

#### Sécurité :
- ✅ Authentification JWT requise
- ✅ Rôle `admin` obligatoire
- ✅ Logging de toutes les actions
- ✅ Métriques Prometheus intégrées

---

### 2️⃣ **Versioning des Clés Redis**

**Fichier**: `backend/ext.py`

#### Principe :

Chaque clé Redis inclut maintenant un **hash de configuration** qui invalide automatiquement les anciens rate limits lors des changements de configuration.

#### Ancienne clé :
```
LIMITER:user:123
LIMITER:ip:192.168.1.1
```

#### Nouvelle clé (avec versioning) :
```
LIMITER:v<hash>:user:123
LIMITER:v<hash>:ip:192.168.1.1
```

Où `<hash>` est un hash MD5 de :
- `RATELIMIT_CONFIG_VERSION` (ex: "v1")
- `ENVIRONMENT` (ex: "development", "production")
- `RATELIMIT_DEFAULT_LIMITS` (ex: "1000 per hour")

#### Avantages :
- ✅ Invalidation automatique lors des changements de configuration
- ✅ Pas besoin de flush manuel Redis lors des déploiements
- ✅ Compatibilité multi-environnements (dev/prod)
- ✅ Traçabilité des versions de configuration

#### Code implémenté :

```python
def get_rate_limit_config_hash() -> str:
    """Génère un hash unique basé sur les configurations de rate limit."""
    import hashlib
    
    version = os.getenv("RATELIMIT_CONFIG_VERSION", "v1")
    environment = os.getenv("ENVIRONMENT", "development")
    default_limits = os.getenv("RATELIMIT_DEFAULT_LIMITS", "1000 per hour")
    
    config_str = f"{version}:{environment}:{default_limits}"
    return hashlib.md5(config_str.encode()).hexdigest()[:8]
```

---

### 3️⃣ **Configuration par Environnement**

**Fichier**: `backend/config.py`

#### Configuration de base (classe `Config`) :

```python
# Stratégie de rate limiting
RATELIMIT_STRATEGY = os.getenv("RATELIMIT_STRATEGY", "moving-window")

# Rate limits par défaut
RATELIMIT_DEFAULT_LIMITS = ["1000 per hour"]
RATELIMIT_DISPATCH_RUN = "30 per hour"
RATELIMIT_DISPATCH_TRIGGER = "50 per hour"
RATELIMIT_COMPANY_DISPATCH_RUN = "10 per minute"
RATELIMIT_COMPANY_DISPATCH_OPTIMIZER = "10 per minute"

# Version de configuration
RATELIMIT_CONFIG_VERSION = os.getenv("RATELIMIT_CONFIG_VERSION", "v1")
```

#### Configuration Development :

```python
class DevelopmentConfig(Config):
    ENVIRONMENT = "development"
    RATELIMIT_DEFAULT_LIMITS = ["100000 per hour"]  # Très permissif
    RATELIMIT_DISPATCH_RUN = "10000 per hour"
    RATELIMIT_DISPATCH_TRIGGER = "10000 per hour"
    RATELIMIT_COMPANY_DISPATCH_RUN = "10000 per hour"
    RATELIMIT_COMPANY_DISPATCH_OPTIMIZER = "10000 per hour"
```

#### Configuration Production :

```python
class ProductionConfig(Config):
    ENVIRONMENT = "production"
    RATELIMIT_DEFAULT_LIMITS = ["1000 per hour"]  # Conservateur
    RATELIMIT_DISPATCH_RUN = "30 per hour"
    RATELIMIT_DISPATCH_TRIGGER = "50 per hour"
    RATELIMIT_COMPANY_DISPATCH_RUN = "10 per minute"
    RATELIMIT_COMPANY_DISPATCH_OPTIMIZER = "10 per minute"
```

#### Avantages :
- ✅ Configuration centralisée dans `config.py`
- ✅ Valeurs différentes par environnement
- ✅ Surcharge possible via variables d'environnement
- ✅ Configuration typée et documentée

---

### 4️⃣ **Métriques Prometheus pour Rate Limits**

**Fichier**: `backend/security/security_metrics.py`

#### Métriques ajoutées :

| Métrique | Type | Description |
|----------|------|-------------|
| `rate_limit_exceeded_total` | Counter | Nombre de dépassements par endpoint et type d'utilisateur |
| `rate_limit_active_keys` | Gauge | Nombre de clés actives dans Redis |
| `rate_limit_flushes_total` | Counter | Nombre de flushes manuels (via admin) |

#### Utilisation dans le code :

```python
# Incrémenter lors d'un dépassement
from backend.security.security_metrics import rate_limit_exceeded_total
rate_limit_exceeded_total.labels(
    endpoint="/api/dispatch/run",
    user_type="authenticated"
).inc()

# Mettre à jour le nombre de clés actives
from backend.security.security_metrics import rate_limit_active_keys
keys = list(redis_client.scan_iter("LIMITER:*"))
rate_limit_active_keys.set(len(keys))
```

#### Dashboards Grafana suggérés :

```promql
# Taux de dépassements par endpoint
rate(rate_limit_exceeded_total[5m])

# Nombre de clés actives Redis
rate_limit_active_keys

# Flushes par admin
rate(rate_limit_flushes_total[1h])
```

---

## 📦 Fichiers Modifiés

| Fichier | Changements |
|---------|-------------|
| `backend/routes/admin.py` | ✨ Nouveau fichier (4 endpoints admin) |
| `backend/app.py` | ➕ Enregistrement du blueprint `admin_bp` |
| `backend/config.py` | ➕ Configuration rate limits par environnement |
| `backend/ext.py` | 🔧 Versioning des clés Redis + config dynamique |
| `backend/security/security_metrics.py` | ➕ 3 nouvelles métriques Prometheus |

---

## 🚀 Utilisation en Production

### 1. Changer la configuration

Incrémenter `RATELIMIT_CONFIG_VERSION` dans les variables d'environnement :

```bash
# Dans docker-compose.production.yml ou .env
RATELIMIT_CONFIG_VERSION=v2
```

Les anciennes clés Redis (v1) seront automatiquement ignorées.

### 2. Flush manuel (si nécessaire)

```bash
# Via endpoint admin
curl -X POST https://atmr-api.example.com/api/admin/rate-limit/flush \
  -H "Authorization: Bearer <admin_token>"
```

### 3. Monitoring

Ajouter des alertes Prometheus :

```yaml
# prometheus/alerts.yml
- alert: RateLimitExceededHigh
  expr: rate(rate_limit_exceeded_total[5m]) > 10
  for: 5m
  annotations:
    summary: "Trop de dépassements de rate limit"
    description: "{{ $value }} dépassements/seconde sur {{ $labels.endpoint }}"

- alert: RateLimitKeysHigh
  expr: rate_limit_active_keys > 100000
  for: 10m
  annotations:
    summary: "Trop de clés de rate limit actives dans Redis"
    description: "{{ $value }} clés actives (mémoire Redis potentiellement saturée)"
```

---

## 🧪 Tests

### Test 1 : Versioning automatique

```bash
# 1. Définir RATELIMIT_CONFIG_VERSION=v1
docker-compose up -d api

# 2. Effectuer des requêtes (rate limit appliqué)
for i in {1..100}; do curl http://localhost:5000/api/health; done

# 3. Changer RATELIMIT_CONFIG_VERSION=v2
# 4. Redémarrer API
docker-compose restart api

# 5. Les requêtes passent à nouveau (nouvelles clés v2)
for i in {1..100}; do curl http://localhost:5000/api/health; done
```

### Test 2 : Endpoint admin

```bash
# 1. Login en tant qu'admin
TOKEN=$(curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@test.com","password":"admin123"}' \
  | jq -r '.access_token')

# 2. Obtenir les stats
curl -X GET http://localhost:5000/api/admin/rate-limit/stats \
  -H "Authorization: Bearer $TOKEN" | jq

# 3. Flush les rate limits
curl -X POST http://localhost:5000/api/admin/rate-limit/flush \
  -H "Authorization: Bearer $TOKEN" | jq
```

### Test 3 : Métriques Prometheus

```bash
# Vérifier que les métriques sont exposées
curl http://localhost:5000/metrics | grep rate_limit
```

---

## 📊 Résultats Attendus

### Avant :
- ❌ Rate limits persistaient en cache après changement de configuration
- ❌ Nécessitait un `docker-compose restart redis` ou flush manuel via redis-cli
- ❌ Pas de visibilité sur les rate limits en production
- ❌ Configuration hardcodée dans le code

### Après :
- ✅ Versioning automatique : changements de config invalidants les anciens rate limits
- ✅ Endpoint admin pour flush/monitoring en temps réel
- ✅ Configuration centralisée et par environnement
- ✅ Métriques Prometheus pour monitoring proactif
- ✅ Logs structurés pour toutes les actions admin

---

## 🔐 Sécurité

### Contrôles d'accès :
- ✅ Tous les endpoints admin nécessitent un token JWT valide
- ✅ Rôle `admin` obligatoire (vérification via `@role_required(UserRole.admin)`)
- ✅ Logging de toutes les actions sensibles (flush, stats)
- ✅ Rate limiting appliqué aux endpoints admin eux-mêmes

### Auditabilité :
- ✅ Tous les flushes sont tracés dans les logs avec l'ID de l'admin
- ✅ Métriques Prometheus incluent l'admin_user_id
- ✅ Possibilité de créer des alertes Sentry/Prometheus sur les flushes fréquents

---

## 📝 Recommendations Futures

### Phase 2 (optionnelle) :

1. **TTL automatiques sur les clés Redis** :
   - Ajouter un TTL par défaut sur toutes les clés de rate limit
   - Éviter l'accumulation de clés anciennes

2. **Dashboard Grafana dédié** :
   - Créer un dashboard pour visualiser les rate limits en temps réel
   - Graphiques : taux de dépassement, clés actives, top endpoints

3. **Circuit breaker sur Redis** :
   - Ajouter un fallback si Redis est indisponible (mode memory)
   - Éviter les erreurs 500 si Redis crash

4. **Rate limiting adaptatif** :
   - Ajuster automatiquement les rate limits selon la charge du système
   - Utiliser les métriques CPU/memory pour moduler les limites

---

## ✅ Validation

**Tests manuels** : ✅ Réalisés  
**Tests automatisés** : ⏳ À implémenter (tests unitaires + intégration)  
**Déployé en dev** : ✅ Oui  
**Déployé en prod** : ⏳ Après validation complète  

---

## 📚 Références

- [Flask-Limiter Documentation](https://flask-limiter.readthedocs.io/)
- [Redis Key Expiration](https://redis.io/commands/expire)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [Rate Limiting Strategies](https://konghq.com/blog/engineering/how-to-design-a-scalable-rate-limiting-algorithm)

---

**Auteur**: AI Assistant (Cursor)  
**Reviewer**: À assigner  
**Statut**: ✅ IMPLÉMENTÉ - EN ATTENTE VALIDATION
