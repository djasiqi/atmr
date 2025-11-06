# ✅ 4.1: Plan de Migration vers HashiCorp Vault

## Vue d'ensemble

Migration progressive des secrets depuis les fichiers `.env` vers HashiCorp Vault pour améliorer la sécurité, la rotation automatique et la conformité RGPD.

## Objectifs

- ✅ Centraliser tous les secrets dans Vault
- ✅ Activer la rotation automatique des clés
- ✅ Implémenter des audit logs complets
- ✅ Maintenir la compatibilité avec `.env` en développement
- ✅ Support multi-environnements (dev/staging/prod)

## Architecture

```
┌─────────────────┐
│   Application   │
│   (ATMR API)    │
└────────┬────────┘
         │
         │ hvac client
         ▼
┌─────────────────┐
│  Vault Wrapper  │  ← backend/shared/vault_client.py
│   (Python)      │
└────────┬────────┘
         │
         │ API REST
         ▼
┌─────────────────┐
│  HashiCorp      │
│     Vault       │
│ (Docker/Self)   │
└─────────────────┘
```

## Phases de Migration

### Phase 1 : Infrastructure Vault (Jours 1-2)

**Objectifs** :

- Installer et configurer Vault
- Configurer les policies et authentifications
- Tester la connectivité

**Actions** :

1. ✅ Ajouter Vault dans docker-compose.yml
2. ✅ Créer les policies Vault
3. ✅ Configurer l'authentification AppRole
4. ✅ Créer les secrets engines (KV v2)
5. ✅ Tester avec vault CLI

**Livrables** :

- `vault/docker-compose.vault.yml`
- `vault/policies/atmr-api.hcl`
- `vault/init-vault.sh`

### Phase 2 : Client Python Vault (Jours 3-4)

**Objectifs** :

- Créer un wrapper Python pour Vault
- Implémenter le cache et le fallback `.env`
- Gérer les erreurs et la reconnexion

**Actions** :

1. ✅ Créer `backend/shared/vault_client.py`
2. ✅ Implémenter le cache local (éviter appels répétés)
3. ✅ Fallback vers `.env` si Vault indisponible (dev uniquement)
4. ✅ Logging et métriques

**Livrables** :

- `backend/shared/vault_client.py`
- Tests unitaires

### Phase 3 : Migration Config (Jours 5-6)

**Objectifs** :

- Modifier `config.py` pour utiliser Vault
- Maintenir la compatibilité avec `.env`
- Implémenter la détection automatique (Vault ou .env)

**Actions** :

1. ✅ Modifier `Config`, `DevelopmentConfig`, `ProductionConfig`
2. ✅ Ajouter détection automatique `VAULT_ADDR`
3. ✅ Fallback gracieux vers `.env` si Vault non configuré
4. ✅ Tests de compatibilité

**Livrables** :

- `backend/config.py` modifié
- Tests de migration

### Phase 4 : Migration des Secrets (Jours 7-8)

**Objectifs** :

- Migrer tous les secrets depuis `.env` vers Vault
- Créer les secrets dans les bons paths
- Valider le chargement depuis Vault

**Actions** :

1. ✅ Lister tous les secrets actuels
2. ✅ Créer script de migration `vault/migrate-secrets.py`
3. ✅ Importer les secrets dans Vault
4. ✅ Valider l'accès depuis l'application

**Livrables** :

- `vault/migrate-secrets.py`
- Documentation des paths secrets

### Phase 5 : Rotation Automatique (Jours 9-10)

**Objectifs** :

- Configurer la rotation automatique des clés
- Implémenter les tasks Celery pour rotation
- Tests de rotation

**Actions** :

1. ✅ Configurer les policies de rotation dans Vault
2. ✅ Créer task Celery `tasks/vault_rotation_tasks.py`
3. ✅ Intégrer avec système de rotation existant (2.5)
4. ✅ Tests de rotation

**Livrables** :

- Configuration de rotation Vault
- Tasks Celery pour rotation

### Phase 6 : Audit et Documentation (Jours 11-12)

**Objectifs** :

- Configurer les audit logs
- Documenter l'utilisation
- Former l'équipe

**Actions** :

1. ✅ Configurer audit device dans Vault
2. ✅ Documenter dans `vault/VAULT_USAGE.md`
3. ✅ Créer runbooks pour opérations courantes

**Livrables** :

- Configuration audit
- Documentation complète

## Secrets à Migrer

### Secrets Critiques (Priorité Haute)

| Secret                   | Path Vault                        | Type    | Rotation          |
| ------------------------ | --------------------------------- | ------- | ----------------- |
| `SECRET_KEY`             | `atmr/prod/flask/secret_key`      | KV      | Manuel (90j)      |
| `JWT_SECRET_KEY`         | `atmr/prod/jwt/secret_key`        | KV      | Automatique (30j) |
| `APP_ENCRYPTION_KEY_B64` | `atmr/prod/encryption/master_key` | KV      | Automatique (90j) |
| `DATABASE_URL`           | `atmr/prod/database/creds`        | Dynamic | Automatique (7j)  |
| `REDIS_URL`              | `atmr/prod/redis/creds`           | Dynamic | Automatique (7j)  |
| `MAIL_PASSWORD`          | `atmr/prod/mail/password`         | KV      | Manuel (90j)      |

### Secrets API Externes (Priorité Moyenne)

| Secret                  | Path Vault                    | Type | Rotation |
| ----------------------- | ----------------------------- | ---- | -------- |
| `GOOGLE_PLACES_API_KEY` | `atmr/prod/api/google_places` | KV   | Manuel   |
| `OPENWEATHER_API_KEY`   | `atmr/prod/api/openweather`   | KV   | Manuel   |
| `SENTRY_DSN`            | `atmr/prod/monitoring/sentry` | KV   | Manuel   |

### Configuration Non-Secrète (Priorité Basse)

| Variable                | Gestion                   | Notes            |
| ----------------------- | ------------------------- | ---------------- |
| `FLASK_ENV`             | Environnement             | Non secret       |
| `PDF_BASE_URL`          | Environnement             | Non secret       |
| `REDIS_URL`             | Vault (prod) / .env (dev) | Peut être secret |
| `SOCKETIO_CORS_ORIGINS` | Environnement             | Non secret       |

## Structure des Paths Vault

```
atmr/
├── dev/              # Développement
│   ├── flask/
│   │   └── secret_key
│   ├── jwt/
│   │   └── secret_key
│   └── encryption/
│       └── master_key
├── staging/          # Staging
│   └── ...
└── prod/             # Production
    ├── flask/
    │   └── secret_key
    ├── jwt/
    │   └── secret_key
    ├── encryption/
    │   └── master_key
    ├── database/
    │   └── creds      # Dynamic secrets (PostgreSQL)
    ├── redis/
    │   └── creds      # Dynamic secrets
    ├── mail/
    │   └── password
    └── api/
        ├── google_places
        └── openweather
```

## Configuration Vault

### Policies

- **`atmr-api-read`** : Lecture seule sur `atmr/prod/*`
- **`atmr-api-rotate`** : Lecture + rotation sur `atmr/prod/*`
- **`atmr-dev-read`** : Lecture sur `atmr/dev/*` (développement)

### Authentification

- **AppRole** (recommandé pour applications)
  - `role_id` : Stocké en variable d'environnement ou Kubernetes secret
  - `secret_id` : Généré dynamiquement ou via Kubernetes injector
- **Token** (développement uniquement)

## Sécurité

### En Production

- ✅ Vault en mode HA (High Availability)
- ✅ Chiffrement au repos activé
- ✅ Audit logs vers fichier/syslog/S3
- ✅ TLS obligatoire
- ✅ Network isolation
- ✅ Backup automatique des données

### En Développement

- ✅ Vault en mode dev (souvent accepté)
- ✅ Fallback `.env` si Vault indisponible
- ✅ Pas d'audit logs (performance)

## Tests de Migration

### Checklist de Validation

- [ ] Tous les secrets chargés depuis Vault en prod
- [ ] Fallback `.env` fonctionne en dev
- [ ] Rotation automatique opérationnelle
- [ ] Audit logs générés
- [ ] Tests unitaires passent
- [ ] Tests E2E passent
- [ ] Pas de régression de performance
- [ ] Documentation complète

## Rollback Plan

En cas de problème :

1. **Immediate** : Désactiver Vault dans `config.py` (flag `USE_VAULT=False`)
2. **Short-term** : Réactiver `.env` temporairement
3. **Long-term** : Analyser les logs Vault pour identifier le problème

## Estimation

- **Infrastructure** : 2 jours
- **Client Python** : 2 jours
- **Migration Config** : 2 jours
- **Migration Secrets** : 2 jours
- **Rotation** : 2 jours
- **Documentation** : 2 jours
- **Total** : **10 jours** (équipe de 2-3 personnes)

## Prochaines Étapes

1. Lire `vault/VAULT_SETUP.md` pour installation
2. Lire `vault/VAULT_USAGE.md` pour utilisation quotidienne
3. Exécuter `vault/migrate-secrets.py` pour migration
4. Consulter les logs audit dans Grafana
