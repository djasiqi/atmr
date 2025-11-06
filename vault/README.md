# ✅ 4.1: HashiCorp Vault pour ATMR

Intégration complète de HashiCorp Vault pour la gestion centralisée des secrets.

## Documentation

- **[QUICK_START.md](QUICK_START.md)** : Démarrage rapide (5 minutes)
- **[VAULT_SETUP.md](VAULT_SETUP.md)** : Guide d'installation et configuration
- **[VAULT_USAGE.md](VAULT_USAGE.md)** : Guide d'utilisation quotidienne
- **[VAULT_MIGRATION_PLAN.md](VAULT_MIGRATION_PLAN.md)** : Plan de migration détaillé
- **[VAULT_PRODUCTION_CONFIG.md](VAULT_PRODUCTION_CONFIG.md)** : Configuration production

## Fichiers

### Configuration

- `docker-compose.vault.yml` : Configuration Docker pour Vault
- `policies/atmr-api-read.hcl` : Policy lecture seule
- `policies/atmr-api-rotate.hcl` : Policy avec rotation

### Scripts

- `migrate-secrets.py` : Script de migration depuis `.env`

## Utilisation Rapide

```bash
# 1. Installer hvac
pip install hvac>=1.2.0

# 2. Lancer Vault
docker-compose -f docker-compose.yml -f vault/docker-compose.vault.yml up -d vault

# 3. Migrer les secrets
python vault/migrate-secrets.py --env-file backend/.env --vault-token dev-root-token
```

## Architecture

L'application charge automatiquement les secrets depuis Vault si `VAULT_ADDR` est configuré, sinon fallback vers `.env`.

## Sécurité

En production :

- ✅ Vault en mode HA
- ✅ TLS obligatoire
- ✅ Audit logs complets
- ✅ Rotation automatique (Celery)
- ✅ Auto-unseal recommandé

## Support

Pour toute question, consulter la documentation ou les logs Vault.
