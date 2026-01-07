"""
Module `security` - Consolidation des services de sécurité ATMR

Ce module regroupe tous les services liés à la sécurité de l'application :
- Authentification (access tokens, refresh tokens)
- Protection CSRF
- Protection anti-spam
- Idempotence des requêtes
- Guards de sécurité
- Rotation des secrets
- Masquage PII (Personally Identifiable Information)

## Migration B2 (7 janvier 2025)

Ce module consolide 10 services fragmentés en un seul module cohérent :
- `access_token_service.py` → `security/authentication.py` (AccessTokenService)
- `refresh_token_service.py` → `security/authentication.py` (RefreshTokenService)
- `csrf_protection.py` → `security/csrf.py`
- `spam_protection.py` → `security/spam.py`
- `idempotency_service.py` → `security/idempotency.py`
- `safety_guards.py` → `security/safety.py`
- `secret_rotation_monitor.py` → `security/secret_rotation.py`
- `pii_masking/` → `security/pii/`

## Usage

```python
# Imports recommandés (nouveaux)
from services.security.authentication import AccessTokenService, RefreshTokenService
from services.security.csrf import generate_csrf_token, validate_csrf_token
from services.security.spam import SpamProtection
from services.security.idempotency import IdempotencyService
from services.security.safety import SafetyGuards
from services.security.secret_rotation import SecretRotationMonitor
from services.security.pii import mask_pii

# Imports de compatibilité (DEPRECATED, à migrer)
# from services.access_token_service import AccessTokenService
# from services.refresh_token_service import RefreshTokenService
```

## Documentation

- Architecture : `docs/SECURITY_ARCHITECTURE.md`
- Migration : `PLAN_CONSOLIDATION_B2_SERVICES.md`

---

**Version :** 1.0.0 (B2 Refactoring)
**Date :** 7 janvier 2025
"""

# Exports publics (à compléter au fur et à mesure de la migration)
# from .authentication import AccessTokenService, RefreshTokenService
# from .csrf import generate_csrf_token, validate_csrf_token, csrf_protection_middleware
# from .spam import SpamProtection
# from .idempotency import IdempotencyService
# from .safety import SafetyGuards
# from .secret_rotation import SecretRotationMonitor

__all__ = [
    # Les exports seront ajoutés au fur et à mesure de la migration
]

__version__ = "1.0.0"
__refactoring__ = "B2 - Services Consolidation"
