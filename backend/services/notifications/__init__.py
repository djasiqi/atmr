"""
Module `notifications` - Consolidation des services de notification ATMR

Ce module regroupe tous les services liés aux notifications :
- Notifications génériques (email, SMS, etc.)
- Push notifications (mobile, web)
- Alertes proactives et monitoring
- Interfaces de notification

## Migration B2 (7 janvier 2025)

Ce module consolide 4 services fragmentés en un seul module cohérent :
- `notification_service.py` → `notifications/core.py`
- `push_service.py` → `notifications/push.py`
- `alerting_service.py` + `proactive_alerts.py` → `notifications/alerts.py`
- `interfaces/notification_interface.py` → `notifications/interfaces.py`

## Usage

```python
# Imports recommandés (nouveaux)
from services.notifications.core import NotificationService
from services.notifications.push import PushService
from services.notifications.alerts import AlertingService, ProactiveAlertsService

# Imports de compatibilité (DEPRECATED, à migrer)
# from services.notification_service import NotificationService
# from services.push_service import PushService
```

## Documentation

- Architecture : `docs/NOTIFICATIONS_ARCHITECTURE.md`
- Migration : `PLAN_CONSOLIDATION_B2_SERVICES.md`

---

**Version :** 1.0.0 (B2 Refactoring)
**Date :** 7 janvier 2025
"""

# ========== Exports publics ==========

# Exports seront ajoutés au fur et à mesure de la migration
# from .core import NotificationService
# from .push import PushService
# from .alerts import AlertingService, ProactiveAlertsService

__all__ = [
    # Les exports seront ajoutés après migration
]

__version__ = "1.0.0"
__refactoring__ = "B2 - Services Consolidation"
