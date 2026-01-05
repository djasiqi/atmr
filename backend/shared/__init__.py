"""Module partagé (shared kernel).

Contient le code partagé entre tous les bounded contexts :
- Événements
- Utilitaires
- Infrastructure partagée (cache, notifications, géolocalisation, etc.)
"""

from shared.error_handling import safe_execute

__all__ = ["safe_execute"]
