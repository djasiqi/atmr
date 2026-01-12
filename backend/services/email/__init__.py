"""
Module `email` - Service d'envoi d'emails pour ATMR

Ce module fournit tous les services liés à l'envoi d'emails :
- Envoi de factures par email via Brevo
- Envoi de rappels de paiement
- Confirmation de paiement
- Validation d'adresses email

## Usage

```python
from services.email.brevo_provider import BrevoEmailProvider

brevo = BrevoEmailProvider()
result = brevo.send_invoice_email(...)
```

---

**Version :** 2.0.0
**Date :** 10 janvier 2026
"""

from .brevo_provider import BrevoEmailProvider
from .email_service import EmailService
from .validators import EmailValidator

__all__ = ["BrevoEmailProvider", "EmailService", "EmailValidator"]

__version__ = "2.0.0"
