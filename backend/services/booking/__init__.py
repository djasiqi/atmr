"""
Module `booking` - Consolidation des services de gestion des réservations

Ce module regroupe les services liés aux réservations et transferts :
- Transferts de réservations entre chauffeurs
- Gestion des factures de transfert
- Intégration avec le bounded context `bookings/`

## Migration B2 (7 janvier 2025)

Ce module consolide 3 services fragmentés :
- `booking_transfer_service.py` → `booking/transfers.py`
- `invoice_transfer_service.py` → `booking/invoices.py`
- (Intégration avec bounded context `bookings/`)

## Usage

```python
# Imports recommandés (nouveaux)
from services.booking.transfers import BookingTransferService
from services.booking.invoices import InvoiceTransferService

# Imports de compatibilité (DEPRECATED, à migrer)
# from services.booking_transfer_service import BookingTransferService
# from services.invoice_transfer_service import InvoiceTransferService
```

## Architecture

Ce module sert de pont entre le bounded context DDD `bookings/` et les services
historiques, permettant une migration progressive vers DDD complet.

## Documentation

- Architecture : `docs/BOOKING_ARCHITECTURE.md`
- Migration : `PLAN_CONSOLIDATION_B2_SERVICES.md`

---

**Version :** 1.0.0 (B2 Refactoring)  
**Date :** 7 janvier 2025
"""

# ========== Exports publics ==========

# Exports seront ajoutés au fur et à mesure de la migration
# from .transfers import BookingTransferService
# from .invoices import InvoiceTransferService

__all__ = [
    # Les exports seront ajoutés après migration
]

__version__ = "1.0.0"
__refactoring__ = "B2 - Services Consolidation"

