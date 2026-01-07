"""
Module `partnerships` - Consolidation des services de gestion des partenariats

Ce module regroupe tous les services liés aux partenariats entre entreprises :
- Gestion des partenariats
- Facturation partenaires
- Génération PDF des factures
- Relevés de compte partenaires
- Statistiques partenaires

## Migration B2 (7 janvier 2025)

Ce module consolide 5 services fragmentés :
- `partnership_service.py` → `partnerships/core.py`
- `partner_invoice_service.py` → `partnerships/invoices.py`
- `partner_invoice_pdf_service.py` → `partnerships/invoices_pdf.py`
- `partnership_statement_service.py` → `partnerships/statements.py`
- `partnership_stats_service.py` → `partnerships/stats.py`

## Usage

```python
# Imports recommandés (nouveaux)
from services.partnerships.core import PartnershipService
from services.partnerships.invoices import PartnerInvoiceService
from services.partnerships.invoices_pdf import PartnerInvoicePDFService
from services.partnerships.statements import PartnershipStatementService
from services.partnerships.stats import PartnershipStatsService

# Imports de compatibilité (DEPRECATED, à migrer)
# from services.partnerships.core import PartnershipService
```

---

**Version :** 1.0.0 (B2 Refactoring)  
**Date :** 7 janvier 2025
"""

__all__ = []

__version__ = "1.0.0"
__refactoring__ = "B2 - Services Consolidation"


