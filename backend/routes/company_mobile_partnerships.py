# backend/routes/company_mobile_partnerships.py
"""Routes pour les partenariats - Version mobile.

Ce module crée un namespace séparé pour les routes de partenariats accessibles
depuis l'application mobile sous /company_mobile/partnerships.

Les handlers réutilisent les mêmes classes que partnerships.py pour éviter
la duplication de code.
"""

from flask_restx import Namespace  # pyright: ignore[reportMissingImports]

# Importer toutes les resources de partnerships.py
from routes.partnerships import (
    PartnershipsForTransfer,
    PartnershipTransfers,
    TransferAccept,
    TransferReject,
    TransfersList,
)

# Créer un nouveau namespace pour l'API mobile
company_mobile_partnerships_ns = Namespace(
    "company_mobile_partnerships",
    description="Partenariats (Mobile)",
)

# ✅ Enregistrer les mêmes routes mais sous le nouveau namespace
company_mobile_partnerships_ns.add_resource(PartnershipsForTransfer, "/for-transfer")
company_mobile_partnerships_ns.add_resource(
    PartnershipTransfers, "/<int:partnership_id>/transfers"
)
company_mobile_partnerships_ns.add_resource(TransfersList, "/transfers")
company_mobile_partnerships_ns.add_resource(
    TransferAccept, "/transfers/<int:transfer_id>/accept"
)
company_mobile_partnerships_ns.add_resource(
    TransferReject, "/transfers/<int:transfer_id>/reject"
)
