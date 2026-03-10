# backend/routes/dispatch/__init__.py
# pyright: reportImportCycles=false
# basedpyright: reportImportCycles=false
"""Module dispatch - Routes pour le dispatch unifié.

Ce module contient toutes les routes liées au dispatch, organisées par fonctionnalité.

✅ **MIGRATION TERMINÉE** : Tous les endpoints ont été extraits depuis `routes/dispatch_routes.py`.
Les endpoints sont maintenant organisés dans des modules séparés par fonctionnalité.

Note: Les cycles d'imports sont intentionnels et gérés par Python (pattern Flask-RESTX standard).
Le namespace est défini avant les imports, ce qui permet à Python de résoudre les cycles correctement.
"""

from flask_restx import Namespace  # pyright: ignore[reportMissingImports]

# Namespace principal pour toutes les routes de dispatch
dispatch_ns = Namespace(
    "company_dispatch", description="Dispatch par journée (contrat unifié)"
)

# Imports des modules pour enregistrer automatiquement les endpoints
# Les classes Resource sont automatiquement enregistrées via les décorateurs @dispatch_ns.route()
# Note: Les imports sont placés après la définition du namespace pour éviter les cycles d'imports
# Les imports sont nécessaires pour déclencher l'enregistrement des endpoints Flask-RESTX
# Les cycles d'imports sont intentionnels et gérés par Python (pattern Flask-RESTX standard)
from routes.dispatch import (  # noqa: E402
    dispatch_advanced,  # type: ignore[reportUnusedImport]
    dispatch_assignments,  # type: ignore[reportUnusedImport]
    dispatch_delays,  # type: ignore[reportUnusedImport]
    dispatch_metrics,  # type: ignore[reportUnusedImport]
    dispatch_optimizer,  # type: ignore[reportUnusedImport]
    dispatch_rl,  # type: ignore[reportUnusedImport]
    dispatch_run,  # type: ignore[reportUnusedImport]
    dispatch_runs,  # type: ignore[reportUnusedImport]
    dispatch_scoring,  # type: ignore[reportUnusedImport]
    dispatch_settings,  # type: ignore[reportUnusedImport]
)

# Exports publics
__all__ = ["dispatch_ns"]
