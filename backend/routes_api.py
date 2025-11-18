# backend/routes_api.py
# ✅ 3.2: Versioning API explicite avec /api/v1/ et /api/v2/
from __future__ import annotations

import os
from typing import Any

from flask import Response, make_response
from flask_restx import Api

from routes.admin import admin_ns
from routes.analytics import analytics_ns  # /analytics
from routes.auth import auth_ns
from routes.bookings import bookings_ns
from routes.clients import clients_ns
from routes.companies import companies_ns
from routes.company_mobile_auth import company_mobile_auth_ns
from routes.company_mobile_dispatch import company_mobile_dispatch_ns
from routes.company_settings import settings_ns
from routes.dispatch_health import dispatch_health_ns  # /company_dispatch_health
from routes.dispatch_routes import dispatch_ns as company_dispatch_ns  # /company_dispatch
from routes.driver import driver_ns
from routes.geocode import geocode_ns
from routes.invoices import invoices_ns
from routes.medical import medical_ns
from routes.messages import messages_ns
from routes.osrm import osrm_ns
from routes.osrm_health import osrm_health_ns
from routes.osrm_metrics import ns_osrm_metrics
from routes.payments import payments_ns
from routes.planning import planning_ns
from routes.prometheus_metrics import prometheus_metrics_ns
from routes.shadow_mode_routes import shadow_mode_bp  # Shadow Mode RL
from routes.utils import utils_ns

authorizations = {"BearerAuth": {"type": "apiKey", "in": "header", "name": "Authorization"}}

# ✅ 3.2: Configuration versioning API
API_PREFIX = os.getenv("API_PREFIX", "/api").rstrip("/") or "/api"
API_DOCS_RAW = os.getenv("API_DOCS", "/docs").strip()
# Ex: "", "off", "false", "0", "none" => désactive
if API_DOCS_RAW.lower() in ("", "off", "false", "0", "none"):
    API_DOCS: bool | str = False
else:
    API_DOCS = API_DOCS_RAW if API_DOCS_RAW.startswith("/") else f"/{API_DOCS_RAW}"

# 👇 Cast vers Any pour satisfaire Pylance (flask-restx accepte bool | str en runtime)
_doc_param: Any = API_DOCS

# ✅ 3.2: API v1 - Toutes les routes existantes migrées ici
_doc_v1: Any = False if API_DOCS is False else f"{API_DOCS}/v1"
api_v1 = Api(
    title="ATMR Transport API v1",
    version="1.0",
    description="API v1 pour la gestion des transports de personnes (dépréciée - migrer vers v2)",
    prefix=f"{API_PREFIX}/v1",
    doc=_doc_v1,
    authorizations=authorizations,
    security="BearerAuth",
    validate=True,
    default_mediatype="application/json",
)

# ✅ 3.2: API v2 - Pour nouvelles routes (vide pour l'instant)
# Note: doc=False pour éviter conflit de route /specs avec api_v1
# Réactiver avec doc=f"{API_DOCS}/v2" quand des routes seront ajoutées
_doc_v2: Any = False  # Désactivé temporairement jusqu'à ce que des routes soient ajoutées
api_v2 = Api(
    title="ATMR Transport API v2",
    version="2.0",
    description="API v2 pour la gestion des transports de personnes (nouvelle version)",
    prefix=f"{API_PREFIX}/v2",
    doc=_doc_v2,
    authorizations=authorizations,
    security="BearerAuth",
    validate=True,
    default_mediatype="application/json",
)

# ✅ 3.2: Migration de tous les namespaces existants vers API v1
# Routes d'authentification
api_v1.add_namespace(auth_ns, path="/auth")

# Routes clients
api_v1.add_namespace(clients_ns, path="/clients")

# Routes admin
api_v1.add_namespace(admin_ns, path="/admin")

# Routes companies
api_v1.add_namespace(companies_ns, path="/companies")

# Routes driver
api_v1.add_namespace(driver_ns, path="/driver")

# Routes bookings
api_v1.add_namespace(bookings_ns, path="/bookings")

# Routes payments
api_v1.add_namespace(payments_ns, path="/payments")

# Routes utils
api_v1.add_namespace(utils_ns, path="/utils")

# Routes messages
api_v1.add_namespace(messages_ns, path="/messages")

# Routes geocode
api_v1.add_namespace(geocode_ns, path="/geocode")

# Routes medical
api_v1.add_namespace(medical_ns, path="/medical")

# Routes invoices
api_v1.add_namespace(invoices_ns, path="/invoices")

# Routes planning
api_v1.add_namespace(planning_ns, path="/planning")

# Routes OSRM
api_v1.add_namespace(osrm_ns, path="/osrm")
api_v1.add_namespace(ns_osrm_metrics, path="/osrm-metrics")
api_v1.add_namespace(osrm_health_ns, path="/osrm")

# Dispatch namespaces
api_v1.add_namespace(company_dispatch_ns, path="/company_dispatch")
api_v1.add_namespace(dispatch_health_ns, path="/company_dispatch_health")
api_v1.add_namespace(company_mobile_auth_ns, path="/company_mobile/auth")
api_v1.add_namespace(company_mobile_dispatch_ns, path="/company_mobile/dispatch")


# Prometheus metrics export
# Ajouter une représentation personnalisée pour text/plain qui accepte les objets Response
@api_v1.representation("text/plain")
@api_v1.representation("text/plain; version=0.0.4; charset=utf-8")
def output_text_plain(data, code, headers=None):
    """Représentation personnalisée pour text/plain qui accepte les objets Response."""
    # Si data est déjà une Response, la retourner directement
    if isinstance(data, Response):
        resp = data
        resp.status_code = code
        if headers:
            resp.headers.update(headers)
        return resp
    # Sinon, créer une nouvelle Response
    resp = make_response(str(data), code)
    resp.headers.update(headers or {})
    return resp


api_v1.add_namespace(prometheus_metrics_ns, path="/prometheus")

# Analytics namespace
api_v1.add_namespace(analytics_ns, path="/analytics")

# Company settings
api_v1.add_namespace(settings_ns, path="/company-settings")

# ✅ 3.2: API v2 - Vide pour l'instant, prête pour nouvelles routes
# Exemple d'ajout futur:
# from routes.v2.bookings import bookings_v2_ns
# api_v2.add_namespace(bookings_v2_ns, path="/bookings")

# ✅ 3.2: API legacy (compatibilité) - Garde /api/* sans version pour transition
# Cette API sera supprimée dans une version future après migration complète
_keep_legacy_api = os.getenv("API_LEGACY_ENABLED", "true").lower() == "true"

if _keep_legacy_api:
    _doc_legacy: Any = False if API_DOCS is False else f"{API_DOCS}/legacy"
    api_legacy = Api(
        title="ATMR Transport API (Legacy - Déprécié)",
        version="1.0",
        description="⚠️ API Legacy - Utiliser /api/v1/ ou /api/v2/ à la place. Cette version sera supprimée dans une version future.",
        prefix=API_PREFIX,
        doc=_doc_legacy,
        serve_spec=False,  # Évite le conflit d'endpoint 'specs' avec api_v1 en tests
        authorizations=authorizations,
        security="BearerAuth",
        validate=True,
        default_mediatype="application/json",
    )

    # Dupliquer tous les namespaces vers API legacy pour compatibilité
    api_legacy.add_namespace(auth_ns, path="/auth")
    api_legacy.add_namespace(clients_ns, path="/clients")
    api_legacy.add_namespace(admin_ns, path="/admin")
    api_legacy.add_namespace(companies_ns, path="/companies")
    api_legacy.add_namespace(driver_ns, path="/driver")
    api_legacy.add_namespace(bookings_ns, path="/bookings")
    api_legacy.add_namespace(payments_ns, path="/payments")
    api_legacy.add_namespace(utils_ns, path="/utils")
    api_legacy.add_namespace(messages_ns, path="/messages")
    api_legacy.add_namespace(geocode_ns, path="/geocode")
    api_legacy.add_namespace(medical_ns, path="/medical")
    api_legacy.add_namespace(invoices_ns, path="/invoices")
    api_legacy.add_namespace(planning_ns, path="/planning")
    api_legacy.add_namespace(osrm_ns, path="/osrm")
    api_legacy.add_namespace(ns_osrm_metrics, path="/osrm-metrics")
    api_legacy.add_namespace(osrm_health_ns, path="/osrm")
    api_legacy.add_namespace(company_dispatch_ns, path="/company_dispatch")
    api_legacy.add_namespace(dispatch_health_ns, path="/company_dispatch_health")
    api_legacy.add_namespace(company_mobile_auth_ns, path="/company_mobile/auth")
    api_legacy.add_namespace(company_mobile_dispatch_ns, path="/company_mobile/dispatch")
    api_legacy.add_namespace(prometheus_metrics_ns, path="/prometheus")
    api_legacy.add_namespace(analytics_ns, path="/analytics")
    api_legacy.add_namespace(settings_ns, path="/company-settings")
else:
    api_legacy = None


def init_namespaces(app):
    """✅ 3.2: Initialise les APIs versionnées v1, v2 et legacy."""
    # Enregistrer le Blueprint Shadow Mode (non-RESTX)
    app.register_blueprint(shadow_mode_bp)

    # ✅ Enregistrer les handlers Socket.IO pour alertes proactives
    from ext import socketio
    from sockets.proactive_alerts import register_proactive_alerts_sockets

    register_proactive_alerts_sockets(socketio)

    # ✅ 3.2: Initialiser API v1 (routes existantes)
    api_v1.init_app(app)
    app.logger.info("[api] ✅ API v1 initialisée: %s/v1", API_PREFIX)

    # ✅ 3.2: Initialiser API v2 seulement quand des routes seront ajoutées
    # Pour l'instant, api_v2 est vide donc on ne l'initialise pas pour éviter conflit /specs
    # Quand des routes seront ajoutées, décommenter:
    # api_v2.init_app(app)
    # app.logger.info("[api] ✅ API v2 initialisée: %s/v2", API_PREFIX)
    app.logger.info("[api] ℹ️  API v2 prête mais non initialisée (vide pour l'instant)")

    # ✅ 3.2: Initialiser API legacy si activée (compatibilité)
    if api_legacy:
        api_legacy.init_app(app)
        app.logger.info("[api] ⚠️  API legacy initialisée: %s (dépréciée - utiliser /v1 ou /v2)", API_PREFIX)
    else:
        app.logger.info("[api] ℹ️  API legacy désactivée (API_LEGACY_ENABLED=false)")

    app.logger.info(
        "[api] Documentation: v1=%s, v2=%s, legacy=%s",
        API_DOCS if API_DOCS else "disabled",
        API_DOCS if API_DOCS else "disabled",
        API_DOCS if (API_DOCS and api_legacy) else "disabled",
    )
