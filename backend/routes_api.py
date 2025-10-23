# backend/routes_api.py
from __future__ import annotations

import os
from typing import Any

from flask_restx import Api

from routes.admin import admin_ns
from routes.analytics import analytics_ns  # /analytics
from routes.auth import auth_ns
from routes.bookings import bookings_ns
from routes.clients import clients_ns
from routes.companies import companies_ns
from routes.company_settings import settings_ns
from routes.dispatch_health import dispatch_health_ns  # /company_dispatch_health
from routes.dispatch_routes import dispatch_ns as company_dispatch_ns  # /company_dispatch
from routes.driver import driver_ns
from routes.geocode import geocode_ns
from routes.invoices import invoices_ns
from routes.medical import medical_ns
from routes.messages import messages_ns
from routes.osrm import osrm_ns
from routes.osrm_metrics import ns_osrm_metrics
from routes.payments import payments_ns
from routes.planning import planning_ns
from routes.shadow_mode_routes import shadow_mode_bp  # Shadow Mode RL
from routes.utils import utils_ns

authorizations = {
    "BearerAuth": {"type": "apiKey", "in": "header", "name": "Authorization"}
}

API_PREFIX = os.getenv("API_PREFIX", "/api").rstrip("/") or "/api"
API_VERSION = os.getenv("API_VERSION", "").strip().lstrip("v")

API_DOCS_RAW = os.getenv("API_DOCS", "/docs").strip()
# Ex: "", "off", "false", "0", "none" => désactive
if API_DOCS_RAW.lower() in ("", "off", "false", "0", "none"):
    API_DOCS: bool | str = False
else:
    API_DOCS = API_DOCS_RAW if API_DOCS_RAW.startswith("/") else f"/{API_DOCS_RAW}"

API_PREFIX_FULL = f"{API_PREFIX}/v{API_VERSION}" if API_VERSION else API_PREFIX

# 👇 Cast vers Any pour satisfaire Pylance (flask-restx accepte bool | str en runtime)
_doc_param: Any = API_DOCS

api = Api(
    title="ATMR Transport API",
    version="1.0",
    description="API pour la gestion des transports de personnes",
    prefix=API_PREFIX_FULL,
    doc=_doc_param,                   # bool (False) ou str, accepté à l’exécution
    authorizations=authorizations,
    security="BearerAuth",
    validate=True,
    default_mediatype="application/json",
)

# Existing namespaces
api.add_namespace(auth_ns, path="/auth")
api.add_namespace(clients_ns, path="/clients")
api.add_namespace(admin_ns, path="/admin")
api.add_namespace(companies_ns, path="/companies")
api.add_namespace(driver_ns, path="/driver")
api.add_namespace(bookings_ns, path="/bookings")
api.add_namespace(payments_ns, path="/payments")
api.add_namespace(utils_ns, path="/utils")
api.add_namespace(messages_ns, path="/messages")
api.add_namespace(geocode_ns, path="/geocode")
api.add_namespace(medical_ns, path="/medical")
api.add_namespace(invoices_ns, path="/invoices")
api.add_namespace(planning_ns, path="/")
api.add_namespace(osrm_ns, path="/osrm")
api.add_namespace(ns_osrm_metrics, path="/osrm-metrics")

# --- Dispatch namespaces
api.add_namespace(company_dispatch_ns, path="/company_dispatch")  # legacy endpoints
api.add_namespace(dispatch_health_ns, path="/company_dispatch_health")  # health monitoring

# --- Analytics namespace
api.add_namespace(analytics_ns, path="/analytics")

api.add_namespace(settings_ns, path="/company-settings")

# AI routes


def init_namespaces(app):
    # Enregistrer le Blueprint Shadow Mode (non-RESTX)
    app.register_blueprint(shadow_mode_bp)
    app.logger.info(
        "[api] init: prefix=%s docs=%s version=%s",
        API_PREFIX_FULL,
        API_DOCS if API_DOCS else "disabled",
        API_VERSION or "none",
    )
    api.init_app(app)
