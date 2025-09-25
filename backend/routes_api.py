# backend/routes_api.py
from flask_restx import Api
import os

# Existing imports …
from routes.auth import auth_ns
from routes.clients import clients_ns
from routes.admin import admin_ns
from routes.companies import companies_ns
from routes.driver import driver_ns
from routes.bookings import bookings_ns
from routes.payments import payments_ns
from routes.messages import messages_ns
from routes.utils import utils_ns
from routes.ai import register_ai_routes
from routes.geocode import geocode_ns
from routes.medical import medical_ns

# --- IMPORTANT: mount both dispatch namespaces with different paths
from routes.dispatch_routes import dispatch_ns as company_dispatch_ns          # /company_dispatch (legacy: run/status/preview/trigger)

authorizations = {
    "BearerAuth": {"type": "apiKey", "in": "header", "name": "Authorization"}
}

API_PREFIX   = os.getenv("API_PREFIX", "/api").rstrip("/") or "/api"
API_VERSION  = os.getenv("API_VERSION", "").strip().lstrip("v")
API_DOCS_RAW = os.getenv("API_DOCS", "/docs").strip()
API_PREFIX_FULL = f"{API_PREFIX}/v{API_VERSION}" if API_VERSION else API_PREFIX
API_DOCS = False if API_DOCS_RAW.lower() in ("", "off", "false", "0") else (
    API_DOCS_RAW if API_DOCS_RAW.startswith("/") else f"/{API_DOCS_RAW}"
)

api = Api(
    title="ATMR Transport API",
    version="1.0",
    description="API pour la gestion des transports de personnes",
    prefix=API_PREFIX_FULL,
    doc=API_DOCS,
    authorizations=authorizations,
    security="BearerAuth",
    validate=True,
    default_mediatype="application/json",
)

# Existing namespaces
api.add_namespace(auth_ns, path='/auth')
api.add_namespace(clients_ns, path='/clients')
api.add_namespace(admin_ns, path='/admin')
api.add_namespace(companies_ns, path='/companies')
api.add_namespace(driver_ns, path='/driver')
api.add_namespace(bookings_ns, path='/bookings')
api.add_namespace(payments_ns, path='/payments')
api.add_namespace(utils_ns, path='/utils')
api.add_namespace(messages_ns, path='/messages')
api.add_namespace(geocode_ns, path='/geocode')
api.add_namespace(medical_ns, path='/medical')

# --- Dispatch namespaces
api.add_namespace(company_dispatch_ns, path='/company_dispatch')  # legacy endpoints (your existing ones)

register_ai_routes(api)

def init_namespaces(app):
    app.logger.info(
        "[api] init: prefix=%s docs=%s version=%s",
        API_PREFIX_FULL, API_DOCS if API_DOCS else "disabled", API_VERSION or "none"
    )
    api.init_app(app)
