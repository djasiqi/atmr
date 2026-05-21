# backend/routes_api.py
# ✅ 3.2: Versioning API explicite avec /api/v1/ et /api/v2/
# ✅ REFACTORING: Imports tardifs pour éviter les cycles d'imports
# Les namespaces de routes sont importés dans init_namespaces() plutôt qu'au niveau du module
# pour éviter le cycle: app.py → routes_api.py → routes/* → services/* → models/* → ext.py
from __future__ import annotations

import os
from typing import Any

from flask import Response, make_response
from flask_restx import Api

# ✅ Imports tardifs des namespaces pour éviter les cycles d'imports
# Les namespaces seront importés dans init_namespaces() au lieu d'ici

authorizations = {
    "BearerAuth": {"type": "apiKey", "in": "header", "name": "Authorization"}
}

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
    description=(
        "API v1 pour la gestion des transports de personnes "
        "(dépréciée - migrer vers v2)"
    ),
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
_doc_v2: Any = (
    False  # Désactivé temporairement jusqu'à ce que des routes soient ajoutées
)
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

# ✅ REFACTORING: Les namespaces sont ajoutés dans init_namespaces() pour éviter les cycles d'imports
# Les add_namespace() ont été déplacés dans init_namespaces() pour que les routes ne soient
# importées qu'au moment de l'initialisation de l'application, pas au chargement du module


# ✅ Gestionnaires d'erreurs JWT pour Flask-RESTX
# ⚠️ IMPORTANT : Flask-RESTX a ses propres gestionnaires d'erreurs qui prennent
# priorité sur les gestionnaires Flask globaux. Il faut donc les enregistrer ici.
@api_v1.errorhandler(Exception)
def handle_jwt_errors_v1(error):
    """Intercepte les erreurs JWT pour retourner 401 au lieu de 500."""
    from jwt.exceptions import (
        ExpiredSignatureError,
        InvalidTokenError,
    )

    if isinstance(error, ExpiredSignatureError):
        return {"error": "token_expired", "message": "Signature has expired"}, 401
    if isinstance(error, InvalidTokenError):
        return {"error": "invalid_token", "message": str(error)}, 422
    # Laisser les autres exceptions être gérées par le handler par défaut
    raise error


@api_v2.errorhandler(Exception)
def handle_jwt_errors_v2(error):
    """Intercepte les erreurs JWT pour retourner 401 au lieu de 500."""
    from jwt.exceptions import (
        ExpiredSignatureError,
        InvalidTokenError,
    )

    if isinstance(error, ExpiredSignatureError):
        return {"error": "token_expired", "message": "Signature has expired"}, 401
    if isinstance(error, InvalidTokenError):
        return {"error": "invalid_token", "message": str(error)}, 422
    # Laisser les autres exceptions être gérées par le handler par défaut
    raise error


# Prometheus metrics export
# Ajouter une représentation personnalisée pour text/plain
# qui accepte les objets Response
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


# ✅ 3.2: API v2 - Vide pour l'instant, prête pour nouvelles routes
# Exemple d'ajout futur:
# from routes.v2.bookings import bookings_v2_ns
# api_v2.add_namespace(bookings_v2_ns, path="/bookings")

# ✅ 3.2: API legacy (compatibilité) - Garde /api/* sans version pour transition
# Cette API sera supprimée dans une version future après migration complète
_keep_legacy_api = os.getenv("API_LEGACY_ENABLED", "true").lower() == "true"

if _keep_legacy_api:
    # Always disable docs/specs for the legacy API to avoid endpoint collisions
    _doc_legacy: Any = False  # Désactiver complètement Swagger UI/docs pour legacy
    api_legacy = Api(
        title="ATMR Transport API (Legacy - Déprécié)",
        version="1.0",
        description=(
            "⚠️ API Legacy - Utiliser /api/v1/ ou /api/v2/ à la place. "
            "Cette version sera supprimée dans une version future."
        ),
        prefix=API_PREFIX,
        doc=_doc_legacy,  # completely disable swagger UI/docs for legacy
        serve_spec=False,  # ensure no spec endpoint is created
        authorizations=authorizations,
        security="BearerAuth",
        validate=True,
        default_mediatype="application/json",
    )

    # ✅ REFACTORING: Les namespaces legacy sont ajoutés dans init_namespaces() pour éviter les cycles d'imports
else:
    api_legacy = None


def init_namespaces(app):
    """✅ 3.2: Initialise les APIs versionnées v1, v2 et legacy.

    ✅ REFACTORING: Les imports des routes sont faits ici plutôt qu'au niveau du module
    pour éviter le cycle: app.py → routes_api.py → routes/* → services/* → models/* → ext.py
    """
    # ruff: noqa: I001 - Imports organisés par domaine fonctionnel
    # ✅ Imports tardifs des namespaces pour éviter les cycles d'imports
    from routes.admin import admin_ns
    from routes.ai_routes import ai_ns
    from routes.analytics import analytics_ns  # /analytics
    from routes.app_version import app_version_ns  # Version check mobile
    from routes.auth import auth_ns
    from routes.billing_review import billing_review_ns  # ✅ P5: Contrôle facturation
    from routes.bookings import bookings_ns
    from routes.clients import clients_ns
    from routes.companies import companies_ns
    from routes.contact import contact_ns
    from routes.demo_requests import (
        admin_demo_accesses_ns,
        admin_demo_requests_ns,
        demo_access_ns,
        demo_requests_ns,
    )
    from routes.company_mobile_auth import company_mobile_auth_ns
    from routes.company_mobile_dispatch import company_mobile_dispatch_ns
    from routes.company_mobile_partnerships import company_mobile_partnerships_ns
    from routes.company_settings import settings_ns
    from routes.dispatch import (
        dispatch_ns as company_dispatch_ns,  # /company_dispatch
    )
    from routes.gateway_auth import gateway_auth_bp
    from routes.dispatch_health import dispatch_health_ns  # /company_dispatch_health
    from routes.driver import driver_ns
    from routes.email import email_ns  # Configuration emails transactionnels (Brevo)
    from routes.geocode import geocode_ns

    __import__("routes.geocode_zones")  # enregistre /geocode/zones sur geocode_ns
    from routes.institutions import institutions_ns  # ✅ Portail institutionnel
    from routes.institution_patients import (
        institution_patients_ns,
    )  # ✅ Patients institution
    from routes.institution_requests import (
        institution_requests_ns,
    )  # ✅ Demandes transport
    from routes.institution_settings import (
        institution_settings_ns,
    )  # ✅ ÉTAPE 4: Paramètres institution
    from routes.institution_billing import (
        institution_billing_ns,
    )  # ✅ ÉTAPE 5: Facturation institution
    from routes.institution_teams import institution_teams_ns  # ✅ Curatelle: équipes
    from routes.institution_notifications import (
        institution_notifications_ns,
    )  # ✅ Notifications in-app
    from routes.booking_messages import booking_messages_ns  # ✅ Mini-chat booking
    from routes.company_notifications import (
        company_notifications_ns,
    )  # ✅ Notifications company
    from routes.company_request_offers import (
        company_offers_ns,
    )  # ✅ ÉTAPE 4: Offres côté company
    from routes.company_security import (
        security_ns,
    )  # ✅ Security Tab V2: audit logs, export, policy
    from routes.invoices import invoices_ns
    from routes.medical import medical_ns
    from routes.messages import messages_ns
    from routes.conversations import conversations_ns
    from routes.messages_hub import messages_hub_ns
    from routes.osrm import osrm_ns
    from routes.osrm_health import osrm_health_ns
    from routes.osrm_metrics import ns_osrm_metrics
    from routes.partnerships import partnerships_ns
    from routes.payments import payments_ns
    from routes.planning import planning_ns
    from routes.pricing import pricing_ns
    from routes.prometheus_metrics import prometheus_metrics_ns
    from routes.realtime_gateway import realtime_gateway_ns
    from routes.secret_rotation_monitoring import secret_rotation_ns
    from routes.security_monitoring import (
        security_monitoring_ns,
    )  # ✅ S3: Monitoring sécurité
    from routes.shadow_mode_routes import shadow_mode_bp  # Shadow Mode RL
    from routes.saferpay_notify import saferpay_notify_bp
    from routes.transport_vouchers import (  # ✅ P3: Bons de transport
        transport_vouchers_ns,
    )
    from routes.platform_ops import platform_ops_ns
    from routes.public_stats import public_stats_ns
    from routes.utils import utils_ns

    # Routes RL (uniquement disponible dans l'environnement RL)
    try:
        from routes.rl_routes import rl_ns
    except ImportError:
        rl_ns = None

    # Enregistrer le Blueprint Shadow Mode (non-RESTX)
    app.register_blueprint(shadow_mode_bp)
    app.register_blueprint(gateway_auth_bp)
    app.register_blueprint(saferpay_notify_bp, url_prefix=f"{API_PREFIX}/v1")

    # ❌ TEMPORAIREMENT DÉSACTIVÉ POUR TEST
    # ✅ Enregistrer les handlers Socket.IO pour alertes proactives
    # from ext import socketio
    # from sockets.proactive_alerts import register_proactive_alerts_sockets
    # register_proactive_alerts_sockets(socketio)

    # ✅ IMPORTANT:
    # api_v1/api_legacy sont des singletons de module. Ils peuvent déjà avoir été
    # initialisés avec une autre app (ex: suite de tests). Il faut donc binder l'API
    # à l'app courante AVANT d'appeler add_namespace(), sinon Flask-RESTX tente
    # d'enregistrer les routes sur l'ancienne app (et échoue après le 1er request).

    # ✅ 3.2: Initialiser API v1 (binding à l'app courante)
    api_v1.init_app(app)
    app.logger.info("[api] ✅ API v1 initialisée: %s/v1", API_PREFIX)

    # ✅ 3.2: Ajouter tous les namespaces à API v1 (déplacé ici pour éviter cycles d'imports)
    # Routes d'authentification
    api_v1.add_namespace(auth_ns, path="/auth")

    # Routes clients
    api_v1.add_namespace(clients_ns, path="/clients")

    # Routes admin
    api_v1.add_namespace(admin_ns, path="/admin")
    api_v1.add_namespace(secret_rotation_ns)
    api_v1.add_namespace(
        security_monitoring_ns, path="/security-monitoring"
    )  # ✅ S3: Monitoring sécurité
    api_v1.add_namespace(platform_ops_ns, path="/platform")

    # Routes companies
    api_v1.add_namespace(companies_ns, path="/companies")

    # Routes app version check (mobile)
    api_v1.add_namespace(app_version_ns, path="/app")

    # Routes driver
    api_v1.add_namespace(driver_ns, path="/driver")

    # Routes bookings
    api_v1.add_namespace(bookings_ns, path="/bookings")
    api_v1.add_namespace(booking_messages_ns, path="/bookings")  # ✅ Mini-chat booking
    api_v1.add_namespace(
        company_notifications_ns, path="/companies/notifications"
    )  # ✅ Notifications company

    # Routes payments
    api_v1.add_namespace(partnerships_ns, path="/partnerships")
    api_v1.add_namespace(payments_ns, path="/payments")

    # Routes utils
    api_v1.add_namespace(utils_ns, path="/utils")

    # Routes messages
    api_v1.add_namespace(messages_ns, path="/messages")
    api_v1.add_namespace(messages_hub_ns, path="/messages")
    api_v1.add_namespace(conversations_ns, path="/conversations")

    # Routes geocode
    api_v1.add_namespace(geocode_ns, path="/geocode")

    # Routes medical
    api_v1.add_namespace(medical_ns, path="/medical")

    # Routes invoices
    api_v1.add_namespace(invoices_ns, path="/invoices")

    # ✅ Portail institutionnel
    api_v1.add_namespace(institutions_ns, path="/institutions")
    api_v1.add_namespace(institution_patients_ns, path="/institutions/patients")
    api_v1.add_namespace(institution_requests_ns, path="/institutions/requests")
    api_v1.add_namespace(
        institution_settings_ns, path="/institutions/settings"
    )  # ✅ ÉTAPE 4
    api_v1.add_namespace(
        institution_billing_ns, path="/institutions/billing"
    )  # ✅ ÉTAPE 5
    api_v1.add_namespace(
        institution_notifications_ns, path="/institutions/notifications"
    )
    api_v1.add_namespace(
        institution_teams_ns, path="/institutions/teams"
    )  # ✅ Curatelle

    # ✅ ÉTAPE 4: Offres côté company
    api_v1.add_namespace(company_offers_ns, path="/company/request-offers")

    # ✅ Security Tab V2: audit logs, export, policy
    api_v1.add_namespace(security_ns, path="/company-settings/security")

    # ✅ P5: Routes contrôle facturation
    api_v1.add_namespace(billing_review_ns, path="/billing")

    # ✅ P3: Routes transport vouchers (bons de transport)
    api_v1.add_namespace(transport_vouchers_ns, path="/transport-vouchers")

    # Routes email (configuration Brevo)
    api_v1.add_namespace(email_ns, path="/email")

    # Routes planning
    api_v1.add_namespace(planning_ns, path="/planning")
    api_v1.add_namespace(pricing_ns, path="/pricing")

    # Routes OSRM
    api_v1.add_namespace(osrm_ns, path="/osrm")
    api_v1.add_namespace(ns_osrm_metrics, path="/osrm-metrics")
    api_v1.add_namespace(osrm_health_ns, path="/osrm")

    # Dispatch namespaces
    api_v1.add_namespace(company_dispatch_ns, path="/company_dispatch")
    api_v1.add_namespace(dispatch_health_ns, path="/company_dispatch_health")
    api_v1.add_namespace(company_mobile_auth_ns, path="/company_mobile/auth")
    api_v1.add_namespace(company_mobile_dispatch_ns, path="/company_mobile/dispatch")

    # ✅ Partnerships namespace sous /company_mobile pour l'application mobile
    # (utilise un namespace séparé pour éviter les conflits d'enregistrement)
    api_v1.add_namespace(
        company_mobile_partnerships_ns, path="/company_mobile/partnerships"
    )

    api_v1.add_namespace(prometheus_metrics_ns, path="/prometheus")
    api_v1.add_namespace(realtime_gateway_ns, path="/realtime-gateway")

    # Analytics namespace
    api_v1.add_namespace(analytics_ns, path="/analytics")

    # Estimation d'itinéraire (dashboard client, POST /ai/optimized-route)
    api_v1.add_namespace(ai_ns, path="/ai")

    # Company settings
    api_v1.add_namespace(settings_ns, path="/company-settings")

    # Routes publiques (sans auth)
    api_v1.add_namespace(public_stats_ns, path="/public/platform-stats")
    api_v1.add_namespace(contact_ns, path="/contact")
    api_v1.add_namespace(demo_requests_ns, path="/demo-requests")
    api_v1.add_namespace(admin_demo_requests_ns, path="/admin/demo_requests")
    api_v1.add_namespace(admin_demo_accesses_ns, path="/admin/demo_accesses")
    api_v1.add_namespace(demo_access_ns, path="/demo_access")

    # SMTP configuration

    # Routes RL (uniquement dans l'environnement RL)
    if rl_ns:
        api_v1.add_namespace(rl_ns, path="/rl")

    # ✅ 3.2: Initialiser API v2 seulement quand des routes seront ajoutées
    # Pour l'instant, api_v2 est vide donc on ne l'initialise pas
    # pour éviter conflit /specs
    # Quand des routes seront ajoutées, décommenter:
    # api_v2.init_app(app)
    # app.logger.info("[api] ✅ API v2 initialisée: %s/v2", API_PREFIX)
    app.logger.info("[api] ℹ️  API v2 prête mais non initialisée (vide pour l'instant)")

    # ✅ 3.2: Initialiser API legacy si activée (compatibilité)
    if api_legacy:
        api_legacy.init_app(app)
        app.logger.info(
            "[api] ⚠️  API legacy initialisée: %s (dépréciée - utiliser /v1 ou /v2)",
            API_PREFIX,
        )
        # ✅ Dupliquer tous les namespaces vers API legacy pour compatibilité
        api_legacy.add_namespace(auth_ns, path="/auth")
        api_legacy.add_namespace(clients_ns, path="/clients")
        api_legacy.add_namespace(admin_ns, path="/admin")
        api_legacy.add_namespace(platform_ops_ns, path="/platform")
        api_legacy.add_namespace(companies_ns, path="/companies")
        api_legacy.add_namespace(partnerships_ns, path="/partnerships")
        api_legacy.add_namespace(driver_ns, path="/driver")
        api_legacy.add_namespace(bookings_ns, path="/bookings")
        api_legacy.add_namespace(booking_messages_ns, path="/bookings")  # ✅ Mini-chat
        api_legacy.add_namespace(
            company_notifications_ns, path="/companies/notifications"
        )  # ✅ Notif company
        api_legacy.add_namespace(payments_ns, path="/payments")
        api_legacy.add_namespace(utils_ns, path="/utils")
        api_legacy.add_namespace(messages_ns, path="/messages")
        api_legacy.add_namespace(messages_hub_ns, path="/messages")
        api_legacy.add_namespace(conversations_ns, path="/conversations")
        api_legacy.add_namespace(geocode_ns, path="/geocode")
        api_legacy.add_namespace(medical_ns, path="/medical")
        api_legacy.add_namespace(invoices_ns, path="/invoices")
        api_legacy.add_namespace(
            institutions_ns, path="/institutions"
        )  # ✅ Portail institutionnel
        api_legacy.add_namespace(institution_patients_ns, path="/institutions/patients")
        api_legacy.add_namespace(institution_requests_ns, path="/institutions/requests")
        api_legacy.add_namespace(planning_ns, path="/planning")
        api_legacy.add_namespace(pricing_ns, path="/pricing")
        api_legacy.add_namespace(osrm_ns, path="/osrm")
        api_legacy.add_namespace(ns_osrm_metrics, path="/osrm-metrics")
        api_legacy.add_namespace(osrm_health_ns, path="/osrm")
        api_legacy.add_namespace(company_dispatch_ns, path="/company_dispatch")
        api_legacy.add_namespace(dispatch_health_ns, path="/company_dispatch_health")
        api_legacy.add_namespace(company_mobile_auth_ns, path="/company_mobile/auth")
        api_legacy.add_namespace(
            company_mobile_dispatch_ns, path="/company_mobile/dispatch"
        )
        # ✅ Partnerships namespace sous /company_mobile pour l'application mobile
        api_legacy.add_namespace(
            company_mobile_partnerships_ns, path="/company_mobile/partnerships"
        )
        api_legacy.add_namespace(prometheus_metrics_ns, path="/prometheus")
        api_legacy.add_namespace(analytics_ns, path="/analytics")
        api_legacy.add_namespace(settings_ns, path="/company-settings")
        api_legacy.add_namespace(contact_ns, path="/contact")
        api_legacy.add_namespace(demo_requests_ns, path="/demo-requests")
        api_legacy.add_namespace(admin_demo_requests_ns, path="/admin/demo_requests")
        api_legacy.add_namespace(admin_demo_accesses_ns, path="/admin/demo_accesses")
        api_legacy.add_namespace(demo_access_ns, path="/demo_access")
    else:
        app.logger.info("[api] ℹ️  API legacy désactivée (API_LEGACY_ENABLED=false)")

    app.logger.info(
        "[api] Documentation: v1=%s, v2=%s, legacy=%s",
        API_DOCS if API_DOCS else "disabled",
        API_DOCS if API_DOCS else "disabled",
        API_DOCS if (API_DOCS and api_legacy) else "disabled",
    )
