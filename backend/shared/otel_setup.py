"""✅ D1: Configuration OpenTelemetry pour traces distribuées.

✅ 2.9: Instrumentation complète Flask/SQLAlchemy/Celery avec propagation contexte W3C.
Exporte traces E2E vers Tempo/Jaeger via OTLP.
"""

from __future__ import annotations

import logging
import os
from typing import Any

# ✅ 2.9: Imports optionnels OpenTelemetry (s'il n'est pas installé, les fonctions sont no-ops)
# pyright: reportMissingImports=false
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.composite import CompositeHTTPPropagator
    from opentelemetry.propagators.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Fallback no-op pour éviter erreurs
    trace = None
    OTLPSpanExporter = None
    CeleryInstrumentor = None
    FlaskInstrumentor = None
    RequestsInstrumentor = None
    SQLAlchemyInstrumentor = None
    set_global_textmap = None
    CompositeHTTPPropagator = None
    TraceContextTextMapPropagator = None
    SERVICE_NAME = None
    SERVICE_VERSION = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None

logger = logging.getLogger(__name__)


def setup_opentelemetry(
    service_name: str = "atmr-backend", service_version: str = "1.0"
) -> None:
    """✅ 2.9: Configure OpenTelemetry avec instrumentation complète.

    Instrumente:
    - Flask (requêtes HTTP avec propagation W3C Trace Context)
    - SQLAlchemy (requêtes DB)
    - Celery (tâches asynchrones)
    - Requests (requêtes HTTP externes)

    Args:
        service_name: Nom du service
        service_version: Version du service
    """
    if not OPENTELEMETRY_AVAILABLE or Resource is None:
        logger.warning(
            "[2.9] OpenTelemetry non installé - installer avec: pip install -r requirements-otel.txt"
        )
        return
    # Créer resource avec métadonnées service
    if Resource is None or TracerProvider is None:
        return
    resource = Resource(
        attributes={
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "deployment.version": os.getenv("DEPLOYMENT_VERSION", "unknown"),
        }
    )

    # Configurer provider de traces
    provider = TracerProvider(resource=resource)

    # Exporter OTLP vers Tempo/Jaeger
    if OTLPSpanExporter is None or BatchSpanProcessor is None or trace is None:
        return
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    try:
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,  # Utiliser HTTP si pas de TLS
        )

        # Ajouter processor batch pour performance
        span_processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(span_processor)

        # Activer le provider
        trace.set_tracer_provider(provider)

        # ✅ 2.9: Propagation W3C Trace Context (pour corrélation E2E)
        if (
            CompositeHTTPPropagator is None
            or TraceContextTextMapPropagator is None
            or set_global_textmap is None
        ):
            return
        propagator = CompositeHTTPPropagator([TraceContextTextMapPropagator()])
        set_global_textmap(propagator)

        # ✅ 2.9: Instrumenter requests pour traces HTTP externes
        if RequestsInstrumentor is None:
            return
        RequestsInstrumentor().instrument()

        logger.info("[2.9] OpenTelemetry configuré (endpoint=%s)", otlp_endpoint)

    except Exception as e:
        logger.warning("[2.9] Échec configuration OpenTelemetry: %s", e)
        # Continuer sans traces si échec


def instrument_flask(app) -> None:
    """✅ 2.9: Instrumente Flask pour traces HTTP avec propagation W3C.

    Args:
        app: Instance Flask
    """
    if not OPENTELEMETRY_AVAILABLE or FlaskInstrumentor is None:
        return
    try:
        FlaskInstrumentor().instrument_app(
            app,
            excluded_urls="health,/metrics",  # Exclure healthchecks
        )
        logger.info("[2.9] Flask instrumenté pour traces HTTP")
    except Exception as e:
        logger.warning("[2.9] Échec instrumentation Flask: %s", e)


def instrument_sqlalchemy(engine) -> None:
    """✅ 2.9: Instrumente SQLAlchemy pour traces requêtes DB.

    Args:
        engine: Engine SQLAlchemy
    """
    if not OPENTELEMETRY_AVAILABLE or SQLAlchemyInstrumentor is None:
        return
    try:
        SQLAlchemyInstrumentor().instrument(
            engine=engine,
            enable_commenter=True,  # Ajouter commentaires SQL avec trace_id
        )
        logger.info("[2.9] SQLAlchemy instrumenté pour traces DB")
    except Exception as e:
        logger.warning("[2.9] Échec instrumentation SQLAlchemy: %s", e)


def instrument_celery(celery_app) -> None:
    """✅ 2.9: Instrumente Celery pour traces tâches asynchrones.

    Args:
        celery_app: Instance Celery
    """
    if not OPENTELEMETRY_AVAILABLE or CeleryInstrumentor is None or trace is None:
        return
    try:
        CeleryInstrumentor().instrument_app(
            celery_app,
            tracer_provider=trace.get_tracer_provider(),
        )
        logger.info("[2.9] Celery instrumenté pour traces tâches")
    except Exception as e:
        logger.warning("[2.9] Échec instrumentation Celery: %s", e)


def get_tracer(name: str):
    """✅ D1: Récupère un tracer nommé.

    Args:
        name: Nom du tracer (ex: "engine", "osrm_client")

    Returns:
        Tracer OpenTelemetry (ou mock si non disponible)
    """
    if not OPENTELEMETRY_AVAILABLE or trace is None:
        # Retourner un mock tracer pour éviter les erreurs
        class MockTracer:
            def start_span(self, name: str, **kwargs):  # noqa: ARG002
                return MockSpan()

            def start_as_current_span(self, name: str, **kwargs):  # noqa: ARG002
                return MockSpan()

        return MockTracer()
    return trace.get_tracer(name)


class MockSpan:
    """Mock span pour éviter erreurs si OpenTelemetry non disponible."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key: str, value: Any):
        pass


def get_current_trace_id() -> str | None:
    """✅ 2.9: Récupère le trace_id actuel pour corrélation logs.

    Returns:
        Trace ID en format hex ou None si pas de trace active
    """
    if not OPENTELEMETRY_AVAILABLE or trace is None:
        return None
    span = trace.get_current_span()
    if span:
        span_context = span.get_span_context()
        if span_context.is_valid:
            return format(span_context.trace_id, "032x")
    return None


def inject_trace_id_to_logs() -> dict[str, Any]:
    """✅ 2.9: Injecte trace_id et span_id dans logs structurés.

    Returns:
        Dictionnaire avec trace_id/span_id pour logs JSON
    """
    if not OPENTELEMETRY_AVAILABLE or trace is None:
        return {}
    span = trace.get_current_span()
    if span:
        span_context = span.get_span_context()
        if span_context.is_valid:
            return {
                "trace_id": format(span_context.trace_id, "032x"),
                "span_id": format(span_context.span_id, "016x"),
            }
    return {}


def create_span(tracer, name: str, **attributes):
    """✅ D1: Crée un span avec attributs.

    Args:
        tracer: Tracer OpenTelemetry
        name: Nom du span
        **attributes: Attributs additionnels

    Returns:
        Span context manager
    """
    span = tracer.start_span(name)

    # Ajouter attributs
    for key, value in attributes.items():
        span.set_attribute(key, str(value))

    return span
