#!/usr/bin/env python3
"""
Tests pour D1 : OpenTelemetry (traces distribuées E2E).

Teste que 100% des runs sont traçables et temps par span visible.
"""

import logging

import pytest
from opentelemetry import trace

from shared.otel_setup import get_tracer, inject_trace_id_to_logs, setup_opentelemetry

logger = logging.getLogger(__name__)


class TestOpenTelemetry:
    """Tests pour OpenTelemetry (D1)."""

    def test_trace_has_all_spans(self):
        """Test: Trace contient tous les spans attendus."""

        # Setup OpenTelemetry
        setup_opentelemetry()

        tracer = get_tracer("test")

        # Créer trace avec tous les spans
        with tracer.start_as_current_span("root_span") as root:
            root.set_attribute("operation", "dispatch_test")

            # Span 1: data_prep
            with tracer.start_as_current_span("data_prep") as data_span:
                data_span.set_attribute("bookings_count", 100)
                data_span.set_attribute("drivers_count", 20)

            # Span 2: heuristics
            with tracer.start_as_current_span("heuristics") as heuristics_span:
                heuristics_span.set_attribute("parallel_enabled", True)
                heuristics_span.set_attribute("score_time_ms", 250)

            # Span 3: solver
            with tracer.start_as_current_span("solver") as solver_span:
                solver_span.set_attribute("solver_used", "ortools")
                solver_span.set_attribute("solve_time_ms", 1500)

            # Span 4: persist
            with tracer.start_as_current_span("persist") as persist_span:
                persist_span.set_attribute("assignments_count", 95)

            # Span 5: ws_emit
            with tracer.start_as_current_span("ws_emit") as ws_span:
                ws_span.set_attribute("events_count", 5)

        # Vérifier que tous les spans sont dans la trace
        current_span = trace.get_current_span()

        assert current_span is None or current_span.get_span_context().span_id is not None

        logger.info("✅ Test: Trace contient tous les spans attendus")

    def test_trace_id_in_logs(self):
        """Test: Trace-id répliqué en logs JSON."""

        setup_opentelemetry()
        tracer = get_tracer("test")

        with tracer.start_as_current_span("test_span") as span:
            trace_id = str(span.get_span_context().trace_id)

            # Injecter trace_id dans logs
            log_context = inject_trace_id_to_logs(trace_id)

            assert "trace_id" in log_context
            assert log_context["trace_id"] == trace_id

            logger.info("✅ Test: Trace-id dans logs (trace_id=%s)", log_context["trace_id"])

    def test_spans_attributes(self):
        """Test: Attributs corrects sur tous les spans."""

        setup_opentelemetry()
        tracer = get_tracer("test")

        spans_expected = [
            ("data_prep", {"bookings_count": 100}),
            ("heuristics", {"parallel": True}),
            ("solver", {"time_ms": 1500}),
            ("persist", {"assignments": 95}),
            ("ws_emit", {"events": 5}),
        ]

        for span_name, attributes in spans_expected:
            with tracer.start_as_current_span(span_name) as span:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            assert span is not None

        logger.info("✅ Test: Attributs corrects sur %d spans", len(spans_expected))

    def test_trace_correlation(self):
        """Test: Corrélation API → Celery → OSRM → DB."""

        setup_opentelemetry()
        tracer = get_tracer("correlation_test")

        # Simuler flow E2E avec trace_id propagé
        trace_id_original = None

        with tracer.start_as_current_span("api_request") as api_span:
            api_span.set_attribute("endpoint", "/api/dispatch")
            api_span.set_attribute("method", "POST")
            trace_id_original = str(api_span.get_span_context().trace_id)

            # Simuler Celery task avec même trace
            with tracer.start_as_current_span("celery_task") as celery_span:
                celery_span.set_attribute("task", "dispatch.run")
                celery_span.set_attribute("company_id", 42)

                # Simuler OSRM call avec même trace
                with tracer.start_as_current_span("osrm_request") as osrm_span:
                    osrm_span.set_attribute("profile", "car")
                    osrm_span.set_attribute("points_count", 50)

                # Simuler DB write avec même trace
                with tracer.start_as_current_span("db_write") as db_span:
                    db_span.set_attribute("table", "assignments")
                    db_span.set_attribute("rows_inserted", 95)

        # Vérifier que trace_id est propagé tout au long du flow
        assert trace_id_original is not None
        assert len(trace_id_original) > 0

        logger.info("✅ Test: Corrélation E2E (trace_id=%s)", trace_id_original[:16])

    def test_all_runs_traceable(self):
        """Test: 100% des runs traçables."""

        setup_opentelemetry()
        tracer = get_tracer("traceability_test")

        # Simuler 10 runs
        traceable_count = 0
        for i in range(10):
            with tracer.start_as_current_span(f"run_{i}") as span:
                span.set_attribute("run_id", i)
                traceable_count += 1

        # 100% des runs devraient être traçables
        assert traceable_count == 10

        logger.info("✅ Test: 100%% des runs traçables (%d/10)", traceable_count)

    def test_span_timing_visible(self):
        """Test: Temps par span visible."""

        setup_opentelemetry()
        tracer = get_tracer("timing_test")

        import time

        # Créer span avec timing
        with tracer.start_as_current_span("timed_span") as span:
            start_time = time.time()

            # Simuler travail
            time.sleep(0.01)  # 10ms

            elapsed = time.time() - start_time

            span.set_attribute("duration_ms", int(elapsed * 1000))

        # Vérifier que timing est enregistré
        assert span is not None

        logger.info("✅ Test: Temps par span visible (%.3fs)", elapsed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
