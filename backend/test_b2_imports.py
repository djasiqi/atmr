#!/usr/bin/env python
"""Script de test rapide pour valider les imports après refactoring B2"""
# ruff: noqa: F401


def test_imports_b2():
    """Teste que tous les modules refactorisés peuvent être importés"""

    print("=" * 60)
    print("TEST DES IMPORTS - REFACTORING B2")
    print("=" * 60)

    errors = []

    # Test Security
    try:
        import services.security.authentication
        import services.security.csrf
        import services.security.idempotency
        import services.security.safety
        import services.security.secret_rotation
        import services.security.spam

        print("✅ services.security OK")
    except Exception as e:
        errors.append(f"❌ services.security: {e}")
        print(f"❌ services.security: {e}")

    # Test Notifications
    try:
        import services.notifications.core
        import services.notifications.proactive
        import services.notifications.push
        import services.notifications.system

        print("✅ services.notifications OK")
    except Exception as e:
        errors.append(f"❌ services.notifications: {e}")
        print(f"❌ services.notifications: {e}")

    # Test Booking
    try:
        import services.booking.invoices
        import services.booking.transfers

        print("✅ services.booking OK")
    except Exception as e:
        errors.append(f"❌ services.booking: {e}")
        print(f"❌ services.booking: {e}")

    # Test ML
    try:
        import services.ml.features
        import services.ml.models.demand_prediction
        import services.ml.models.eta_delay
        import services.ml.models.registry
        import services.ml.monitoring

        print("✅ services.ml OK")
    except Exception as e:
        errors.append(f"❌ services.ml: {e}")
        print(f"❌ services.ml: {e}")

    # Test Dispatch
    try:
        import services.dispatch.agent
        import services.dispatch.auto_reassignment
        import services.dispatch.planning
        import services.dispatch.utils

        print("✅ services.dispatch OK")
    except Exception as e:
        errors.append(f"❌ services.dispatch: {e}")
        print(f"❌ services.dispatch: {e}")

    # Test Geolocation
    try:
        import services.geolocation.core
        import services.geolocation.geofencing
        import services.geolocation.maps
        import services.geolocation.osrm

        print("✅ services.geolocation OK")
    except Exception as e:
        errors.append(f"❌ services.geolocation: {e}")
        print(f"❌ services.geolocation: {e}")

    # Test Partnerships
    try:
        import services.partnerships.core
        import services.partnerships.invoices
        import services.partnerships.invoices_pdf
        import services.partnerships.statements
        import services.partnerships.stats

        print("✅ services.partnerships OK")
    except Exception as e:
        errors.append(f"❌ services.partnerships: {e}")
        print(f"❌ services.partnerships: {e}")

    # Test Documents
    try:
        import services.documents.clamav
        import services.documents.pdf
        import services.documents.qrbill
        import services.documents.validation

        print("✅ services.documents OK")
    except Exception as e:
        errors.append(f"❌ services.documents: {e}")
        print(f"❌ services.documents: {e}")

    # Test Monitoring
    try:
        import services.monitoring.db_metrics
        import services.monitoring.prometheus
        import services.monitoring.slo
        import services.monitoring.websocket_metrics

        print("✅ services.monitoring OK")
    except Exception as e:
        errors.append(f"❌ services.monitoring: {e}")
        print(f"❌ services.monitoring: {e}")

    # Test Events
    try:
        import services.events.fanout
        import services.events.handlers
        import services.events.registry

        print("✅ services.events OK")
    except Exception as e:
        errors.append(f"❌ services.events: {e}")
        print(f"❌ services.events: {e}")

    # Test Infrastructure
    try:
        import services.infrastructure.ab_testing
        import services.infrastructure.cache
        import services.infrastructure.db_context
        import services.infrastructure.factories
        import services.infrastructure.feature_flags

        print("✅ services.infrastructure OK")
    except Exception as e:
        errors.append(f"❌ services.infrastructure: {e}")
        print(f"❌ services.infrastructure: {e}")

    # Test External
    try:
        import services.external.ai
        import services.external.holidays
        import services.external.weather

        print("✅ services.external OK")
    except Exception as e:
        errors.append(f"❌ services.external: {e}")
        print(f"❌ services.external: {e}")

    # Test Business
    try:
        import services.business.delay_tools
        import services.business.eta
        import services.business.vacations

        print("✅ services.business OK")
    except Exception as e:
        errors.append(f"❌ services.business: {e}")
        print(f"❌ services.business: {e}")

    # Test Realtime
    try:
        import services.realtime.socketio

        print("✅ services.realtime OK")
    except Exception as e:
        errors.append(f"❌ services.realtime: {e}")
        print(f"❌ services.realtime: {e}")

    print("=" * 60)
    if errors:
        print(f"❌ {len(errors)} ERREUR(S) DÉTECTÉE(S)")
        for error in errors:
            print(f"  {error}")
        return False

    print("✅ TOUS LES IMPORTS B2 FONCTIONNENT CORRECTEMENT!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys

    success = test_imports_b2()
    sys.exit(0 if success else 1)
