#!/usr/bin/env python
"""Script de test rapide pour valider les imports après refactoring B2"""


def test_imports_b2():
    """Teste que tous les modules refactorisés peuvent être importés"""

    print("=" * 60)
    print("TEST DES IMPORTS - REFACTORING B2")
    print("=" * 60)

    errors = []

    # Test Security
    try:
        from services.security.authentication import (
            AccessTokenService,
            RefreshTokenService,
        )
        from services.security.csrf import generate_csrf_token
        from services.security.spam import SpamProtectionService

        print("✅ services.security OK")
    except Exception as e:
        errors.append(f"❌ services.security: {e}")
        print(f"❌ services.security: {e}")

    # Test Notifications
    try:
        from services.notifications.core import NotificationService
        from services.notifications.push import PushService
        from services.notifications.system import AlertingService

        print("✅ services.notifications OK")
    except Exception as e:
        errors.append(f"❌ services.notifications: {e}")
        print(f"❌ services.notifications: {e}")

    # Test Booking
    try:
        from services.booking.transfers import BookingTransferService
        from services.booking.invoices import InvoiceTransferService

        print("✅ services.booking OK")
    except Exception as e:
        errors.append(f"❌ services.booking: {e}")
        print(f"❌ services.booking: {e}")

    # Test ML
    try:
        from services.ml.features import MLFeaturesService
        from services.ml.monitoring import MLMonitoringService
        from services.ml.models.demand_prediction import DemandPredictionModel

        print("✅ services.ml OK")
    except Exception as e:
        errors.append(f"❌ services.ml: {e}")
        print(f"❌ services.ml: {e}")

    # Test Dispatch
    try:
        from services.dispatch.planning import PlanningService
        from services.dispatch.auto_reassignment import AutoReassignmentService

        print("✅ services.dispatch OK")
    except Exception as e:
        errors.append(f"❌ services.dispatch: {e}")
        print(f"❌ services.dispatch: {e}")

    # Test Geolocation
    try:
        from services.geolocation.osrm import build_distance_matrix_osrm_with_cb
        from services.geolocation.maps import geocode_address

        print("✅ services.geolocation OK")
    except Exception as e:
        errors.append(f"❌ services.geolocation: {e}")
        print(f"❌ services.geolocation: {e}")

    # Test Partnerships
    try:
        from services.partnerships.core import PartnershipService
        from services.partnerships.invoices import PartnerInvoiceService

        print("✅ services.partnerships OK")
    except Exception as e:
        errors.append(f"❌ services.partnerships: {e}")
        print(f"❌ services.partnerships: {e}")

    # Test Documents
    try:
        from services.documents.pdf import PDFService
        from services.documents.qrbill import QRBillService

        print("✅ services.documents OK")
    except Exception as e:
        errors.append(f"❌ services.documents: {e}")
        print(f"❌ services.documents: {e}")

    # Test Monitoring
    try:
        from services.monitoring.prometheus import setup_prometheus_metrics
        from services.monitoring.slo import check_slo_health

        print("✅ services.monitoring OK")
    except Exception as e:
        errors.append(f"❌ services.monitoring: {e}")
        print(f"❌ services.monitoring: {e}")

    # Test Events
    try:
        from services.events.fanout import EventFanoutService
        from services.events.registry import register_event_handler

        print("✅ services.events OK")
    except Exception as e:
        errors.append(f"❌ services.events: {e}")
        print(f"❌ services.events: {e}")

    # Test Infrastructure
    try:
        from services.infrastructure.cache import invalidate_cache
        from services.infrastructure.feature_flags import is_feature_enabled

        print("✅ services.infrastructure OK")
    except Exception as e:
        errors.append(f"❌ services.infrastructure: {e}")
        print(f"❌ services.infrastructure: {e}")

    # Test External
    try:
        from services.external.weather import WeatherService
        from services.external.holidays import HolidaysService

        print("✅ services.external OK")
    except Exception as e:
        errors.append(f"❌ services.external: {e}")
        print(f"❌ services.external: {e}")

    # Test Business
    try:
        from services.business.eta import ETAService
        from services.business.vacations import VacationService

        print("✅ services.business OK")
    except Exception as e:
        errors.append(f"❌ services.business: {e}")
        print(f"❌ services.business: {e}")

    # Test Realtime
    try:
        from services.realtime.socketio import socketio

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
    else:
        print("✅ TOUS LES IMPORTS B2 FONCTIONNENT CORRECTEMENT!")
        print("=" * 60)
        return True


if __name__ == "__main__":
    import sys

    success = test_imports_b2()
    sys.exit(0 if success else 1)
