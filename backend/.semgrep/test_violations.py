"""
Fichier de test pour valider les règles Semgrep architecturales.

⚠️ CE FICHIER CONTIENT VOLONTAIREMENT DES VIOLATIONS pour tester le linter.
NE PAS UTILISER COMME EXEMPLE DE CODE !

Usage:
    cd backend
    semgrep --config=.semgrep/rules/architecture.yml .semgrep/test_violations.py
"""


# Définitions factices pour éviter les erreurs de linting
class _MockDB:
    session = None


class _MockSession:
    def add(self, obj):
        pass

    def commit(self):
        pass


db = _MockDB()
db.session = _MockSession()


def get_booking(booking_id):
    """Mock function."""
    pass


# ============================================================================
# TEST 1: VIOLATION - Import direct depuis models/ dans bounded context
# Règle: no-direct-model-import-in-bounded-contexts
# ============================================================================
from models import Booking, Driver  # noqa: E402  # ❌ DOIT être détecté comme violation

# Utiliser Driver pour éviter warning unused import
_unused_driver = Driver


# ============================================================================
# TEST 2: VIOLATION - Query ORM direct dans application/
# Règle: no-direct-orm-query-in-application-api
# ============================================================================
def create_booking_wrong(data):
    """❌ Violation: Query ORM direct dans use-case."""
    booking = Booking.query.get(data["id"])  # ❌ DOIT être détecté
    db.session.add(booking)  # ❌ DOIT être détecté
    db.session.commit()  # ❌ DOIT être détecté
    return booking


# ============================================================================
# TEST 3: VIOLATION - Logique métier dans use-case
# Règle: business-logic-leak-in-use-case
# ============================================================================
def confirm_booking_wrong(booking_id):
    """❌ Violation: Logique métier dans use-case."""
    booking = get_booking(booking_id)

    # ❌ DOIT être détecté: Logique métier (validation d'état)
    if booking.status == "pending":
        booking.status = "confirmed"
    else:
        raise ValueError("Cannot confirm non-pending booking")

    return booking


# ============================================================================
# TEST 4: CODE CORRECT (ne doit PAS être détecté)
# ============================================================================
def create_booking_correct(data):
    """✅ Correct: Utilise repository."""
    from bookings.infrastructure.booking_repository import BookingRepository

    repo = BookingRepository()
    return repo.save(data)


# ============================================================================
# TEST 5: CODE CORRECT dans infrastructure/ (exception autorisée)
# ============================================================================
# Ce fichier simule infrastructure/booking_repository.py
# Les imports models/ sont autorisés ici
from models import Booking as BookingORM  # noqa: E402  # ✅ OK dans infrastructure/


class BookingRepository:
    """✅ Repository: import models/ autorisé."""

    def find_by_id(self, booking_id):
        # ✅ OK: Query ORM dans repository
        return BookingORM.query.get(booking_id)
