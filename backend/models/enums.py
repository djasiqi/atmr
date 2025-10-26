# models/enums.py
"""Tous les enums utilisés dans les models.
Extrait depuis models.py (lignes 121-243).
"""
from enum import Enum as PyEnum


class UserRole(str, PyEnum):
    ADMIN = "ADMIN"
    CLIENT = "CLIENT"
    DRIVER = "DRIVER"
    COMPANY = "COMPANY"
    admin = ADMIN
    client = CLIENT
    driver = DRIVER
    company = COMPANY


class BookingStatus(str, PyEnum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    ASSIGNED = "ASSIGNED"
    EN_ROUTE = "EN_ROUTE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    RETURN_COMPLETED = "RETURN_COMPLETED"
    CANCELED = "CANCELED"
    @classmethod
    def choices(cls): return [e.value for e in cls]


class PaymentStatus(str, PyEnum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    @classmethod
    def choices(cls): return [e.value for e in cls]


class GenderEnum(str, PyEnum):
    HOMME = "HOMME"
    FEMME = "FEMME"
    AUTRE = "AUTRE"
    homme = HOMME
    femme = FEMME
    autre = AUTRE


class ClientType(str, PyEnum):
    SELF_SERVICE = "SELF_SERVICE"
    PRIVATE = "PRIVATE"
    CORPORATE = "CORPORATE"


class InvoiceStatus(str, PyEnum):
    DRAFT = "draft"
    SENT = "sent"
    PARTIALLY_PAID = "partially_paid"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"
    @classmethod
    def choices(cls): return [e.value for e in cls]


class InvoiceLineType(str, PyEnum):
    RIDE = "ride"
    LATE_FEE = "late_fee"
    REMINDER_FEE = "reminder_fee"
    CUSTOM = "custom"
    @classmethod
    def choices(cls): return [e.value for e in cls]


class PaymentMethod(str, PyEnum):
    BANK_TRANSFER = "bank_transfer"
    CASH = "cash"
    CARD = "card"
    ADJUSTMENT = "adjustment"
    @classmethod
    def choices(cls): return [e.value for e in cls]


class DriverType(PyEnum):
    REGULAR = "REGULAR"
    EMERGENCY = "EMERGENCY"


class DriverState(str, PyEnum):
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"
    OFFLINE = "OFFLINE"


class VacationType(str, PyEnum):
    VACANCES = "VACANCES"
    MALADIE = "MALADIE"
    CONGES = "CONGES"
    AUTRE = "AUTRE"


class SenderRole(str, PyEnum):
    DRIVER = "DRIVER"
    COMPANY = "COMPANY"
    driver = DRIVER
    company = COMPANY


class RealtimeEventType(str, PyEnum):
    LOCATION_UPDATE = "LOCATION_UPDATE"
    STATUS_CHANGE = "STATUS_CHANGE"
    ASSIGNMENT_DELTA = "ASSIGNMENT_DELTA"
    DELAY_DETECTED = "DELAY_DETECTED"


class RealtimeEntityType(str, PyEnum):
    DRIVER = "DRIVER"
    BOOKING = "BOOKING"
    ASSIGNMENT = "ASSIGNMENT"


class AssignmentStatus(str, PyEnum):
    SCHEDULED = "SCHEDULED"
    EN_ROUTE_PICKUP = "EN_ROUTE_PICKUP"
    ARRIVED_PICKUP = "ARRIVED_PICKUP"
    ONBOARD = "ONBOARD"
    EN_ROUTE_DROPOFF = "EN_ROUTE_DROPOFF"
    ARRIVED_DROPOFF = "ARRIVED_DROPOFF"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    NO_SHOW = "NO_SHOW"
    REASSIGNED = "REASSIGNED"


class DispatchStatus(str, PyEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ========== ENUMS PLANNING / DRIVER ==========

class ShiftType(str, PyEnum):
    REGULAR = "regular"
    STANDBY = "standby"
    REPLACEMENT = "replacement"
    TRAINING = "training"
    MAINTENANCE = "maintenance"


class ShiftStatus(str, PyEnum):
    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class UnavailabilityReason(str, PyEnum):
    VACATION = "vacation"
    SICK = "sick"
    BREAK = "break"
    PERSONAL = "personal"
    OTHER = "other"


class BreakType(str, PyEnum):
    MANDATORY = "mandatory"
    OPTIONAL = "optional"


class DispatchMode(str, PyEnum):
    """Modes de fonctionnement du système de dispatch.
    - MANUAL: Assignations 100% manuelles, aucune automatisation
    - SEMI_AUTO: Dispatch sur demande ou périodique, validation manuelle des suggestions
    - FULLY_AUTO: Système 100% autonome avec application automatique des suggestions.
    """

    MANUAL = "manual"
    SEMI_AUTO = "semi_auto"
    FULLY_AUTO = "fully_auto"
