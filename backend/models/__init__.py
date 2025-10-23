# models/__init__.py
"""
Point d'entrée principal du package models.
Importe tous les models extraits depuis les fichiers individuels.
"""

# ========== Import db (requis par les routes) ==========
from ext import db

# ========== ML & Autonomous Systems (après les autres modèles) ==========
from .ab_test_result import ABTestResult
from .autonomous_action import AutonomousAction

# ========== ÉTAPE 1 : Import helpers & enums ==========
from .base import (
    _as_bool,
    _as_dt,
    _as_float,
    _as_int,
    _as_str,
    _coerce_enum,
    _encryption_key,
    _encryption_key_str,
    _iso,
)
from .booking import Booking
from .client import Client
from .company import Company
from .dispatch import Assignment, DailyStats, DispatchMetrics, DispatchRun, DriverStatus, RealtimeEvent
from .driver import (
    CompanyPlanningSettings,
    Driver,
    DriverBreak,
    DriverPreference,
    DriverShift,
    DriverUnavailability,
    DriverVacation,
    DriverWeeklyTemplate,
    DriverWorkingConfig,
)
from .enums import (
    AssignmentStatus,
    BookingStatus,
    BreakType,
    ClientType,
    DispatchMode,
    DispatchStatus,
    DriverState,
    DriverType,
    GenderEnum,
    InvoiceLineType,
    InvoiceStatus,
    PaymentMethod,
    PaymentStatus,
    RealtimeEntityType,
    RealtimeEventType,
    SenderRole,
    ShiftStatus,
    ShiftType,
    UnavailabilityReason,
    UserRole,
    VacationType,
)
from .invoice import CompanyBillingSettings, Invoice, InvoiceLine, InvoicePayment, InvoiceReminder, InvoiceSequence
from .medical import FavoritePlace, MedicalEstablishment, MedicalService
from .message import Message
from .ml_prediction import MLPrediction
from .payment import Payment
from .rl_feedback import RLFeedback
from .rl_suggestion_metric import RLSuggestionMetric

# ========== ÉTAPE 2 : Import models extraits ==========
from .user import User
from .vehicle import Vehicle

# ========== EXPORTS ==========
__all__ = [
    # Core
    'db',

    # Helpers
    '_as_dt', '_as_str', '_as_float', '_as_int', '_as_bool', '_iso', '_coerce_enum',
    '_encryption_key', '_encryption_key_str',

    # Enums
    'UserRole', 'BookingStatus', 'PaymentStatus', 'GenderEnum', 'ClientType',
    'InvoiceStatus', 'InvoiceLineType', 'PaymentMethod', 'DriverType', 'DriverState',
    'VacationType', 'SenderRole', 'RealtimeEventType', 'RealtimeEntityType',
    'AssignmentStatus', 'DispatchStatus', 'DispatchMode',
    'ShiftType', 'ShiftStatus', 'UnavailabilityReason', 'BreakType',

    # Models principaux
    'User', 'Company', 'Vehicle', 'Client', 'Booking', 'Payment',

    # Models Driver & Planning
    'Driver', 'DriverShift', 'DriverUnavailability', 'DriverWeeklyTemplate',
    'DriverBreak', 'DriverPreference', 'DriverVacation',
    'DriverWorkingConfig', 'CompanyPlanningSettings',

    # Models Facturation
    'Invoice', 'InvoiceLine', 'InvoicePayment', 'InvoiceReminder',
    'CompanyBillingSettings', 'InvoiceSequence',

    # Models Communication & Lieux
    'Message', 'FavoritePlace', 'MedicalEstablishment', 'MedicalService',

    # Models Dispatch & Temps Réel
    'DispatchRun', 'Assignment', 'DriverStatus', 'RealtimeEvent',
    'DispatchMetrics', 'DailyStats',

    # Models ML & Autonomous
    'MLPrediction', 'AutonomousAction', 'ABTestResult', 'RLSuggestionMetric', 'RLFeedback',
]
