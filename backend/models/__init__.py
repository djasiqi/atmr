# models/__init__.py
"""Point d'entrée principal du package models.
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
from .profiling_metrics import ProfilingMetrics  # ✅ 3.4: Profiling automatique
from .rl_feedback import RLFeedback
from .rl_suggestion import RLSuggestion
from .rl_suggestion_metric import RLSuggestionMetric

# A3: TaskFailure pour DLQ
from .secret_rotation import SecretRotation
from .task_failure import TaskFailure

# ========== ÉTAPE 2 : Import models extraits ==========
from .user import User
from .vehicle import Vehicle

# ========== EXPORTS ==========
__all__ = [
    "ABTestResult",
    "Assignment",
    "AssignmentStatus",
    "AutonomousAction",
    "Booking",
    "BookingStatus",
    "BreakType",
    "Client",
    "ClientType",
    "Company",
    "CompanyBillingSettings",
    "CompanyPlanningSettings",
    "DailyStats",
    "DispatchMetrics",
    "DispatchMode",
    # Models Dispatch & Temps Réel
    "DispatchRun",
    "DispatchStatus",
    # Models Driver & Planning
    "Driver",
    "DriverBreak",
    "DriverPreference",
    "DriverShift",
    "DriverState",
    "DriverStatus",
    "DriverType",
    "DriverUnavailability",
    "DriverVacation",
    "DriverWeeklyTemplate",
    "DriverWorkingConfig",
    "FavoritePlace",
    "GenderEnum",
    # Models Facturation
    "Invoice",
    "InvoiceLine",
    "InvoiceLineType",
    "InvoicePayment",
    "InvoiceReminder",
    "InvoiceSequence",
    "InvoiceStatus",
    # Models ML & Autonomous
    "MLPrediction",
    "MedicalEstablishment",
    "MedicalService",
    # Models Communication & Lieux
    "Message",
    "Payment",
    "PaymentMethod",
    "PaymentStatus",
    "ProfilingMetrics",  # ✅ 3.4: Profiling automatique
    "RLFeedback",
    "RLSuggestion",
    "RLSuggestionMetric",
    "RealtimeEntityType",
    "RealtimeEvent",
    "RealtimeEventType",
    "SecretRotation",  # Monitoring rotations de secrets
    "SenderRole",
    "ShiftStatus",
    "ShiftType",
    "TaskFailure",  # A3: DLQ model
    "UnavailabilityReason",
    # Models principaux
    "User",
    # Enums
    "UserRole",
    "VacationType",
    "Vehicle",
    "_as_bool",
    # Helpers
    "_as_dt",
    "_as_float",
    "_as_int",
    "_as_str",
    "_coerce_enum",
    "_encryption_key",
    "_encryption_key_str",
    "_iso",
    # Core
    "db",
]
