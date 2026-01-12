# models/__init__.py
# ruff: noqa: I001, RUF022
"""Point d'entrée principal du package models.
Importe tous les models extraits depuis les fichiers individuels.
"""

# ========== Import db (requis par les routes) ==========
from ext import db

# ========== ML & Autonomous Systems (après les autres modèles) ==========
from .ab_test_result import ABTestResult
from .app_version_config import AppVersionConfig
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
from .booking_transfer import BookingTransfer
from .client import Client
from .company import Company
from .delay_event import DelayEvent
from .dispatch import (
    Assignment,
    DailyStats,
    DispatchMetrics,
    DispatchRun,
    DriverStatus,
    RealtimeEvent,
)
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
    PartnershipStatus,
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
from .billing_profile import CompanyBillingProfile
from .eta_accuracy_log import EtaAccuracyLog
from .invoice import (
    CompanyBillingSettings,
    Invoice,
    InvoiceLine,
    InvoicePayment,
    InvoiceReminder,
    InvoiceSequence,
)
from .medical import FavoritePlace, MedicalEstablishment, MedicalService
from .message import Message
from .ml_prediction import MLPrediction
from .partner_invoice import PartnerInvoice
from .partnership import Partnership
from .password_history import PasswordHistory  # ✅ S3: Historique mots de passe
from .payment import Payment
from .profiling_metrics import ProfilingMetrics  # ✅ 3.4: Profiling automatique
from .refresh_token import RefreshToken
from .rl_feedback import RLFeedback
from .rl_suggestion import RLSuggestion
from .rl_suggestion_metric import RLSuggestionMetric
from .secret_rotation import SecretRotation
from .task_failure import TaskFailure
from .trip_tracking import TripTracking
from .trip_tracking_archive import TripTrackingArchive
from .user import User
from .vehicle import Vehicle

# ========== EXPORTS ==========
__all__ = [
    "ABTestResult",
    "AppVersionConfig",
    "Assignment",
    "AssignmentStatus",
    "AutonomousAction",
    "Booking",
    "BookingStatus",
    "BookingTransfer",
    "BreakType",
    "Client",
    "ClientType",
    "Company",
    "CompanyBillingSettings",
    "CompanyBillingProfile",
    "CompanyPlanningSettings",
    "DailyStats",
    "DelayEvent",  # ✅ 3.5.1: Historique événements retards
    "DispatchMetrics",
    "DispatchMode",
    "DispatchRun",
    "DispatchStatus",
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
    "EtaAccuracyLog",
    "FavoritePlace",
    "GenderEnum",
    "Invoice",
    "InvoiceLine",
    "InvoiceLineType",
    "InvoicePayment",
    "InvoiceReminder",
    "InvoiceSequence",
    "InvoiceStatus",
    "MLPrediction",
    "MedicalEstablishment",
    "MedicalService",
    "Message",
    "PartnerInvoice",
    "Partnership",
    "PartnershipStatus",
    "PasswordHistory",  # ✅ S3: Historique mots de passe
    "Payment",
    "PaymentMethod",
    "PaymentStatus",
    "ProfilingMetrics",
    "RLFeedback",
    "RLSuggestion",
    "RLSuggestionMetric",
    "RealtimeEntityType",
    "RealtimeEvent",
    "RealtimeEventType",
    "RefreshToken",
    "SecretRotation",
    "SenderRole",
    "ShiftStatus",
    "ShiftType",
    "TaskFailure",
    "TripTracking",  # ✅ 3.3.3: Historique trajets
    "TripTrackingArchive",  # ✅ 3.5.2: Archive positions (partitionnée)
    "UnavailabilityReason",
    "User",
    "UserRole",
    "VacationType",
    "Vehicle",
    "_as_bool",
    "_as_dt",
    "_as_float",
    "_as_int",
    "_as_str",
    "_coerce_enum",
    "_encryption_key",
    "_encryption_key_str",
    "_iso",
    "db",
]
