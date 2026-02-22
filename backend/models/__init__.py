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
from .booking_message import BookingMessage, BookingMessageSender
from .booking_transfer import BookingTransfer
from .client import Client
from .client_stay import ClientStay
from .clinic_billing_party_mapping import ClinicBillingPartyMapping
from .company import Company
from .transport_voucher import TransportVoucher, TransportVoucherFile
from .delay_event import DelayEvent
from .device_token import DeviceToken
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
    BillingIntent,
    BookingStatus,
    BreakType,
    BillingReviewStatus,
    BillingPartyType,
    BillingSource,
    ClientType,
    DispatchMode,
    DispatchStatus,
    DriverState,
    DriverType,
    GenderEnum,
    InstitutionRole,
    InvoiceBillingStrategy,
    InvoiceLineType,
    InvoiceStatus,
    LocationType,
    MissionType,
    OfferMode,  # ✅ ÉTAPE 4: Mode d'offre
    OfferStatus,  # ✅ ÉTAPE 4: Statut offre
    PartnershipStatus,
    PaymentMethod,
    PaymentStatus,
    RealtimeEntityType,
    RealtimeEventType,
    RequestStatus,
    SenderRole,
    ShiftStatus,
    ShiftType,
    TransportVoucherStatus,
    TransportVoucherType,
    UnavailabilityReason,
    UserRole,
    VacationType,
)
from .billing_profile import CompanyBillingProfile
from .billing_audit_log import BillingAuditLog
from .billing_party import BillingParty, ClientBillingParty
from .curator_team import CuratorTeam, CuratorTeamMember
from .institution import Institution
from .institution_api_key import (
    InstitutionApiKey,
    VALID_SCOPES as INSTITUTION_API_VALID_SCOPES,
    generate_api_key,
    hash_api_key,
    validate_scopes,
)
from .company_notification import CompanyNotification
from .institution_notification import InstitutionNotification
from .institution_patient import InstitutionPatient
from .patient_identity import (
    PatientAuditLog,
    PatientIdentity,
    PatientIdentityLink,
    PatientLinkSuggestion,
    PatientMatchRejection,
    PatientSyncEvent,
)
from .institution_settings import InstitutionSettings
from .institution_transport_preference import InstitutionTransportPreference  # ✅ ÉTAPE 4
from .request_offer import RequestOffer  # ✅ ÉTAPE 4
from .transport_request import TransportRequest
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
    "BookingMessage",
    "BookingMessageSender",
    "BookingStatus",
    "BookingTransfer",
    "BreakType",
    "BillingIntent",  # ✅ Intention facturation
    "BillingParty",
    "BillingPartyType",
    "BillingAuditLog",
    "BillingReviewStatus",
    "BillingSource",
    "ClientBillingParty",
    "Client",
    "ClientStay",
    "ClinicBillingPartyMapping",
    "ClientType",
    "Company",
    "CuratorTeam",
    "CuratorTeamMember",
    "CompanyBillingSettings",
    "CompanyBillingProfile",
    "CompanyNotification",
    "CompanyPlanningSettings",
    "DailyStats",
    "DelayEvent",  # ✅ 3.5.1: Historique événements retards
    "DeviceToken",  # ✅ Support multi-device pour push notifications
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
    "Institution",  # ✅ Portail institutionnel
    "InstitutionApiKey",  # ✅ API Keys DPI
    "InstitutionNotification",  # ✅ Notifications in-app
    "InstitutionPatient",  # ✅ Patients institution
    "PatientAuditLog",  # ✅ Curatelle: audit log
    "PatientIdentity",  # ✅ Curatelle: master index
    "PatientIdentityLink",  # ✅ Curatelle: liens entités
    "PatientLinkSuggestion",  # ✅ Curatelle: suggestions de lien patient
    "PatientMatchRejection",  # ✅ Curatelle: rejets matching
    "PatientSyncEvent",  # ✅ Curatelle: outbox sync
    "InstitutionSettings",  # ✅ P1: Settings institution
    "InstitutionRole",  # ✅ Rôles institution
    "InstitutionTransportPreference",  # ✅ ÉTAPE 4: Préférences transport
    "INSTITUTION_API_VALID_SCOPES",  # ✅ Scopes API DPI
    "generate_api_key",  # ✅ Utilitaire API Keys
    "hash_api_key",  # ✅ Utilitaire API Keys
    "validate_scopes",  # ✅ Utilitaire API Keys
    "Invoice",
    "InvoiceBillingStrategy",
    "InvoiceLine",
    "InvoiceLineType",
    "InvoicePayment",
    "InvoiceReminder",
    "InvoiceSequence",
    "InvoiceStatus",
    "LocationType",  # ✅ Type de lieu (institution/domicile/other)
    "MissionType",  # ✅ Type mission transport
    "MLPrediction",
    "OfferMode",  # ✅ ÉTAPE 4: Mode d'offre
    "OfferStatus",  # ✅ ÉTAPE 4: Statut offre
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
    "RequestOffer",  # ✅ ÉTAPE 4: Offres de transport
    "RequestStatus",  # ✅ Statut demande transport
    "SecretRotation",
    "SenderRole",
    "ShiftStatus",
    "ShiftType",
    "TaskFailure",
    "TransportRequest",  # ✅ Demandes transport institution
    "TransportVoucher",
    "TransportVoucherFile",
    "TransportVoucherStatus",
    "TransportVoucherType",
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
