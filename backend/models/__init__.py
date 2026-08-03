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
from .activation_email_delivery import (
    ActivationEmailDelivery,
    BrevoWebhookEvent,
)
from .activation_session import ActivationSession
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
from .demo_access import DemoAccess
from .demo_request import DemoRequest
from .device_token import DeviceToken
from .driver_device_health_event import DriverDeviceHealthEvent
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
    BookingCreatedVia,
    BookingStatus,
    BreakType,
    BillingReviewStatus,
    BillingPartyType,
    BillingSource,
    ClientType,
    ManagementMode,
    DispatchOfferStatus,
    DispatchMode,
    DispatchTriggerOrigin,
    DispatchStatus,
    DriverState,
    DriverType,
    GenderEnum,
    GeoUnitType,
    InstitutionRole,
    InvoiceBillingStrategy,
    InvoiceLineType,
    InvoiceStatus,
    LocationType,
    MissionType,
    OfferMode,  # ✅ ÉTAPE 4: Mode d'offre
    OfferStatus,  # ✅ ÉTAPE 4: Statut offre
    PartnershipStatus,
    BillingOriginSource,
    BookingBillingOrigin,
    CommissionCancellationPolicy,
    LegalForm,
    PartnerAgreementStatus,
    PlatformBillingAccessState,
    PlatformBillingLineType,
    PlatformBillingPeriodStatus,
    PlatformBillingStateSource,
    PlatformDunningCaseStatus,
    PlatformDunningEventStatus,
    PlatformDunningEventType,
    PlatformIssuedInvoiceStatus,
    PlatformStatementItemType,
    PlatformStatementStatus,
    PlatformSupportEntryCategory,
    SubscriptionPricingMode,
    PaymentMethod,
    PaymentStatus,
    PricingModelType,
    RealtimeEntityType,
    RealtimeEventType,
    RequestStatus,
    CarrierSource,
    SenderRole,
    ServiceCoverageMode,
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
from .booking_change_event import BookingChangeAcknowledgement, BookingChangeEvent
from .booking_change_request import (
    BookingChangeRequest,
    BookingChangeRequestStatus,
    TransportActionEffectStatus,
    TransportActionNextActor,
    TransportActionStatus,
    TransportActionType,
)
from .transport_action_exchange import (
    TransportActionExchange,
    TransportActionExchangeDecision,
)
from .billing_party import BillingParty, ClientBillingParty
from .contact_request import ContactRequest
from .curator_team import CuratorTeam, CuratorTeamMember
from .institution import Institution
from .institution_reserved_username import InstitutionReservedUsername
from .institution_user_audit_event import InstitutionUserAuditEvent
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
from .institution_transport_preference import (
    InstitutionTransportPreference,
)  # ✅ ÉTAPE 4
from .request_offer import RequestOffer  # ✅ ÉTAPE 4
from .geo_unit import GeoUnit
from .service_area_pricing import (
    DispatchOffer,
    PlatformZone,
    PlatformZoneMembership,
    PlatformZoneSet,
    PricingProfile,
    PricingProfileVersion,
    ServiceArea,
)
from .transport_request import TransportRequest
from .transport_request_leg import TransportRequestLeg
from .transport_timeline_event import TransportTimelineEvent
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
from .conversation import Conversation
from .conversation_participant import ConversationParticipant
from .message import Message
from .message_read import MessageRead
from .messaging_enums import (
    ConversationContext,
    ConversationType,
    ParticipantRole,
)
from .ml_prediction import MLPrediction
from .partner_invoice import PartnerInvoice
from .partnership import Partnership
from .control_plane import (
    ControlPlaneAnomaly,
    ControlPlaneEntityOverride,
    OrganizationMembership,
    OrganizationServiceEntitlement,
    PermissionCatalog,
    PlatformOrganization,
    RoleTemplate,
    RoleTemplatePermission,
    ServiceCatalog,
)
from .platform_admin_permission_grant import PlatformAdminPermissionGrant
from .platform_billing import (
    BookingBillingOriginAudit,
    CompanyPlatformBillingConfig,
    PlatformBillingCreditor,
    PlatformBillingPeriod,
    PlatformBillingStatementItem,
    PlatformDunningCase,
    PlatformDunningEvent,
    PlatformInvoice,
    PlatformInvoiceDunningHold,
    PlatformInvoiceLine,
    PlatformInvoicePayment,
    PlatformIssuedInvoice,
    PlatformPartnerAgreement,
    PlatformPartnerAgreementSequence,
    PlatformSubscriptionPricing,
    PlatformSubscriptionPricingGrid,
    PlatformSubscriptionPricingTier,
    PlatformSupportEntry,
)
from .platform_client_indicative_fare_config import (
    PlatformClientIndicativeFareConfig,
)
from .platform_change_request import PlatformChangeRequest
from .platform_runbook_execution import PlatformRunbookExecution
from .password_history import PasswordHistory  # ✅ S3: Historique mots de passe
from .payment import Payment
from .worldline_webhook_event import WorldlineWebhookEvent
from .profiling_metrics import ProfilingMetrics  # ✅ 3.4: Profiling automatique
from .refresh_token import RefreshToken
from .mobile_device_session import AuthRotationResult, MobileDeviceSession, MobileDeviceSessionStatus
from .rl_feedback import RLFeedback
from .rl_suggestion import RLSuggestion
from .rl_suggestion_metric import RLSuggestionMetric
from .secret_rotation import SecretRotation
from .task_failure import TaskFailure
from .tracking_ingest_event import (
    TrackingDerivedRepairPending,
    TrackingIngestEvent,
)
from .tracking_shadow_observation import TrackingShadowObservation
from .tracking_session import (
    DriverLocationEnrichment,
    DriverLocationEvent,
    TrackingEventOutbox,
    TrackingSequenceGap,
    TrackingSession,
    TrackingSessionState,
)
from .trip_tracking import TripTracking
from .trip_tracking_archive import TripTrackingArchive
from .user import User
from .vehicle import Vehicle

# ========== EXPORTS ==========
__all__ = [
    "ABTestResult",
    "ActivationEmailDelivery",
    "ActivationSession",
    "AppVersionConfig",
    "BrevoWebhookEvent",
    "Assignment",
    "AssignmentStatus",
    "AutonomousAction",
    "Booking",
    "BookingMessage",
    "BookingMessageSender",
    "BookingCreatedVia",
    "BookingStatus",
    "BookingTransfer",
    "BreakType",
    "BillingIntent",  # ✅ Intention facturation
    "BillingParty",
    "BillingPartyType",
    "BillingAuditLog",
    "BookingChangeAcknowledgement",
    "BookingChangeEvent",
    "BookingChangeRequest",
    "BookingChangeRequestStatus",
    "TransportActionExchange",
    "TransportActionExchangeDecision",
    "TransportActionStatus",
    "TransportActionEffectStatus",
    "TransportActionType",
    "TransportActionNextActor",
    "BillingReviewStatus",
    "BillingSource",
    "ClientBillingParty",
    "Client",
    "ClientStay",
    "ClinicBillingPartyMapping",
    "ClientType",
    "ManagementMode",
    "Company",
    "ContactRequest",
    "ControlPlaneAnomaly",
    "ControlPlaneEntityOverride",
    "CuratorTeam",
    "CuratorTeamMember",
    "CompanyBillingSettings",
    "CompanyBillingProfile",
    "CompanyNotification",
    "CompanyPlanningSettings",
    "DailyStats",
    "DelayEvent",  # ✅ 3.5.1: Historique événements retards
    "DemoAccess",
    "DemoRequest",
    "DeviceToken",  # ✅ Support multi-device pour push notifications
    "DriverDeviceHealthEvent",
    "DispatchMetrics",
    "DispatchOffer",
    "DispatchOfferStatus",
    "DispatchMode",
    "DispatchTriggerOrigin",
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
    "GeoUnit",
    "GeoUnitType",
    "Institution",
    "InstitutionReservedUsername",
    "InstitutionUserAuditEvent",  # ✅ Portail institutionnel
    "InstitutionApiKey",  # ✅ API Keys DPI
    "InstitutionNotification",  # ✅ Notifications in-app
    "InstitutionPatient",  # ✅ Patients institution
    "OrganizationMembership",
    "OrganizationServiceEntitlement",
    "PermissionCatalog",
    "PlatformOrganization",
    "RoleTemplate",
    "RoleTemplatePermission",
    "ServiceCatalog",
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
    "Conversation",
    "ConversationContext",
    "ConversationParticipant",
    "ConversationType",
    "Message",
    "MessageRead",
    "ParticipantRole",
    "PartnerInvoice",
    "Partnership",
    "PlatformAdminPermissionGrant",
    "BookingBillingOriginAudit",
    "PlatformBillingCreditor",
    "PlatformBillingLineType",
    "PlatformBillingPeriod",
    "PlatformBillingPeriodStatus",
    "PlatformBillingStatementItem",
    "PlatformChangeRequest",
    "CompanyPlatformBillingConfig",
    "PlatformDunningCase",
    "PlatformDunningEvent",
    "PlatformInvoice",
    "PlatformInvoiceDunningHold",
    "PlatformInvoiceLine",
    "PlatformInvoicePayment",
    "PlatformIssuedInvoice",
    "PlatformPartnerAgreement",
    "PlatformPartnerAgreementSequence",
    "PlatformClientIndicativeFareConfig",
    "PlatformRunbookExecution",
    "PlatformSubscriptionPricing",
    "PlatformSubscriptionPricingGrid",
    "PlatformSubscriptionPricingTier",
    "PlatformSupportEntry",
    "PlatformSupportEntryCategory",
    "BookingBillingOrigin",
    "BillingOriginSource",
    "PlatformBillingAccessState",
    "PlatformBillingStateSource",
    "PlatformDunningCaseStatus",
    "PlatformDunningEventStatus",
    "PlatformDunningEventType",
    "PlatformStatementStatus",
    "PlatformStatementItemType",
    "PlatformIssuedInvoiceStatus",
    "SubscriptionPricingMode",
    "CommissionCancellationPolicy",
    "LegalForm",
    "PartnerAgreementStatus",
    "PartnershipStatus",
    "PasswordHistory",  # ✅ S3: Historique mots de passe
    "Payment",
    "WorldlineWebhookEvent",
    "PaymentMethod",
    "PaymentStatus",
    "PricingModelType",
    "PlatformZone",
    "PlatformZoneMembership",
    "PlatformZoneSet",
    "PricingProfile",
    "PricingProfileVersion",
    "ProfilingMetrics",
    "RLFeedback",
    "RLSuggestion",
    "RLSuggestionMetric",
    "RealtimeEntityType",
    "RealtimeEvent",
    "RealtimeEventType",
    "RefreshToken",
    "MobileDeviceSession",
    "MobileDeviceSessionStatus",
    "AuthRotationResult",
    "RequestOffer",  # ✅ ÉTAPE 4: Offres de transport
    "RequestStatus",  # ✅ Statut demande transport
    "CarrierSource",  # ✅ Mode d'exécution LIRIE / externe
    "SecretRotation",
    "SenderRole",
    "ServiceArea",
    "ServiceCoverageMode",
    "ShiftStatus",
    "ShiftType",
    "TaskFailure",
    "TransportRequest",  # ✅ Demandes transport institution
    "TransportRequestLeg",
    "TransportTimelineEvent",
    "TransportVoucher",
    "TransportVoucherFile",
    "TransportVoucherStatus",
    "TransportVoucherType",
    "TrackingDerivedRepairPending",
    "TrackingIngestEvent",
    "TrackingShadowObservation",
    "TrackingSession",
    "TrackingSessionState",
    "TrackingSequenceGap",
    "TrackingEventOutbox",
    "DriverLocationEvent",
    "DriverLocationEnrichment",
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
