"""✅ Centralisation de tous les schémas Marshmallow pour validation.

Ce module exporte tous les schémas de validation organisés par domaine,
permettant des imports simplifiés et une meilleure maintenabilité.

Usage:
    from schemas import LoginSchema, RegisterSchema, BookingCreateSchema
    from schemas.validation_utils import validate_request, handle_validation_error
"""

# ============================================================
# Admin Schemas
# ============================================================
from schemas.admin_schemas import (
    AutonomousActionReviewSchema,
    AutonomousActionsListQuerySchema,
    UserRoleUpdateSchema,
)

# ============================================================
# Alert Schemas
# ============================================================
from schemas.alert_schemas import ClearAlertHistorySchema

# ============================================================
# Analytics Schemas
# ============================================================
from schemas.analytics_schemas import (
    AnalyticsDashboardQuerySchema,
    AnalyticsExportQuerySchema,
    AnalyticsInsightsQuerySchema,
    AnalyticsWeeklySummaryQuerySchema,
)

# ============================================================
# Auth Schemas
# ============================================================
from schemas.auth_schemas import (
    ChangePasswordSchema,
    LoginSchema,
    RefreshTokenSchema,
    RegisterSchema,
)

# ============================================================
# Booking Schemas
# ============================================================
from schemas.booking_schemas import (
    BookingCreateSchema,
    BookingListSchema,
    BookingUpdateSchema,
)

# ============================================================
# Client Schemas
# ============================================================
from schemas.client_schemas import ClientUpdateSchema

# ============================================================
# Company Schemas
# ============================================================
from schemas.company_schemas import (
    ClientCreateSchema,
    CompanyUpdateSchema,
    DriverCreateSchema,
    DriverVacationCreateSchema,
    ManualBookingCreateSchema,
    VehicleUpdateSchema,
)

# ============================================================
# Dispatch Schemas
# ============================================================
from schemas.dispatch_schemas import (
    AssignmentSchema,
    BookingSchema,
    DispatchProblemSchema,
    DispatchResultSchema,
    DispatchRunRequestSchema,
    DispatchRunSchema,
    DispatchSuggestionSchema,
    DriverSchema,
)

# ============================================================
# Driver Schemas
# ============================================================
from schemas.driver_schemas import DriverProfileUpdateSchema

# ============================================================
# Invoice Schemas
# ============================================================
from schemas.invoice_schemas import (
    BillingSettingsUpdateSchema,
    InvoiceGenerateSchema,
)

# ============================================================
# Medical Schemas
# ============================================================
from schemas.medical_schemas import (
    MedicalEstablishmentQuerySchema,
    MedicalServiceQuerySchema,
)

# ============================================================
# Payment Schemas
# ============================================================
from schemas.payment_schemas import (
    PaymentCreateSchema,
    PaymentStatusUpdateSchema,
)

# ============================================================
# Planning Schemas
# ============================================================
from schemas.planning_schemas import (
    PlanningShiftsQuerySchema,
    PlanningUnavailabilityQuerySchema,
    PlanningWeeklyTemplateQuerySchema,
)

# ============================================================
# Query Schemas
# ============================================================
from schemas.query_schemas import (
    DateRangeQuerySchema,
    FilterQuerySchema,
    LimitOffsetQuerySchema,
    PaginationQuerySchema,
    SearchQuerySchema,
)

# ============================================================
# Secret Rotation Schemas
# ============================================================
from schemas.secret_rotation_schemas import SecretRotationMonitoringQuerySchema

# ============================================================
# Validation Utilities
# ============================================================
from schemas.validation_utils import (
    EMAIL_VALIDATOR,
    ISO8601_DATE_REGEX,
    ISO8601_DATETIME_REGEX,
    PASSWORD_VALIDATOR,
    PHONE_VALIDATOR,
    USERNAME_VALIDATOR,
    handle_validation_error,
    validate_query_params,
    validate_request,
)

# ============================================================
# Exports publics
# ============================================================
__all__ = [  # noqa: RUF022
    "AnalyticsDashboardQuerySchema",
    "AnalyticsExportQuerySchema",
    "AnalyticsInsightsQuerySchema",
    "AnalyticsWeeklySummaryQuerySchema",
    "AssignmentSchema",
    "AutonomousActionReviewSchema",
    "AutonomousActionsListQuerySchema",
    "BillingSettingsUpdateSchema",
    "BookingCreateSchema",
    "BookingListSchema",
    "BookingSchema",
    "BookingUpdateSchema",
    "ChangePasswordSchema",
    "ClearAlertHistorySchema",
    "ClientCreateSchema",
    "ClientUpdateSchema",
    "CompanyUpdateSchema",
    "DispatchProblemSchema",
    "DispatchResultSchema",
    "DispatchRunRequestSchema",
    "DispatchRunSchema",
    "DispatchSuggestionSchema",
    "DriverCreateSchema",
    "DriverProfileUpdateSchema",
    "DriverSchema",
    "DriverVacationCreateSchema",
    "EMAIL_VALIDATOR",
    "InvoiceGenerateSchema",
    "ISO8601_DATE_REGEX",
    "ISO8601_DATETIME_REGEX",
    "LoginSchema",
    "ManualBookingCreateSchema",
    "MedicalEstablishmentQuerySchema",
    "MedicalServiceQuerySchema",
    "PASSWORD_VALIDATOR",
    "PaymentCreateSchema",
    "PaymentStatusUpdateSchema",
    "PHONE_VALIDATOR",
    "PlanningShiftsQuerySchema",
    "PlanningUnavailabilityQuerySchema",
    "PlanningWeeklyTemplateQuerySchema",
    "RefreshTokenSchema",
    "RegisterSchema",
    "SearchQuerySchema",
    "SecretRotationMonitoringQuerySchema",
    "USERNAME_VALIDATOR",
    "UserRoleUpdateSchema",
    "VehicleUpdateSchema",
    "DateRangeQuerySchema",
    "FilterQuerySchema",
    "LimitOffsetQuerySchema",
    "PaginationQuerySchema",
    "handle_validation_error",
    "validate_query_params",
    "validate_request",
]
