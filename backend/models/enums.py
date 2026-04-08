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
    INSTITUTION = "INSTITUTION"
    admin = ADMIN
    client = CLIENT
    driver = DRIVER
    company = COMPANY
    institution = INSTITUTION


class InstitutionRole(str, PyEnum):
    """Rôles spécifiques au sein d'une institution.

    - ADMIN: Gestion complète de l'institution
    - REQUESTER: Peut créer et suivre des demandes de transport
    - READER: Lecture seule (suivi des demandes)
    - BILLING: Accès aux fonctions de facturation
    - CURATOR: Curateur (curatelle) — gère demandes + facturation pour ses protégés assignés
    """

    ADMIN = "institution_admin"
    REQUESTER = "institution_requester"
    READER = "institution_reader"
    BILLING = "institution_billing"
    CURATOR = "institution_curator"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class RequestStatus(str, PyEnum):
    """Statut d'une demande de transport institutionnelle.

    - DRAFT: Brouillon, pas encore envoyé aux transporteurs
    - SENT: Envoyé à un ou plusieurs transporteurs
    - CANCELLED: Annulé par l'institution
    - EXPIRED: Aucune réponse dans le délai imparti
    - ACCEPTED: Accepté par un transporteur
    - CONVERTED: Converti en Booking
    """

    DRAFT = "DRAFT"
    SENT = "SENT"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    ACCEPTED = "ACCEPTED"
    CONVERTED = "CONVERTED"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]

    @classmethod
    def editable_statuses(cls) -> list["RequestStatus"]:
        """Statuts permettant la modification de la demande."""
        return [cls.DRAFT, cls.SENT]

    @classmethod
    def cancellable_statuses(cls) -> list["RequestStatus"]:
        """Statuts permettant l'annulation."""
        return [cls.DRAFT, cls.SENT, cls.ACCEPTED]


class MissionType(str, PyEnum):
    """Type de mission de transport.

    - PATIENT_TRANSPORT: Transport de patient
    - MATERIAL_DELIVERY: Livraison de matériel/documents
    """

    PATIENT_TRANSPORT = "patient_transport"
    MATERIAL_DELIVERY = "material_delivery"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class BillingIntent(str, PyEnum):
    """Intention de facturation pour une demande.

    - PATIENT: Le patient paie
    - INSTITUTION: L'institution paie
    - CURATOR: Le curateur paie
    - SPC: Service de Protection des Citoyens
    - OTHER: Autre payeur
    """

    PATIENT = "patient"
    INSTITUTION = "institution"
    CURATOR = "curator"
    SPC = "spc"
    OTHER = "other"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class ScheduledTimeType(str, PyEnum):
    """Type d'horaire pour une demande de transport.

    - DEPARTURE: L'heure indiquée est l'heure de départ souhaitée.
      Le transporteur doit être prêt à cette heure.
    - ARRIVAL: L'heure indiquée est l'heure du rendez-vous (arrivée).
      Le transporteur doit proposer une heure de prise en charge
      pour arriver à temps.
    """

    DEPARTURE = "departure"
    ARRIVAL = "arrival"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class LocationType(str, PyEnum):
    """Type de lieu (départ ou arrivée) pour une demande de transport.

    - INSTITUTION: L'institution elle-même (clinique, EMS, hôpital)
    - DOMICILE: Domicile du patient
    - OTHER: Autre lieu (adresse libre)
    """

    INSTITUTION = "institution"
    DOMICILE = "domicile"
    OTHER = "other"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


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
    def choices(cls):
        return [e.value for e in cls]


class PaymentStatus(str, PyEnum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


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
    def choices(cls):
        return [e.value for e in cls]


class InvoiceLineType(str, PyEnum):
    # Valeurs alignées sur PostgreSQL (enum invoice_line_type)
    RIDE = "RIDE"
    MATERIAL_DELIVERY = "material_delivery"  # Livraison matériel (ajouté en minuscules)
    LATE_FEE = "LATE_FEE"
    REMINDER_FEE = "REMINDER_FEE"
    CUSTOM = "CUSTOM"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class BillingReviewStatus(str, PyEnum):
    """Workflow de contrôle de la décision de facturation (pré-facturation).

    Note: valeurs en snake_case car exposées côté API/UI.
    """

    DRAFT = "draft"
    NEEDS_REVIEW = "needs_review"
    READY = "ready"
    LOCKED = "locked"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class InvoiceBillingStrategy(str, PyEnum):
    """Stratégie de facturation (niveau facture).

    Objectif: rendre les factures requêtables/uniques sans dépendre de JSON `meta`.
    Note: valeurs en snake_case car exposées côté API/UI.
    """

    S1_PATIENT = "s1_patient"
    S2_CLINIC_MONTHLY = "s2_clinic_monthly"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class PaymentMethod(str, PyEnum):
    BANK_TRANSFER = "bank_transfer"
    CASH = "cash"
    CARD = "card"
    ADJUSTMENT = "adjustment"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


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


class CancelReason(str, PyEnum):
    """Raison d'annulation d'une réservation."""

    CANCEL = "CANCEL"  # Annulation réelle, facturée (par défaut)
    RELEASE = "RELEASE"  # Libération pour réassignation, non facturée
    # Nouvelles raisons avec justification
    CLIENT_REQUEST = "CLIENT_REQUEST"  # Client a demandé l'annulation → facturée
    CLIENT_NO_SHOW = "CLIENT_NO_SHOW"  # Client ne s'est pas présenté → facturée
    COMPANY_ISSUE = "COMPANY_ISSUE"  # Problème entreprise → non facturée
    DELAY = "DELAY"  # Retard important → non facturée
    VEHICLE_ISSUE = "VEHICLE_ISSUE"  # Problème véhicule → non facturée
    OTHER = "OTHER"  # Autre raison → non facturée par défaut


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


class PlatformBillingPeriodStatus(str, PyEnum):
    """Période de facturation plateforme LIRIE (brouillon / figée)."""

    DRAFT = "draft"
    LOCKED = "locked"


class PlatformBillingLineType(str, PyEnum):
    """Lignes de relevé plateforme (distinct de InvoiceLineType transporteur→client)."""

    SUBSCRIPTION = "subscription"
    INSTITUTION_COMMISSION = "institution_commission"
    SUPPORT_TIME = "support_time"
    ADJUSTMENT = "adjustment"


class PlatformSupportEntryCategory(str, PyEnum):
    """Catégorie d’entrée support facturable plateforme."""

    SUPPORT = "support"
    TRAINING = "training"
    CONFIGURATION = "configuration"
    OTHER = "other"


class TransferModel(str, PyEnum):
    """Modèles de transfert de courses entre entreprises.
    - SUBCONTRACT: A facture client, B facture A (sous-traitance classique)
    - ASSIGN_TO_PARTNER: B facture directement le client (cession)
    - MARKETPLACE: Plateforme comme tiers (pour plus tard)
    """

    SUBCONTRACT = "SUBCONTRACT"
    ASSIGN_TO_PARTNER = "ASSIGN_TO_PARTNER"
    MARKETPLACE = "MARKETPLACE"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class PartnershipStatus(str, PyEnum):
    """Statut d'une demande de partenariat."""

    PENDING = "PENDING"  # En attente d'acceptation
    ACCEPTED = "ACCEPTED"  # Acceptée (partenariat actif)
    REJECTED = "REJECTED"  # Refusée


class TransferStatus(str, PyEnum):
    """Statut d'un transfert de course."""

    PENDING = "PENDING"  # En attente d'acceptation
    ACCEPTED = "ACCEPTED"  # Accepté par le partenaire
    REJECTED = "REJECTED"  # Refusé par le partenaire
    COMPLETED = "COMPLETED"  # Course terminée et validée
    CANCELLED = "CANCELLED"  # Annulé

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class BillingPartyType(str, PyEnum):
    """Types de tiers payeur (données métier).

    Note: on utilise des valeurs "snake_case" car elles sont exposées côté API/UI.
    """

    PATIENT = "patient"
    CLINIC = "clinic"
    EMS = "ems"
    HOSPITAL = "hospital"
    CURATORSHIP = "curatorship"
    OPAD = "opad"
    LAWYER = "lawyer"
    FAMILY = "family"
    INSURANCE = "insurance"
    OTHER = "other"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class BillingSource(str, PyEnum):
    """Source de la décision de facturation (traçabilité).

    Permet de justifier "pourquoi" une course est facturée à tel payeur.
    Note: valeurs en snake_case car exposées côté API/UI.
    """

    DEFAULT_CLIENT = "default_client"  # Défaut du client (curatelle, etc.)
    TRANSPORT_VOUCHER = "transport_voucher"  # Bon de transport validé
    CLIENT_STAY = "client_stay"  # Séjour actif (hospitalisation)
    MANUAL_OVERRIDE = "manual_override"  # Correction manuelle backoffice
    IMPORT = "import"  # Import CSV/Excel
    SYSTEM_RULE = "system_rule"  # Règle système automatique

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class TransportVoucherType(str, PyEnum):
    """Type de bon de transport."""

    CLINIC = "clinic"  # Bon émis par une clinique
    INSURANCE = "insurance"  # Bon assurance
    OTHER = "other"  # Autre type

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class TransportVoucherStatus(str, PyEnum):
    """Statut de validation d'un bon de transport."""

    DRAFT = "draft"  # Brouillon (création)
    SUBMITTED = "submitted"  # Soumis (en attente de validation)
    VALIDATED = "validated"  # Validé (peut être utilisé pour facturation)
    REJECTED = "rejected"  # Rejeté (non valide)
    EXPIRED = "expired"  # Expiré (hors période de validité)

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


# ========== ENUMS REQUEST OFFERS (ÉTAPE 4) ==========


class OfferStatus(str, PyEnum):
    """Statut d'une offre de transport envoyée à une entreprise.

    - PENDING: En attente de réponse
    - ACCEPTED: Acceptée par l'entreprise
    - REJECTED: Refusée par l'entreprise
    - UNAVAILABLE: Indisponible (acceptée par une autre entreprise)
    - EXPIRED: Expirée (timeout sans réponse)
    """

    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNAVAILABLE = "UNAVAILABLE"
    EXPIRED = "EXPIRED"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class OfferMode(str, PyEnum):
    """Mode d'envoi d'une offre.

    - SEQUENTIAL: Envoi séquentiel selon préférences (1 par 1 avec timeout)
    - BROADCAST: Envoi simultané à toutes les entreprises éligibles
    """

    SEQUENTIAL = "sequential"
    BROADCAST = "broadcast"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class GeoUnitType(str, PyEnum):
    COUNTRY = "country"
    CANTON = "canton"
    DISTRICT = "district"
    COMMUNE = "commune"
    ZIPCODE = "zipcode"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class ServiceCoverageMode(str, PyEnum):
    A_STRICT = "A_STRICT"
    B_PICKUP_ONLY = "B_PICKUP_ONLY"
    C_INTRA_ONLY = "C_INTRA_ONLY"
    D_NATIONAL = "D_NATIONAL"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class PricingModelType(str, PyEnum):
    FLAT = "flat"
    ZONE = "zone"
    DISTANCE = "distance"
    HYBRID = "hybrid"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class DispatchOfferStatus(str, PyEnum):
    PROPOSED = "PROPOSED"
    ACCEPTED = "ACCEPTED"
    DECLINED = "DECLINED"
    EXPIRED = "EXPIRED"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]
