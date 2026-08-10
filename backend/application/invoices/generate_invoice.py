"""Use-case: générer une facture.

Ce use case migre la logique métier de InvoiceService.generate_invoice()
vers l'architecture DDD.
"""

from __future__ import annotations  # noqa: I001

# pyright: reportUnusedImport=false, reportUnusedVariable=false, reportGeneralTypeIssues=false, reportUnusedFunction=false
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError

from ext import db
from shared.constants import ErrorCodes
from infrastructure.invoices.invoice_calculator import (
    InvoiceCalculator,
    round_to_5_cents,
)
from application.invoices.invoice_line_description import (
    build_merged_round_trip_invoice_line_description_from_segments,
)
from application.invoices.ride_line_draft import (
    MaterialDeliveryPriceNotConfiguredError,
    RideLineDraft,
    compute_ride_line_draft,
)
from application.invoices.round_trip_billing_lock import (
    filter_bookings_open_for_new_invoice_line,
    round_trip_component_id_sets,
)
from application.invoices.invoice_pdf_state import mark_pdf_failed, mark_pdf_ready
from infrastructure.invoices.invoice_description_builder import (
    InvoiceDescriptionBuilder,
)
from infrastructure.invoices.invoice_number_generator import InvoiceNumberGenerator
from models import (
    BillingParty,
    Booking,
    Invoice,
    InvoiceLine,
    InvoiceLineType,
    InvoiceStatus,
)
from models.enums import InvoiceBillingStrategy
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)
from repositories.invoice_line_repository import InvoiceLineRepository
from repositories.invoice_repository import InvoiceRepository
from repositories.invoice_sequence_repository import InvoiceSequenceRepository
from services.billing.billing_party_linker import (
    get_or_create_billing_party_for_legacy_bill_to_client,
    resolve_billing_party_for_clinic,
)
from services.documents.pdf import PDFService
from services.documents.qrbill import QRBillService

logger = logging.getLogger(__name__)

PERIOD_MONTH_THRESHOLD = 12
MAX_BOOKING_IDS_SHOWN = 10  # Limite le nombre d'IDs affichés dans les messages d'erreur
ROUND_TRIP_MERGE_MIN_SEGMENTS = 2
# Taille exacte d'un A/R commercial (aller + retour). Les composantes plus
# grandes correspondent a une chaine de trajets independants et ne doivent pas
# etre regroupees sur une seule ligne de facture (ex. 4 trajets enchaines a la
# meme date pour le meme patient = 4 lignes distinctes).
ROUND_TRIP_MERGE_EXACT_SEGMENTS = 2


def _is_strict_reverse_round_trip(segments: list[Any]) -> bool:
    """Vrai A/R commercial : 2 segments avec adresses inversees (A->B et B->A)
    ou lien explicite parent / retour. Refuse les chaines A->B + B->C.
    """
    if len(segments) != ROUND_TRIP_MERGE_EXACT_SEGMENTS:
        return False
    from application.invoices.round_trip_booking_pairs import (
        normalize_address_for_round_trip_comparison,
    )

    a, b = segments
    pid_a = getattr(a, "parent_booking_id", None)
    pid_b = getattr(b, "parent_booking_id", None)
    try:
        if pid_a is not None and int(pid_a) == int(getattr(b, "id", -1)):
            return True
        if pid_b is not None and int(pid_b) == int(getattr(a, "id", -1)):
            return True
    except (TypeError, ValueError):
        pass
    a_pick = normalize_address_for_round_trip_comparison(
        getattr(a, "pickup_location", "") or ""
    )
    a_drop = normalize_address_for_round_trip_comparison(
        getattr(a, "dropoff_location", "") or ""
    )
    b_pick = normalize_address_for_round_trip_comparison(
        getattr(b, "pickup_location", "") or ""
    )
    b_drop = normalize_address_for_round_trip_comparison(
        getattr(b, "dropoff_location", "") or ""
    )
    if not (a_pick and a_drop and b_pick and b_drop):
        return False
    return a_pick == b_drop and a_drop == b_pick


@dataclass(frozen=True, slots=True)
class GenerateInvoiceInput:
    """Input pour générer une facture.

    Attributes:
        company_id: ID de l'entreprise
        client_id: ID du bénéficiaire / compte porteur (optionnel si billing_opportunity_key)
        period_year: Année de facturation (ex: 2025)
        period_month: Mois de facturation (1-12)
        billing_party_id: ID du destinataire de facturation unifié (BillingParty).
            Doit appartenir à l'entreprise (company_id).
        bill_to_client_id: ID du payeur (clinique/institution).
            Si None, client_id paie directement
        clinic_company_id: ID d'une clinique (Company) pour facturation mensuelle (S2).
            Résolu via ClinicBillingPartyMapping -> BillingParty.
        reservation_ids: Liste d'IDs de réservations spécifiques.
            Si None, prend toutes les réservations non facturées
        overrides: Dict facultatif {reservation_id: {amount, vat_rate, note}}
        billing_opportunity_key: Clé opportunité sujet|payeur (autorité serveur)
        excluded_booking_ids: IDs à exclure lors de la génération par opportunité
    """

    company_id: int
    client_id: int | None
    period_year: int
    period_month: int
    billing_party_id: int | None = None
    bill_to_client_id: int | None = None
    clinic_company_id: int | None = None
    reservation_ids: list[int] | None = None
    overrides: dict[str, Any] | None = None
    global_discount_percent: float | None = None
    global_discount_note: str | None = None
    billing_opportunity_key: str | None = None
    excluded_booking_ids: list[int] | None = None


@dataclass(frozen=True, slots=True)
class GenerateInvoiceOutput:
    """Output pour générer une facture.

    Attributes:
        success: True si l'opération a réussi
        invoice_id: ID de la facture créée (si succès)
        invoice: Facture créée (si succès)
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    invoice_id: int | None = None
    invoice: Any | None = None  # Invoice model
    error: dict[str, str] | None = None
    status_code: int | None = None


class GenerateInvoiceUseCase:
    """Use-case Application: générer une facture pour un client et une période.

    Ce use case migre la logique métier de InvoiceService.generate_invoice()
    vers l'architecture DDD en utilisant les repositories et services d'infrastructure.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        invoice_repo: InvoiceRepository | None = None,
        invoice_line_repo: InvoiceLineRepository | None = None,
        invoice_sequence_repo: InvoiceSequenceRepository | None = None,
        billing_settings_repo: CompanyBillingSettingsRepository | None = None,
        booking_repo: BookingRepository | None = None,
        client_repo: ClientRepository | None = None,
        invoice_number_generator: InvoiceNumberGenerator | None = None,
        invoice_calculator: InvoiceCalculator | None = None,
        description_builder: InvoiceDescriptionBuilder | None = None,
        pdf_service: PDFService | None = None,
    ):
        """Initialise le use case avec injection de dépendances.

        Args:
            invoice_repo: Repository pour Invoice
            invoice_line_repo: Repository pour InvoiceLine
            invoice_sequence_repo: Repository pour InvoiceSequence
            billing_settings_repo: Repository pour CompanyBillingSettings
            booking_repo: Repository pour Booking
            client_repo: Repository pour Client
            invoice_number_generator: Générateur de numéro de facture
            invoice_calculator: Calculateur de facturation
            description_builder: Constructeur de descriptions
            pdf_service: Service de génération PDF
        """
        self.invoice_repo = invoice_repo or InvoiceRepository()
        self.invoice_line_repo = invoice_line_repo or InvoiceLineRepository()
        self.invoice_sequence_repo = (
            invoice_sequence_repo or InvoiceSequenceRepository()
        )
        self.billing_settings_repo = (
            billing_settings_repo or CompanyBillingSettingsRepository()
        )
        self.booking_repo = booking_repo or BookingRepository()
        self.client_repo = client_repo or ClientRepository()
        self.invoice_number_generator = (
            invoice_number_generator or InvoiceNumberGenerator()
        )
        self.invoice_calculator = invoice_calculator or InvoiceCalculator()
        self.description_builder = description_builder or InvoiceDescriptionBuilder()
        self.pdf_service = pdf_service or PDFService()

    def execute(self, input_data: GenerateInvoiceInput) -> GenerateInvoiceOutput:
        """Génère une nouvelle facture pour un client et une période.

        Args:
            input_data: Données d'entrée du use case

        Returns:
            GenerateInvoiceOutput avec la facture créée
        """
        generated_pdf_url: str | None = None
        try:
            from application.invoices.billing_opportunities import (
                build_billing_subject_snapshot,
                build_recipient_snapshot,
                load_eligible_bookings_for_opportunity,
                parse_billing_opportunity_key,
                resolve_recipient_status,
            )
            from application.invoices.subject_identity import resolve_subject_identity
            from models.institution_patient import InstitutionPatient
            from sqlalchemy import and_

            HTTP_409_CONFLICT = 409
            HTTP_422_UNPROCESSABLE = 422

            # Garde-fou: un seul mode "destinataire" à la fois.
            destinations = [
                bool(input_data.billing_party_id),
                bool(input_data.bill_to_client_id),
                bool(input_data.clinic_company_id),
            ]
            if sum(destinations) > 1:
                raise ValueError(
                    "Fournir un seul parmi billing_party_id, bill_to_client_id, clinic_company_id."
                )

            if not input_data.billing_opportunity_key and not input_data.client_id:
                raise ValueError(
                    "client_id ou billing_opportunity_key est requis pour générer une facture."
                )

            # PR3 : résolution opportunité (autorité serveur)
            opportunity_reservations: list[Booking] | None = None
            institution_patient_id_for_invoice: int | None = None
            opportunity_meta: dict[str, Any] | None = None
            effective_client_id: int | None = input_data.client_id
            opportunity_billing_party_id: int | None = None

            if input_data.billing_opportunity_key:
                parsed = parse_billing_opportunity_key(
                    input_data.billing_opportunity_key
                )
                if parsed.billing_party_id is None or parsed.subject_id is None:
                    raise ValueError("billing_opportunity_key incomplet")
                excluded = {
                    int(x) for x in (input_data.excluded_booking_ids or []) if x
                }
                opportunity_reservations = load_eligible_bookings_for_opportunity(
                    company_id=input_data.company_id,
                    parsed=parsed,
                    period_year=input_data.period_year,
                    period_month=input_data.period_month,
                    excluded_booking_ids=excluded,
                )
                if not opportunity_reservations:
                    raise ValueError(
                        "Aucune réservation éligible pour cette opportunité / période"
                    )

                if parsed.billing_party_id is None:
                    return GenerateInvoiceOutput(
                        success=False,
                        error={
                            "error": (
                                "Destinataire (BillingParty) manquant sur cette opportunité. "
                                "Rattachez un payeur aux transports avant de générer."
                            ),
                            "recipient_status": "incomplete",
                            "billing_opportunity_key": parsed.opportunity_key,
                        },
                        status_code=HTTP_422_UNPROCESSABLE,
                    )
                bp = BillingParty.query.filter_by(
                    id=int(parsed.billing_party_id),
                    company_id=input_data.company_id,
                    is_active=True,
                ).first()
                if not bp:
                    raise ValueError(
                        "BillingParty introuvable, inactif, ou n'appartient pas à l'entreprise."
                    )
                opportunity_billing_party_id = bp.id

                sample = opportunity_reservations[0]
                subject = resolve_subject_identity(sample)
                if subject.status != "resolved":
                    return GenerateInvoiceOutput(
                        success=False,
                        error={
                            "error": (
                                "Identité patient non résolue (needs_review) : "
                                "impossible de générer la facture."
                            ),
                            "identity_status": subject.status,
                            "subject_key": subject.key,
                        },
                        status_code=HTTP_422_UNPROCESSABLE,
                    )

                ip: InstitutionPatient | None = getattr(
                    sample, "institution_patient", None
                )
                if ip is None and parsed.subject_type == "institution_patient":
                    ip = db.session.get(InstitutionPatient, parsed.subject_id)

                display = (bp.display_name or "").strip()
                if ip is not None:
                    display = (
                        f"{ip.first_name or ''} {ip.last_name or ''}".strip() or display
                    )
                recipient_status = resolve_recipient_status(
                    billing_party=bp,
                    institution_patient=ip,
                    display_name=display,
                )
                if recipient_status != "ready":
                    return GenerateInvoiceOutput(
                        success=False,
                        error={
                            "error": (
                                "Adresse de facturation incomplète pour le destinataire "
                                "(nom + rue + NPA + ville requis pour un patient)."
                            ),
                            "recipient_status": recipient_status,
                            "billing_opportunity_key": parsed.opportunity_key,
                        },
                        status_code=HTTP_422_UNPROCESSABLE,
                    )

                carriers = {
                    int(b.client_id)
                    for b in opportunity_reservations
                    if getattr(b, "client_id", None) is not None
                }
                if effective_client_id is None:
                    if len(carriers) == 1:
                        effective_client_id = next(iter(carriers))
                    elif subject.carrier_client_id is not None:
                        effective_client_id = int(subject.carrier_client_id)
                    elif carriers:
                        effective_client_id = sorted(carriers)[0]
                if effective_client_id is None:
                    raise ValueError(
                        "Impossible de déterminer le client porteur (carrier) pour l'opportunité."
                    )

                if parsed.subject_type == "institution_patient":
                    institution_patient_id_for_invoice = parsed.subject_id

                # Max 1 DRAFT par (company, sujet, party, période)
                draft_q = Invoice.query.filter(
                    and_(
                        Invoice.company_id == input_data.company_id,
                        Invoice.billing_party_id == parsed.billing_party_id,
                        Invoice.period_year == input_data.period_year,
                        Invoice.period_month == input_data.period_month,
                        Invoice.status == InvoiceStatus.DRAFT,
                    )
                )
                if institution_patient_id_for_invoice is not None:
                    draft_q = draft_q.filter(
                        Invoice.institution_patient_id
                        == institution_patient_id_for_invoice
                    )
                else:
                    draft_q = draft_q.filter(
                        Invoice.client_id == effective_client_id,
                        Invoice.institution_patient_id.is_(None),
                    )
                existing_draft = (
                    draft_q.order_by(Invoice.id.asc()).with_for_update().first()
                )
                if existing_draft is not None:
                    return GenerateInvoiceOutput(
                        success=False,
                        error={
                            "error": (
                                "Une facture brouillon existe déjà pour ce sujet, "
                                "ce destinataire et cette période. "
                                "Complétez-la ou annulez-la avant d'en créer une nouvelle."
                            ),
                            "existing_invoice_id": existing_draft.id,
                            "existing_invoice_number": existing_draft.invoice_number,
                        },
                        status_code=HTTP_409_CONFLICT,
                    )

                opportunity_meta = {
                    "billing_subject_snapshot": build_billing_subject_snapshot(
                        subject,
                        display_name=display,
                        institution_patient=ip,
                        client=getattr(sample, "client", None),
                    ),
                    "recipient_snapshot": build_recipient_snapshot(
                        bp,
                        recipient_status=recipient_status,
                        institution_patient=ip,
                    ),
                    "billing_opportunity_key": parsed.opportunity_key,
                }

            # 1. Récupérer les paramètres de facturation
            billing_settings_dto = self.billing_settings_repo.find_or_create(
                input_data.company_id
            )

            # 2. Traiter les overrides
            overrides_map: dict[int, dict[str, Any]] = {}
            if input_data.overrides:
                for key, value in input_data.overrides.items():
                    try:
                        reservation_id = int(key)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(value, dict):
                        overrides_map[reservation_id] = value

            # 3. Résoudre le destinataire de facturation
            billing_party_id: int | None = None
            payer_needs_review_reason: str | None = None
            if opportunity_billing_party_id is not None:
                billing_party_id = opportunity_billing_party_id
            elif input_data.billing_party_id:
                bp = BillingParty.query.filter_by(
                    id=input_data.billing_party_id,
                    company_id=input_data.company_id,
                    is_active=True,
                ).first()
                if not bp:
                    raise ValueError(
                        "BillingParty introuvable, inactif, ou n'appartient pas à l'entreprise."
                    )
                billing_party_id = bp.id
            elif input_data.bill_to_client_id:
                bill_to_client = self.client_repo.find_model_by_id_and_company(
                    input_data.bill_to_client_id, input_data.company_id
                )
                if not bill_to_client:
                    msg = "Client payeur non trouvé"
                    raise ValueError(msg)
                if not bool(bill_to_client.is_institution):
                    logger.warning(
                        "Le client %s n'est pas marqué comme institution",
                        input_data.bill_to_client_id,
                    )

                # ✅ Nouveau: créer/associer un BillingParty (destinataire unifié)
                bp = get_or_create_billing_party_for_legacy_bill_to_client(
                    company_id=input_data.company_id,
                    bill_to_client_id=input_data.bill_to_client_id,
                )
                billing_party_id = bp.id if bp is not None else None
            elif input_data.clinic_company_id:
                # ✅ S2: Validation du mapping clinique → billing_party (prérequis pour S2)
                bp = resolve_billing_party_for_clinic(
                    company_id=input_data.company_id,
                    clinic_company_id=input_data.clinic_company_id,
                )
                if bp is None:
                    # Mapping manquant : refuser la génération et mettre les bookings en NEEDS_REVIEW
                    payer_needs_review_reason = (
                        "Destinataire de facturation clinique non configuré "
                        "(mapping clinique → billing_party manquant). "
                        "Veuillez configurer le mapping dans les paramètres de facturation."
                    )
                    billing_party_id = None
                    # Mettre les bookings en NEEDS_REVIEW avant de lever l'erreur
                    # (sera fait plus tard dans le code après récupération des reservations)
                else:
                    billing_party_id = bp.id
            else:
                # Facturation directe au client : si le client a un tiers payeur par défaut, l'utiliser
                # (PDF affichera "Client c/o Tiers payeur" + adresse du tiers)
                if effective_client_id is not None:
                    from services.billing.client_stay_resolver import (
                        resolve_default_billing_party_for_client,
                    )

                    bp = resolve_default_billing_party_for_client(
                        client_id=effective_client_id,
                        company_id=input_data.company_id,
                    )
                    if bp is not None:
                        billing_party_id = bp.id

            # 4. Récupérer les réservations
            target_statuses = ["COMPLETED", "RETURN_COMPLETED", "CANCELED"]
            if opportunity_reservations is not None:
                reservations = opportunity_reservations
                bookings_by_id = {r.id: r for r in reservations}
            elif input_data.reservation_ids:
                # Mode sélection manuelle : accepter aussi les annulations facturables
                booking_dtos = self.booking_repo.find_by_ids(input_data.reservation_ids)
                # Charger les modèles Booking pour les annulations (vérifier is_cancellation_billable / billing_override_reason)
                bookings_by_id = {
                    b.id: b
                    for b in Booking.query.filter(
                        Booking.id.in_(input_data.reservation_ids)
                    ).all()
                }
                filtered_booking_dtos = []
                for dto in booking_dtos:
                    if dto.client_id != effective_client_id:
                        continue
                    if getattr(dto, "invoice_line_id", None) is not None:
                        continue
                    # Annulation : n'accepter que si facturable (montant > 0 et motif)
                    if dto.status.value == "CANCELED":
                        booking = bookings_by_id.get(dto.id)
                        if not booking or not (getattr(dto, "amount", 0) or 0) > 0:
                            continue
                        is_billable = getattr(
                            booking, "is_cancellation_billable", False
                        ) is True or (
                            getattr(booking, "billing_override_reason", None)
                            and str(
                                getattr(booking, "billing_override_reason", "") or ""
                            ).strip()
                        )
                        if not is_billable:
                            continue
                        is_owner = dto.company_id == input_data.company_id
                        if is_owner:
                            filtered_booking_dtos.append(dto)
                        continue
                    # COMPLETED / RETURN_COMPLETED
                    if dto.status.value not in ("COMPLETED", "RETURN_COMPLETED"):
                        continue
                    is_owner = dto.company_id == input_data.company_id
                    is_executor = (
                        getattr(dto, "executing_company_id", None)
                        == input_data.company_id
                    )
                    if is_owner:
                        filtered_booking_dtos.append(dto)
                    elif is_executor:
                        from models.booking_transfer import BookingTransfer
                        from models.enums import TransferModel, TransferStatus

                        transfer = BookingTransfer.query.filter_by(
                            booking_id=int(dto.id),
                            executing_company_id=input_data.company_id,
                            transfer_model=TransferModel.ASSIGN_TO_PARTNER,
                            is_validated=True,
                            status=TransferStatus.COMPLETED,
                        ).first()
                        if transfer:
                            filtered_booking_dtos.append(dto)

                booking_ids = [dto.id for dto in filtered_booking_dtos]
                if not booking_ids:
                    msg = "Aucune réservation valide dans la sélection"
                    raise ValueError(msg)
                eligible_period = self.booking_repo.find_models_eligible_for_billing_period_by_company_and_client(
                    company_id=input_data.company_id,
                    client_id=int(effective_client_id),
                    period_year=input_data.period_year,
                    period_month=input_data.period_month,
                    billed_to_type=None,
                )
                selected_ids = {int(x) for x in booking_ids}
                comps = round_trip_component_id_sets(
                    eligible_period,
                    amount_ht_fn=None,
                )
                expanded_ids: set[int] = set(selected_ids)
                for comp in comps:
                    if comp & selected_ids:
                        expanded_ids |= comp
                eligible_by_id = {int(b.id): b for b in eligible_period}
                for sid in selected_ids:
                    if sid in bookings_by_id:
                        eligible_by_id.setdefault(sid, bookings_by_id[sid])
                expanded_bookings = [
                    eligible_by_id[i] for i in expanded_ids if i in eligible_by_id
                ]
                reservations = filter_bookings_open_for_new_invoice_line(
                    expanded_bookings
                )
                if not reservations:
                    msg = (
                        "Aucune réservation facturable : les courses sélectionnées appartiennent "
                        "à un aller-retour dont un segment est déjà facturé sur une facture valide, "
                        "ou la sélection ne contient aucune course éligible."
                    )
                    raise ValueError(msg)
                if len(booking_ids) < len(input_data.reservation_ids):
                    logger.warning(
                        (
                            "Certaines réservations ne sont pas valides ou "
                            "déjà facturées. Demandé: %s, éligibles après filtre: %s"
                        ),
                        len(input_data.reservation_ids),
                        len(booking_ids),
                    )
                bookings_by_id = {r.id: r for r in reservations}
            else:
                # Mode automatique : récupérer toutes les réservations de la période
                start_date = datetime(
                    input_data.period_year, input_data.period_month, 1
                )
                end_date = (
                    datetime(input_data.period_year + 1, 1, 1)
                    if input_data.period_month == PERIOD_MONTH_THRESHOLD
                    else datetime(
                        input_data.period_year, input_data.period_month + 1, 1
                    )
                )
                eligible_owner = self.booking_repo.find_models_eligible_for_billing_period_by_company_and_client(
                    company_id=input_data.company_id,
                    client_id=int(effective_client_id),
                    period_year=input_data.period_year,
                    period_month=input_data.period_month,
                    billed_to_type="patient",
                )

                # Pour ASSIGN_TO_PARTNER : inclure aussi les bookings où
                # l'entreprise est exécutante (y compris déjà facturés pour le graphe A/R)
                from models.booking_transfer import BookingTransfer
                from models.enums import TransferModel, TransferStatus

                assigned_bookings = (
                    Booking.query.join(BookingTransfer)
                    .filter(
                        and_(
                            Booking.executing_company_id == input_data.company_id,
                            Booking.company_id != input_data.company_id,
                            Booking.client_id == effective_client_id,
                            Booking.status.in_(target_statuses),
                            Booking.scheduled_time >= start_date,
                            Booking.scheduled_time < end_date,
                            BookingTransfer.transfer_model
                            == TransferModel.ASSIGN_TO_PARTNER,
                            BookingTransfer.is_validated.is_(True),
                            BookingTransfer.status == TransferStatus.COMPLETED,
                        )
                    )
                    .all()
                )
                merged_for_lock: dict[int, Booking] = {
                    int(b.id): b for b in eligible_owner
                }
                for b in assigned_bookings:
                    merged_for_lock[int(b.id)] = b
                reservations = filter_bookings_open_for_new_invoice_line(
                    list(merged_for_lock.values())
                )
                # Même accès qu'en mode reservation_ids (annulation / frais d'annulation)
                bookings_by_id = {r.id: r for r in reservations}

            if not reservations:
                msg = "Aucune réservation trouvée pour cette période"
                raise ValueError(msg)

            # ✅ Pré-vérification : livraisons matériel sans description
            missing_desc_ids = [
                r.id
                for r in reservations
                if (getattr(r, "mission_type", None) or "patient_transport")
                == "material_delivery"
                and not (getattr(r, "delivery_description", None) or "").strip()
            ]
            if missing_desc_ids:
                msg = (
                    "Certaines livraisons matériel n'ont pas de description. "
                    "Veuillez renseigner le champ « Description de la livraison » "
                    f"pour les réservations #{', #'.join(map(str, missing_desc_ids[:MAX_BOOKING_IDS_SHOWN]))}"
                    + (
                        f" (et {len(missing_desc_ids) - MAX_BOOKING_IDS_SHOWN} autres)"
                        if len(missing_desc_ids) > MAX_BOOKING_IDS_SHOWN
                        else ""
                    )
                    + " avant de générer la facture."
                )
                logger.warning(
                    "Livraisons matériel sans description: booking_ids=%s",
                    missing_desc_ids,
                )
                return GenerateInvoiceOutput(
                    success=False,
                    error={
                        "error": msg,
                        "error_code": ErrorCodes.MATERIAL_DELIVERY_DESCRIPTION_REQUIRED,
                        "details": {
                            "field": "delivery_description",
                            "booking_ids": missing_desc_ids,
                        },
                    },
                    status_code=400,
                )

            # ✅ P2.2: Résoudre automatiquement le payeur selon la date (séjour actif + payeur par défaut)
            from models.enums import BillingReviewStatus
            from services.billing.client_stay_resolver import (
                detect_billing_conflict_with_stay,
                find_active_stay_for_booking,
                resolve_billing_party_for_booking,
            )

            for r in reservations:
                # Résoudre le payeur automatiquement selon la date (gère transitions temporelles)
                payer_resolved = resolve_billing_party_for_booking(
                    booking=r,
                    company_id=input_data.company_id,
                )

                # Ne pas overwrite patient → clinique en facture client directe (clinic_company_id absent).
                # L'utilisateur a explicitement choisi "facturation client" ; conserver l'override "facturer patient"
                # pour les hospitalisés (sinon, après annulation, les transports repassent en "facturer clinique").
                if payer_resolved and input_data.clinic_company_id is not None:
                    # Appliquer seulement si le booking n'a pas déjà un payeur explicite
                    current_billed_to_type = (
                        getattr(r, "billed_to_type", None) or "patient"
                    ).lower()
                    current_billing_party_id = getattr(r, "billing_party_id", None)

                    # Si pas de payeur explicite, utiliser la résolution automatique
                    if (
                        current_billed_to_type == "patient"
                        and not current_billing_party_id
                    ):
                        try:
                            if payer_resolved.get("billed_to_type"):
                                r.billed_to_type = payer_resolved["billed_to_type"]
                            if payer_resolved.get("billed_to_company_id"):
                                r.billed_to_company_id = payer_resolved[
                                    "billed_to_company_id"
                                ]
                            if payer_resolved.get("billing_party_id"):
                                r.billing_party_id = payer_resolved["billing_party_id"]
                            # ✅ P4: Renseigner billing_source et billing_source_ref pour traçabilité
                            if payer_resolved.get("billing_source"):
                                from models.enums import BillingSource

                                billing_source_str = payer_resolved["billing_source"]
                                try:
                                    # Les valeurs retournées sont déjà en snake_case (comme l'enum)
                                    r.billing_source = BillingSource(billing_source_str)
                                except (ValueError, AttributeError) as e:
                                    logger.warning(
                                        "[GenerateInvoice] Booking %s: billing_source invalide: %s (%s)",
                                        r.id,
                                        billing_source_str,
                                        e,
                                    )
                            if payer_resolved.get("billing_source_ref"):
                                r.billing_source_ref = payer_resolved[
                                    "billing_source_ref"
                                ]
                            logger.info(
                                (
                                    "[GenerateInvoice] Booking %s: payeur résolu automatiquement "
                                    "(billing_party_id=%s, type=%s, source=%s, source_ref=%s)"
                                ),
                                r.id,
                                payer_resolved.get("billing_party_id"),
                                payer_resolved.get("billed_to_type", "patient"),
                                payer_resolved.get("billing_source"),
                                payer_resolved.get("billing_source_ref"),
                            )
                        except Exception:
                            # Best effort: certains DTO/mock peuvent ne pas exposer ces champs
                            pass

                # Détecter les conflits séjour vs payeur actuel
                stay = find_active_stay_for_booking(booking=r)
                if stay:
                    has_conflict, conflict_reason = detect_billing_conflict_with_stay(
                        booking=r,
                        stay=stay,
                        company_id=input_data.company_id,
                    )
                    if has_conflict:
                        try:
                            r.billing_review_status = BillingReviewStatus.NEEDS_REVIEW
                            r.billing_override_reason = conflict_reason
                            logger.warning(
                                (
                                    "[GenerateInvoice] Booking %s: conflit détecté avec séjour "
                                    "(stay_id=%s) → marqué NEEDS_REVIEW: %s"
                                ),
                                r.id,
                                stay.id,
                                conflict_reason,
                            )
                        except Exception:
                            # Best effort
                            pass

            # ✅ P1.4: si payeur non-patient mais pas de BillingParty, forcer NEEDS_REVIEW
            # ✅ S2: Si mapping clinique manquant, refuser la génération et mettre en NEEDS_REVIEW
            if payer_needs_review_reason:
                for r in reservations:
                    try:
                        r.billing_review_status = BillingReviewStatus.NEEDS_REVIEW
                        r.billing_override_reason = payer_needs_review_reason
                    except Exception:
                        # Best effort: certains DTO/mock peuvent ne pas exposer ces champs
                        pass
                # Si c'est un problème de mapping S2, refuser la génération
                if input_data.clinic_company_id and not billing_party_id:
                    db.session.commit()  # Sauvegarder les changements de statut
                    msg = (
                        f"Impossible de générer la facture : mapping clinique → billing_party manquant "
                        f"pour clinic_company_id={input_data.clinic_company_id}. "
                        f"Les bookings ont été mis en NEEDS_REVIEW. "
                        f"Veuillez configurer le mapping dans les paramètres de facturation."
                    )
                    raise ValueError(msg)

            # 5. Générer le numéro de facture
            sequence_dto = self.invoice_sequence_repo.find_or_create(
                input_data.company_id,
                input_data.period_year,
                input_data.period_month,
            )
            sequence_dto = self.invoice_sequence_repo.increment_sequence(
                sequence_dto.id
            )
            invoice_number = self.invoice_number_generator.generate(
                company_id=input_data.company_id,
                period_year=input_data.period_year,
                period_month=input_data.period_month,
                billing_settings=billing_settings_dto,
                sequence=sequence_dto,
            )

            # 6. Calculer la TVA
            vat_applicable_setting = billing_settings_dto.vat_applicable
            vat_rate_setting = billing_settings_dto.vat_rate

            logger.debug(
                "TVA settings pour company %s: applicable=%s, rate=%s",
                input_data.company_id,
                vat_applicable_setting,
                vat_rate_setting,
            )

            vat_rate_valid = False
            if vat_rate_setting is not None:
                try:
                    test_rate = Decimal(str(vat_rate_setting))
                    vat_rate_valid = test_rate > Decimal("0")
                except (InvalidOperation, ValueError, TypeError):
                    logger.warning(
                        "Taux TVA invalide pour company %s: %s",
                        input_data.company_id,
                        vat_rate_setting,
                    )
                    vat_rate_valid = False

            vat_applicable = vat_applicable_setting and vat_rate_valid
            default_vat_rate = Decimal("0")

            if vat_applicable and vat_rate_valid:
                try:
                    default_vat_rate = Decimal(str(vat_rate_setting)).quantize(
                        Decimal("0.01")
                    )
                    logger.info(
                        "TVA activée pour company %s avec taux %s%%",
                        input_data.company_id,
                        default_vat_rate,
                    )
                except (InvalidOperation, ValueError, TypeError) as e:
                    logger.error(
                        "Erreur conversion taux TVA pour company %s: %s",
                        input_data.company_id,
                        e,
                    )
                    default_vat_rate = Decimal("0")
                    vat_applicable = False
            else:
                logger.debug(
                    "TVA désactivée pour company %s (applicable=%s, rate_valid=%s)",
                    input_data.company_id,
                    vat_applicable_setting,
                    vat_rate_valid,
                )

            vat_label = billing_settings_dto.vat_label or "TVA"
            vat_number = billing_settings_dto.vat_number

            if effective_client_id is None:
                raise ValueError("client_id manquant pour la génération de facture")

            # 7. Récupérer les infos du client pour les descriptions
            client = self.client_repo.find_model_by_id_with_user(
                effective_client_id, input_data.company_id
            )
            patient_name = ""
            if opportunity_meta and isinstance(
                opportunity_meta.get("billing_subject_snapshot"), dict
            ):
                patient_name = (
                    opportunity_meta["billing_subject_snapshot"].get("display_name")
                    or ""
                ).strip()
            # Pour les clients institution : utiliser customer_name du premier booking
            if (
                not patient_name
                and client
                and getattr(client, "is_institution", False)
                and reservations
            ):
                for _r in reservations:
                    if getattr(_r, "customer_name", None):
                        patient_name = _r.customer_name
                        break
            if not patient_name and client and client.user:
                patient_name = (
                    f"{client.user.first_name} {client.user.last_name}".strip()
                )
            if not patient_name:
                patient_name = f"Client #{effective_client_id}"

            # 8. Créer la facture
            two_places = Decimal("0.01")
            meta: dict[str, Any] | None = None
            if payer_needs_review_reason:
                meta = {
                    "billing_review_status": "needs_review",
                    "needs_review_reason": payer_needs_review_reason,
                    "payer_resolution": {
                        "mode": "clinic_company_id",
                        "clinic_company_id": input_data.clinic_company_id,
                    },
                }
            if opportunity_meta:
                meta = {**(meta or {}), **opportunity_meta}
            # ✅ S2: Déterminer si c'est une facture S2 (clinique mensuelle multi-patients)
            # S2 = clinic_company_id fourni ET plusieurs clients différents dans les bookings
            unique_client_ids = {
                r.client_id for r in reservations if hasattr(r, "client_id")
            }
            is_s2 = (
                input_data.clinic_company_id is not None and len(unique_client_ids) > 1
            )

            invoice_data = {
                "company_id": input_data.company_id,
                "client_id": effective_client_id,
                "bill_to_client_id": input_data.bill_to_client_id,
                "billing_party_id": billing_party_id,
                "billed_to_company_id": input_data.clinic_company_id,
                "billing_strategy": (
                    InvoiceBillingStrategy.S2_CLINIC_MONTHLY
                    if is_s2
                    else InvoiceBillingStrategy.S1_PATIENT
                ),
                "period_month": input_data.period_month,
                "period_year": input_data.period_year,
                "invoice_number": invoice_number,
                "currency": "CHF",
                "issued_at": datetime.now(UTC),
                "due_date": datetime.now(UTC)
                + timedelta(days=billing_settings_dto.payment_terms_days),
                "status": InvoiceStatus.DRAFT,
                "subtotal_amount": Decimal("0.00"),
                "vat_total_amount": Decimal("0.00"),
                "total_amount": Decimal("0.00"),
                "balance_due": Decimal("0.00"),
                "meta": meta,
                "institution_patient_id": institution_patient_id_for_invoice,
            }
            invoice_dto = self.invoice_repo.create(invoice_data)
            # Récupérer le modèle pour les opérations suivantes
            invoice = Invoice.query.get(invoice_dto.id)
            if invoice is None:
                msg = "Erreur lors de la création de la facture"
                raise ValueError(msg)

            # 9. Créer les lignes de facture
            subtotal = Decimal("0.00")
            vat_total = Decimal("0.00")
            vat_breakdown: dict[str, dict[str, Decimal]] = {}
            ride_line_entries: list[tuple[InvoiceLine, Decimal, Decimal]] = []

            # Remise globale : parser le % avant la boucle pour normaliser les overrides legacy
            gd_pct_early: Decimal | None = None
            _gd_raw_loop = getattr(input_data, "global_discount_percent", None)
            if _gd_raw_loop is not None:
                try:
                    _pe = Decimal(str(_gd_raw_loop))
                    if Decimal("0") < _pe <= Decimal("100"):
                        gd_pct_early = _pe
                except (InvalidOperation, ValueError, TypeError):
                    gd_pct_early = None

            drafts: dict[int, RideLineDraft] = {}
            for reservation in reservations:
                try:
                    drafts[reservation.id] = compute_ride_line_draft(
                        reservation,
                        two_places=two_places,
                        billing_settings_dto=billing_settings_dto,
                        overrides_map=overrides_map,
                        bookings_by_id=bookings_by_id,
                        gd_pct_early=gd_pct_early,
                        patient_name=patient_name,
                        bill_to_client_id=input_data.bill_to_client_id,
                        clinic_company_id=input_data.clinic_company_id,
                        billing_party_id=billing_party_id,
                        default_vat_rate=default_vat_rate,
                        vat_applicable=vat_applicable,
                        description_builder=self.description_builder,
                    )
                except MaterialDeliveryPriceNotConfiguredError as e:
                    msg = (
                        f"Impossible de facturer : configurez le prix fixe livraison "
                        f"dans Paramètres > Facturation (réservation #{e.booking_id})."
                    )
                    logger.error(msg)
                    return GenerateInvoiceOutput(
                        success=False,
                        error={
                            "error": msg,
                            "error_code": ErrorCodes.MATERIAL_DELIVERY_PRICE_NOT_CONFIGURED,
                            "details": {
                                "field": "material_delivery_price_fixed",
                                "booking_id": str(e.booking_id),
                            },
                        },
                        status_code=400,
                    )

            ride_bookings = [
                r
                for r in reservations
                if (getattr(r, "mission_type", None) or "patient_transport")
                != "material_delivery"
            ]

            def rt_amount_ht(booking: Booking) -> Decimal:
                return drafts[int(booking.id)].base_amount

            round_trip_group_sets_raw = [
                s
                for s in round_trip_component_id_sets(
                    ride_bookings, amount_ht_fn=rt_amount_ht
                )
                if len(s) == ROUND_TRIP_MERGE_EXACT_SEGMENTS
            ]
            round_trip_group_sets = []
            for _s in round_trip_group_sets_raw:
                _segs = [bookings_by_id[i] for i in _s if i in bookings_by_id]
                if _is_strict_reverse_round_trip(_segs):
                    round_trip_group_sets.append(_s)
            in_round_trip_merge: set[int] = set()
            round_trip_primary_by_booking_id: dict[int, int] = {}
            for comp in round_trip_group_sets:
                in_round_trip_merge |= comp
                segs_b = [bookings_by_id[i] for i in comp if i in bookings_by_id]
                if len(segs_b) != ROUND_TRIP_MERGE_EXACT_SEGMENTS:
                    continue
                pri_b = min(
                    segs_b,
                    key=lambda b: (
                        b.scheduled_time or datetime.min.replace(tzinfo=UTC),
                        int(b.id),
                    ),
                )
                for sb in segs_b:
                    round_trip_primary_by_booking_id[int(sb.id)] = int(pri_b.id)

            def _append_ride_accounting(
                base_amount: Decimal, vat_amt: Decimal, rate: Decimal
            ) -> None:
                nonlocal subtotal, vat_total
                subtotal += base_amount
                vat_total += vat_amt
                rate_key = f"{rate.normalize()}"
                if rate_key not in vat_breakdown:
                    vat_breakdown[rate_key] = {
                        "base": Decimal("0.00"),
                        "vat": Decimal("0.00"),
                    }
                vat_breakdown[rate_key]["base"] += base_amount
                vat_breakdown[rate_key]["vat"] += vat_amt

            def emit_single_line(reservation: Booking, d: RideLineDraft) -> None:
                vat_amount, total_with_vat = self.invoice_calculator.calculate_vat(
                    d.base_amount, d.line_vat_rate
                )
                line_data = {
                    "invoice_id": invoice.id,
                    "type": d.line_type,
                    "description": d.description,
                    "qty": Decimal("1"),
                    "unit_price": d.base_amount,
                    "line_total": d.base_amount,
                    "vat_rate": d.line_vat_rate
                    if d.line_vat_rate > Decimal("0")
                    else None,
                    "vat_amount": vat_amount,
                    "total_with_vat": total_with_vat,
                    "adjustment_note": d.line_adjustment_note,
                    "reservation_id": reservation.id,
                }
                line_dto = self.invoice_line_repo.create(line_data)
                line_model = db.session.get(InvoiceLine, line_dto.id)
                if line_model is not None:
                    ride_line_entries.append(
                        (line_model, d.base_amount, d.line_vat_rate)
                    )
                reservation.invoice_line_id = line_dto.id
                reservation.updated_at = datetime.now(UTC)
                _append_ride_accounting(d.base_amount, vat_amount, d.line_vat_rate)

            def _booking_scheduled_date_iso(b: Booking) -> str | None:
                st = getattr(b, "scheduled_time", None)
                if st is None:
                    return None
                if hasattr(st, "date"):
                    return st.date().isoformat()
                return None

            def emit_merged_round_trip_group(segments: list[Booking]) -> None:
                if len(segments) < ROUND_TRIP_MERGE_MIN_SEGMENTS:
                    return
                ordered = sorted(
                    segments,
                    key=lambda b: (
                        b.scheduled_time or datetime.min.replace(tzinfo=UTC),
                        int(b.id),
                    ),
                )
                primary = ordered[0]
                d_pri = drafts[int(primary.id)]
                base_total = round_to_5_cents(
                    sum(drafts[int(b.id)].base_amount for b in segments)
                )
                rate = d_pri.line_vat_rate
                for b in segments:
                    if int(b.id) == int(primary.id):
                        continue
                    if drafts[int(b.id)].line_vat_rate != rate:
                        logger.warning(
                            "Segments A/R groupe incluant #%s : taux TVA différents — "
                            "taux du segment principal #%s",
                            b.id,
                            primary.id,
                        )
                vat_amount, total_with_vat = self.invoice_calculator.calculate_vat(
                    base_total, rate
                )
                merged_desc = (
                    build_merged_round_trip_invoice_line_description_from_segments(
                        ordered,
                        primary_segment_description=d_pri.description,
                    )
                )
                others = [int(b.id) for b in ordered if int(b.id) != int(primary.id)]
                sec_descriptions = [
                    drafts[int(b.id)].description[:500]
                    for b in ordered
                    if int(b.id) != int(primary.id)
                ]
                line_meta: dict[str, Any] = {
                    "is_round_trip_leg": True,
                    "transport_type": "A/R",
                    "billing_unit": "round_trip",
                    "booking_ids": [int(b.id) for b in ordered],
                    "round_trip_secondary_reservation_ids": others,
                    "secondary_segment_descriptions": sec_descriptions,
                    "merged_segment_count": len(segments),
                }
                if others:
                    line_meta["round_trip_secondary_reservation_id"] = others[0]
                    line_meta["secondary_segment_description"] = sec_descriptions[0]
                d_start = _booking_scheduled_date_iso(ordered[0])
                d_end = _booking_scheduled_date_iso(ordered[-1])
                if d_start:
                    line_meta["service_date"] = d_start
                    line_meta["service_date_iso"] = d_start
                if d_end and d_end != d_start:
                    line_meta["service_date_end"] = d_end
                    line_meta["service_date_iso_end"] = d_end
                line_data = {
                    "invoice_id": invoice.id,
                    "type": InvoiceLineType.RIDE,
                    "description": merged_desc[:500],
                    "qty": Decimal("1"),
                    "unit_price": base_total,
                    "line_total": base_total,
                    "vat_rate": rate if rate > Decimal("0") else None,
                    "vat_amount": vat_amount,
                    "total_with_vat": total_with_vat,
                    "adjustment_note": d_pri.line_adjustment_note,
                    "reservation_id": primary.id,
                    "line_meta": line_meta,
                }
                line_dto = self.invoice_line_repo.create(line_data)
                line_model = db.session.get(InvoiceLine, line_dto.id)
                if line_model is not None:
                    ride_line_entries.append((line_model, base_total, rate))
                for b in segments:
                    b.invoice_line_id = line_dto.id
                    b.updated_at = datetime.now(UTC)
                _append_ride_accounting(base_total, vat_amount, rate)

            for reservation in reservations:
                d = drafts[reservation.id]
                if d.mission_type == "material_delivery":
                    emit_single_line(reservation, d)
                    continue
                rid = int(reservation.id)
                if rid in in_round_trip_merge:
                    primary_rid = round_trip_primary_by_booking_id.get(rid)
                    if primary_rid is None or primary_rid != rid:
                        continue
                    grp_bookings = [
                        bookings_by_id[i]
                        for i in next(gs for gs in round_trip_group_sets if rid in gs)
                        if i in bookings_by_id
                    ]
                    if len(grp_bookings) != ROUND_TRIP_MERGE_EXACT_SEGMENTS:
                        emit_single_line(reservation, d)
                        continue
                    emit_merged_round_trip_group(grp_bookings)
                else:
                    emit_single_line(reservation, d)

            # 10. Mettre à jour les totaux de la facture
            gross_subtotal = round_to_5_cents(subtotal)

            gd_note_raw = getattr(input_data, "global_discount_note", None)
            gd_note_stripped = (
                str(gd_note_raw).strip()[:500]
                if isinstance(gd_note_raw, str) and str(gd_note_raw).strip()
                else None
            )

            gd_pct_dec: Decimal | None = None
            gd_pct_raw = getattr(input_data, "global_discount_percent", None)
            if gd_pct_raw is not None:
                try:
                    _p = Decimal(str(gd_pct_raw))
                    if Decimal("0") < _p <= Decimal("100"):
                        gd_pct_dec = _p
                except (InvalidOperation, ValueError, TypeError):
                    gd_pct_dec = None

            apply_global_discount = (
                gd_pct_dec is not None
                and gross_subtotal > 0
                and len(ride_line_entries) > 0
            )

            global_discount_meta: dict[str, Any] | None = None

            if apply_global_discount:
                assert gd_pct_dec is not None
                # Remise globale : un seul arrondi 0,05 sur le montant (pas % ni ligne par ligne).
                discount_ht = min(
                    round_to_5_cents(gross_subtotal * gd_pct_dec / Decimal("100")),
                    gross_subtotal,
                )
                net_ht = round_to_5_cents(gross_subtotal - discount_ht)

                desc = f"Remise commerciale {gd_pct_dec.normalize()!s} %"
                if gd_note_stripped:
                    desc = (
                        f"{desc} — appliquée sur le sous-total HT des prestations"
                        f" — {gd_note_stripped[:300]}"
                    )
                self.invoice_line_repo.create(
                    {
                        "invoice_id": invoice.id,
                        "type": InvoiceLineType.CUSTOM,
                        "description": desc[:500],
                        "qty": Decimal("1"),
                        "unit_price": -discount_ht,
                        "line_total": -discount_ht,
                        "vat_rate": None,
                        "vat_amount": Decimal("0.00"),
                        "total_with_vat": -discount_ht,
                        "adjustment_note": None,
                        "reservation_id": None,
                    }
                )

                vat_breakdown = {}
                vat_total = Decimal("0.00")
                if vat_applicable:
                    running_net = Decimal("0.00")
                    n_lines = len(ride_line_entries)
                    for idx, (lm, base_amt, rate_line) in enumerate(ride_line_entries):
                        if idx < n_lines - 1:
                            net_i = round_to_5_cents(base_amt * net_ht / gross_subtotal)
                        else:
                            net_i = round_to_5_cents(net_ht - running_net)
                        running_net += net_i
                        vat_i, twv_i = self.invoice_calculator.calculate_vat(
                            net_i, rate_line
                        )
                        lm.vat_amount = vat_i
                        lm.total_with_vat = twv_i
                        vat_total += vat_i
                        rate_key = f"{rate_line.normalize()}"
                        if rate_key not in vat_breakdown:
                            vat_breakdown[rate_key] = {
                                "base": Decimal("0.00"),
                                "vat": Decimal("0.00"),
                            }
                        vat_breakdown[rate_key]["base"] += net_i
                        vat_breakdown[rate_key]["vat"] += vat_i
                else:
                    for lm, _, _ in ride_line_entries:
                        lm.vat_amount = Decimal("0.00")
                        lm.total_with_vat = lm.line_total

                subtotal = net_ht
                vat_total = round_to_5_cents(vat_total)
                total = round_to_5_cents(subtotal + vat_total)
                vat_total = total - subtotal
                if vat_total < 0:
                    vat_total = Decimal("0.00")

                global_discount_meta = {
                    "percent": float(gd_pct_dec),
                    "amount_ht": float(discount_ht),
                    "subtotal_before_ht": float(gross_subtotal),
                }
                if gd_note_stripped:
                    global_discount_meta["note"] = gd_note_stripped
            else:
                subtotal = gross_subtotal
                vat_total = round_to_5_cents(vat_total)
                total = round_to_5_cents(subtotal + vat_total)
                vat_total = total - subtotal
                if vat_total < 0:
                    vat_total = Decimal("0.00")

            invoice.subtotal_amount = subtotal
            invoice.vat_total_amount = vat_total
            invoice.total_amount = total
            invoice.balance_due = total
            vat_payload: dict[str, dict[str, float]] = {
                rate: {
                    "base": float(values["base"].quantize(two_places)),
                    "vat": float(values["vat"].quantize(two_places)),
                }
                for rate, values in vat_breakdown.items()
            }
            invoice.vat_breakdown = cast(Any, vat_payload)

            # Mettre à jour les métadonnées
            current_meta: dict[str, Any] = {}
            if isinstance(invoice.meta, dict):
                current_meta = dict(invoice.meta)
            current_meta["vat"] = {
                "applicable": vat_applicable and (default_vat_rate > Decimal("0")),
                "default_rate": float(default_vat_rate),
                "label": vat_label,
                "number": vat_number,
            }
            if global_discount_meta is not None:
                current_meta["global_discount"] = global_discount_meta
            invoice.meta = cast(Any, current_meta)

            # 11. Générer et sauvegarder la référence QR (si pas déjà présente)
            if not invoice.qr_reference:
                qrbill_service = QRBillService()
                qr_ref = qrbill_service._get_payment_reference(invoice)
                if qr_ref:
                    invoice.qr_reference = qr_ref
                    logger.debug(
                        "Référence QR sauvegardée pour facture %s: %s",
                        invoice.invoice_number,
                        qr_ref,
                    )

            # 12. Commit métier (facture + lignes + réservations). PDF fichier dans une 2e transaction.
            db.session.commit()

            # 13. PDF + meta.pdf (``generate_invoice_pdf`` recharge la facture après ``expire_all``).
            generated_pdf_url = None
            try:
                pdf_url = self.pdf_service.generate_invoice_pdf(invoice)
                generated_pdf_url = pdf_url or None
                if pdf_url:
                    mark_pdf_ready(invoice, pdf_url)
                else:
                    mark_pdf_failed(invoice, "PDF_EMPTY")
                db.session.commit()
            except Exception as pdf_err:
                logger.exception(
                    "Échec génération ou persistance PDF après création facture invoice_id=%s",
                    getattr(invoice, "id", None),
                )
                try:
                    mark_pdf_failed(invoice, str(pdf_err))
                    db.session.commit()
                except Exception:
                    db.session.rollback()
                    logger.exception(
                        "Échec persistance mark_pdf_failed invoice_id=%s",
                        getattr(invoice, "id", None),
                    )

            if input_data.bill_to_client_id:
                logger.info(
                    "Facture générée: %s pour client %s (facturée à institution %s)",
                    invoice_number,
                    effective_client_id,
                    input_data.bill_to_client_id,
                )
            else:
                logger.info(
                    "Facture générée: %s pour client %s",
                    invoice_number,
                    effective_client_id,
                )

            return GenerateInvoiceOutput(
                success=True, invoice_id=invoice.id, invoice=invoice
            )

        except (OperationalError, DBAPIError) as e:
            if generated_pdf_url:
                logger.warning(
                    "Rollback après génération PDF — fichier possiblement orphelin (pdf_url=%s)",
                    generated_pdf_url,
                )
            db.session.rollback()
            err_msg = str(e).lower()
            orig = getattr(e, "orig", None)
            pgcode = getattr(orig, "pgcode", None) if orig else None
            # DataError (invalid enum) : pgcode 22P02 ou "invalid input value for enum"
            # Ne pas confondre avec CHECK constraint (ck_booking_material_delivery_description)
            is_enum_error = pgcode == "22P02" or (
                "invalid input value for enum" in err_msg
                and "invoice_line_type" in err_msg
            )
            if is_enum_error:
                logger.error(
                    "Enum invoice_line_type non à jour (migration manquante?): %s",
                    str(e),
                )
                return GenerateInvoiceOutput(
                    success=False,
                    error={
                        "error": (
                            "Configuration base de données incomplète. "
                            "Exécutez les migrations (alembic upgrade head)."
                        ),
                        "error_code": "INVOICE_LINE_TYPE_MIGRATION_REQUIRED",
                        "details": {
                            "enum_type": "invoice_line_type",
                            "expected_value": "material_delivery",
                        },
                    },
                    status_code=400,
                )
            logger.error(
                "Erreur DB lors de la génération de la facture (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return GenerateInvoiceOutput(
                success=False,
                error={"error": "Erreur de base de données"},
                status_code=500,
            )
        except IntegrityError as e:
            if generated_pdf_url:
                logger.warning(
                    "Rollback après génération PDF — fichier possiblement orphelin (pdf_url=%s)",
                    generated_pdf_url,
                )
            db.session.rollback()
            err_msg = str(e).lower()
            # CHECK constraint : livraison sans description (priorité sur enum)
            if "ck_booking_material_delivery_description" in err_msg:
                logger.warning(
                    "Livraison matériel sans description (CHECK constraint): %s",
                    str(e),
                )
                return GenerateInvoiceOutput(
                    success=False,
                    error={
                        "error": (
                            "Une livraison matériel doit avoir une description. "
                            "Veuillez renseigner le champ « Description de la livraison »."
                        ),
                        "error_code": ErrorCodes.MATERIAL_DELIVERY_DESCRIPTION_REQUIRED,
                        "details": {"field": "delivery_description"},
                    },
                    status_code=400,
                )
            # Enum non migré (ex: material_delivery absent de invoice_line_type)
            if (
                "invoice_line_type" in err_msg
                and "invalid input value for enum" in err_msg
            ):
                logger.error(
                    "Enum invoice_line_type non à jour (migration manquante?): %s",
                    str(e),
                )
                return GenerateInvoiceOutput(
                    success=False,
                    error={
                        "error": (
                            "Configuration base de données incomplète. "
                            "Exécutez les migrations (alembic upgrade head)."
                        ),
                        "error_code": "INVOICE_LINE_TYPE_MIGRATION_REQUIRED",
                        "details": {
                            "enum_type": "invoice_line_type",
                            "expected_value": "material_delivery",
                        },
                    },
                    status_code=400,
                )
            logger.error(
                "Erreur d'intégrité DB lors de la génération de la facture: %s",
                str(e),
            )
            return GenerateInvoiceOutput(
                success=False,
                error={"error": "Erreur de base de données"},
                status_code=500,
            )
        except (KeyError, AttributeError) as e:
            db.session.rollback()
            logger.warning(
                "Erreur de mapping/template (KeyError/AttributeError): %s", e
            )
            details: dict[str, str | int | None] = {}
            if isinstance(e, KeyError) and e.args:
                details["line_type"] = str(e.args[0])
            elif isinstance(e, AttributeError) and e.args:
                details["attribute"] = str(e.args[0])
            return GenerateInvoiceOutput(
                success=False,
                error={
                    "error": "Erreur de configuration (type de ligne non supporté)",
                    "error_code": "UNKNOWN_LINE_TYPE",
                    "details": details if details else None,
                },
                status_code=400,
            )
        except ValueError as e:
            if generated_pdf_url:
                logger.warning(
                    "Rollback après génération PDF — fichier possiblement orphelin (pdf_url=%s)",
                    generated_pdf_url,
                )
            db.session.rollback()
            logger.warning(
                "Erreur de validation lors de la génération de facture: %s", e
            )
            return GenerateInvoiceOutput(
                success=False,
                error={"error": str(e)},
                status_code=400,
            )
        except Exception:
            if generated_pdf_url:
                logger.warning(
                    "Rollback après génération PDF — fichier possiblement orphelin (pdf_url=%s)",
                    generated_pdf_url,
                )
            db.session.rollback()
            logger.exception("Erreur inattendue lors de la génération de la facture")
            return GenerateInvoiceOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
