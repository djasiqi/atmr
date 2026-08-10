"""Use-case: générer une facture clinique mensuelle unique (S2).

Ce use case génère UNE SEULE facture pour tous les patients d'une clinique
sur une période donnée, avec support des exceptions (include/exclude clients).
"""

from __future__ import annotations  # noqa: I001

# pyright: reportUnusedImport=false, reportUnusedVariable=false, reportGeneralTypeIssues=false, reportUnusedFunction=false
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from sqlalchemy import and_, exists, or_
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError
from sqlalchemy.orm import aliased

from ext import db
from shared.constants import ErrorCodes
from infrastructure.invoices.invoice_calculator import (
    InvoiceCalculator,
    round_to_5_cents,
)
from application.invoices.billable_amount import calculate_billable_booking_amount
from application.invoices.invoice_booking_units import (
    collect_explicit_peer_ids_to_load,
    resolve_invoice_booking_units,
)
from application.invoices.invoice_line_description import (
    build_invoice_line_description_clinic_monthly,
    build_merged_round_trip_invoice_line_description_from_segments,
)
from application.invoices.invoice_pdf_state import mark_pdf_failed, mark_pdf_ready
from application.invoices.subject_identity import resolve_subject_identity
from infrastructure.invoices.invoice_description_builder import (
    InvoiceDescriptionBuilder,
)
from infrastructure.invoices.invoice_number_generator import InvoiceNumberGenerator
from models import Booking, ClientStay, Invoice, InvoiceLineType, InvoiceStatus
from models.enums import BookingStatus, InvoiceBillingStrategy
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)
from repositories.invoice_line_repository import InvoiceLineRepository
from repositories.invoice_repository import InvoiceRepository
from repositories.invoice_sequence_repository import InvoiceSequenceRepository
from services.billing.billing_party_linker import resolve_billing_party_for_clinic
from services.billing.clinic_s2_eligibility import clinic_s2_billed_to_company_predicate
from services.documents.pdf import PDFService

logger = logging.getLogger(__name__)

PERIOD_MONTH_THRESHOLD = 12


def _booking_service_date_iso(reservation: Booking) -> str | None:
    """Date calendaire du trajet (reservation) pour l'apercu / PDF facture clinique."""
    st = getattr(reservation, "scheduled_time", None)
    if st is None:
        return None
    try:
        if hasattr(st, "date"):
            return st.date().isoformat()
    except Exception:
        return None
    return None


HTTP_409_CONFLICT = 409  # HTTP Conflict (déjà générée)
MAX_BOOKING_IDS_SHOWN = 10  # Limite le nombre d'IDs affichés dans les messages d'erreur


@dataclass(frozen=True, slots=True)
class GenerateClinicMonthlyInvoiceInput:
    """Input pour générer une facture clinique mensuelle unique (S2).

    Attributes:
        company_id: ID de l'entreprise
        clinic_company_id: ID de la clinique (Company) payeuse
        period_year: Année de facturation (ex: 2025)
        period_month: Mois de facturation (1-12)
        include_client_ids: Liste optionnelle d'IDs clients à inclure (exception partielle)
        exclude_client_ids: Liste optionnelle d'IDs clients à exclure
        overrides: Dict facultatif {reservation_id: {amount, vat_rate, note}}
    """

    company_id: int
    clinic_company_id: int
    period_year: int
    period_month: int
    include_client_ids: list[int] | None = None
    exclude_client_ids: list[int] | None = None
    reservation_ids: list[int] | None = None
    overrides: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class GenerateClinicMonthlyInvoiceOutput:
    """Output pour générer une facture clinique mensuelle unique (S2).

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


class GenerateClinicMonthlyInvoiceUseCase:
    """Use-case Application: générer une facture clinique mensuelle unique (S2).

    Génère UNE SEULE facture pour tous les patients d'une clinique sur une période,
    avec support des exceptions (include/exclude clients).
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
        """Initialise le use case avec injection de dépendances."""
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

    def execute(
        self, input_data: GenerateClinicMonthlyInvoiceInput
    ) -> GenerateClinicMonthlyInvoiceOutput:
        """Génère une facture clinique mensuelle unique (S2).

        Args:
            input_data: Données d'entrée du use case

        Returns:
            GenerateClinicMonthlyInvoiceOutput avec la facture créée
        """
        try:
            # ✅ 1. Validation include/exclude (priorité: include > exclude, interdire les deux)
            if input_data.include_client_ids and input_data.exclude_client_ids:
                msg = (
                    "Ne pas fournir à la fois include_client_ids et exclude_client_ids. "
                    "Utilisez include_client_ids pour limiter à certains patients, "
                    "ou exclude_client_ids pour exclure certains patients."
                )
                logger.warning(msg)
                return GenerateClinicMonthlyInvoiceOutput(
                    success=False,
                    error={"error": msg},
                    status_code=400,
                )

            # Validation: include_client_ids doit être non vide si fourni
            if input_data.include_client_ids is not None:
                if not input_data.include_client_ids:
                    msg = "include_client_ids ne peut pas être vide. Sélectionnez au moins un patient ou retirez la limitation."
                    logger.warning(msg)
                    return GenerateClinicMonthlyInvoiceOutput(
                        success=False,
                        error={"error": msg},
                        status_code=400,
                    )
                # Vérifier que tous les IDs sont des entiers uniques
                try:
                    include_ids = [
                        int(client_id) for client_id in input_data.include_client_ids
                    ]
                    if len(include_ids) != len(set(include_ids)):
                        msg = "include_client_ids contient des doublons"
                        logger.warning(msg)
                        return GenerateClinicMonthlyInvoiceOutput(
                            success=False,
                            error={"error": msg},
                            status_code=400,
                        )
                    input_data = GenerateClinicMonthlyInvoiceInput(
                        company_id=input_data.company_id,
                        clinic_company_id=input_data.clinic_company_id,
                        period_year=input_data.period_year,
                        period_month=input_data.period_month,
                        include_client_ids=include_ids,
                        exclude_client_ids=input_data.exclude_client_ids,
                        overrides=input_data.overrides,
                    )
                except (ValueError, TypeError):
                    msg = "include_client_ids doit contenir uniquement des entiers valides"
                    logger.warning(msg)
                    return GenerateClinicMonthlyInvoiceOutput(
                        success=False,
                        error={"error": msg},
                        status_code=400,
                    )

            # Validation: exclude_client_ids doit être non vide si fourni
            if input_data.exclude_client_ids is not None:
                if not input_data.exclude_client_ids:
                    msg = "exclude_client_ids ne peut pas être vide. Sélectionnez au moins un patient à exclure ou retirez l'exclusion."
                    logger.warning(msg)
                    return GenerateClinicMonthlyInvoiceOutput(
                        success=False,
                        error={"error": msg},
                        status_code=400,
                    )
                # Vérifier que tous les IDs sont des entiers uniques
                try:
                    exclude_ids = [
                        int(client_id) for client_id in input_data.exclude_client_ids
                    ]
                    if len(exclude_ids) != len(set(exclude_ids)):
                        msg = "exclude_client_ids contient des doublons"
                        logger.warning(msg)
                        return GenerateClinicMonthlyInvoiceOutput(
                            success=False,
                            error={"error": msg},
                            status_code=400,
                        )
                    input_data = GenerateClinicMonthlyInvoiceInput(
                        company_id=input_data.company_id,
                        clinic_company_id=input_data.clinic_company_id,
                        period_year=input_data.period_year,
                        period_month=input_data.period_month,
                        include_client_ids=input_data.include_client_ids,
                        exclude_client_ids=exclude_ids,
                        overrides=input_data.overrides,
                    )
                except (ValueError, TypeError):
                    msg = "exclude_client_ids doit contenir uniquement des entiers valides"
                    logger.warning(msg)
                    return GenerateClinicMonthlyInvoiceOutput(
                        success=False,
                        error={"error": msg},
                        status_code=400,
                    )

            # 2. Vérifier l'anti-doublon: une seule facture S2 DRAFT par (company_id, clinic_company_id, year, month)
            # ✅ Si une facture DRAFT existe → 409 (compléter celle-ci d'abord)
            # ✅ Si une facture SENT/PAID existe → autoriser une facture complémentaire
            existing_draft_invoice = Invoice.query.filter(
                and_(
                    Invoice.company_id == input_data.company_id,
                    Invoice.billed_to_company_id == input_data.clinic_company_id,
                    Invoice.period_year == input_data.period_year,
                    Invoice.period_month == input_data.period_month,
                    Invoice.billing_strategy
                    == InvoiceBillingStrategy.S2_CLINIC_MONTHLY,
                    Invoice.status == InvoiceStatus.DRAFT,
                )
            ).first()

            if existing_draft_invoice:
                # ✅ UX: Retourner l'ID de la facture brouillon existante pour permettre l'ouverture
                month_names = [
                    "Janvier",
                    "Février",
                    "Mars",
                    "Avril",
                    "Mai",
                    "Juin",
                    "Juillet",
                    "Août",
                    "Septembre",
                    "Octobre",
                    "Novembre",
                    "Décembre",
                ]
                month_name = (
                    month_names[input_data.period_month - 1]
                    if 1 <= input_data.period_month <= PERIOD_MONTH_THRESHOLD
                    else str(input_data.period_month)
                )
                msg = (
                    f"Une facture clinique mensuelle (S2) en brouillon existe déjà pour {month_name} {input_data.period_year}. "
                    f"Numéro: {existing_draft_invoice.invoice_number}. "
                    f"Complétez-la ou annulez-la avant d'en créer une nouvelle."
                )
                logger.info(
                    (
                        "⚠️ S2 invoice 409 conflict (draft exists): company_id=%s, clinic_company_id=%s, "
                        "period=%s-%02d, existing_invoice_id=%s, existing_invoice_number=%s"
                    ),
                    input_data.company_id,
                    input_data.clinic_company_id,
                    input_data.period_year,
                    input_data.period_month,
                    existing_draft_invoice.id,
                    existing_draft_invoice.invoice_number,
                )
                logger.warning(msg)
                return GenerateClinicMonthlyInvoiceOutput(
                    success=False,
                    error={
                        "error": msg,
                        "existing_invoice_id": existing_draft_invoice.id,
                        "existing_invoice_number": existing_draft_invoice.invoice_number,
                    },
                    status_code=HTTP_409_CONFLICT,  # ✅ 409 Conflict pour "brouillon existant"
                )

            # 2. Récupérer les paramètres de facturation
            billing_settings_dto = self.billing_settings_repo.find_or_create(
                input_data.company_id
            )

            # 3. Résoudre le billing_party pour la clinique
            bp = resolve_billing_party_for_clinic(
                company_id=input_data.company_id,
                clinic_company_id=input_data.clinic_company_id,
            )
            if bp is None:
                msg = (
                    f"Destinataire de facturation clinique non configuré "
                    f"(mapping clinique → billing_party manquant pour clinic_company_id={input_data.clinic_company_id}). "
                    f"Veuillez configurer le mapping dans les paramètres de facturation."
                )
                logger.error(msg)
                return GenerateClinicMonthlyInvoiceOutput(
                    success=False,
                    error={"error": msg},
                    status_code=400,
                )
            billing_party_id = bp.id

            # 4. Récupérer toutes les réservations éligibles du mois
            # pour clinic_company_id, billed_to_type='clinic', invoice_line_id is null
            start_date = datetime(input_data.period_year, input_data.period_month, 1)
            end_date = (
                datetime(input_data.period_year + 1, 1, 1)
                if input_data.period_month == PERIOD_MONTH_THRESHOLD
                else datetime(input_data.period_year, input_data.period_month + 1, 1)
            )

            # ✅ 4. Récupérer toutes les réservations éligibles du mois
            # Scope strict: billed_to_type='clinic', clinic_company_id match, invoice_line_id null, status target_statuses
            # ✅ Scope strict: billed_to_type='clinic' suffit (exclut automatiquement les overrides patient)
            # Le critère réel = billed_to_type == 'clinic' (pas besoin de vérifier billing_source)
            # Inclut COMPLETED, RETURN_COMPLETED et CANCELED facturables UNIQUEMENT si client hospitalisé
            # (annulations billables à la clinique ; legacy / non billables → pas en facture)
            stay_overlaps_booking = exists().where(
                ClientStay.client_id == Booking.client_id,
                ClientStay.company_id == input_data.clinic_company_id,
                ClientStay.status == "active",
                ClientStay.start_date <= Booking.scheduled_time,
                or_(
                    ClientStay.end_date.is_(None),
                    ClientStay.end_date >= Booking.scheduled_time,
                ),
            )
            # ✅ Annulations : uniquement billables + aller (pas le retour) ; amount > 0 = garde-fou anti-ligne zéro
            canceled_condition = (
                (Booking.status == "CANCELED")
                & (Booking.is_cancellation_billable == True)  # noqa: E712
                & (Booking.amount > 0)
                & stay_overlaps_booking
                & (Booking.is_return == False)  # noqa: E712 — SQLAlchemy column comparison
            )
            query = Booking.query.filter(
                Booking.company_id == input_data.company_id,
                clinic_s2_billed_to_company_predicate(
                    input_data.clinic_company_id, input_data.company_id
                ),
                Booking.billed_to_type
                == "clinic",  # ✅ Strict: uniquement facturation clinique (exclut automatiquement les overrides patient)
                or_(
                    Booking.status.in_(["COMPLETED", "RETURN_COMPLETED"]),
                    canceled_condition,
                ),
                Booking.invoice_line_id.is_(None),  # ✅ Pas encore facturé
                Booking.scheduled_time >= start_date,
                Booking.scheduled_time < end_date,
            )

            # Appliquer les filtres include/exclude (priorité: include > exclude)
            if input_data.include_client_ids:
                query = query.filter(
                    Booking.client_id.in_(input_data.include_client_ids)
                )
            elif input_data.exclude_client_ids:
                query = query.filter(
                    ~Booking.client_id.in_(input_data.exclude_client_ids)
                )

            # ✅ Si l'aller est annulé, le retour n'est pas facturable : exclure les retours dont le parent (aller) est CANCELED
            ParentBooking = aliased(Booking)
            query = query.outerjoin(
                ParentBooking, ParentBooking.id == Booking.parent_booking_id
            )
            query = query.filter(
                or_(
                    Booking.is_return == False,  # noqa: E712
                    ParentBooking.id.is_(None),
                    ParentBooking.status != BookingStatus.CANCELED.value,
                )
            )

            reservations = query.order_by(Booking.scheduled_time.asc()).all()

            # C3 : charger pairs explicites hors fenêtre mensuelle (retour mois suivant)
            peer_ids = collect_explicit_peer_ids_to_load(reservations)
            present_ids = {int(r.id) for r in reservations}
            parent_ids = {
                int(r.parent_booking_id)
                for r in reservations
                if getattr(r, "parent_booking_id", None) is not None
            }
            child_peers = (
                Booking.query.filter(
                    Booking.company_id == input_data.company_id,
                    Booking.parent_booking_id.in_(present_ids),
                    Booking.billed_to_type == "clinic",
                ).all()
                if present_ids
                else []
            )
            missing_parents = (
                Booking.query.filter(Booking.id.in_(peer_ids | parent_ids)).all()
                if (peer_ids or parent_ids)
                else []
            )
            by_id: dict[int, Booking] = {int(r.id): r for r in reservations}
            for extra in list(child_peers) + list(missing_parents):
                by_id[int(extra.id)] = extra

            # Filet orphelin : si le pair explicite est déjà lié à une facture active,
            # ne pas proposer le segment ouvert comme nouvelle ligne.
            from application.invoices.round_trip_billing_lock import (
                booking_has_blocking_invoice_line,
            )

            open_candidates: list[Booking] = []
            for b in by_id.values():
                if b.invoice_line_id is not None and booking_has_blocking_invoice_line(
                    b
                ):
                    continue
                if b.invoice_line_id is not None:
                    continue
                pid = getattr(b, "parent_booking_id", None)
                if pid is not None and int(pid) in by_id:
                    parent = by_id[int(pid)]
                    if (
                        parent.invoice_line_id is not None
                        and booking_has_blocking_invoice_line(parent)
                    ):
                        continue
                # enfants déjà facturés bloquent le parent ouvert
                blocked_by_child = False
                for other in by_id.values():
                    if (
                        getattr(other, "parent_booking_id", None) == b.id
                        and other.invoice_line_id is not None
                        and booking_has_blocking_invoice_line(other)
                    ):
                        blocked_by_child = True
                        break
                if blocked_by_child:
                    continue
                # période : ancre = principal (non-retour ou parent) dans le mois
                st = getattr(b, "scheduled_time", None)
                is_return = bool(getattr(b, "is_return", False)) or (
                    getattr(b, "parent_booking_id", None) is not None
                )
                if is_return and pid is not None and int(pid) in by_id:
                    anchor = by_id[int(pid)]
                    ast = getattr(anchor, "scheduled_time", None)
                    if (
                        (ast is None or not (start_date <= ast < end_date))
                        and (st is None or not (start_date <= st < end_date))
                        and int(b.id)
                        not in ({int(x) for x in (input_data.reservation_ids or [])})
                    ):
                        # retour hors mois dont l'aller n'est pas ancré manuellement
                        continue
                elif st is None or not (start_date <= st < end_date):
                    if int(b.id) not in (
                        {int(x) for x in (input_data.reservation_ids or [])}
                    ):
                        # pair hors période chargé pour l'unité uniquement
                        pass
                open_candidates.append(b)

            # Reconstruire la liste ancrée période : bookings dont l'ancre est dans le mois
            period_anchor_ids: set[int] = set()
            for b in open_candidates:
                st = getattr(b, "scheduled_time", None)
                if st is not None and start_date <= st < end_date:
                    period_anchor_ids.add(int(b.id))
                pid = getattr(b, "parent_booking_id", None)
                if pid is not None and int(pid) in by_id:
                    parent = by_id[int(pid)]
                    pst = getattr(parent, "scheduled_time", None)
                    if pst is not None and start_date <= pst < end_date:
                        period_anchor_ids.add(int(parent.id))
                        period_anchor_ids.add(int(b.id))

            if input_data.reservation_ids:
                wanted = {int(x) for x in input_data.reservation_ids}
                # expand pairs explicites
                for b in list(by_id.values()):
                    if int(b.id) in wanted:
                        pid = getattr(b, "parent_booking_id", None)
                        if pid is not None:
                            wanted.add(int(pid))
                    if (
                        getattr(b, "parent_booking_id", None) is not None
                        and int(b.parent_booking_id) in wanted
                    ):
                        wanted.add(int(b.id))
                period_anchor_ids &= wanted
                # inclure pairs hors période des sélectionnés
                for wid in list(wanted):
                    if wid in by_id:
                        period_anchor_ids.add(wid)

            scope_bookings = [by_id[i] for i in sorted(period_anchor_ids) if i in by_id]
            # n'émettre que les bookings encore ouverts
            scope_bookings = [
                b
                for b in scope_bookings
                if b.invoice_line_id is None or not booking_has_blocking_invoice_line(b)
            ]
            scope_bookings = [b for b in scope_bookings if b.invoice_line_id is None]

            def _amount_ht(b: Booking) -> Decimal:
                return calculate_billable_booking_amount(
                    b, billing_settings=billing_settings_dto
                ).amount_ht

            units = resolve_invoice_booking_units(
                selected_ids=None,
                scope_bookings=scope_bookings,
                subject_key_fn=lambda bk: resolve_subject_identity(bk).key,
                amount_ht_fn=_amount_ht,
                expand_explicit_peers=True,
            )

            # Verrou SQL : FOR UPDATE sur tous les segments des unités
            all_unit_ids: list[int] = sorted(
                {bid for u in units for bid in u.booking_ids}
            )
            if all_unit_ids:
                locked = (
                    Booking.query.filter(Booking.id.in_(all_unit_ids))
                    .order_by(Booking.id.asc())
                    .with_for_update()
                    .all()
                )
                by_id.update({int(b.id): b for b in locked})
                # Revalider ouverture
                still_open_units = []
                for u in units:
                    segs = [by_id[i] for i in u.booking_ids if i in by_id]
                    if len(segs) != len(u.booking_ids):
                        continue
                    if any(s.invoice_line_id is not None for s in segs):
                        continue
                    still_open_units.append(u)
                units = still_open_units

            reservations = [
                by_id[bid] for u in units for bid in u.booking_ids if bid in by_id
            ]

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
                return GenerateClinicMonthlyInvoiceOutput(
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

            # ✅ Gérer le cas "0 lignes" avec message clair (422 Unprocessable Entity)
            if not reservations:
                month_names = [
                    "Janvier",
                    "Février",
                    "Mars",
                    "Avril",
                    "Mai",
                    "Juin",
                    "Juillet",
                    "Août",
                    "Septembre",
                    "Octobre",
                    "Novembre",
                    "Décembre",
                ]
                month_name = (
                    month_names[input_data.period_month - 1]
                    if 1 <= input_data.period_month <= PERIOD_MONTH_THRESHOLD
                    else str(input_data.period_month)
                )
                if input_data.include_client_ids:
                    msg = (
                        f"Aucun transport clinique éligible pour la période {month_name} {input_data.period_year} "
                        f"pour les patients sélectionnés. "
                        f"Vérifiez que les patients ont des transports facturés à la clinique (billed_to_type='clinic')."
                    )
                else:
                    msg = (
                        f"Aucun transport clinique éligible pour la clinique "
                        f"{input_data.clinic_company_id} sur la période {month_name} {input_data.period_year}. "
                        f"Vérifiez que les patients ont des transports facturés à la clinique (billed_to_type='clinic')."
                    )
                logger.warning(msg)
                return GenerateClinicMonthlyInvoiceOutput(
                    success=False,
                    error={"error": msg},
                    status_code=422,  # ✅ 422 Unprocessable Entity pour "aucune donnée éligible"
                )

            # 5. Traiter les overrides
            overrides_map: dict[int, dict[str, Any]] = {}
            if input_data.overrides:
                for key, value in input_data.overrides.items():
                    try:
                        reservation_id = int(key)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(value, dict):
                        overrides_map[reservation_id] = value

            # 6. Générer le numéro de facture
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

            # 7. Calculer la TVA
            vat_applicable_setting = billing_settings_dto.vat_applicable
            vat_rate_setting = billing_settings_dto.vat_rate

            vat_rate_valid = False
            if vat_rate_setting is not None:
                try:
                    test_rate = Decimal(str(vat_rate_setting))
                    vat_rate_valid = test_rate > Decimal("0")
                except (InvalidOperation, ValueError, TypeError):
                    vat_rate_valid = False

            vat_applicable = vat_applicable_setting and vat_rate_valid
            default_vat_rate = Decimal("0")

            if vat_applicable and vat_rate_valid:
                try:
                    default_vat_rate = Decimal(str(vat_rate_setting)).quantize(
                        Decimal("0.01")
                    )
                except (InvalidOperation, ValueError, TypeError):
                    default_vat_rate = Decimal("0")
                    vat_applicable = False

            vat_label = billing_settings_dto.vat_label or "TVA"
            vat_number = billing_settings_dto.vat_number

            # 8. Créer la facture (S2)
            # Pour S2, on utilise le premier client comme client_id principal
            # (nécessaire pour la structure Invoice, mais toutes les lignes incluront le patient_name)
            first_client_id = reservations[0].client_id

            two_places = Decimal("0.01")
            # ✅ Assert: billed_to_company_id doit toujours être non-null en S2
            if not input_data.clinic_company_id:
                msg = "clinic_company_id est requis et ne peut pas être null pour S2"
                logger.error(msg)
                return GenerateClinicMonthlyInvoiceOutput(
                    success=False,
                    error={"error": msg},
                    status_code=400,
                )

            invoice_data = {
                "company_id": input_data.company_id,
                "client_id": first_client_id,  # Client principal (premier de la liste)
                "bill_to_client_id": None,  # S2 utilise billed_to_company_id
                "billing_party_id": billing_party_id,
                "billed_to_company_id": input_data.clinic_company_id,  # ✅ Toujours non-null en S2
                "billing_strategy": InvoiceBillingStrategy.S2_CLINIC_MONTHLY,
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
                "meta": {
                    "s2_clinic_monthly": True,
                    "clinic_company_id": input_data.clinic_company_id,
                    "total_patients": len({r.client_id for r in reservations}),
                    "total_reservations": len(reservations),
                },
            }

            # ✅ Assert final: vérifier que billed_to_company_id est bien défini avant création
            if invoice_data["billed_to_company_id"] is None:
                msg = (
                    "Erreur interne: billed_to_company_id est null pour une facture S2"
                )
                logger.error(msg)
                return GenerateClinicMonthlyInvoiceOutput(
                    success=False,
                    error={"error": msg},
                    status_code=500,
                )
            # ✅ Créer la facture avec catch IntegrityError pour race condition
            try:
                invoice_dto = self.invoice_repo.create(invoice_data)
                invoice = Invoice.query.get(invoice_dto.id)
                if invoice is None:
                    msg = "Erreur lors de la création de la facture"
                    raise ValueError(msg)

                # ✅ Assert final: vérifier que billed_to_company_id est bien défini après création
                if invoice.billed_to_company_id is None:
                    db.session.rollback()
                    msg = "Erreur interne: billed_to_company_id est null après création de la facture S2"
                    logger.error(msg)
                    return GenerateClinicMonthlyInvoiceOutput(
                        success=False,
                        error={"error": msg},
                        status_code=500,
                    )
                logger.debug(
                    "Facture S2 créée avec billed_to_company_id=%s (clinic_company_id=%s)",
                    invoice.billed_to_company_id,
                    input_data.clinic_company_id,
                )
            except IntegrityError as e:
                # ✅ Race condition: une autre requête a créé la facture entre le SELECT et l'INSERT
                # La contrainte unique DB a bloqué l'insertion
                db.session.rollback()
                logger.warning(
                    "Tentative de création d'une facture S2 en doublon (race condition): %s",
                    str(e),
                )
                # Récupérer la facture existante pour retourner son ID
                existing_invoice = Invoice.query.filter(
                    and_(
                        Invoice.company_id == input_data.company_id,
                        Invoice.billed_to_company_id == input_data.clinic_company_id,
                        Invoice.period_year == input_data.period_year,
                        Invoice.period_month == input_data.period_month,
                        Invoice.billing_strategy
                        == InvoiceBillingStrategy.S2_CLINIC_MONTHLY,
                        Invoice.status != InvoiceStatus.CANCELLED,
                    )
                ).first()
                if existing_invoice:
                    month_names = [
                        "Janvier",
                        "Février",
                        "Mars",
                        "Avril",
                        "Mai",
                        "Juin",
                        "Juillet",
                        "Août",
                        "Septembre",
                        "Octobre",
                        "Novembre",
                        "Décembre",
                    ]
                    month_name = (
                        month_names[input_data.period_month - 1]
                        if 1 <= input_data.period_month <= PERIOD_MONTH_THRESHOLD
                        else str(input_data.period_month)
                    )
                    msg = (
                        f"Facture clinique mensuelle (S2) déjà générée pour {month_name} {input_data.period_year}. "
                        f"Numéro: {existing_invoice.invoice_number}"
                    )
                    # ✅ Log INFO erreur 409: existing_invoice_id, existing_invoice_number
                    logger.info(
                        (
                            "⚠️ S2 invoice 409 conflict (race condition): company_id=%s, clinic_company_id=%s, "
                            "period=%s-%02d, existing_invoice_id=%s, existing_invoice_number=%s"
                        ),
                        input_data.company_id,
                        input_data.clinic_company_id,
                        input_data.period_year,
                        input_data.period_month,
                        existing_invoice.id,
                        existing_invoice.invoice_number,
                    )
                    return GenerateClinicMonthlyInvoiceOutput(
                        success=False,
                        error={
                            "error": msg,
                            "existing_invoice_id": existing_invoice.id,
                            "existing_invoice_number": existing_invoice.invoice_number,
                        },
                        status_code=HTTP_409_CONFLICT,  # ✅ 409 Conflict
                    )
                # Si on ne trouve pas la facture (cas rare), retourner une erreur générique
                msg = "Une facture clinique mensuelle (S2) existe déjà pour cette clinique et cette période."
                return GenerateClinicMonthlyInvoiceOutput(
                    success=False,
                    error={"error": msg},
                    status_code=HTTP_409_CONFLICT,
                )

            # 9. Créer les lignes de facture (1 ligne / unité A/R ou simple)
            subtotal = Decimal("0.00")
            vat_total = Decimal("0.00")
            vat_breakdown: dict[str, dict[str, Decimal]] = {}

            from application.invoices.invoice_line_description import (
                resolve_s2_clinic_line_patient_name,
            )

            client_cache: dict[int, dict[str, Any]] = {}

            for unit in units:
                segments = [by_id[i] for i in unit.booking_ids if i in by_id]
                if not segments:
                    continue
                primary = next(
                    (s for s in segments if int(s.id) == unit.primary_booking_id),
                    segments[0],
                )
                for reservation in segments:
                    if (
                        reservation.billed_to_company_id != input_data.clinic_company_id
                        and reservation.billed_to_type == "clinic"
                    ):
                        reservation.billed_to_company_id = input_data.clinic_company_id

                seg_amounts: list[Decimal] = []
                for reservation in segments:
                    mission_type = (
                        getattr(reservation, "mission_type", None)
                        or "patient_transport"
                    )
                    if mission_type == "material_delivery":
                        fixed_price = billing_settings_dto.material_delivery_price_fixed
                        if fixed_price is None or fixed_price <= 0:
                            msg = (
                                "Impossible de facturer : configurez le prix fixe livraison "
                                f"dans Paramètres > Facturation (réservation #{reservation.id})."
                            )
                            return GenerateClinicMonthlyInvoiceOutput(
                                success=False,
                                error={
                                    "error": msg,
                                    "error_code": ErrorCodes.MATERIAL_DELIVERY_PRICE_NOT_CONFIGURED,
                                    "details": {
                                        "field": "material_delivery_price_fixed",
                                        "booking_id": str(reservation.id),
                                    },
                                },
                                status_code=400,
                            )
                    override = overrides_map.get(reservation.id)
                    ba = calculate_billable_booking_amount(
                        reservation,
                        billing_settings=billing_settings_dto,
                        override=override
                        if mission_type != "material_delivery"
                        else None,
                    ).amount_ht
                    seg_amounts.append(ba)

                base_amount = round_to_5_cents(sum(seg_amounts, Decimal("0.00")))
                override = overrides_map.get(primary.id)

                client = self.client_repo.find_model_by_id_with_user(
                    primary.client_id, input_data.company_id
                )
                if client and getattr(client, "is_institution", False):
                    patient_name = resolve_s2_clinic_line_patient_name(client, primary)
                    patient_id = primary.client_id
                else:
                    if primary.client_id not in client_cache:
                        patient_name = resolve_s2_clinic_line_patient_name(
                            client, primary
                        )
                        patient_id = (
                            client.id if client is not None else primary.client_id
                        )
                        client_cache[primary.client_id] = {
                            "name": patient_name,
                            "id": patient_id,
                        }
                    patient_info = client_cache[primary.client_id]
                    patient_name = patient_info["name"]
                    patient_id = patient_info["id"]

                line_vat_rate = Decimal("0")
                if vat_applicable:
                    if override and override.get("vat_rate") is not None:
                        try:
                            override_vat_rate = Decimal(
                                str(override["vat_rate"])
                            ).quantize(Decimal("0.01"))
                            if override_vat_rate > Decimal("0"):
                                line_vat_rate = override_vat_rate
                        except (InvalidOperation, ValueError, TypeError):
                            line_vat_rate = default_vat_rate
                    else:
                        line_vat_rate = default_vat_rate

                vat_amount, total_with_vat = self.invoice_calculator.calculate_vat(
                    base_amount, line_vat_rate
                )

                mission_type = (
                    getattr(primary, "mission_type", None) or "patient_transport"
                )
                is_delivery = mission_type == "material_delivery"
                if unit.kind == "round_trip" and len(segments) >= 2:
                    ordered = sorted(
                        segments,
                        key=lambda b: (
                            b.scheduled_time or datetime.min.replace(tzinfo=UTC),
                            int(b.id),
                        ),
                    )
                    pri_desc = build_invoice_line_description_clinic_monthly(
                        ordered[0],
                        description_builder=self.description_builder,
                    )
                    description = (
                        build_merged_round_trip_invoice_line_description_from_segments(
                            ordered,
                            primary_segment_description=pri_desc,
                        )
                    )
                    billing_unit = "round_trip"
                else:
                    description = build_invoice_line_description_clinic_monthly(
                        primary,
                        description_builder=self.description_builder,
                    )
                    billing_unit = "single"

                line_type = (
                    InvoiceLineType.MATERIAL_DELIVERY
                    if is_delivery
                    else InvoiceLineType.RIDE
                )
                booking_ids = [int(s.id) for s in segments]
                line_meta: dict[str, Any] = {
                    "patient_id": patient_id,
                    "patient_client_id": primary.client_id,
                    "service_date": _booking_service_date_iso(primary),
                    "billing_unit": billing_unit,
                    "booking_ids": booking_ids,
                    "primary_booking_id": int(primary.id),
                }
                if patient_name and str(patient_name).strip():
                    line_meta["patient_name"] = patient_name
                if unit.kind == "round_trip":
                    line_meta["is_round_trip_leg"] = True
                    line_meta["transport_type"] = "A/R"
                    others = [i for i in booking_ids if i != int(primary.id)]
                    line_meta["round_trip_secondary_reservation_ids"] = others
                    if others:
                        line_meta["round_trip_secondary_reservation_id"] = others[0]

                line_data = {
                    "invoice_id": invoice.id,
                    "type": line_type,
                    "description": description[:500] if description else description,
                    "qty": Decimal("1"),
                    "unit_price": base_amount,
                    "line_total": base_amount,
                    "vat_rate": line_vat_rate if line_vat_rate > Decimal("0") else None,
                    "vat_amount": vat_amount,
                    "total_with_vat": total_with_vat,
                    "adjustment_note": (
                        str(override["note"])[:500]
                        if override and override.get("note")
                        else None
                    ),
                    "reservation_id": primary.id,
                    "line_meta": line_meta,
                }
                line_dto = self.invoice_line_repo.create(line_data)

                for reservation in segments:
                    reservation.invoice_line_id = line_dto.id
                    reservation.updated_at = datetime.now(UTC)

                subtotal += base_amount
                vat_total += vat_amount
                rate_key = f"{line_vat_rate.normalize()}"
                if rate_key not in vat_breakdown:
                    vat_breakdown[rate_key] = {
                        "base": Decimal("0.00"),
                        "vat": Decimal("0.00"),
                    }
                vat_breakdown[rate_key]["base"] += base_amount
                vat_breakdown[rate_key]["vat"] += vat_amount

            # 10. Mettre à jour les totaux de la facture
            subtotal = round_to_5_cents(subtotal)
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
            invoice.meta = cast(Any, current_meta)

            # 11. Commit métier (facture + lignes + réservations), puis PDF dans une 2e transaction
            db.session.commit()

            # 12. PDF + meta.pdf (le service recharge la facture après expire_all)
            try:
                pdf_url = self.pdf_service.generate_invoice_pdf(invoice)
                if pdf_url:
                    mark_pdf_ready(invoice, pdf_url)
                else:
                    mark_pdf_failed(invoice, "PDF_EMPTY")
                db.session.commit()
            except Exception as pdf_err:
                logger.exception(
                    "Échec génération ou persistance PDF après facture clinique mensuelle invoice_id=%s",
                    getattr(invoice, "id", None),
                )
                try:
                    mark_pdf_failed(invoice, str(pdf_err))
                    db.session.commit()
                except Exception:
                    db.session.rollback()
                    logger.exception(
                        "Échec persistance mark_pdf_failed clinique invoice_id=%s",
                        getattr(invoice, "id", None),
                    )

            # 13. Vérification post-commit : s'assurer que invoice_line_id
            # est bien persisté sur chaque booking (filet de sécurité)
            unfixed = []
            for reservation in reservations:
                db.session.refresh(reservation)
                if reservation.invoice_line_id is None:
                    unfixed.append(reservation.id)
            if unfixed:
                logger.error(
                    "⚠️ S2 post-commit: %d booking(s) avec invoice_line_id=None "
                    "après commit ! IDs: %s. Tentative de réparation.",
                    len(unfixed),
                    unfixed[:10],
                )
                # Réparation : retrouver la ligne par reservation_id
                from models import InvoiceLine as ILModel

                for bid in unfixed:
                    il = ILModel.query.filter_by(
                        reservation_id=bid, invoice_id=invoice.id
                    ).first()
                    if il:
                        bk = Booking.query.get(bid)
                        if bk:
                            bk.invoice_line_id = il.id
                            logger.info(
                                "  Repaired booking %s → invoice_line_id=%s",
                                bid,
                                il.id,
                            )
                db.session.commit()

            # ✅ Log INFO succès S2: company_id, clinic_company_id, period, line_count, total
            logger.info(
                (
                    "✅ S2 invoice generated: company_id=%s, clinic_company_id=%s, "
                    "period=%s-%02d, line_count=%s, total=%.2f CHF, invoice_id=%s"
                ),
                input_data.company_id,
                input_data.clinic_company_id,
                input_data.period_year,
                input_data.period_month,
                len(reservations),
                float(invoice.total_amount),
                invoice.id,
            )

            return GenerateClinicMonthlyInvoiceOutput(
                success=True, invoice_id=invoice.id, invoice=invoice
            )

        except (OperationalError, DBAPIError) as e:
            db.session.rollback()
            err_msg = str(e).lower()
            orig = getattr(e, "orig", None)
            pgcode = getattr(orig, "pgcode", None) if orig else None
            is_enum_error = pgcode == "22P02" or (
                "invalid input value for enum" in err_msg
                and "invoice_line_type" in err_msg
            )
            if is_enum_error:
                logger.error(
                    "Enum invoice_line_type non à jour (migration manquante?): %s",
                    str(e),
                )
                return GenerateClinicMonthlyInvoiceOutput(
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
                "Erreur DB lors de la génération de la facture clinique mensuelle (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return GenerateClinicMonthlyInvoiceOutput(
                success=False,
                error={"error": "Erreur de base de données"},
                status_code=500,
            )
        except IntegrityError as e:
            db.session.rollback()
            err_msg = str(e).lower()
            if "ck_booking_material_delivery_description" in err_msg:
                logger.warning(
                    "Livraison matériel sans description (CHECK constraint): %s",
                    str(e),
                )
                return GenerateClinicMonthlyInvoiceOutput(
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
            if (
                "invoice_line_type" in err_msg
                and "invalid input value for enum" in err_msg
            ):
                logger.error(
                    "Enum invoice_line_type non à jour (migration manquante?): %s",
                    str(e),
                )
                return GenerateClinicMonthlyInvoiceOutput(
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
                "Erreur d'intégrité DB lors de la génération de la facture clinique mensuelle: %s",
                str(e),
            )
            return GenerateClinicMonthlyInvoiceOutput(
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
            return GenerateClinicMonthlyInvoiceOutput(
                success=False,
                error={
                    "error": "Erreur de configuration (type de ligne non supporté)",
                    "error_code": "UNKNOWN_LINE_TYPE",
                    "details": details if details else None,
                },
                status_code=400,
            )
        except ValueError as e:
            db.session.rollback()
            logger.warning(
                "Erreur de validation lors de la génération de facture clinique mensuelle: %s",
                e,
            )
            return GenerateClinicMonthlyInvoiceOutput(
                success=False,
                error={"error": str(e)},
                status_code=400,
            )
        except Exception:
            db.session.rollback()
            logger.exception(
                "Erreur inattendue lors de la génération de la facture clinique mensuelle"
            )
            return GenerateClinicMonthlyInvoiceOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
