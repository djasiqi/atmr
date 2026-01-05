"""Use-case: générer une facture.

Ce use case migre la logique métier de InvoiceService.generate_invoice()
vers l'architecture DDD.
"""

from __future__ import annotations  # noqa: I001

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, cast

from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError

from ext import db
from infrastructure.invoices.invoice_calculator import (
    InvoiceCalculator,
    round_to_5_cents,
)
from infrastructure.invoices.invoice_description_builder import (
    InvoiceDescriptionBuilder,
)
from infrastructure.invoices.invoice_number_generator import InvoiceNumberGenerator
from models import Booking, Invoice, InvoiceLineType, InvoiceStatus
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)
from repositories.invoice_line_repository import InvoiceLineRepository
from repositories.invoice_repository import InvoiceRepository
from repositories.invoice_sequence_repository import InvoiceSequenceRepository
from services.pdf_service import PDFService

logger = logging.getLogger(__name__)

PERIOD_MONTH_THRESHOLD = 12


@dataclass(frozen=True, slots=True)
class GenerateInvoiceInput:
    """Input pour générer une facture.

    Attributes:
        company_id: ID de l'entreprise
        client_id: ID du bénéficiaire du service (patient)
        period_year: Année de facturation (ex: 2025)
        period_month: Mois de facturation (1-12)
        bill_to_client_id: ID du payeur (clinique/institution). Si None, client_id paie directement
        reservation_ids: Liste d'IDs de réservations spécifiques. Si None, prend toutes les réservations non facturées
        overrides: Dict facultatif {reservation_id: {amount, vat_rate, note}}
    """

    company_id: int
    client_id: int
    period_year: int
    period_month: int
    bill_to_client_id: int | None = None
    reservation_ids: list[int] | None = None
    overrides: dict[str, Any] | None = None


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
        try:
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

            # 3. Vérifier le client payeur si fourni
            if input_data.bill_to_client_id:
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

            # 4. Récupérer les réservations
            target_statuses = ["COMPLETED", "RETURN_COMPLETED"]
            if input_data.reservation_ids:
                # Mode sélection manuelle
                booking_dtos = self.booking_repo.find_by_ids(input_data.reservation_ids)
                filtered_booking_dtos = []
                for dto in booking_dtos:
                    # Vérifier les conditions de base
                    if (
                        dto.client_id == input_data.client_id
                        and dto.status.value in target_statuses
                        and getattr(dto, "invoice_line_id", None) is None
                    ):
                        # Pour SUBCONTRACT : l'entreprise propriétaire peut facturer
                        # Pour ASSIGN_TO_PARTNER : l'entreprise exécutante peut facturer
                        is_owner = dto.company_id == input_data.company_id
                        is_executor = (
                            getattr(dto, "executing_company_id", None)
                            == input_data.company_id
                        )

                        if is_owner:
                            # Entreprise propriétaire : peut facturer pour SUBCONTRACT ou si pas de transfert
                            # Pour SUBCONTRACT, l'entreprise propriétaire facture toujours le client
                            # même si la course a été exécutée par une autre entreprise
                            filtered_booking_dtos.append(dto)
                        elif is_executor:
                            # Entreprise exécutante : peut facturer uniquement pour ASSIGN_TO_PARTNER
                            # Vérifier qu'il y a un transfert ASSIGN_TO_PARTNER validé
                            from models.booking_transfer import BookingTransfer
                            from models.enums import TransferModel, TransferStatus

                            transfer = BookingTransfer.query.filter_by(
                                booking_id=dto.id,
                                executing_company_id=input_data.company_id,
                                transfer_model=TransferModel.ASSIGN_TO_PARTNER,
                                is_validated=True,
                                status=TransferStatus.COMPLETED,
                            ).first()
                            if transfer:
                                filtered_booking_dtos.append(dto)

                booking_ids = [dto.id for dto in filtered_booking_dtos]
                reservations = (
                    Booking.query.filter(Booking.id.in_(booking_ids)).all()
                    if booking_ids
                    else []
                )
                if len(reservations) != len(input_data.reservation_ids):
                    logger.warning(
                        (
                            "Certaines réservations ne sont pas valides ou "
                            "déjà facturées. Demandé: %s, Trouvé: %s"
                        ),
                        len(input_data.reservation_ids),
                        len(reservations),
                    )
                if not reservations:
                    msg = "Aucune réservation valide dans la sélection"
                    raise ValueError(msg)
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
                reservations = self.booking_repo.find_by_company_and_client_and_period(
                    company_id=input_data.company_id,
                    client_id=input_data.client_id,
                    start_date=start_date,
                    end_date=end_date,
                    statuses=target_statuses,
                )
                # Filtrer celles déjà facturées
                reservations = [
                    r
                    for r in reservations
                    if getattr(r, "invoice_line_id", None) is None
                ]

                # Pour ASSIGN_TO_PARTNER : inclure aussi les bookings où l'entreprise est exécutante
                from sqlalchemy import and_

                from models.booking_transfer import BookingTransfer
                from models.enums import TransferModel, TransferStatus

                assigned_bookings = (
                    Booking.query.join(BookingTransfer)
                    .filter(
                        and_(
                            Booking.executing_company_id == input_data.company_id,
                            Booking.company_id != input_data.company_id,
                            Booking.client_id == input_data.client_id,
                            Booking.status.in_(target_statuses),
                            Booking.invoice_line_id.is_(None),
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
                # Ajouter les bookings assignés qui ne sont pas déjà dans la liste
                existing_ids = {r.id for r in reservations}
                reservations.extend(
                    [b for b in assigned_bookings if b.id not in existing_ids]
                )

            if not reservations:
                msg = "Aucune réservation trouvée pour cette période"
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

            # 7. Récupérer les infos du client pour les descriptions
            client = self.client_repo.find_model_by_id_with_user(
                input_data.client_id, input_data.company_id
            )
            patient_name = ""
            if client and client.user:
                patient_name = (
                    f"{client.user.first_name} {client.user.last_name}".strip()
                )
            if not patient_name:
                patient_name = f"Client #{input_data.client_id}"

            # 8. Créer la facture
            two_places = Decimal("0.01")
            invoice_data = {
                "company_id": input_data.company_id,
                "client_id": input_data.client_id,
                "bill_to_client_id": input_data.bill_to_client_id,
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

            for reservation in reservations:
                base_amount = Decimal(str(reservation.amount or 0)).quantize(two_places)
                override = overrides_map.get(reservation.id)
                if override and "amount" in override and override["amount"] is not None:
                    try:
                        base_amount = Decimal(str(override["amount"])).quantize(
                            two_places, rounding=ROUND_HALF_UP
                        )
                    except (InvalidOperation, ValueError, TypeError):
                        logger.warning(
                            "Montant override invalide pour réservation %s",
                            reservation.id,
                        )

                # Déterminer le taux de TVA pour cette ligne
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
                            logger.warning(
                                "TVA override invalide pour réservation %s",
                                reservation.id,
                            )
                            line_vat_rate = default_vat_rate
                    else:
                        line_vat_rate = default_vat_rate

                # Arrondir base_amount à 5 centimes avant de calculer la TVA
                base_amount = round_to_5_cents(base_amount)

                # Calculer TVA et total avec TVA
                vat_amount, total_with_vat = self.invoice_calculator.calculate_vat(
                    base_amount, line_vat_rate
                )

                # Construire la description
                description = self.description_builder.build_description(
                    pickup_location=reservation.pickup_location or "",
                    dropoff_location=reservation.dropoff_location or "",
                    patient_name=patient_name if input_data.bill_to_client_id else None,
                    bill_to_client_id=input_data.bill_to_client_id,
                )

                # Créer la ligne
                line_data = {
                    "invoice_id": invoice.id,
                    "type": InvoiceLineType.RIDE,
                    "description": description,
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
                    "reservation_id": reservation.id,
                }
                line_dto = self.invoice_line_repo.create(line_data)

                # Lier la réservation à la ligne de facture
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
            # Arrondir les totaux à 5 centimes pour éviter les montants comme 10.12 ou 11.13
            # Arrondir le subtotal à 5 centimes
            subtotal = round_to_5_cents(subtotal)
            # Arrondir la TVA totale à 5 centimes
            vat_total = round_to_5_cents(vat_total)
            # Calculer le total et l'arrondir à 5 centimes
            total = round_to_5_cents(subtotal + vat_total)
            # Ajuster la TVA totale pour qu'elle corresponde au total arrondi
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

            # 11. Générer le PDF
            pdf_url = self.pdf_service.generate_invoice_pdf(invoice)
            invoice.pdf_url = pdf_url

            # 12. Commit de la transaction
            db.session.commit()

            if input_data.bill_to_client_id:
                logger.info(
                    "Facture générée: %s pour client %s (facturée à institution %s)",
                    invoice_number,
                    input_data.client_id,
                    input_data.bill_to_client_id,
                )
            else:
                logger.info(
                    "Facture générée: %s pour client %s",
                    invoice_number,
                    input_data.client_id,
                )

            return GenerateInvoiceOutput(
                success=True, invoice_id=invoice.id, invoice=invoice
            )

        except (OperationalError, DBAPIError, IntegrityError) as e:
            db.session.rollback()
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
        except ValueError as e:
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
            db.session.rollback()
            logger.exception("Erreur inattendue lors de la génération de la facture")
            return GenerateInvoiceOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
