"""Use-case: générer une facture clinique mensuelle unique (S2).

Ce use case génère UNE SEULE facture pour tous les patients d'une clinique
sur une période donnée, avec support des exceptions (include/exclude clients).
"""

from __future__ import annotations  # noqa: I001

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, cast

from sqlalchemy import and_
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
from models.enums import InvoiceBillingStrategy
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)
from repositories.invoice_line_repository import InvoiceLineRepository
from repositories.invoice_repository import InvoiceRepository
from repositories.invoice_sequence_repository import InvoiceSequenceRepository
from services.billing.billing_party_linker import resolve_billing_party_for_clinic
from services.documents.pdf import PDFService

logger = logging.getLogger(__name__)

PERIOD_MONTH_THRESHOLD = 12
HTTP_409_CONFLICT = 409  # HTTP Conflict (déjà générée)


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

    def execute(  # noqa: PLR0911
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
                    include_ids = [int(client_id) for client_id in input_data.include_client_ids]
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
                    exclude_ids = [int(client_id) for client_id in input_data.exclude_client_ids]
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

            # 2. Vérifier l'anti-doublon: une seule facture S2 par (company_id, clinic_company_id, year, month)
            # ✅ Check SELECT avant INSERT (optimisation)
            existing_invoice = Invoice.query.filter(
                and_(
                    Invoice.company_id == input_data.company_id,
                    Invoice.billed_to_company_id == input_data.clinic_company_id,
                    Invoice.period_year == input_data.period_year,
                    Invoice.period_month == input_data.period_month,
                    Invoice.billing_strategy == InvoiceBillingStrategy.S2_CLINIC_MONTHLY,
                    Invoice.status != InvoiceStatus.CANCELLED,
                )
            ).first()

            if existing_invoice:
                # ✅ UX: Retourner l'ID de la facture existante pour permettre l'ouverture
                month_names = [
                    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
                    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
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
                        "⚠️ S2 invoice 409 conflict: company_id=%s, clinic_company_id=%s, "
                        "period=%s-%02d, existing_invoice_id=%s, existing_invoice_number=%s"
                    ),
                    input_data.company_id,
                    input_data.clinic_company_id,
                    input_data.period_year,
                    input_data.period_month,
                    existing_invoice.id,
                    existing_invoice.invoice_number,
                )
                logger.warning(msg)
                return GenerateClinicMonthlyInvoiceOutput(
                    success=False,
                    error={
                        "error": msg,
                        "existing_invoice_id": existing_invoice.id,
                        "existing_invoice_number": existing_invoice.invoice_number,
                    },
                    status_code=HTTP_409_CONFLICT,  # ✅ 409 Conflict pour "déjà générée"
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

            # ✅ 4. Récupérer toutes les réservations éligibles du mois
            # Scope strict: billed_to_type='clinic', clinic_company_id match, invoice_line_id null, status target_statuses
            # ✅ Scope strict: billed_to_type='clinic' suffit (exclut automatiquement les overrides patient)
            # Le critère réel = billed_to_type == 'clinic' (pas besoin de vérifier billing_source)
            target_statuses = ["COMPLETED", "RETURN_COMPLETED"]
            query = Booking.query.filter(
                Booking.company_id == input_data.company_id,
                Booking.billed_to_company_id == input_data.clinic_company_id,
                Booking.billed_to_type == "clinic",  # ✅ Strict: uniquement facturation clinique (exclut automatiquement les overrides patient)
                Booking.status.in_(target_statuses),  # ✅ Même liste que S1
                Booking.invoice_line_id.is_(None),  # ✅ Pas encore facturé
                Booking.scheduled_time >= start_date,
                Booking.scheduled_time < end_date,
            )

            # Appliquer les filtres include/exclude (priorité: include > exclude)
            if input_data.include_client_ids:
                query = query.filter(Booking.client_id.in_(input_data.include_client_ids))
            elif input_data.exclude_client_ids:
                query = query.filter(~Booking.client_id.in_(input_data.exclude_client_ids))

            reservations = query.order_by(Booking.scheduled_time.asc()).all()

            # ✅ Gérer le cas "0 lignes" avec message clair (422 Unprocessable Entity)
            if not reservations:
                month_names = [
                    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
                    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
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
                msg = "Erreur interne: billed_to_company_id est null pour une facture S2"
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
                        Invoice.billing_strategy == InvoiceBillingStrategy.S2_CLINIC_MONTHLY,
                        Invoice.status != InvoiceStatus.CANCELLED,
                    )
                ).first()
                if existing_invoice:
                    month_names = [
                        "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
                        "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
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

            # 9. Créer les lignes de facture (multi-clients)
            subtotal = Decimal("0.00")
            vat_total = Decimal("0.00")
            vat_breakdown: dict[str, dict[str, Decimal]] = {}

            # ✅ Cache pour les noms de patients (snapshot au moment de la génération)
            client_cache: dict[int, dict[str, Any]] = {}  # {client_id: {"name": str, "id": int}}

            for reservation in reservations:
                # ✅ Récupérer le nom du patient (snapshot pour traçabilité juridique)
                # Format standardisé: "NOM Prénom" pour cohérence PDF
                if reservation.client_id not in client_cache:
                    client = self.client_repo.find_model_by_id_with_user(
                        reservation.client_id, input_data.company_id
                    )
                    patient_name = ""
                    patient_id = reservation.client_id
                    if client and client.user:
                        # ✅ Format standardisé: "NOM Prénom" (majuscules pour nom, capitalisé pour prénom)
                        first_name = (client.user.first_name or "").strip()
                        last_name = (client.user.last_name or "").strip()
                        if last_name and first_name:
                            # Format: "NOM Prénom" (nom en majuscules, prénom capitalisé)
                            patient_name = f"{last_name.upper()} {first_name.capitalize()}".strip()
                        elif last_name:
                            patient_name = last_name.upper()
                        elif first_name:
                            patient_name = first_name.capitalize()
                        else:
                            patient_name = client.user.username or f"Client #{reservation.client_id}"
                        patient_id = client.id
                    if not patient_name:
                        patient_name = f"Client #{reservation.client_id}"
                    # ✅ Snapshot: stocker patient_id + patient_name au moment de la génération
                    client_cache[reservation.client_id] = {
                        "name": patient_name,
                        "id": patient_id,
                    }

                patient_info = client_cache[reservation.client_id]
                patient_name = patient_info["name"]
                patient_id = patient_info["id"]

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
                            line_vat_rate = default_vat_rate
                    else:
                        line_vat_rate = default_vat_rate

                # Arrondir base_amount à 5 centimes avant de calculer la TVA
                base_amount = round_to_5_cents(base_amount)

                # Calculer TVA et total avec TVA
                vat_amount, total_with_vat = self.invoice_calculator.calculate_vat(
                    base_amount, line_vat_rate
                )

                # Construire la description avec le nom du patient (S2)
                description = self.description_builder.build_description(
                    pickup_location=reservation.pickup_location or "",
                    dropoff_location=reservation.dropoff_location or "",
                    patient_name=patient_name,  # ✅ S2: toujours inclure le patient_name
                    bill_to_client_id=None,  # S2 utilise billed_to_company_id
                )

                # ✅ Créer la ligne avec métadonnées patient (snapshot juridique)
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
                    "meta": {
                        # ✅ Snapshot patient: stocker patient_id + patient_name au moment de la génération
                        # Le PDF lira uniquement ces valeurs snapshot (pas de recalcul)
                        "patient_id": patient_id,
                        "patient_name": patient_name,  # Snapshot du nom au moment de la génération
                        "patient_client_id": reservation.client_id,  # Pour référence
                    },
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

            # 11. Générer le PDF
            pdf_url = self.pdf_service.generate_invoice_pdf(invoice)
            invoice.pdf_url = pdf_url

            # 12. Commit de la transaction
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

        except (OperationalError, DBAPIError, IntegrityError) as e:
            db.session.rollback()
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
