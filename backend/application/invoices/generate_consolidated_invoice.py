"""Use-case: générer des factures consolidées.

Ce use case migre la logique métier de InvoiceService.generate_consolidated_invoice()
vers l'architecture DDD.
"""

from __future__ import annotations  # noqa: I001

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError

from models import Invoice, InvoiceStatus
from application.invoices.generate_invoice import (
    GenerateInvoiceUseCase,
    GenerateInvoiceInput,
)
from repositories.invoice_repository import InvoiceRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GenerateConsolidatedInvoiceInput:
    """Input pour générer des factures consolidées.

    Attributes:
        company_id: ID de l'entreprise
        client_ids: Liste des IDs de clients (patients)
        period_year: Année de facturation
        period_month: Mois de facturation
        bill_to_client_id: ID du client payeur (institution)
        client_reservations: Mapping client_id -> liste de reservation_ids (optionnel)
        overrides: Paramètres de remplacement (optionnel)
    """

    company_id: int
    client_ids: list[int]
    period_year: int
    period_month: int
    bill_to_client_id: int | None
    client_reservations: dict[int, list[int]] | None = None
    overrides: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class GenerateConsolidatedInvoiceOutput:
    """Output pour générer des factures consolidées.

    Attributes:
        success: True si l'opération a réussi
        invoices: Liste des factures créées (si succès)
        errors: Liste des erreurs par client (si échec partiel)
        success_count: Nombre de factures créées avec succès
        error_count: Nombre d'erreurs
        error: Dictionnaire d'erreurs globales (si échec total)
        status_code: Code HTTP (si échec total)
    """

    success: bool
    invoices: list[Any] | None = None  # List of Invoice models
    errors: list[dict[str, Any]] | None = None
    success_count: int = 0
    error_count: int = 0
    error: dict[str, str] | None = None
    status_code: int | None = None


class GenerateConsolidatedInvoiceUseCase:
    """Use-case Application: générer des factures consolidées pour plusieurs
    clients.

    Ce use case migre la logique métier de
    InvoiceService.generate_consolidated_invoice()
    vers l'architecture DDD. Il génère plusieurs factures pour différents clients
    mais toutes adressées à une institution.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        generate_invoice_use_case: GenerateInvoiceUseCase | None = None,
        invoice_repo: InvoiceRepository | None = None,
    ):
        """Initialise le use case avec injection de dépendances.

        Args:
            generate_invoice_use_case: Use case pour générer une facture individuelle
            invoice_repo: Repository pour Invoice
        """
        self.generate_invoice_use_case = (
            generate_invoice_use_case or GenerateInvoiceUseCase()
        )
        self.invoice_repo = invoice_repo or InvoiceRepository()

    def execute(
        self, input_data: GenerateConsolidatedInvoiceInput
    ) -> GenerateConsolidatedInvoiceOutput:
        """Génère des factures consolidées pour plusieurs clients.

        Args:
            input_data: Données d'entrée du use case

        Returns:
            GenerateConsolidatedInvoiceOutput avec les factures créées et les erreurs
        """
        invoices = []
        errors = []

        for client_id in input_data.client_ids:
            try:
                # 1. Vérifier qu'une facture non annulée n'existe pas déjà
                # pour ce client et cette période avec le même bill_to_client_id
                filter_conditions = [
                    Invoice.company_id == input_data.company_id,
                    Invoice.client_id == client_id,
                    Invoice.period_year == input_data.period_year,
                    Invoice.period_month == input_data.period_month,
                    Invoice.status != InvoiceStatus.CANCELLED,
                ]
                # Prendre en compte le bill_to_client_id :
                # None = facturation directe, sinon facturation tierce
                if input_data.bill_to_client_id is None:
                    filter_conditions.append(Invoice.bill_to_client_id.is_(None))
                else:
                    filter_conditions.append(
                        Invoice.bill_to_client_id == input_data.bill_to_client_id
                    )
                existing_invoice = Invoice.query.filter(
                    and_(*filter_conditions)
                ).first()

                if existing_invoice:
                    logger.warning(
                        (
                            "Facture déjà existante pour client %s, période %s/%s, "
                            "bill_to_client_id=%s"
                        ),
                        client_id,
                        input_data.period_month,
                        input_data.period_year,
                        input_data.bill_to_client_id,
                    )
                    errors.append(
                        {
                            "client_id": client_id,
                            "error": "Facture déjà existante pour cette période",
                        }
                    )
                    continue

                # 2. Récupérer les IDs de réservations pour ce client si fourni
                reservation_ids_for_client = None
                if (
                    input_data.client_reservations
                    and client_id in input_data.client_reservations
                ):
                    reservation_ids_for_client = input_data.client_reservations[
                        client_id
                    ]

                # 3. Générer la facture pour ce client
                generate_input = GenerateInvoiceInput(
                    company_id=input_data.company_id,
                    client_id=client_id,
                    period_year=input_data.period_year,
                    period_month=input_data.period_month,
                    bill_to_client_id=input_data.bill_to_client_id,
                    reservation_ids=reservation_ids_for_client,
                    overrides=input_data.overrides,
                )
                generate_result = self.generate_invoice_use_case.execute(generate_input)

                if not generate_result.success:
                    errors.append(
                        {
                            "client_id": client_id,
                            "error": generate_result.error
                            or {"error": "Erreur inconnue"},
                        }
                    )
                    continue

                if generate_result.invoice:
                    invoices.append(generate_result.invoice)

            except ValueError as e:
                # Erreur de validation métier
                logger.warning(
                    "Impossible de créer facture pour client %s: %s", client_id, e
                )
                errors.append({"client_id": client_id, "error": str(e)})
                continue
            except (OperationalError, DBAPIError, IntegrityError) as e:
                # Erreurs DB attendues : connexion, contraintes, timeout
                logger.error(
                    "Erreur DB pour client %s (DB error: %s): %s",
                    client_id,
                    type(e).__name__,
                    e,
                )
                errors.append({"client_id": client_id, "error": f"Erreur DB: {e!s}"})
                continue
            except Exception:
                # Erreur inattendue : logger avec trace complète
                logger.exception("Erreur inattendue pour client %s", client_id)
                errors.append(
                    {"client_id": client_id, "error": "Erreur interne inattendue"}
                )
                continue

        logger.info(
            (
                "Facturation consolidée: %s factures créées, %s erreurs "
                "pour institution %s"
            ),
            len(invoices),
            len(errors),
            input_data.bill_to_client_id,
        )

        return GenerateConsolidatedInvoiceOutput(
            success=True,
            invoices=invoices,
            errors=errors if errors else None,
            success_count=len(invoices),
            error_count=len(errors),
        )
