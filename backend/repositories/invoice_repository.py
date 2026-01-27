"""Repository pour l'accès aux données Invoice."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, cast

from sqlalchemy.orm import joinedload

from domain.invoice_dto import InvoiceDTO, InvoiceLineDTO
from models import Booking, Invoice, InvoiceStatus

logger = __import__("logging").getLogger(__name__)


class InvoiceRepositoryPort(Protocol):
    """Port (interface) pour le repository Invoice.

    Cette interface définit le contrat que doit respecter toute implémentation
    du repository. Elle permet de découpler la couche Application de l'implémentation
    concrète (SQLAlchemy, MongoDB, etc.).
    """

    def find_by_id_and_company(
        self, invoice_id: int, company_id: int
    ) -> InvoiceDTO | None:
        """Trouve une facture par son ID et company_id.

        Args:
            invoice_id: ID de la facture
            company_id: ID de l'entreprise

        Returns:
            InvoiceDTO ou None si non trouvée
        """
        ...

    def find_by_id_with_lines(
        self, invoice_id: int, company_id: int
    ) -> InvoiceDTO | None:
        """Trouve une facture par son ID avec eager loading des lignes.

        Args:
            invoice_id: ID de la facture
            company_id: ID de l'entreprise

        Returns:
            InvoiceDTO avec lines chargées ou None si non trouvée
        """
        ...

    def find_by_client_id_and_company(
        self, client_id: int, company_id: int
    ) -> list[InvoiceDTO]:
        """Trouve les factures d'un client pour une entreprise.

        Args:
            client_id: ID du client
            company_id: ID de l'entreprise

        Returns:
            Liste de InvoiceDTO triées par created_at décroissant
        """
        ...


class InvoiceRepository:
    """Repository SQLAlchemy pour Invoice.

    Implémentation concrète du port InvoiceRepositoryPort utilisant SQLAlchemy.
    Cette classe convertit les modèles SQLAlchemy en DTOs pour maintenir
    le découplage avec la couche Application.
    """

    def _to_dto(self, invoice: Invoice, include_lines: bool = False) -> InvoiceDTO:
        """Convertit un modèle SQLAlchemy Invoice en DTO.

        Args:
            invoice: Modèle SQLAlchemy Invoice
            include_lines: Si True, inclut les lignes de facture (si chargées)

        Returns:
            InvoiceDTO correspondant
        """
        lines = None
        if include_lines and invoice.lines:
            lines = [
                InvoiceLineDTO(
                    id=line.id,
                    invoice_id=line.invoice_id,
                    line_type=line.type.value
                    if hasattr(line.type, "value")
                    else str(line.type),
                    description=line.description,
                    quantity=line.qty,
                    unit_price=line.unit_price,
                    line_total=line.line_total,
                    vat_rate=line.vat_rate,
                    vat_amount=line.vat_amount,
                    total_with_vat=line.total_with_vat,
                    adjustment_note=line.adjustment_note,
                    reservation_id=line.reservation_id,
                )
                for line in invoice.lines
            ]

        return InvoiceDTO(
            id=invoice.id,
            company_id=invoice.company_id,
            client_id=invoice.client_id,
            bill_to_client_id=invoice.bill_to_client_id,
            period_month=invoice.period_month,
            period_year=invoice.period_year,
            invoice_number=invoice.invoice_number,
            currency=cast(str, invoice.currency),
            subtotal_amount=invoice.subtotal_amount,
            late_fee_amount=invoice.late_fee_amount,
            reminder_fee_amount=invoice.reminder_fee_amount,
            vat_total_amount=invoice.vat_total_amount,
            total_amount=invoice.total_amount,
            amount_paid=invoice.amount_paid,
            balance_due=invoice.balance_due,
            issued_at=cast(datetime | None, invoice.issued_at),
            due_date=invoice.due_date,
            sent_at=invoice.sent_at,
            paid_at=invoice.paid_at,
            status=invoice.status,
            lines=lines,
        )

    def find_by_id_and_company(
        self, invoice_id: int, company_id: int
    ) -> InvoiceDTO | None:
        """Trouve une facture par son ID et company_id.

        Args:
            invoice_id: ID de la facture
            company_id: ID de l'entreprise

        Returns:
            InvoiceDTO ou None si non trouvée
        """
        invoice = Invoice.query.filter_by(id=invoice_id, company_id=company_id).first()
        if invoice is None:
            return None
        return self._to_dto(invoice)

    def find_by_id_with_lines(
        self, invoice_id: int, company_id: int
    ) -> InvoiceDTO | None:
        """Trouve une facture par son ID avec eager loading des lignes.

        Args:
            invoice_id: ID de la facture
            company_id: ID de l'entreprise

        Returns:
            InvoiceDTO avec lines chargées ou None si non trouvée
        """
        invoice = (
            Invoice.query.filter_by(id=invoice_id, company_id=company_id)
            .options(joinedload(Invoice.lines))
            .first()
        )
        if invoice is None:
            return None
        return self._to_dto(invoice, include_lines=True)

    def find_by_client_id_and_company(
        self, client_id: int, company_id: int
    ) -> list[InvoiceDTO]:
        """Trouve les factures d'un client pour une entreprise.

        Args:
            client_id: ID du client
            company_id: ID de l'entreprise

        Returns:
            Liste de InvoiceDTO triées par created_at décroissant
        """
        invoices = (
            Invoice.query.filter_by(client_id=client_id, company_id=company_id)
            .order_by(Invoice.created_at.desc())
            .all()
        )
        return [self._to_dto(inv) for inv in invoices]

    # Méthodes legacy - retournent des modèles SQLAlchemy pour compatibilité
    def find_by_company_id_with_lines(self, company_id: int) -> list[Invoice]:
        """Trouve toutes les factures d'une entreprise avec eager loading des lignes.

        Args:
            company_id: ID de l'entreprise

        Returns:
            Liste de Invoice avec lines chargées
        """
        return (
            Invoice.query.options(joinedload(Invoice.lines))
            .join(Booking)
            .filter(Booking.company_id == company_id)
            .all()
        )

    def find_models_by_client_id_and_company(
        self, client_id: int, company_id: int
    ) -> list[Invoice]:
        """Trouve les factures d'un client pour une entreprise (retourne les
        modèles SQLAlchemy).

        Args:
            client_id: ID du client
            company_id: ID de l'entreprise

        Returns:
            Liste de Invoice triées par created_at décroissant (modèles SQLAlchemy)

        Note:
            Méthode legacy - utiliser find_by_client_id_and_company() pour
            obtenir des DTOs
        """
        return (
            Invoice.query.filter_by(client_id=client_id, company_id=company_id)
            .order_by(Invoice.created_at.desc())
            .all()
        )

    def count_by_client_id(self, client_id: int) -> int:
        """Compte les factures d'un client.

        Args:
            client_id: ID du client

        Returns:
            Nombre de factures
        """
        from sqlalchemy import or_

        return Invoice.query.filter(
            or_(
                Invoice.client_id == client_id,
                Invoice.bill_to_client_id == client_id,
            )
        ).count()

    def find_models_by_company_with_eager_loading(
        self, company_id: int
    ) -> list[Invoice]:
        """Trouve toutes les factures d'une entreprise avec eager loading complet.

        Args:
            company_id: ID de l'entreprise

        Returns:
            Liste de Invoice avec client, bill_to_client, lines et payments chargés
        """
        from models import Client

        return (
            Invoice.query.options(
                joinedload(Invoice.client).joinedload(Client.user),
                joinedload(Invoice.bill_to_client).joinedload(Client.user),
                joinedload(Invoice.lines),
                joinedload(Invoice.payments),
            )
            .filter(Invoice.company_id == company_id)
            .all()
        )

    def find_model_by_id_and_company(
        self, invoice_id: int, company_id: int
    ) -> Invoice | None:
        """Trouve une facture par son ID et company_id (retourne le modèle SQLAlchemy).

        Args:
            invoice_id: ID de la facture
            company_id: ID de l'entreprise

        Returns:
            Invoice ou None si non trouvée
        """
        return Invoice.query.filter_by(id=invoice_id, company_id=company_id).first()

    def find_model_by_id_with_eager_loading(
        self, invoice_id: int, company_id: int
    ) -> Invoice | None:
        """Trouve une facture par son ID avec eager loading (retourne le
        modèle SQLAlchemy).

        Args:
            invoice_id: ID de la facture
            company_id: ID de l'entreprise

        Returns:
            Invoice ou None si non trouvée, avec client et lines chargés
        """
        return (
            Invoice.query.filter_by(id=invoice_id, company_id=company_id)
            .options(
                joinedload(Invoice.client),
                joinedload(Invoice.lines),
            )
            .first()
        )

    def find_models_by_company_with_filters_query(
        self,
        company_id: int,
        status: InvoiceStatus | None = None,
        client_id: int | None = None,
        year: int | None = None,
        month: int | None = None,
        with_balance: bool = False,
        with_reminders: bool = False,
        search_query: str | None = None,
    ):
        """Retourne une query Invoice filtrée par company avec filtres optionnels.

        Args:
            company_id: ID de l'entreprise
            status: Statut de la facture (optionnel)
            client_id: ID du client (optionnel)
            year: Année (optionnel)
            month: Mois (optionnel)
            with_balance: Si True, filtre les factures avec solde > 0 (optionnel)
            with_reminders: Si True, filtre les factures avec rappels > 0 (optionnel)
            search_query: Recherche textuelle sur numéro, nom client, etc. (optionnel)

        Returns:
            Query SQLAlchemy filtrée avec eager loading complet
        """
        from models import Client, User

        query = Invoice.query.options(
            joinedload(Invoice.client).joinedload(Client.user),
            joinedload(Invoice.bill_to_client).joinedload(Client.user),
            joinedload(Invoice.lines),
            joinedload(Invoice.payments),
        ).filter(Invoice.company_id == company_id)

        if status:
            query = query.filter_by(status=status)

        if client_id:
            query = query.filter(Invoice.client_id == client_id)

        if year:
            query = query.filter(Invoice.period_year == year)

        if month:
            query = query.filter(Invoice.period_month == month)

        if with_balance:
            # balance_due > 0
            query = query.filter(Invoice.balance_due > 0)

        if with_reminders:
            # reminder_level > 0
            query = query.filter(Invoice.reminder_level > 0)

        if search_query:
            from sqlalchemy import or_
            from sqlalchemy.orm import aliased

            # Alias pour distinguer client (patient) et institution (payeur)
            PatientClient = aliased(Client)
            BillToClient = aliased(Client)
            PatientUser = aliased(User)

            # Jointure avec le client (patient)
            query = query.join(PatientClient, Invoice.client_id == PatientClient.id)
            query = query.join(PatientUser, PatientClient.user_id == PatientUser.id)

            # Jointure OPTIONNELLE avec l'institution payeuse (bill_to_client)
            query = query.outerjoin(
                BillToClient, Invoice.bill_to_client_id == BillToClient.id
            )

            like = f"%{search_query}%"
            query = query.filter(
                or_(
                    Invoice.invoice_number.ilike(like),
                    PatientUser.first_name.ilike(like),
                    PatientUser.last_name.ilike(like),
                    PatientUser.username.ilike(like),
                    BillToClient.institution_name.ilike(like),
                )
            )

        return query

    def count_all(self) -> int:
        """Compte toutes les factures.

        Returns:
            Nombre total de factures
        """
        return Invoice.query.count()

    def create(self, invoice_data: dict[str, Any]) -> InvoiceDTO:
        """Crée une nouvelle facture.

        Args:
            invoice_data: Dictionnaire avec les données de la facture
                (company_id, client_id, invoice_number, etc.)

        Returns:
            InvoiceDTO créé

        Side-effects:
            - DB: Crée Invoice et commit
        """
        from datetime import UTC, datetime, timedelta
        from decimal import Decimal

        from models.enums import InvoiceBillingStrategy, InvoiceStatus

        invoice = Invoice()
        invoice.company_id = invoice_data["company_id"]
        invoice.client_id = invoice_data["client_id"]
        invoice.bill_to_client_id = invoice_data.get("bill_to_client_id")
        invoice.billing_party_id = invoice_data.get("billing_party_id")
        invoice.billing_strategy = invoice_data.get(
            "billing_strategy", InvoiceBillingStrategy.S1_PATIENT
        )
        invoice.billed_to_company_id = invoice_data.get("billed_to_company_id")
        invoice.period_month = invoice_data["period_month"]
        invoice.period_year = invoice_data["period_year"]
        invoice.invoice_number = invoice_data["invoice_number"]
        invoice.currency = invoice_data.get("currency", "CHF")
        invoice.issued_at = invoice_data.get("issued_at", datetime.now(UTC))
        invoice.due_date = invoice_data.get(
            "due_date",
            datetime.now(UTC)
            + timedelta(days=invoice_data.get("payment_terms_days", 30)),
        )
        invoice.status = invoice_data.get("status", InvoiceStatus.DRAFT)
        invoice.subtotal_amount = invoice_data.get("subtotal_amount", Decimal("0.00"))
        invoice.vat_total_amount = invoice_data.get("vat_total_amount", Decimal("0.00"))
        invoice.total_amount = invoice_data.get("total_amount", Decimal("0.00"))
        invoice.balance_due = invoice_data.get("balance_due", invoice.total_amount)
        invoice.vat_breakdown = invoice_data.get("vat_breakdown")
        invoice.meta = invoice_data.get("meta")

        from ext import db

        db.session.add(invoice)
        db.session.flush()  # Pour obtenir l'ID

        return self._to_dto(invoice)
