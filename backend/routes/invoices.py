import json
import logging
import os
import traceback
from datetime import UTC, datetime
from decimal import Decimal

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import (
    Namespace,
    Resource,
    fields,
    reqparse,
)
from sqlalchemy import func, or_, select
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError
from sqlalchemy.orm import joinedload

# ✅ DDD: Utilise use cases au lieu d'adapters
from application.invoices import (
    CancelInvoiceUseCase,
    DuplicateInvoiceUseCase,
    GenerateConsolidatedInvoiceUseCase,
    GenerateInvoicePdfUseCase,
    GenerateInvoiceReminderUseCase,
    GenerateInvoiceUseCase,
    GetInvoiceUseCase,
    SendInvoiceByEmailInput,
    SendInvoiceByEmailUseCase,
    SendReminderByEmailInput,
    SendReminderByEmailUseCase,
)
from application.invoices.duplicate_invoice import DuplicateInvoiceInput
from application.invoices.generate_invoice_reminder import (
    GenerateInvoiceReminderInput,
)
from ext import limiter, role_required
from middleware.trace_id import get_trace_id
from models import (
    Booking,
    Client,
    ClientStay,
    Company,
    Invoice,
    InvoiceLine,
    InvoicePayment,
    User,
    db,
)
from models.enums import BookingStatus, ClientType, InvoiceStatus, PaymentMethod
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from services.security.idempotency import IdempotencyService
from shared.error_handlers import APIErrorHandler
from shared.response_helpers import paginated_response, success_response

# Note: Modèles utilisés pour requêtes complexes (subqueries, joins)
# TODO: Migrer vers repositories quand les méthodes nécessaires seront disponibles

# Constantes pour éviter les valeurs magiques
SOLDE_ZERO = 0
BALANCE_DUE_ZERO = 0
REMINDER_LEVEL_ZERO = 0
AMOUNT_ZERO = 0
CURRENT_REMINDER_FEE_ZERO = 0
AMOUNT_PAID_ZERO = 0
PERIOD_MONTH_THRESHOLD = 12

# Configuration du logger
logger = logging.getLogger(__name__)

# Namespace pour les factures
invoices_ns = Namespace("invoices", description="Gestion des factures")

# ✅ P0: Modèles d'erreur standardisés
api_error_model = create_api_error_model(invoices_ns)
validation_error_model = create_validation_error_model(invoices_ns)
not_found_error_model = create_not_found_error_model(invoices_ns)
permission_error_model = create_permission_error_model(invoices_ns)

# Modèles de sérialisation
invoice_model = invoices_ns.model(
    "Invoice",
    {
        "id": fields.Integer(required=True),
        "company_id": fields.Integer(required=True),
        "client_id": fields.Integer(required=True),
        "period_month": fields.Integer(required=True),
        "period_year": fields.Integer(required=True),
        "invoice_number": fields.String(required=True),
        "currency": fields.String(required=True),
        "subtotal_amount": fields.Float(required=True),
        "late_fee_amount": fields.Float(required=True),
        "reminder_fee_amount": fields.Float(required=True),
        "total_amount": fields.Float(required=True),
        "amount_paid": fields.Float(required=True),
        "balance_due": fields.Float(required=True),
        "issued_at": fields.DateTime(required=True),
        "due_date": fields.DateTime(required=True),
        "sent_at": fields.DateTime(required=False),
        "paid_at": fields.DateTime(required=False),
        "cancelled_at": fields.DateTime(required=False),
        "status": fields.String(required=True),
        "reminder_level": fields.Integer(required=True),
        "last_reminder_at": fields.DateTime(required=False),
        "pdf_url": fields.String(required=False),
        "qr_reference": fields.String(required=False),
        "client": fields.Nested(
            invoices_ns.model(
                "Client",
                {
                    "id": fields.Integer(required=True),
                    "first_name": fields.String(required=False),
                    "last_name": fields.String(required=False),
                    "username": fields.String(required=False),
                },
            )
        ),
    },
)

invoice_line_model = invoices_ns.model(
    "InvoiceLine",
    {
        "id": fields.Integer(required=True),
        "type": fields.String(required=True),
        "description": fields.String(required=True),
        "qty": fields.Float(required=True),
        "unit_price": fields.Float(required=True),
        "line_total": fields.Float(required=True),
        "reservation_id": fields.Integer(required=False),
    },
)

payment_model = invoices_ns.model(
    "Payment",
    {
        "amount": fields.Float(required=True),
        "paid_at": fields.DateTime(required=True),
        "method": fields.String(required=True),
        "reference": fields.String(required=False),
    },
)

reminder_model = invoices_ns.model(
    "Reminder",
    {
        "level": fields.Integer(required=True),
    },
)

send_email_model = invoices_ns.model(
    "SendEmail",
    {
        "recipient_email": fields.String(
            required=False,
            allow_null=True,
            description="Email du destinataire (optionnel, utilise client.contact_email par défaut)",
        ),
        "force_regenerate_pdf": fields.Boolean(
            required=False,
            default=False,
            description="Regénérer le PDF même s'il existe déjà",
        ),
        "send_method": fields.String(
            required=False,
            default="email",
            enum=["email", "paper"],
            description="Méthode d'envoi (email ou papier)",
        ),
    },
)

billing_settings_model = invoices_ns.model(
    "BillingSettings",
    {
        "payment_terms_days": fields.Integer(
            required=False,
            allow_null=True,
            minimum=0,
            maximum=365,
            description="Jours de paiement (0-365)",
        ),
        "overdue_fee": fields.Float(
            required=False,
            allow_null=True,
            minimum=0,
            description="Frais de retard (>= 0)",
        ),
        "reminder1fee": fields.Float(
            required=False,
            allow_null=True,
            minimum=0,
            description="Frais rappel 1 (>= 0)",
        ),
        "reminder2fee": fields.Float(
            required=False,
            allow_null=True,
            minimum=0,
            description="Frais rappel 2 (>= 0)",
        ),
        "reminder3fee": fields.Float(
            required=False,
            allow_null=True,
            minimum=0,
            description="Frais rappel 3 (>= 0)",
        ),
        "reminder_schedule_days": fields.Raw(
            required=False, description="Planification des rappels (liste de jours)"
        ),
        "auto_reminders_enabled": fields.Boolean(
            required=False, description="Activer rappels automatiques"
        ),
        "email_sender": fields.String(
            required=False,
            allow_null=True,
            max_length=254,
            description="Email expéditeur",
        ),
        "invoice_number_format": fields.String(
            required=False, max_length=50, description="Format numéro facture"
        ),
        "invoice_prefix": fields.String(
            required=False, max_length=20, description="Préfixe numéro facture"
        ),
        "iban": fields.String(
            required=False,
            allow_null=True,
            pattern="^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$",
            description="IBAN (format: CH9300762011623852957)",
        ),
        "qr_iban": fields.String(
            required=False,
            allow_null=True,
            pattern="^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$",
            description="QR IBAN",
        ),
        "esr_ref_base": fields.String(
            required=False, allow_null=True, max_length=26, description="Référence ESR"
        ),
        "invoice_message_template": fields.String(
            required=False,
            allow_null=True,
            max_length=1000,
            description="Template message facture",
        ),
        "reminder1template": fields.String(
            required=False,
            allow_null=True,
            max_length=1000,
            description="Template rappel 1",
        ),
        "reminder2template": fields.String(
            required=False,
            allow_null=True,
            max_length=1000,
            description="Template rappel 2",
        ),
        "reminder3template": fields.String(
            required=False,
            allow_null=True,
            max_length=1000,
            description="Template rappel 3",
        ),
        "legal_footer": fields.String(
            required=False,
            allow_null=True,
            max_length=2000,
            description="Pied de page légal",
        ),
        "pdf_template_variant": fields.String(
            required=False, max_length=50, description="Variant template PDF"
        ),
    },
)

# Modèle Swagger pour génération de facture
invoice_generate_model = invoices_ns.model(
    "InvoiceGenerate",
    {
        "client_id": fields.Integer(
            description="ID client unique (optionnel si client_ids utilisé)", minimum=1
        ),
        "client_ids": fields.List(
            fields.Integer(description="ID client", minimum=1),
            description="Liste d'IDs clients (au moins 1 élément)",
        ),
        "billing_party_id": fields.Integer(
            description="ID BillingParty (destinataire unifié) - alternative à bill_to_client_id",
            minimum=1,
            allow_null=True,
        ),
        "bill_to_client_id": fields.Integer(
            description="ID client payeur (facturation tierce)",
            minimum=1,
            allow_null=True,
        ),
        "clinic_company_id": fields.Integer(
            description="ID clinique payeuse (Company) - alternative à bill_to_client_id",
            minimum=1,
            allow_null=True,
        ),
        "period_year": fields.Integer(
            required=True, minimum=2000, maximum=2100, description="Année (2000-2100)"
        ),
        "period_month": fields.Integer(
            required=True, minimum=1, maximum=12, description="Mois (1-12)"
        ),
        "client_reservations": fields.Raw(
            description="Sélection manuelle: {client_id: [reservation_ids]}"
        ),
        "reservation_ids": fields.List(
            fields.Integer(description="ID réservation"),
            description="Liste d'IDs réservations (optionnel)",
        ),
    },
)

# Parser pour les filtres
filter_parser = reqparse.RequestParser()
filter_parser.add_argument("status", type=str, help="Statut de la facture")
filter_parser.add_argument("client_id", type=int, help="ID du client")
filter_parser.add_argument("year", type=int, help="Année")
filter_parser.add_argument("month", type=int, help="Mois")
filter_parser.add_argument("q", type=str, help="Recherche textuelle")
filter_parser.add_argument("page", type=int, default=1, help="Page")
filter_parser.add_argument("per_page", type=int, default=20, help="Éléments par page")
filter_parser.add_argument("with_balance", type=bool, help="Avec solde > SOLDE_ZERO")
filter_parser.add_argument("with_reminders", type=bool, help="Avec rappels en cours")


@invoices_ns.route("/companies/<int:company_id>/invoices")
class InvoicesList(Resource):
    def get(self, company_id):
        """Récupère la liste des factures avec filtres et pagination."""
        logger.info("🚀 InvoicesList.get() company_id=%s", company_id)

        args = request.args
        status_raw = (args.get("status") or "").strip().lower()
        client_id = args.get("client_id", type=int)
        year = args.get("year", type=int)
        month = args.get("month", type=int)
        q = (args.get("q") or "").strip()
        with_balance = args.get("with_balance") in ("1", "true", "True", "on")
        with_reminders = args.get("with_reminders") in ("1", "true", "True", "on")
        page = args.get("page", default=1, type=int)
        per_page = args.get("per_page", default=20, type=int)

        # ✅ Utilisation du repository pour la requête avec filtres dynamiques
        from repositories.invoice_repository import InvoiceRepository

        # Status mapping frontend -> enum value
        status_map = {
            "draft": InvoiceStatus.DRAFT,
            "sent": InvoiceStatus.SENT,
            "partially_paid": InvoiceStatus.PARTIALLY_PAID,
            "paid": InvoiceStatus.PAID,
            "overdue": InvoiceStatus.OVERDUE,
            "cancelled": InvoiceStatus.CANCELLED,
        }
        status_enum = status_map.get(status_raw) if status_raw else None

        invoice_repo = InvoiceRepository()

        query = invoice_repo.find_models_by_company_with_filters_query(
            company_id=company_id,
            status=status_enum,
            client_id=client_id,
            year=year,
            month=month,
            with_balance=with_balance,
            with_reminders=with_reminders,
            search_query=q if q else None,
        )

        # Calculer les stats sur TOUTES les factures filtrées AVANT le tri et
        # la pagination
        from sqlalchemy import desc, func

        # IMPORTANT: Créer une copie de la query pour les stats (with_entities
        # modifie la query). Utiliser from_statement ou créer une nouvelle query
        # depuis la même base
        stats_base_query = invoice_repo.find_models_by_company_with_filters_query(
            company_id=company_id,
            status=status_enum,
            client_id=client_id,
            year=year,
            month=month,
            with_balance=with_balance,
            with_reminders=with_reminders,
            search_query=q if q else None,
        )

        # Stats avec toutes les factures filtrées (sans tri ni pagination)
        stats_query = stats_base_query.with_entities(
            func.sum(Invoice.total_amount).label("total_issued"),
            func.sum(Invoice.amount_paid).label("total_paid"),
            func.sum(Invoice.balance_due).label("total_balance"),
            func.count(Invoice.id)
            .filter(Invoice.status == InvoiceStatus.OVERDUE)
            .label("overdue_count"),
        )
        stats_result = stats_query.first()

        # Exclure les factures annulées du total émis (utiliser stats_base_query)
        cancelled_query = stats_base_query.filter(
            Invoice.status != InvoiceStatus.CANCELLED
        )
        total_issued_cancelled = cancelled_query.with_entities(
            func.sum(Invoice.total_amount).label("total_issued")
        ).first()
        total_issued = (
            float(total_issued_cancelled.total_issued or 0.0)
            if total_issued_cancelled and total_issued_cancelled.total_issued
            else 0.0
        )
        total_paid = (
            float(stats_result.total_paid or 0.0)
            if stats_result and stats_result.total_paid
            else 0.0
        )
        total_balance = (
            float(stats_result.total_balance or 0.0)
            if stats_result and stats_result.total_balance
            else 0.0
        )
        overdue_count = (
            int(stats_result.overdue_count or 0)
            if stats_result and stats_result.overdue_count
            else 0
        )

        # Tri par émission récente et pagination
        # IMPORTANT: Les joinedload sur relations one-to-many (lines, payments)
        # créent des doublons
        # Solution: Créer une copie de la query SANS les joinedload pour la
        # pagination, puis charger les relations après avec subqueryload
        # Créer une query de base sans les options de chargement pour éviter les
        # doublons. On clone la query en récupérant uniquement les filtres WHERE

        # Créer une nouvelle query Invoice avec les mêmes filtres mais sans les
        # joinedload. On utilise query.statement pour obtenir les filtres WHERE,
        # mais cela ne fonctionne pas bien
        # Meilleure approche: créer une query basique avec les mêmes filtres
        pagination_query = Invoice.query.filter(Invoice.company_id == company_id)

        # Appliquer les mêmes filtres que la query originale
        if status_enum:
            pagination_query = pagination_query.filter_by(status=status_enum)
        if client_id:
            pagination_query = pagination_query.filter(Invoice.client_id == client_id)
        if year:
            pagination_query = pagination_query.filter(Invoice.period_year == year)
        if month:
            pagination_query = pagination_query.filter(Invoice.period_month == month)
        if with_balance:
            pagination_query = pagination_query.filter(Invoice.balance_due > 0)
        if with_reminders:
            pagination_query = pagination_query.filter(Invoice.reminder_level > 0)
        if q:
            # Pour la recherche, on doit réappliquer les jointures
            from sqlalchemy import or_
            from sqlalchemy.orm import aliased

            # Client et User sont déjà importés en haut du fichier
            PatientClient = aliased(Client)
            BillToClient = aliased(Client)
            PatientUser = aliased(User)

            pagination_query = pagination_query.join(
                PatientClient, Invoice.client_id == PatientClient.id
            )
            pagination_query = pagination_query.join(
                PatientUser, PatientClient.user_id == PatientUser.id
            )
            pagination_query = pagination_query.outerjoin(
                BillToClient, Invoice.bill_to_client_id == BillToClient.id
            )

            like = f"%{q}%"
            pagination_query = pagination_query.filter(
                or_(
                    Invoice.invoice_number.ilike(like),
                    PatientUser.first_name.ilike(like),
                    PatientUser.last_name.ilike(like),
                    PatientUser.username.ilike(like),
                    BillToClient.institution_name.ilike(like),
                )
            )

        # Créer une sous-requête pour obtenir les IDs paginés (sans doublons)
        # IMPORTANT: PostgreSQL exige que les colonnes ORDER BY soient dans le
        # SELECT avec DISTINCT
        # On sélectionne id et issued_at, on trie, puis on ne garde que id dans
        # la sous-requête externe
        # Note: Si issued_at est NULL, il sera trié en dernier (NULLS LAST par
        # défaut en PostgreSQL)
        ids_subquery = (
            pagination_query.with_entities(Invoice.id, Invoice.issued_at)
            .distinct()
            .order_by(desc(Invoice.issued_at).nulls_last())
            .subquery()
        )

        # Paginer sur les IDs
        ids_pagination = db.session.query(ids_subquery.c.id).paginate(
            page=page, per_page=per_page, error_out=False
        )
        paginated_ids = [row[0] for row in ids_pagination.items]

        # Charger les objets complets avec les relations pour les IDs paginés
        # IMPORTANT: Utiliser subqueryload au lieu de joinedload pour les
        # relations one-to-many pour éviter les doublons qui faussent la
        # pagination
        if paginated_ids:
            from sqlalchemy.orm import subqueryload

            query = (
                Invoice.query.options(
                    joinedload(Invoice.client).joinedload(Client.user),
                    joinedload(Invoice.bill_to_client).joinedload(Client.user),
                    subqueryload(
                        Invoice.lines
                    ),  # Utiliser subqueryload pour one-to-many
                    subqueryload(
                        Invoice.payments
                    ),  # Utiliser subqueryload pour one-to-many
                )
                .filter(Invoice.id.in_(paginated_ids))
                .order_by(desc(Invoice.issued_at).nulls_last())
            )

        else:
            # Aucun ID trouvé, créer une query vide
            query = Invoice.query.filter(Invoice.id.in_([]))

        # Charger les objets complets
        items = query.all() if paginated_ids else []

        # Créer un objet de pagination manuel pour compatibilité avec le reste du code
        class PaginationObject:
            """Objet de pagination manuel pour compatibilité avec le code existant."""

            def __init__(self, items, total, page, per_page):  # pyright: ignore[reportMissingSuperCall]
                self.items = items
                self.total = total
                self.page = page
                self.per_page = per_page
                self.pages = (total + per_page - 1) // per_page if per_page > 0 else 0
                # Attributs pour compatibilité avec Flask-SQLAlchemy pagination
                self.has_next = page < self.pages
                self.has_prev = page > 1
                self.next_num = page + 1 if self.has_next else None
                self.prev_num = page - 1 if self.has_prev else None

        # Calculer le total pour la pagination (utiliser ids_pagination.total qui est déjà calculé)
        total_count = ids_pagination.total if paginated_ids else 0

        result_invoices = [inv.to_dict() for inv in items]

        # ✅ Inclure les factures partenaires (PartnerInvoice)
        # Une facture partenaire appartient à l'entreprise si :
        # - L'entreprise est owner_company_id ou partner_company_id du partenariat
        # - ET l'entreprise est executing_company_id dans les transferts associés à la facture
        from models.booking_transfer import BookingTransfer
        from models.company import Company
        from models.enums import TransferStatus
        from models.partner_invoice import (
            PartnerInvoice,
            PartnerInvoiceStatus,
            partner_invoice_transfers,
        )
        from models.partnership import Partnership

        # Récupérer tous les partenariats où l'entreprise est impliquée
        partnerships = Partnership.query.filter(
            (Partnership.owner_company_id == company_id)
            | (Partnership.partner_company_id == company_id)
        ).all()

        # Mapping des statuts PartnerInvoice vers Invoice pour le filtrage
        partner_status_map = {
            "draft": PartnerInvoiceStatus.DRAFT,
            "sent": PartnerInvoiceStatus.SENT,
            "partially_paid": PartnerInvoiceStatus.PARTIALLY_PAID,
            "paid": PartnerInvoiceStatus.PAID,
            "overdue": PartnerInvoiceStatus.OVERDUE,
            "cancelled": PartnerInvoiceStatus.CANCELLED,
        }
        partner_status_enum = partner_status_map.get(status_raw) if status_raw else None

        partner_invoice_ids = []
        for partnership in partnerships:
            # Récupérer les factures partenaires pour ce partenariat avec filtres
            partner_invoices_query = PartnerInvoice.query.filter_by(
                partnership_id=partnership.id
            )

            # Appliquer le filtre de statut si spécifié
            if partner_status_enum is not None:
                partner_invoices_query = partner_invoices_query.filter_by(
                    status=partner_status_enum
                )

            # Appliquer le filtre d'année si spécifié
            if year:
                partner_invoices_query = partner_invoices_query.filter_by(
                    period_year=year
                )

            # Appliquer le filtre de mois si spécifié
            if month:
                partner_invoices_query = partner_invoices_query.filter_by(
                    period_month=month
                )

            partner_invoices = partner_invoices_query.all()

            for partner_invoice in partner_invoices:
                # Vérifier que l'entreprise est executing_company_id dans les transferts
                # associés à cette facture
                transfers = (
                    db.session.query(BookingTransfer)
                    .join(
                        partner_invoice_transfers,
                        BookingTransfer.id
                        == partner_invoice_transfers.c.booking_transfer_id,
                    )
                    .filter(
                        partner_invoice_transfers.c.partner_invoice_id
                        == partner_invoice.id,
                        BookingTransfer.executing_company_id == company_id,
                        BookingTransfer.status == TransferStatus.COMPLETED,
                    )
                    .count()
                )

                if transfers > 0:
                    partner_invoice_ids.append(partner_invoice.id)

        # Charger les factures partenaires avec les relations nécessaires
        partner_invoices_items = []
        if partner_invoice_ids:
            partner_invoices_items = (
                PartnerInvoice.query.options(
                    joinedload(PartnerInvoice.partnership).joinedload(
                        Partnership.owner_company
                    ),
                    joinedload(PartnerInvoice.partnership).joinedload(
                        Partnership.partner_company
                    ),
                )
                .filter(PartnerInvoice.id.in_(partner_invoice_ids))
                .all()
            )

        # #region agent log (désactivé par défaut, activable via DEBUG_AGENT_LOGS=1)
        import os

        if os.getenv("DEBUG_AGENT_LOGS", "0") == "1":
            import json
            from pathlib import Path

            try:
                debug_log_path = os.getenv("DEBUG_AGENT_LOG_PATH", ".cursor/debug.log")
                with Path(debug_log_path).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "E",
                                "location": "routes/invoices.py:InvoicesList.get",
                                "message": "Factures partenaires trouvées",
                                "data": {
                                    "company_id": company_id,
                                    "partner_invoice_ids": partner_invoice_ids,
                                    "partner_invoices_count": len(
                                        partner_invoices_items
                                    ),
                                },
                                "timestamp": int(__import__("time").time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
        # #endregion

        # Convertir les factures partenaires en format compatible avec Invoice
        # pour l'affichage dans le tableau
        partner_invoices_dict = []
        for pi in partner_invoices_items:
            # Calculer amount_paid et balance_due
            amount_paid = float(pi.amount_paid or 0)
            balance_due = float(pi.total_amount) - amount_paid

            # Si le statut n'est pas à jour, le mettre à jour en fonction des paiements
            # (pour les factures existantes qui n'ont pas encore amount_paid)
            if pi.status != PartnerInvoiceStatus.CANCELLED:
                if balance_due <= 0:
                    if pi.status != PartnerInvoiceStatus.PAID:
                        pi.status = PartnerInvoiceStatus.PAID
                        if not pi.paid_at:
                            pi.paid_at = datetime.now(UTC)
                elif amount_paid > 0:
                    if pi.status not in (
                        PartnerInvoiceStatus.PARTIALLY_PAID,
                        PartnerInvoiceStatus.PAID,
                    ):
                        pi.status = PartnerInvoiceStatus.PARTIALLY_PAID
                elif pi.status == PartnerInvoiceStatus.PAID:
                    # Si le statut est PAID mais qu'il n'y a pas de paiement, revenir à SENT
                    pi.status = PartnerInvoiceStatus.SENT
                    pi.paid_at = None

            # Appliquer le filtre with_balance si spécifié
            if with_balance and balance_due <= 0:
                continue

            # Déterminer quelle entreprise afficher (l'autre entreprise du partenariat)
            partner_company_name = None
            if pi.partnership:
                if (
                    pi.partnership.owner_company_id == company_id
                    and pi.partnership.partner_company
                ):
                    # L'entreprise actuelle est owner, afficher partner
                    partner_company_name = pi.partnership.partner_company.name
                elif (
                    pi.partnership.partner_company_id == company_id
                    and pi.partnership.owner_company
                ):
                    # L'entreprise actuelle est partner, afficher owner
                    partner_company_name = pi.partnership.owner_company.name

            # Si le nom n'est toujours pas déterminé, essayer une approche alternative
            if not partner_company_name and pi.partnership:
                # En dernier recours, récupérer le nom de l'autre entreprise
                # en utilisant directement les IDs
                other_company_id = None
                if pi.partnership.owner_company_id == company_id:
                    other_company_id = pi.partnership.partner_company_id
                elif pi.partnership.partner_company_id == company_id:
                    other_company_id = pi.partnership.owner_company_id

                if other_company_id:
                    other_company = Company.query.get(other_company_id)
                    if other_company:
                        partner_company_name = other_company.name

            # Appliquer le filtre de recherche textuelle (q) si spécifié
            if q:
                search_lower = q.lower()
                invoice_number_match = (
                    pi.invoice_number.lower().find(search_lower) >= 0
                    if pi.invoice_number
                    else False
                )
                company_name_match = (
                    partner_company_name.lower().find(search_lower) >= 0
                    if partner_company_name
                    else False
                )
                if not invoice_number_match and not company_name_match:
                    continue  # Ne pas inclure cette facture si elle ne correspond pas à la recherche

            # Créer un dictionnaire compatible avec Invoice.to_dict()
            # mais avec un type spécial pour identifier les factures partenaires
            partner_invoices_dict.append(
                {
                    "id": pi.id,
                    "invoice_number": pi.invoice_number,
                    "period_year": pi.period_year,
                    "period_month": pi.period_month,
                    "total_amount": float(pi.total_amount),
                    "amount_paid": amount_paid,
                    "balance_due": balance_due,
                    "status": pi.status,
                    "issued_at": pi.issued_at.isoformat() if pi.issued_at else None,
                    "due_date": pi.due_date.isoformat() if pi.due_date else None,
                    "paid_at": pi.paid_at.isoformat() if pi.paid_at else None,
                    "pdf_url": pi.pdf_url,
                    "currency": pi.currency,
                    "client": {
                        "id": None,
                        "first_name": "",
                        "last_name": "",
                        "username": "",
                        "is_institution": True,
                        "institution_name": partner_company_name
                        or "Entreprise partenaire",
                    },
                    "bill_to_client": None,
                    "lines": [],
                    "payments": [],
                    "reminders": [],
                    "reminder_level": 0,  # Les factures partenaires n'ont pas de rappels
                    "last_reminder_at": None,  # Les factures partenaires n'ont pas de rappels
                    "is_partner_invoice": True,  # Flag pour identifier les factures partenaires
                    "partnership_id": pi.partnership_id,
                }
            )

        # Combiner les factures normales et partenaires
        all_invoices = result_invoices + partner_invoices_dict

        # Trier par issued_at (les plus récentes en premier)
        all_invoices.sort(
            key=lambda x: (
                datetime.fromisoformat(x["issued_at"].replace("Z", "+00:00"))
                if x.get("issued_at")
                else datetime.min.replace(tzinfo=UTC)
            ),
            reverse=True,
        )

        # Paginer manuellement sur toutes les factures (normales + partenaires)
        total_count_with_partners = len(all_invoices)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_invoices = all_invoices[start_idx:end_idx]

        # Calculer le total pour la pagination
        total_count = total_count_with_partners

        pagination = PaginationObject(paginated_invoices, total_count, page, per_page)

        # Construire les liens de pagination
        links = {}
        if pagination.has_next:
            links["next"] = (
                f"/api/invoices?page={pagination.page + 1}&per_page={pagination.per_page}"
            )
        if pagination.has_prev:
            links["prev"] = (
                f"/api/invoices?page={pagination.page - 1}&per_page={pagination.per_page}"
            )
        links["first"] = f"/api/invoices?page=1&per_page={pagination.per_page}"
        if pagination.pages > 0:
            links["last"] = (
                f"/api/invoices?page={pagination.pages}&per_page={pagination.per_page}"
            )

        # Retourner réponse paginée avec stats
        response_data = paginated_response(
            items=paginated_invoices,
            total=pagination.total or 0,
            page=pagination.page,
            per_page=pagination.per_page,
            links=links if links else None,
        )
        # Calculer les stats incluant les factures partenaires
        partner_total_issued = sum(
            float(pi.total_amount)
            for pi in partner_invoices_items
            if pi.status != PartnerInvoiceStatus.CANCELLED
        )
        partner_total_paid = sum(
            float(pi.total_amount)
            for pi in partner_invoices_items
            if pi.status == PartnerInvoiceStatus.PAID
        )
        partner_total_balance = sum(
            float(pi.total_amount)
            for pi in partner_invoices_items
            if pi.status != PartnerInvoiceStatus.PAID
        )
        partner_overdue_count = sum(
            1
            for pi in partner_invoices_items
            if pi.status == PartnerInvoiceStatus.OVERDUE
        )

        # Ajouter les stats au response_data (incluant factures normales + partenaires)
        response_data[0]["stats"] = {
            "total_issued": total_issued + partner_total_issued,
            "total_paid": total_paid + partner_total_paid,
            "total_balance": total_balance + partner_total_balance,
            "overdue_count": overdue_count + partner_overdue_count,
        }
        return response_data


@invoices_ns.route("/companies/<int:company_id>/invoices/debug")
class InvoicesDebug(Resource):
    """Endpoint de debug temporaire pour identifier les factures."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, company_id):
        """Debug: Liste toutes les factures de l'entreprise."""
        from sqlalchemy import func

        from models.enums import InvoiceStatus

        all_invoices = Invoice.query.filter_by(company_id=company_id).all()
        non_cancelled = (
            Invoice.query.filter_by(company_id=company_id)
            .filter(Invoice.status != InvoiceStatus.CANCELLED)
            .all()
        )

        total_calculated = (
            db.session.query(func.sum(Invoice.total_amount))
            .filter_by(company_id=company_id)
            .filter(Invoice.status != InvoiceStatus.CANCELLED)
            .scalar()
        )

        # Test avec les filtres year=2026, month=None
        from repositories.invoice_repository import InvoiceRepository

        repo = InvoiceRepository()
        filtered_query = repo.find_models_by_company_with_filters_query(
            company_id=company_id,
            year=2026,
            month=None,
        )
        filtered_invoices = filtered_query.all()

        result = {
            "total_invoices": len(all_invoices),
            "non_cancelled_count": len(non_cancelled),
            "total_calculated": float(total_calculated or 0.0),
            "filtered_2026_count": len(filtered_invoices),
            "invoices": [
                {
                    "id": inv.id,
                    "period_year": inv.period_year,
                    "period_month": inv.period_month,
                    "total_amount": float(inv.total_amount),
                    "status": inv.status.value
                    if hasattr(inv.status, "value")
                    else str(inv.status),
                    "issued_at": inv.issued_at.isoformat() if inv.issued_at else None,
                    "client_id": inv.client_id,
                }
                for inv in all_invoices
            ],
            "filtered_2026": [
                {
                    "id": inv.id,
                    "period_year": inv.period_year,
                    "period_month": inv.period_month,
                    "total_amount": float(inv.total_amount),
                    "status": inv.status.value
                    if hasattr(inv.status, "value")
                    else str(inv.status),
                }
                for inv in filtered_invoices
            ],
        }
        return result, 200


@invoices_ns.route("/companies/<int:company_id>/billing-settings")
class CompanyBillingSettingsResource(Resource):
    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, company_id):
        """Récupère les paramètres de facturation d'une entreprise."""
        # ✅ DDD: Utilise use-case au lieu de service directement
        from routes.companies import _get_current_company_via_use_case

        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response or not company:
            return error_response, status_code

        # Vérifier que l'ID de l'entreprise correspond
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except (ValueError, TypeError, OverflowError):
            # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
            cid = None
        except Exception:
            # Erreur inattendue : logger et utiliser None
            logger.debug("Unexpected error converting company.id to int: %s", cid_obj)
            cid = None
        if cid != company_id:
            return APIErrorHandler.handle_permission_error(
                "Entreprise non trouvée ou accès refusé",
                logger_instance=logger,
            )

        # ✅ Récupérer directement le modèle SQLAlchemy (pas via repository qui retourne un DTO)
        from models import CompanyBillingSettings

        billing_settings = CompanyBillingSettings.query.filter_by(
            company_id=company_id
        ).first()
        if not billing_settings:
            # Créer des paramètres par défaut si non existants
            billing_settings = CompanyBillingSettings()
            billing_settings.company_id = company_id
            db.session.add(billing_settings)
            db.session.commit()
        return success_response(data=billing_settings.to_dict())

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    @invoices_ns.expect(billing_settings_model, validate=False)
    def put(self, company_id):  # noqa: PLR0911
        """Met à jour les paramètres de facturation d'une entreprise."""
        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
                cid = None
            except Exception:
                # Erreur inattendue : logger et utiliser None
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Entreprise non trouvée ou accès refusé",
                    logger_instance=logger,
                )

            # ✅ Récupérer directement le modèle SQLAlchemy (pas via repository qui retourne un DTO)
            from models import CompanyBillingSettings

            billing_settings = CompanyBillingSettings.query.filter_by(
                company_id=company_id
            ).first()
            if not billing_settings:
                return APIErrorHandler.handle_not_found(
                    "Paramètres de facturation",
                    company_id if "company_id" in locals() else None,
                    logger,
                )

            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import (
                ValidationError,
            )

            from schemas.invoice_schemas import BillingSettingsUpdateSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            try:
                validated_data = validate_request(
                    BillingSettingsUpdateSchema(), data, strict=False
                )
            except ValidationError as e:
                return handle_validation_error(e)

            logger.info(
                "Données reçues pour les paramètres de facturation: %s", validated_data
            )

            # Convertir les chaînes vides en None pour les champs numériques
            for field in [
                "payment_terms_days",
                "overdue_fee",
                "reminder1fee",
                "reminder2fee",
                "reminder3fee",
            ]:
                if field in validated_data:
                    value = validated_data[field]
                    if value == "" or value is None:
                        setattr(billing_settings, field, None)
                    else:
                        try:
                            setattr(
                                billing_settings,
                                field,
                                float(value) if "." in str(value) else int(value),
                            )
                        except (ValueError, TypeError):
                            logger.warning("Valeur invalide pour %s: %s", field, value)
                            setattr(billing_settings, field, None)

            # Mettre à jour les autres champs - utilise données validées
            # email d'envoi des factures
            if "billing_email" in validated_data or "email_sender" in validated_data:
                billing_settings.email_sender = validated_data.get(
                    "billing_email",
                    validated_data.get("email_sender", billing_settings.email_sender),
                )
            if "invoice_prefix" in validated_data:
                billing_settings.invoice_prefix = validated_data["invoice_prefix"]
            if "invoice_number_format" in validated_data:
                billing_settings.invoice_number_format = validated_data[
                    "invoice_number_format"
                ]
            if "iban" in validated_data:
                billing_settings.iban = validated_data["iban"]
            if "qr_iban" in validated_data:
                billing_settings.qr_iban = validated_data["qr_iban"]
            # esr_ref_base dans le schéma, colonne esr_ref_base dans le modèle
            if "esr_ref_base" in validated_data:
                billing_settings.esr_ref_base = (
                    validated_data.get("esr_ref_base") or None
                )

            # planning des rappels: accepter dict ou string JSON, ou tableau
            # ordonné
            if "reminder_schedule_days" in validated_data:
                sched = validated_data["reminder_schedule_days"]
                try:
                    if isinstance(sched, str):
                        sched = json.loads(sched)
                    # Normaliser en dict str->int ex: {"1": 30, "2": 10, "3":
                    # 5}
                    if isinstance(sched, list):
                        # ex [30,10,5]
                        sched = {str(i + 1): int(v) for i, v in enumerate(sched)}
                    elif isinstance(sched, dict):
                        sched = {str(k): int(v) for k, v in sched.items()}
                    billing_settings.reminder_schedule_days = sched
                except json.JSONDecodeError as e:
                    # Erreurs de parsing JSON attendues (doit être avant ValueError)
                    logger.warning(
                        "reminder_schedule_days invalide (JSON decode error), valeur ignorée: %s",
                        e,
                    )
                except (ValueError, TypeError, KeyError) as e:
                    # Erreurs de parsing/validation attendues : valeurs invalides, types incorrects
                    logger.warning(
                        "reminder_schedule_days invalide (validation error: %s), valeur ignorée",
                        type(e).__name__,
                    )
                except Exception as e:
                    # Erreur inattendue : logger mais continuer
                    logger.warning(
                        "reminder_schedule_days invalide (unexpected error: %s), valeur ignorée",
                        e,
                    )

            # auto_reminders_enabled si fourni - utilise données validées
            if "auto_reminders_enabled" in validated_data:
                billing_settings.auto_reminders_enabled = validated_data[
                    "auto_reminders_enabled"
                ]

            if "vat_applicable" in validated_data:
                billing_settings.vat_applicable = bool(validated_data["vat_applicable"])

            if "vat_rate" in validated_data:
                rate_value = validated_data.get("vat_rate")
                if rate_value is None:
                    billing_settings.vat_rate = None
                else:
                    try:
                        billing_settings.vat_rate = Decimal(str(rate_value))
                    except (ValueError, TypeError) as e:
                        # Erreurs de conversion Decimal attendues
                        logger.warning(
                            "vat_rate invalide (validation error: %s), valeur ignorée: %s",
                            type(e).__name__,
                            rate_value,
                        )
                        billing_settings.vat_rate = None
                    except Exception as e:
                        # Erreur inattendue : logger et utiliser None
                        logger.warning(
                            "vat_rate invalide (unexpected error: %s), valeur ignorée: %s",
                            e,
                            rate_value,
                        )
                        billing_settings.vat_rate = None

            if "vat_label" in validated_data:
                billing_settings.vat_label = validated_data.get("vat_label")

            if "vat_number" in validated_data:
                billing_settings.vat_number = validated_data.get("vat_number")

            logger.info("Paramètres mis à jour avec succès")
            db.session.commit()
            return success_response(data=billing_settings.to_dict())

        except (OperationalError, DBAPIError, IntegrityError) as e:
            # Erreurs DB attendues : connexion, contraintes, timeout
            logger.error(
                "Erreur DB lors de la mise à jour des paramètres (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : données invalides, attributs manquants
            logger.error(
                "Erreur de validation lors de la mise à jour des paramètres (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            db.session.rollback()
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            # Erreur inattendue : logger avec trace complète
            logger.exception("Erreur inattendue lors de la mise à jour des paramètres")
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route("/companies/<int:company_id>/clients/eligible")
class EligibleClients(Resource):
    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    @invoices_ns.param("search", "Recherche par prénom, nom ou email", type="string")
    @invoices_ns.param(
        "limit",
        "Nombre maximum de clients retournés (1-200)",
        type="integer",
        default=50,
        minimum=1,
        maximum=200,
    )
    @invoices_ns.param(
        "billed_to_type",
        "Filtrer par type de facturation: 'patient' = facturation directe au client uniquement",
        type="string",
    )
    def get(self, company_id: int):
        """Liste les clients ayant des trajets non facturés,
        avec possibilité de recherche. Si billed_to_type=patient, uniquement les transports à facturer au patient."""
        # ✅ DDD: Utilise use-case au lieu de service directement
        from routes.companies import _get_current_company_via_use_case

        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response or not company:
            return error_response, status_code

        # Vérifier que l'ID de l'entreprise correspond
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except (ValueError, TypeError, OverflowError):
            # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
            cid = None
        except Exception:
            # Erreur inattendue : logger et utiliser None
            logger.debug("Unexpected error converting company.id to int: %s", cid_obj)
            cid = None
        if cid != company_id:
            return APIErrorHandler.handle_permission_error(
                "Entreprise non trouvée ou accès refusé",
                logger_instance=logger,
            )

        search = (request.args.get("search") or "").strip()
        period_year = request.args.get("year", type=int)
        period_month = request.args.get("month", type=int)
        period_month_threshold = PERIOD_MONTH_THRESHOLD
        bill_to_client_id = request.args.get("bill_to_client_id", type=int)
        clinic_company_id = request.args.get("clinic_company_id", type=int)
        billed_to_type = (request.args.get("billed_to_type") or "").strip().lower()
        try:
            limit = int(request.args.get("limit", 50))
        except ValueError:
            limit = 50
        limit = max(1, min(limit, 200))

        # Log pour debug
        logger.info(
            "🔍 [EligibleClients] Paramètres reçus: company_id=%s, year=%s, month=%s, search=%s, limit=%s, billed_to_type=%s",
            company_id,
            period_year,
            period_month,
            search,
            limit,
            billed_to_type or None,
        )

        if bill_to_client_id and not clinic_company_id:
            institution = Client.query.filter_by(
                id=bill_to_client_id, company_id=company_id, is_institution=True
            ).first()
            if institution:
                clinic_company_id = getattr(
                    institution, "default_billed_to_company_id", None
                )
                if not clinic_company_id and institution.institution_name:
                    clinic_company = Company.query.filter_by(
                        name=institution.institution_name
                    ).first()
                    if clinic_company:
                        clinic_company_id = clinic_company.id

        # ✅ Pour les courses transférées :
        # - L'entreprise propriétaire (company_id) peut facturer le client
        # - L'entreprise exécutante (executing_company_id) peut facturer l'entreprise propriétaire
        # Ici, on cherche les clients avec courses où l'entreprise est propriétaire
        # ✅ Unbilled = pas de ligne facture OU ligne liée à une facture annulée (récupération après annulation)
        unbilled_query = (
            db.session.query(
                Booking.client_id.label("client_id"),
                func.count(Booking.id).label("unbilled_count"),
                func.coalesce(func.sum(Booking.amount), 0).label("unbilled_total_amount"),
                func.max(func.coalesce(Booking.completed_at, Booking.scheduled_time)).label(
                    "last_ride_at"
                ),
            )
            .outerjoin(InvoiceLine, Booking.invoice_line_id == InvoiceLine.id)
            .outerjoin(Invoice, InvoiceLine.invoice_id == Invoice.id)
            .filter(
                Booking.company_id == company_id,  # Entreprise propriétaire
                Booking.status.in_(
                    [
                        BookingStatus.COMPLETED.value,
                        BookingStatus.RETURN_COMPLETED.value,
                    ]
                ),
                or_(
                    Booking.invoice_line_id.is_(None),
                    Invoice.status == InvoiceStatus.CANCELLED,
                ),
            )
        )

        # Facturation directe au client : uniquement les transports à facturer au patient
        if billed_to_type == "patient":
            unbilled_query = unbilled_query.filter(Booking.billed_to_type == "patient")

        # S2 / facturation clinique : uniquement les clients avec ≥1 transport à facturer à la clinique
        # (exclut les patients qui n'ont que des transports "patient", ex. après annulation facture client)
        if clinic_company_id and billed_to_type != "patient":
            unbilled_query = unbilled_query.filter(
                Booking.billed_to_type == "clinic",
                Booking.billed_to_company_id == clinic_company_id,
            )

        # Filtrer par période si fournie
        if period_year and period_month:
            # Créer des dates timezone-naive pour la comparaison (scheduled_time est timezone-naive dans la DB)
            start_date = datetime(period_year, period_month, 1)
            if period_month == period_month_threshold:
                end_date = datetime(period_year + 1, 1, 1)
            else:
                end_date = datetime(period_year, period_month + 1, 1)

            unbilled_query = unbilled_query.filter(
                Booking.scheduled_time >= start_date,
                Booking.scheduled_time < end_date,
            )

        unbilled_subquery = unbilled_query.group_by(Booking.client_id).subquery()
        # Note: Les courses transférées ont toujours company_id = entreprise propriétaire,
        # donc elles sont incluses automatiquement dans cette requête

        query = (
            db.session.query(
                Client,
                unbilled_subquery.c.unbilled_count,
                unbilled_subquery.c.unbilled_total_amount,
                unbilled_subquery.c.last_ride_at,
            )
            .join(unbilled_subquery, Client.id == unbilled_subquery.c.client_id)
            .options(joinedload(Client.user))
            .filter(
                Client.company_id == company_id,
                Client.is_institution.is_(False),
                Client.is_active.is_(True),
                Client.client_type != ClientType.SELF_SERVICE,
            )
        )

        if clinic_company_id:
            stay_query = db.session.query(ClientStay.client_id).filter(
                ClientStay.company_id == clinic_company_id,
                ClientStay.status == "active",
            )
            if period_year and period_month:
                start_date = datetime(period_year, period_month, 1)
                if period_month == period_month_threshold:
                    end_date = datetime(period_year + 1, 1, 1)
                else:
                    end_date = datetime(period_year, period_month + 1, 1)
                stay_query = stay_query.filter(
                    ClientStay.start_date <= end_date,
                    (ClientStay.end_date.is_(None))
                    | (ClientStay.end_date >= start_date),
                )
            stay_subquery = stay_query.distinct().subquery()
            query = query.join(stay_subquery, Client.id == stay_subquery.c.client_id)

        if search:
            pattern = f"%{search.lower()}%"
            query = query.join(User, Client.user).filter(
                or_(
                    func.lower(User.first_name).like(pattern),
                    func.lower(User.last_name).like(pattern),
                    func.lower(User.email).like(pattern),
                )
            )

        results = (
            query.order_by(unbilled_subquery.c.last_ride_at.desc()).limit(limit).all()
        )

        clients = []
        for client, unbilled_count, unbilled_total_amount, last_ride_at in results:
            payload = client.serialize
            payload["unbilled_count"] = int(unbilled_count or 0)
            # HT ; string "125.00" pour éviter imprécisions float (SUM Numeric/Decimal)
            raw = float(unbilled_total_amount or 0)
            payload["unbilled_total_amount"] = f"{round(raw, 2):.2f}"
            payload["last_ride_at"] = (
                last_ride_at.isoformat() if isinstance(last_ride_at, datetime) else None
            )
            clients.append(payload)

        # Log pour debug
        logger.info(
            "🔍 [EligibleClients] Résultats: %s clients trouvés pour company_id=%s, year=%s, month=%s",
            len(clients),
            company_id,
            period_year,
            period_month,
        )

        return success_response(data={"clients": clients, "total": len(clients)})


@invoices_ns.route("/companies/<int:company_id>/clinic-monthly-totals")
class ClinicMonthlyTotals(Resource):
    """Endpoint pour récupérer les totaux des transports éligibles pour une facture clinique mensuelle."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    @invoices_ns.param("year", "Année (ex: 2025)", type="integer", required=True)
    @invoices_ns.param("month", "Mois (1-12)", type="integer", required=True)
    @invoices_ns.param("clinic_company_id", "ID de la clinique", type="integer", required=True)
    @invoices_ns.param("include_client_ids", "IDs clients à inclure (optionnel, séparés par virgule)", type="string")
    def get(self, company_id: int):  # noqa: PLR0911
        """Récupère les totaux des transports éligibles pour une facture clinique mensuelle.

        Retourne:
        - total_eligible: Nombre de transports éligibles (billed_to_type='clinic')
        - total_amount_eligible: Montant total des transports éligibles
        - total_excluded: Nombre de transports exclus (billed_to_type='patient')
        - total_amount_excluded: Montant total des transports exclus
        """
        try:
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                cid = None
            except Exception:
                logger.debug("Unexpected error converting company.id to int: %s", cid_obj)
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Entreprise non trouvée ou accès refusé",
                    logger_instance=logger,
                )

            # Récupérer les paramètres
            period_year = request.args.get("year", type=int)
            period_month = request.args.get("month", type=int)
            clinic_company_id = request.args.get("clinic_company_id", type=int)
            include_client_ids_param = request.args.get("include_client_ids", type=str)

            if not period_year or not period_month or not clinic_company_id:
                return APIErrorHandler.handle_validation_error(
                    "Les paramètres year, month et clinic_company_id sont requis",
                    logger_instance=logger,
                )

            # Parser include_client_ids si fourni
            include_client_ids = None
            if include_client_ids_param:
                try:
                    include_client_ids = [int(x.strip()) for x in include_client_ids_param.split(",") if x.strip()]
                except (ValueError, TypeError):
                    include_client_ids = None

            # Calculer les dates de période
            start_date = datetime(period_year, period_month, 1)
            if period_month == PERIOD_MONTH_THRESHOLD:
                end_date = datetime(period_year + 1, 1, 1)
            else:
                end_date = datetime(period_year, period_month + 1, 1)

            target_statuses = [
                BookingStatus.COMPLETED.value,
                BookingStatus.RETURN_COMPLETED.value,
            ]
            # Règle métier : eligible (clinique) et excluded (patient) partagent le même périmètre
            # (status COMPLETED/RETURN_COMPLETED, période [start_date, end_date)) pour cohérence S2.

            # ✅ Hardening: safe Number(amount||0) pour éviter NaN
            def safe_amount(amount):
                """Convertit un montant en float de manière sûre, retourne 0.0 si invalide."""
                if amount is None:
                    return 0.0
                try:
                    return float(amount)
                except (ValueError, TypeError):
                    return 0.0

            # ✅ Transports éligibles (billed_to_type='clinic') - filtre strict
            eligible_query = Booking.query.filter(
                Booking.company_id == company_id,
                Booking.billed_to_company_id == clinic_company_id,
                Booking.billed_to_type == "clinic",
                Booking.status.in_(target_statuses),
                Booking.invoice_line_id.is_(None),
                Booking.scheduled_time >= start_date,
                Booking.scheduled_time < end_date,
            )

            # Appliquer le filtre include_client_ids si fourni
            if include_client_ids:
                eligible_query = eligible_query.filter(Booking.client_id.in_(include_client_ids))

            # Calculer les totaux des transports éligibles
            eligible_bookings = eligible_query.all()
            total_eligible = len(eligible_bookings)
            total_amount_eligible = sum(safe_amount(b.amount) for b in eligible_bookings)

            # eligible_client_ids : liste plate de client_id (DISTINCT), pour exclusions
            stmt = (
                select(Booking.client_id)
                .where(
                    Booking.company_id == company_id,
                    Booking.billed_to_company_id == clinic_company_id,
                    Booking.billed_to_type == "clinic",
                    Booking.status.in_(target_statuses),
                    Booking.invoice_line_id.is_(None),
                    Booking.scheduled_time >= start_date,
                    Booking.scheduled_time < end_date,
                )
                .distinct()
            )
            if include_client_ids:
                stmt = stmt.where(Booking.client_id.in_(include_client_ids))
            eligible_client_ids = db.session.scalars(stmt).all()

            # Transports exclus (billed_to_type=patient, même période).
            # Priorité : include_client_ids si fourni, sinon eligible_client_ids, sinon vide.
            # Jamais de booking patient si le client n'a aucun booking clinique eligible.
            # On évite toute requête exclusions (et tout .in_([])) quand les deux sont vides.
            excluded_query = Booking.query.filter(
                Booking.company_id == company_id,
                Booking.billed_to_type == "patient",
                Booking.status.in_(target_statuses),
                Booking.invoice_line_id.is_(None),
                Booking.scheduled_time >= start_date,
                Booking.scheduled_time < end_date,
            )
            excluded_bookings: list[Booking] = []
            total_excluded = 0
            total_amount_excluded = 0.0

            if include_client_ids:
                excluded_query = excluded_query.filter(
                    Booking.client_id.in_(include_client_ids)
                )
                excluded_bookings = excluded_query.all()
                total_excluded = len(excluded_bookings)
                total_amount_excluded = sum(safe_amount(b.amount) for b in excluded_bookings)
            elif eligible_client_ids:
                excluded_query = excluded_query.filter(
                    Booking.client_id.in_(eligible_client_ids)
                )
                excluded_bookings = excluded_query.all()
                total_excluded = len(excluded_bookings)
                total_amount_excluded = sum(safe_amount(b.amount) for b in excluded_bookings)
            else:
                # Aucune requête exclusions : excluded_bookings reste []
                pass

            if os.getenv("BILLING_DEBUG", "0") == "1":
                _excl_fmt = (
                    "[ClinicMonthlyExcluded] booking_id=%s billed_to_type=%s "
                    "billed_to_company_id=%s billing_party_id=%s status=%s invoice_line_id=%s"
                )
                for b in excluded_bookings:
                    logger.info(
                        _excl_fmt,
                        b.id,
                        getattr(b, "billed_to_type", None),
                        getattr(b, "billed_to_company_id", None),
                        getattr(b, "billing_party_id", None),
                        getattr(b, "status", None),
                        getattr(b, "invoice_line_id", None),
                    )

            # Retourner aussi les détails des transports exclus pour affichage
            excluded_details = []
            for booking in excluded_bookings:
                excluded_details.append({
                    "id": booking.id,
                    "date": booking.scheduled_time.isoformat() if booking.scheduled_time else None,
                    "scheduled_time": booking.scheduled_time.isoformat() if booking.scheduled_time else None,  # ✅ Alias pour compatibilité frontend
                    "pickup_location": booking.pickup_location,
                    "pickup_address": booking.pickup_location,  # ✅ Alias pour compatibilité frontend
                    "dropoff_location": booking.dropoff_location,
                    "dropoff_address": booking.dropoff_location,  # ✅ Alias pour compatibilité frontend
                    "customer_name": booking.customer_name,
                    "client_name": booking.customer_name,  # ✅ Alias pour compatibilité frontend
                    "amount": float(booking.amount or 0),
                    "billed_to_type": booking.billed_to_type,
                    "billed_to_company_id": booking.billed_to_company_id,
                    "billing_source": booking.billing_source.value if booking.billing_source else None,
                    "status": booking.status.value if booking.status else None,
                    # ✅ Ajouter invoice_line_id et billing_review_status pour safety rules S2
                    "invoice_line_id": booking.invoice_line_id,
                    # ✅ Normaliser billing_review_status en lowercase pour cohérence frontend
                    "billing_review_status": (
                        booking.billing_review_status.value.lower() if booking.billing_review_status else None
                    ),
                })

            return success_response(data={
                "total_eligible": total_eligible,
                "total_amount_eligible": total_amount_eligible,
                "total_excluded": total_excluded,
                "total_amount_excluded": total_amount_excluded,
                "excluded_bookings": excluded_details,  # ✅ Détails des transports exclus
            })

        except (OperationalError, DBAPIError) as e:
            logger.error(
                "Erreur DB lors de la récupération des totaux clinique mensuelle (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(
                "Erreur de validation lors de la récupération des totaux (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            logger.exception(
                "Erreur inattendue lors de la récupération des totaux clinique mensuelle"
            )
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route("/companies/<int:company_id>/invoices/generate")
class GenerateInvoice(Resource):
    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    @limiter.limit("10 per minute")
    @invoices_ns.expect(invoice_generate_model, validate=False)
    def post(self, company_id):
        """Génère une ou plusieurs factures avec support de la facturation tierce."""
        # Variables pour stocker le résultat
        result = None
        status_code = 200

        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Continuer avec la génération de facture
            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import (
                ValidationError,
            )

            from schemas.invoice_schemas import InvoiceGenerateSchema
            from schemas.validation_utils import validate_request

            try:
                validated_data = validate_request(
                    InvoiceGenerateSchema(), data, strict=False
                )
            except ValidationError as e:
                result, status_code = APIErrorHandler.handle_validation_error(
                    str(e),
                    logger_instance=logger,
                )
                return result, status_code

            client_id = validated_data.get("client_id")
            # NOUVEAU: pour facturation groupée
            client_ids = validated_data.get("client_ids", [])
            # NOUVEAU: support facturation tierce
            bill_to_client_id = validated_data.get("bill_to_client_id")
            # NOUVEAU: support facturation clinique via Company (source Booking.billed_to_company_id)
            clinic_company_id = validated_data.get("clinic_company_id")
            # NOUVEAU: support BillingParty explicite (destinataire unifié)
            billing_party_id = validated_data.get("billing_party_id")
            period_year = validated_data["period_year"]
            period_month = validated_data["period_month"]
            # ✅ S2: Mode de génération
            mode = validated_data.get("mode", "standard")
            # ✅ S2: Exceptions pour facture clinique mensuelle
            include_client_ids = validated_data.get("include_client_ids")
            exclude_client_ids = validated_data.get("exclude_client_ids")

            # period_year et period_month sont déjà validés par le schema,
            # donc toujours présents
            # ✅ DDD: Utilise adapter au lieu de service directement
            client_reservations = validated_data.get("client_reservations")
            overrides = validated_data.get("overrides")

            # ✅ Cas 0: Facture clinique mensuelle unique (S2)
            if mode == "clinic_monthly":
                if not clinic_company_id:
                    result, status_code = APIErrorHandler.handle_validation_error(
                        "clinic_company_id est requis pour mode='clinic_monthly'",
                        logger_instance=logger,
                    )
                    return result, status_code

                logger.info(
                    "Génération facture clinique mensuelle (S2): clinique %s, période %s/%s",
                    clinic_company_id,
                    period_month,
                    period_year,
                )

                from application.invoices.generate_clinic_monthly_invoice import (
                    GenerateClinicMonthlyInvoiceInput,
                    GenerateClinicMonthlyInvoiceUseCase,
                )

                uc = GenerateClinicMonthlyInvoiceUseCase()
                input_data = GenerateClinicMonthlyInvoiceInput(
                    company_id=company_id,
                    clinic_company_id=clinic_company_id,
                    period_year=period_year,
                    period_month=period_month,
                    include_client_ids=include_client_ids,
                    exclude_client_ids=exclude_client_ids,
                    overrides=overrides,
                )
                invoice_result = uc.execute(input_data)

                if not invoice_result.success:
                    # ✅ S2: Si erreur 409 (déjà générée), retourner l'invoice_id existante pour UX
                    HTTP_409_CONFLICT = 409
                    if invoice_result.status_code == HTTP_409_CONFLICT:
                        result = invoice_result.error or {"error": "Facture déjà générée"}
                        status_code = HTTP_409_CONFLICT
                    else:
                        result, status_code = (
                            invoice_result.error,
                            invoice_result.status_code or 400,
                        )
                elif invoice_result.invoice:
                    result = invoice_result.invoice.to_dict()
                    status_code = 201
                elif invoice_result.invoice_id:
                    from repositories.invoice_repository import InvoiceRepository

                    invoice_repo = InvoiceRepository()
                    invoice_model = invoice_repo.find_model_by_id_and_company(
                        invoice_result.invoice_id, company_id
                    )
                    if invoice_model:
                        result = invoice_model.to_dict()
                        status_code = 201
                    else:
                        result = {"error": "Facture générée mais non trouvée"}
                        status_code = 500
                else:
                    result = {"error": "Facture générée mais non retournée"}
                    status_code = 500
                return result, status_code

            # Cas 1: Facturation groupée de plusieurs clients vers une
            # institution
            if client_ids and bill_to_client_id:
                logger.info(
                    ("Génération factures consolidées: %s clients vers institution %s"),
                    len(client_ids),
                    bill_to_client_id,
                )

                # Vérifier que l'institution existe et appartient à
                # l'entreprise
                # Imports locaux pour éviter dépendances circulaires
                from repositories.client_repository import ClientRepository

                client_repo = ClientRepository()
                institution = client_repo.find_model_by_id_and_company(
                    bill_to_client_id, company_id
                )
                if not institution:
                    result, status_code = APIErrorHandler.handle_not_found(
                        "Institution",
                        bill_to_client_id if "bill_to_client_id" in locals() else None,
                        logger,
                    )
                elif not bool(institution.is_institution):
                    result, status_code = APIErrorHandler.handle_validation_error(
                        "Le client sélectionné n'est pas une institution",
                        logger_instance=logger,
                    )
                else:
                    # ✅ DDD: Générer les factures via use case
                    from application.invoices.generate_consolidated_invoice import (
                        GenerateConsolidatedInvoiceInput,
                    )

                    uc = GenerateConsolidatedInvoiceUseCase()
                    input_data = GenerateConsolidatedInvoiceInput(
                        company_id=company_id,
                        client_ids=client_ids,
                        period_year=period_year,
                        period_month=period_month,
                        bill_to_client_id=bill_to_client_id,
                        client_reservations=client_reservations,
                        overrides=overrides,
                    )
                    invoice_result = uc.execute(input_data)

                    if not invoice_result.success:
                        result, status_code = (
                            invoice_result.error,
                            invoice_result.status_code or 400,
                        )
                    else:
                        result = {
                            "message": (
                                f"{invoice_result.success_count} "
                                f"facture(s) générée(s), "
                                f"{invoice_result.error_count} erreur(s)"
                            ),
                            "invoices": [
                                inv.to_dict() for inv in (invoice_result.invoices or [])
                            ],
                            "errors": invoice_result.errors,
                            "success_count": invoice_result.success_count,
                            "error_count": invoice_result.error_count,
                        }
                        status_code = 201

            # Cas 2: Facturation simple (avec ou sans tierce)
            elif client_id:
                # Validation: un seul mode "destinataire" à la fois
                destinations = [
                    bool(billing_party_id),
                    bool(bill_to_client_id),
                    bool(clinic_company_id),
                ]
                if sum(destinations) > 1:
                    result, status_code = APIErrorHandler.handle_validation_error(
                        "Fournir un seul parmi billing_party_id, bill_to_client_id, clinic_company_id",
                        logger_instance=logger,
                    )
                    return result, status_code

                # Vérifier que le client appartient à l'entreprise
                # Imports locaux pour éviter dépendances circulaires
                from repositories.client_repository import ClientRepository

                client_repo = ClientRepository()
                client = client_repo.find_model_by_id_and_company(client_id, company_id)
                if not client:
                    result, status_code = APIErrorHandler.handle_not_found(
                        "Client",
                        client_id if "client_id" in locals() else None,
                        logger,
                    )
                # Si facturation tierce, vérifier l'institution
                elif bill_to_client_id:
                    institution = client_repo.find_model_by_id_and_company(
                        bill_to_client_id, company_id
                    )
                    if not institution:
                        result, status_code = APIErrorHandler.handle_not_found(
                            "Institution payeuse",
                            bill_to_client_id
                            if "bill_to_client_id" in locals()
                            else None,
                            logger,
                        )
                    elif not bool(institution.is_institution):
                        result, status_code = APIErrorHandler.handle_validation_error(
                            "Le client sélectionné n'est pas une institution",
                            logger_instance=logger,
                        )
                    else:
                        # ✅ DDD: Générer la facture via use case
                        from application.invoices import GenerateInvoiceInput

                        uc = GenerateInvoiceUseCase()
                        input_data = GenerateInvoiceInput(
                            company_id=company_id,
                            client_id=client_id,
                            period_year=period_year,
                            period_month=period_month,
                            billing_party_id=None,
                            bill_to_client_id=bill_to_client_id,
                            clinic_company_id=None,
                            reservation_ids=validated_data.get("reservation_ids"),
                            overrides=overrides,
                        )
                        invoice_result = uc.execute(input_data)
                        if not invoice_result.success:
                            result, status_code = (
                                invoice_result.error,
                                invoice_result.status_code or 400,
                            )
                        elif invoice_result.invoice:
                            result = invoice_result.invoice.to_dict()
                            status_code = 201
                        elif invoice_result.invoice_id:
                            # Facture créée mais non retournée (cas normal pour brouillon)
                            # Récupérer la facture depuis la base de données
                            invoice_repo = (
                                ClientRepository()
                            )  # Utiliser le repo déjà importé
                            from repositories.invoice_repository import (
                                InvoiceRepository,
                            )

                            invoice_repo = InvoiceRepository()
                            invoice_model = invoice_repo.find_model_by_id_and_company(
                                invoice_result.invoice_id, company_id
                            )
                            if invoice_model:
                                result = invoice_model.to_dict()
                                status_code = 201
                            else:
                                result = {"error": "Facture générée mais non trouvée"}
                                status_code = 500
                        else:
                            result = {"error": "Facture générée mais non retournée"}
                            status_code = 500
                else:
                    # ✅ DDD: Générer la facture sans facturation tierce via use case
                    from application.invoices import GenerateInvoiceInput

                    uc = GenerateInvoiceUseCase()
                    input_data = GenerateInvoiceInput(
                        company_id=company_id,
                        client_id=client_id,
                        period_year=period_year,
                        period_month=period_month,
                        billing_party_id=billing_party_id,
                        bill_to_client_id=bill_to_client_id,
                        clinic_company_id=clinic_company_id,
                        reservation_ids=validated_data.get("reservation_ids"),
                        overrides=overrides,
                    )
                    invoice_result = uc.execute(input_data)
                    if not invoice_result.success:
                        result, status_code = (
                            invoice_result.error,
                            invoice_result.status_code or 400,
                        )
                    elif invoice_result.invoice:
                        result = invoice_result.invoice.to_dict()
                        status_code = 201
                    elif invoice_result.invoice_id:
                        # Facture créée mais non retournée (cas normal pour brouillon)
                        # Récupérer la facture depuis la base de données
                        from repositories.invoice_repository import InvoiceRepository

                        invoice_repo = InvoiceRepository()
                        invoice_model = invoice_repo.find_model_by_id_and_company(
                            invoice_result.invoice_id, company_id
                        )
                        if invoice_model:
                            result = invoice_model.to_dict()
                            status_code = 201
                        else:
                            result = {"error": "Facture générée mais non trouvée"}
                            status_code = 500
                    else:
                        result = {"error": "Facture générée mais non retournée"}
                        status_code = 500

            else:
                result, status_code = APIErrorHandler.handle_validation_error(
                    "client_id ou client_ids requis",
                    logger_instance=logger,
                )

        except ValueError as e:
            # Erreur de validation métier
            logger.error("Erreur de validation: %s", str(e))
            result, status_code = APIErrorHandler.handle_validation_error(
                str(e),
                logger_instance=logger,
            )
        except (OperationalError, DBAPIError, IntegrityError) as e:
            # Erreurs DB attendues : connexion, contraintes, timeout
            error_trace = traceback.format_exc()
            logger.error(
                "Erreur DB lors de la génération de facture (DB error: %s): %s\n%s",
                type(e).__name__,
                str(e),
                error_trace,
            )
            result, status_code = APIErrorHandler.handle_exception(e, logger)
        except Exception as e:
            # Erreur inattendue : logger avec trace complète
            error_trace = traceback.format_exc()
            logger.exception(
                "Erreur inattendue lors de la génération de facture: %s\n%s",
                str(e),
                error_trace,
            )
            result, status_code = APIErrorHandler.handle_exception(e, logger)

        return result, status_code


@invoices_ns.route("/companies/<int:company_id>/invoices/<int:invoice_id>")
class InvoiceDetail(Resource):
    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, company_id, invoice_id):  # noqa: PLR0911
        """Récupère les détails d'une facture."""
        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
                cid = None
            except Exception:
                # Erreur inattendue : logger et utiliser None
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Entreprise non trouvée ou accès refusé",
                    logger_instance=logger,
                )

            # Validation
            if invoice_id <= 0:
                return APIErrorHandler.handle_validation_error(
                    "invoice_id must be positive",
                    field="invoice_id",
                    logger_instance=logger,
                )

            # ✅ DDD: Utiliser le use case pour récupérer la facture
            from repositories.invoice_repository import InvoiceRepository

            invoice_repo = InvoiceRepository()
            from application.invoices import GetInvoiceInput

            uc = GetInvoiceUseCase(invoice_repo=invoice_repo)
            input_data = GetInvoiceInput(invoice_id=invoice_id, company_id=company_id)
            result = uc.execute(input_data)

            if not result.found:
                return APIErrorHandler.handle_validation_error(
                    result.error.get("message", "Erreur inconnue")
                    if result.error
                    else "Erreur inconnue",
                    logger_instance=logger,
                )

            if result.invoice is None:
                return APIErrorHandler.handle_not_found(
                    "Facture",
                    resource_id=invoice_id,
                    logger_instance=logger,
                )

            return success_response(data=result.invoice.to_dict())

        except (OperationalError, DBAPIError) as e:
            # Erreurs DB attendues : connexion, timeout
            logger.error(
                "Erreur DB lors de la récupération de la facture (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : données invalides, attributs manquants
            logger.error(
                "Erreur de validation lors de la récupération de la facture (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            # Erreur inattendue : logger avec trace complète
            logger.exception("Erreur inattendue lors de la récupération de la facture")
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route("/companies/<int:company_id>/invoices/<int:invoice_id>/send")
class SendInvoice(Resource):
    """Endpoint pour envoyer une facture par email ou la marquer comme envoyée."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    @invoices_ns.expect(send_email_model, validate=False)
    @invoices_ns.response(200, "Facture envoyée avec succès")
    @invoices_ns.response(
        400, "Erreur de validation (email invalide, client sans email)"
    )
    @invoices_ns.response(404, "Facture non trouvée")
    @invoices_ns.response(500, "Erreur lors de l'envoi")
    def post(self, company_id, invoice_id):  # noqa: PLR0911
        """
        Envoie une facture par email ou la marque comme envoyée (papier).

        **Méthodes d'envoi** :
        - `email` (défaut) : Envoie par email au client
        - `paper` : Marque uniquement comme envoyée (envoi papier manuel)

        **Body** :
        ```json
        {
            "recipient_email": "client@example.com",  // Optionnel
            "force_regenerate_pdf": false,
            "send_method": "email"  // ou "paper"
        }
        ```
        """
        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                cid = None
            except Exception:
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Non autorisé",
                    logger_instance=logger,
                )

            # Parser le body
            data = request.get_json() or {}
            send_method = data.get("send_method", "email")
            recipient_email = data.get("recipient_email")
            force_regenerate_pdf = data.get("force_regenerate_pdf", False)

            from models.partner_invoice import (
                PartnerInvoice,
                PartnerInvoiceStatus,
            )

            # Facture partenaire : seul "paper" (marquer envoyée) est supporté
            if (
                send_method == "email"
                and PartnerInvoice.query.filter_by(
                    id=invoice_id, executing_company_id=company_id
                ).first()
            ):
                return APIErrorHandler.handle_validation_error(
                    "Pour les factures partenaires, utilisez 'Marquer comme envoyée' (send_method=paper).",
                    logger_instance=logger,
                )

            if send_method == "email":
                # Envoi par email avec use case
                send_use_case = SendInvoiceByEmailUseCase()
                input_data = SendInvoiceByEmailInput(
                    invoice_id=invoice_id,
                    recipient_email=recipient_email,
                    force_regenerate_pdf=force_regenerate_pdf,
                )

                result = send_use_case.execute(input_data)

                if not result.success:
                    return {
                        "success": False,
                        "error": result.error,
                    }, result.status_code

                return success_response(
                    data={
                        "invoice_id": result.invoice_id,
                        "recipient": result.recipient,
                        "sent_at": result.sent_at.isoformat()
                        if result.sent_at
                        else None,
                        "send_method": "email",
                    },
                    message=f"Facture envoyée par email à {result.recipient}",
                )

            # send_method == "paper" : Marquer comme envoyée
            # Facture partenaire (PartnerInvoice) : seul executing_company peut envoyer, uniquement si DRAFT
            partner_invoice = PartnerInvoice.query.filter_by(
                id=invoice_id, executing_company_id=company_id
            ).first()
            if partner_invoice:
                if partner_invoice.status != PartnerInvoiceStatus.DRAFT:
                    return APIErrorHandler.handle_validation_error(
                        "Déjà envoyée ou statut invalide (utilisez une facture en brouillon)",
                        logger_instance=logger,
                    )
                from services.partnerships.invoices import PartnerInvoiceService

                svc = PartnerInvoiceService()
                try:
                    pi = svc.mark_as_sent(invoice_id, company_id)
                except ValueError as e:
                    return APIErrorHandler.handle_validation_error(
                        str(e), logger_instance=logger
                    )
                return success_response(
                    data={
                        "invoice_id": invoice_id,
                        "sent_at": (
                            pi.sent_at.isoformat() if pi.sent_at else None
                        ),
                        "send_method": "paper",
                    },
                    message="Facture marquée comme envoyée (courrier papier)",
                )

            # Facture classique (Invoice)
            from repositories.invoice_repository import InvoiceRepository

            invoice_repo = InvoiceRepository()
            invoice = invoice_repo.find_model_by_id_and_company(
                invoice_id, company_id
            )
            if not invoice:
                return APIErrorHandler.handle_not_found(
                    "Facture",
                    invoice_id,
                    logger,
                )

            invoice.status = InvoiceStatus.SENT
            invoice.sent_at = datetime.now(UTC)
            db.session.commit()

            sent_at_iso = (
                invoice.sent_at.isoformat() if invoice.sent_at else None
            )
            return success_response(
                data={
                    "invoice_id": invoice_id,
                    "sent_at": sent_at_iso,
                    "send_method": "paper",
                },
                message="Facture marquée comme envoyée (courrier papier)",
            )

        except (OperationalError, DBAPIError, IntegrityError) as e:
            logger.error(
                "Erreur DB lors de l'envoi de la facture (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(
                "Erreur de validation lors de l'envoi de la facture (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            db.session.rollback()
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            logger.exception("Erreur inattendue lors de l'envoi de la facture")
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route("/companies/<int:company_id>/invoices/<int:invoice_id>/payments")
class InvoicePayments(Resource):
    """Endpoint pour enregistrer un paiement."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    @invoices_ns.expect(payment_model)
    @invoices_ns.response(200, "Paiement enregistré avec succès")
    @invoices_ns.response(400, "Erreur de validation", validation_error_model)
    @invoices_ns.response(401, "Non authentifié", permission_error_model)
    @invoices_ns.response(403, "Non autorisé", permission_error_model)
    @invoices_ns.response(404, "Facture non trouvée", not_found_error_model)
    @invoices_ns.response(
        409, "Paiement déjà enregistré (idempotency)", api_error_model
    )
    @invoices_ns.response(500, "Erreur serveur", api_error_model)
    def post(self, company_id, invoice_id):  # noqa: PLR0911
        """Enregistrer un paiement pour une facture.

        ✅ P0: Support idempotency-key pour éviter les doublons de paiement.
        """
        try:
            # ✅ P0: Vérifier idempotency-key (CRITIQUE pour les paiements)
            idempotency_key = IdempotencyService.get_idempotency_key_from_request()
            if idempotency_key:
                cached_response = IdempotencyService.check_key(idempotency_key)
                if cached_response[0]:  # Key exists
                    logger.info(
                        "Idempotency key found for payment, returning cached response",
                        extra={
                            "trace_id": get_trace_id(),
                            "idempotency_key": idempotency_key,
                            "invoice_id": invoice_id,
                        },
                    )
                    return cached_response[1], 200

            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                # ✅ P0: Ajouter trace_id dans l'erreur
                trace_id = get_trace_id()
                if isinstance(error_response, dict):
                    error_response["trace_id"] = trace_id
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond et récupérer la facture
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
                cid = None
            except Exception:
                # Erreur inattendue : logger et utiliser None
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None

            # Utiliser le repository pour récupérer la facture
            from repositories.invoice_repository import InvoiceRepository

            invoice_repo = InvoiceRepository()
            invoice = invoice_repo.find_model_by_id_and_company(invoice_id, company_id)

            # Si ce n'est pas une facture normale, vérifier si c'est une facture partenaire
            if not invoice:
                from models.booking_transfer import BookingTransfer
                from models.enums import TransferStatus
                from models.partner_invoice import (
                    PartnerInvoice,
                    PartnerInvoiceStatus,
                    partner_invoice_transfers,
                )

                # Chercher une facture partenaire
                partner_invoice = PartnerInvoice.query.get(invoice_id)
                if partner_invoice:
                    # Vérifier que l'entreprise est associée à cette facture partenaire
                    transfers_count = (
                        db.session.query(BookingTransfer)
                        .join(
                            partner_invoice_transfers,
                            BookingTransfer.id
                            == partner_invoice_transfers.c.booking_transfer_id,
                        )
                        .filter(
                            partner_invoice_transfers.c.partner_invoice_id
                            == partner_invoice.id,
                            BookingTransfer.executing_company_id == company_id,
                            BookingTransfer.status == TransferStatus.COMPLETED,
                        )
                        .count()
                    )

                    if transfers_count > 0:
                        # Gérer le paiement pour une facture partenaire
                        data = request.get_json()
                        raw_amount = data.get("amount", 0)
                        try:
                            amount = Decimal(str(raw_amount))
                            if amount <= AMOUNT_ZERO:
                                return APIErrorHandler.handle_validation_error(
                                    "Le montant doit être positif",
                                    field="amount",
                                    logger_instance=logger,
                                )
                        except (ValueError, TypeError, Exception):
                            return APIErrorHandler.handle_validation_error(
                                "Montant invalide",
                                field="amount",
                                logger_instance=logger,
                            )

                        # Vérifier que la facture n'est pas déjà payée
                        if partner_invoice.status == PartnerInvoiceStatus.PAID:
                            return APIErrorHandler.handle_validation_error(
                                "La facture partenaire est déjà payée.",
                                logger_instance=logger,
                            ), 400

                        # Calculer le solde dû
                        balance_due = (
                            partner_invoice.total_amount - partner_invoice.amount_paid
                        )

                        # Récupérer le type d'excédent (crédit ou pourboire)
                        is_tip = data.get(
                            "is_tip", False
                        )  # Par défaut, c'est un crédit

                        # Calculer le montant excédentaire si le paiement dépasse le solde dû
                        excess_amount = Decimal("0")
                        if amount > balance_due:
                            excess_amount = amount - balance_due
                            # Si c'est un pourboire, l'ajouter au tip_amount
                            if is_tip:
                                partner_invoice.tip_amount = (
                                    partner_invoice.tip_amount or Decimal("0")
                                ) + excess_amount
                            else:
                                # Sinon, c'est un crédit à déduire de la prochaine facture
                                # Stocker le crédit dans la facture actuelle
                                # Le crédit sera récupéré lors de la génération de la prochaine facture
                                partner_invoice.credit_balance = (
                                    partner_invoice.credit_balance or Decimal("0")
                                ) + excess_amount

                        # Mettre à jour amount_paid (cumulatif)
                        # Pour le paiement, on utilise seulement le montant nécessaire pour payer la facture
                        payment_amount = min(amount, balance_due)
                        current_amount_paid = float(partner_invoice.amount_paid or 0)
                        new_amount_paid = current_amount_paid + float(payment_amount)
                        partner_invoice.amount_paid = Decimal(str(new_amount_paid))

                        # Mettre à jour le statut en fonction du montant payé
                        if new_amount_paid >= float(partner_invoice.total_amount):
                            partner_invoice.status = PartnerInvoiceStatus.PAID
                            partner_invoice.paid_at = datetime.now(UTC)
                        elif new_amount_paid > 0:
                            partner_invoice.status = PartnerInvoiceStatus.PARTIALLY_PAID
                            # Ne pas mettre paid_at si ce n'est pas complètement payé
                            if partner_invoice.paid_at:
                                partner_invoice.paid_at = None

                        db.session.commit()

                        # ✅ P0: Ajouter trace_id dans la réponse (facture partenaire)
                        trace_id = get_trace_id()
                        logger.info(
                            "✅ Paiement partenaire enregistré: invoice_id=%s, amount=%s, company_id=%s",
                            invoice_id,
                            float(amount),
                            company_id,
                            extra={
                                "trace_id": trace_id,
                                "invoice_id": invoice_id,
                                "amount": float(amount),
                                "company_id": company_id,
                            },
                        )

                        response_data = {
                            "message": "Paiement enregistré avec succès",
                            "invoice": partner_invoice.to_dict(),
                            "trace_id": trace_id,
                        }

                        # ✅ P0: Stocker la réponse pour idempotency
                        if idempotency_key:
                            IdempotencyService.store_response(
                                idempotency_key, response_data, 200
                            )

                        return response_data, 200

            # Combiner les vérifications : ID entreprise et existence facture
            if cid != company_id or not invoice:
                if cid != company_id:
                    return APIErrorHandler.handle_permission_error(
                        "Non autorisé",
                        logger_instance=logger,
                    )
                return APIErrorHandler.handle_not_found(
                    "Facture",
                    invoice_id,
                    logger,
                )

            # Valider le montant (combiner validation format et valeur)
            data = request.get_json()
            raw_amount = data.get("amount")

            # Calculer le solde restant dû pour la facture
            current_paid = Decimal(str(invoice.amount_paid or 0))
            total_amount = Decimal(str(invoice.total_amount or 0))
            balance_due = total_amount - current_paid

            # Si aucun montant n'est fourni, utiliser le solde restant (paiement complet)
            if raw_amount is None or raw_amount == "":
                amount = balance_due
                logger.info(
                    "Paiement complet sans montant spécifié, utilisation du solde restant: %s",
                    float(amount),
                )
            else:
                try:
                    amount = Decimal(str(raw_amount))
                    # Vérifier que le montant est positif
                    if amount <= AMOUNT_ZERO:
                        return APIErrorHandler.handle_validation_error(
                            "Le montant doit être positif",
                            field="amount",
                            logger_instance=logger,
                        )
                    # Limiter le montant au solde restant (éviter surpaiement)
                    if amount > balance_due:
                        logger.warning(
                            "Montant saisi (%s) supérieur au solde restant (%s), limitation au solde",
                            float(amount),
                            float(balance_due),
                        )
                        amount = balance_due
                except (ValueError, TypeError, Exception):
                    return APIErrorHandler.handle_validation_error(
                        "Montant invalide",
                        field="amount",
                        logger_instance=logger,
                    )
            method = data.get("method", "bank_transfer")
            # Normaliser le libellé/valeur provenant du frontend (labels FR ou
            # constantes uppercase)
            method_map = {
                "virement bancaire": "bank_transfer",
                "virement": "bank_transfer",
                "bank_transfer": "bank_transfer",
                "bank-transfer": "bank_transfer",
                "bank transfer": "bank_transfer",
                "cash": "cash",
                "espèces": "cash",
                "especes": "cash",
                "carte": "card",
                "card": "card",
                "adjustment": "adjustment",
            }
            raw_method = str(method).strip() if method is not None else "bank_transfer"
            method_norm = method_map.get(raw_method.lower(), raw_method.lower())
            if method_norm not in PaymentMethod.choices():
                method_norm = "bank_transfer"
            payment_method = PaymentMethod(method_norm)
            method_value = payment_method.value
            logger.info(
                "Paiement: method reçu='%s', normalisé='%s', value='%s'",
                method,
                method_norm,
                method_value,
            )
            waive_reminder_fees = bool(data.get("waive_reminder_fees", False))

            # Optionnel: annuler les frais de rappel avant d'appliquer le
            # paiement
            if waive_reminder_fees and hasattr(invoice, "reminder_fee_amount"):
                try:
                    current_reminder_fee = Decimal(
                        str(invoice.reminder_fee_amount or 0)
                    )
                except (ValueError, TypeError) as e:
                    # Erreurs de conversion Decimal attendues : valeur invalide, type incorrect
                    logger.debug(
                        "Failed to convert reminder_fee_amount to Decimal (expected: %s), using 0",
                        type(e).__name__,
                    )
                    current_reminder_fee = Decimal("0")
                except Exception as e:
                    # Erreur inattendue : logger et utiliser 0
                    logger.debug(
                        "Unexpected error converting reminder_fee_amount to Decimal: %s, using 0",
                        e,
                    )
                    current_reminder_fee = Decimal("0")
                if current_reminder_fee > CURRENT_REMINDER_FEE_ZERO:
                    invoice.reminder_fee_amount = Decimal("0")
                    # Recalcule le total si un champ total existe
                    if (
                        hasattr(invoice, "subtotal_amount")
                        and hasattr(invoice, "late_fee_amount")
                        and hasattr(invoice, "total_amount")
                    ):
                        subtotal = Decimal(str(invoice.subtotal_amount or 0))
                        late_fee = Decimal(str(invoice.late_fee_amount or 0))
                        invoice.total_amount = subtotal + late_fee

            # ✅ VENTILATION AUTOMATIQUE : Détecter si c'est un paiement pour un rappel consolidé
            from models import InvoiceReminder

            # Constante pour la tolérance de comparaison de montants
            AMOUNT_TOLERANCE = Decimal("0.01")

            open_reminder = (
                InvoiceReminder.query.filter_by(
                    invoice_id=invoice.id,
                    status="OPEN",
                )
                .order_by(InvoiceReminder.level.desc())  # Prendre le dernier rappel
                .first()
            )

            reminder_id = None
            principal_payment_amount = amount
            reminder_fee_payment_amount = Decimal("0.00")

            if open_reminder and abs(float(amount) - float(open_reminder.total_due)) < float(AMOUNT_TOLERANCE):
                # ✅ Paiement correspond au total du rappel consolidé → ventilation automatique
                logger.info(
                    "Paiement pour rappel consolidé détecté: reminder_id=%s, total_due=%s, amount=%s",
                    open_reminder.id,
                    float(open_reminder.total_due),
                    float(amount),
                )

                reminder_id = open_reminder.id
                principal_payment_amount = open_reminder.principal_amount
                reminder_fee_payment_amount = open_reminder.reminder_fee_amount

                # Marquer le rappel comme payé
                open_reminder.status = "PAID"
                open_reminder.paid_at = datetime.now(UTC)

                logger.info(
                    "Ventilation automatique: principal=%s CHF → facture, frais=%s CHF → rappel",
                    float(principal_payment_amount),
                    float(reminder_fee_payment_amount),
                )

            # Créer le paiement (montant total pour l'export CSV, mais ventilé en interne)
            payment = InvoicePayment()
            payment.invoice_id = invoice.id
            payment.amount = amount  # Montant total payé (pour export CSV)
            payment.method = (
                method_value  # passer la valeur ENUM attendue par la colonne SAEnum
            )
            payment.paid_at = datetime.now(UTC)
            payment.reminder_id = reminder_id  # Lien vers le rappel si applicable
            db.session.add(payment)

            # ✅ Mettre à jour le montant payé de la facture (SEULEMENT le principal)
            # Les frais de rappel sont gérés séparément via le rappel
            invoice.amount_paid = current_paid + principal_payment_amount
            invoice.balance_due = total_amount - invoice.amount_paid

            # Mettre à jour le statut
            if invoice.balance_due <= BALANCE_DUE_ZERO:
                invoice.status = InvoiceStatus.PAID
                invoice.paid_at = datetime.now(UTC)
            elif invoice.amount_paid > AMOUNT_PAID_ZERO:
                invoice.status = InvoiceStatus.PARTIALLY_PAID

            db.session.commit()

            # ✅ P0: Ajouter trace_id dans la réponse
            trace_id = get_trace_id()
            logger.info(
                "✅ Paiement enregistré avec succès: invoice_id=%s, amount=%s, company_id=%s",
                invoice_id,
                float(amount),
                company_id,
                extra={
                    "trace_id": trace_id,
                    "invoice_id": invoice_id,
                    "amount": float(amount),
                    "company_id": company_id,
                },
            )

            response_data = success_response(
                data={
                    "balance_due": float(invoice.balance_due),
                    "amount_paid": float(invoice.amount_paid),
                    "status": invoice.status.value,
                    "trace_id": trace_id,
                },
                message="Paiement enregistré",
            )

            # ✅ P0: Stocker la réponse pour idempotency
            if idempotency_key:
                # success_response retourne toujours un tuple (data, status_code)
                IdempotencyService.store_response(
                    idempotency_key,
                    response_data[0],
                    200,
                )

            return response_data

        except (OperationalError, DBAPIError, IntegrityError) as e:
            # Erreurs DB attendues : connexion, contraintes, timeout
            logger.error(
                "Erreur DB lors de l'enregistrement du paiement (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : données invalides, attributs manquants
            logger.error(
                "Erreur de validation lors de l'enregistrement du paiement (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            db.session.rollback()
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            # Erreur inattendue : logger avec trace complète
            logger.exception("Erreur inattendue lors de l'enregistrement du paiement")
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route("/companies/<int:company_id>/invoices/<int:invoice_id>/reminders")
class InvoiceReminders(Resource):
    """Endpoint pour générer un rappel."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id, invoice_id):  # noqa: PLR0911
        """Générer un rappel pour une facture."""
        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
                cid = None
            except Exception:
                # Erreur inattendue : logger et utiliser None
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Non autorisé",
                    logger_instance=logger,
                )

            # Utiliser le repository pour récupérer la facture
            from repositories.invoice_repository import InvoiceRepository

            invoice_repo = InvoiceRepository()
            invoice = invoice_repo.find_model_by_id_and_company(invoice_id, company_id)
            if not invoice:
                return APIErrorHandler.handle_not_found(
                    "Facture",
                    invoice_id if "invoice_id" in locals() else None,
                    logger,
                )

            data = request.get_json()
            level = int(data.get("level", 1))

            # ✅ DDD: Générer le rappel via use case
            uc_reminder = GenerateInvoiceReminderUseCase()
            reminder_result = uc_reminder.execute(
                GenerateInvoiceReminderInput(invoice_id=invoice_id, level=level)
            )

            if reminder_result.success and reminder_result.reminder:
                # ✅ IMPORTANT: Ne JAMAIS régénérer le PDF de la facture initiale
                # Le rappel a son propre PDF stocké dans reminder.pdf_url
                # La facture initiale (invoice.pdf_url) reste INTACTE

                import os
                REMINDER_DEBUG = os.getenv("REMINDER_DEBUG", "0") == "1"

                if REMINDER_DEBUG:
                    logger.info(
                        (
                            "[REMINDER_DEBUG] Rappel généré: invoice_id=%s, level=%s, "
                            "invoice.pdf_url=%s (INTACT), reminder.pdf_url=%s"
                        ),
                        invoice.id,
                        level,
                        invoice.pdf_url,
                        reminder_result.reminder.pdf_url,
                    )

                return {
                    "message": f"Rappel niveau {level} généré avec succès",
                    "reminder_level": invoice.reminder_level,
                    "reminder_id": reminder_result.reminder.id,
                    "reminder_pdf_url": reminder_result.reminder.pdf_url,
                    "invoice_pdf_url": invoice.pdf_url,  # PDF initial intact
                }, 200
            return reminder_result.error or APIErrorHandler.handle_validation_error(
                "Impossible de générer le rappel",
                logger_instance=logger,
            ), reminder_result.status_code or 400

        except (OperationalError, DBAPIError, IntegrityError) as e:
            # Erreurs DB attendues : connexion, contraintes, timeout
            logger.error(
                "Erreur DB lors de la génération du rappel (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : données invalides, attributs manquants
            logger.error(
                "Erreur de validation lors de la génération du rappel (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            # Erreur inattendue : logger avec trace complète
            logger.exception("Erreur inattendue lors de la génération du rappel")
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route(
    "/companies/<int:company_id>/invoices/<int:invoice_id>/reminders/<int:reminder_id>/send"
)
class SendReminder(Resource):
    """Endpoint pour envoyer un rappel par email."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    @invoices_ns.expect(send_email_model, validate=False)
    @invoices_ns.response(200, "Rappel envoyé avec succès")
    @invoices_ns.response(
        400, "Erreur de validation (email invalide, client sans email)"
    )
    @invoices_ns.response(404, "Rappel non trouvé")
    @invoices_ns.response(500, "Erreur lors de l'envoi")
    def post(self, company_id, invoice_id, reminder_id):  # noqa: PLR0911, ARG002
        """
        Envoie un rappel de paiement par email.

        **Body** :
        ```json
        {
            "recipient_email": "client@example.com",  // Optionnel
            "force_regenerate_pdf": false
        }
        ```
        """
        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                cid = None
            except Exception:
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Non autorisé",
                    logger_instance=logger,
                )

            # Parser le body
            data = request.get_json() or {}
            recipient_email = data.get("recipient_email")
            force_regenerate_pdf = data.get("force_regenerate_pdf", False)

            # Envoi par email avec use case
            send_use_case = SendReminderByEmailUseCase()
            input_data = SendReminderByEmailInput(
                reminder_id=reminder_id,
                recipient_email=recipient_email,
                force_regenerate_pdf=force_regenerate_pdf,
            )

            result = send_use_case.execute(input_data)

            if not result.success:
                return {
                    "success": False,
                    "error": result.error,
                }, result.status_code

            return success_response(
                data={
                    "reminder_id": result.reminder_id,
                    "invoice_id": result.invoice_id,
                    "recipient": result.recipient,
                    "sent_at": result.sent_at.isoformat() if result.sent_at else None,
                },
                message=f"Rappel envoyé par email à {result.recipient}",
            )

        except (OperationalError, DBAPIError, IntegrityError) as e:
            logger.error(
                "Erreur DB lors de l'envoi du rappel (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(
                "Erreur de validation lors de l'envoi du rappel (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            db.session.rollback()
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            logger.exception("Erreur inattendue lors de l'envoi du rappel")
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route(
    "/companies/<int:company_id>/invoices/<int:invoice_id>/regenerate-pdf"
)
class RegenerateInvoicePdf(Resource):
    """Endpoint pour régénérer le PDF d'une facture."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id, invoice_id):  # noqa: PLR0911
        """Régénérer le PDF d'une facture."""
        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
                cid = None
            except Exception:
                # Erreur inattendue : logger et utiliser None
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Non autorisé",
                    logger_instance=logger,
                )

            # Utiliser le repository pour récupérer la facture
            from models.booking_transfer import BookingTransfer
            from models.enums import TransferStatus
            from models.partner_invoice import PartnerInvoice, partner_invoice_transfers
            from repositories.invoice_repository import InvoiceRepository

            invoice_repo = InvoiceRepository()
            invoice = invoice_repo.find_model_by_id_and_company(invoice_id, company_id)

            # Si ce n'est pas une facture normale, vérifier si c'est une facture partenaire
            if not invoice:
                # Chercher une facture partenaire
                partner_invoice = PartnerInvoice.query.get(invoice_id)
                if partner_invoice:
                    # Vérifier que l'entreprise est associée à cette facture partenaire
                    # (via les transferts où elle est executing_company_id)
                    transfers_count = (
                        db.session.query(BookingTransfer)
                        .join(
                            partner_invoice_transfers,
                            BookingTransfer.id
                            == partner_invoice_transfers.c.booking_transfer_id,
                        )
                        .filter(
                            partner_invoice_transfers.c.partner_invoice_id
                            == partner_invoice.id,
                            BookingTransfer.executing_company_id == company_id,
                            BookingTransfer.status == TransferStatus.COMPLETED,
                        )
                        .count()
                    )

                    if transfers_count > 0:
                        # Régénérer le PDF pour les factures partenaires
                        from services.partnerships.invoices import (
                            PartnerInvoiceService,
                        )

                        try:
                            partner_service = PartnerInvoiceService()
                            pdf_url = partner_service.regenerate_pdf(partner_invoice.id)
                            return {"message": "PDF régénéré", "pdf_url": pdf_url}
                        except ValueError as e:
                            return APIErrorHandler.handle_validation_error(
                                str(e), logger_instance=logger
                            ), 400
                        except Exception as e:
                            logger.exception(
                                "Erreur lors de la régénération PDF pour facture partenaire %s",
                                partner_invoice.id,
                            )
                            return APIErrorHandler.handle_exception(e, logger)

                return APIErrorHandler.handle_not_found(
                    "Facture",
                    invoice_id if "invoice_id" in locals() else None,
                    logger,
                )

            # ✅ PROTECTION IMMUTABILITÉ: Vérifier si la facture est "verrouillée"
            from models.enums import InvoiceStatus

            locked_statuses = {InvoiceStatus.SENT, InvoiceStatus.PARTIALLY_PAID, InvoiceStatus.PAID}
            if invoice.status in locked_statuses:
                logger.warning(
                    (
                        "[PDF PROTECTION] Tentative de régénération PDF pour facture verrouillée: "
                        "invoice_id=%s, status=%s, pdf_url=%s. Opération refusée."
                    ),
                    invoice.id,
                    invoice.status.value,
                    invoice.pdf_url,
                )
                return APIErrorHandler.handle_validation_error(
                    (
                        f"Impossible de régénérer le PDF: la facture est {invoice.status.value} "
                        "et ne peut plus être modifiée."
                    ),
                    logger_instance=logger,
                ), 400

            # ✅ DDD: Régénérer le PDF via use case
            uc = GenerateInvoicePdfUseCase()
            pdf_result = uc.execute(invoice=invoice)

            if pdf_result.ok and pdf_result.pdf_url:
                # ✅ PROTECTION: Double vérification avant d'écraser invoice.pdf_url
                if invoice.pdf_url and invoice.pdf_url != pdf_result.pdf_url:
                    logger.warning(
                        (
                            "[PDF PROTECTION] Tentative d'écrasement PDF existant: "
                            "invoice_id=%s, ancien=%s, nouveau=%s. Opération refusée."
                        ),
                        invoice.id,
                        invoice.pdf_url,
                        pdf_result.pdf_url,
                    )
                    return APIErrorHandler.handle_validation_error(
                        "Un PDF existe déjà pour cette facture. Utilisez un flag explicite pour forcer la régénération.",
                        logger_instance=logger,
                    ), 400

                invoice.pdf_url = pdf_result.pdf_url
                db.session.commit()
                return {"message": "PDF régénéré", "pdf_url": pdf_result.pdf_url}
            return pdf_result.error or APIErrorHandler.handle_validation_error(
                "Impossible de régénérer le PDF",
                logger_instance=logger,
            ), pdf_result.status_code or 400

        except (OperationalError, DBAPIError, IntegrityError) as e:
            # Erreurs DB attendues : connexion, contraintes, timeout
            logger.error(
                "Erreur DB lors de la régénération du PDF (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : données invalides, attributs manquants
            logger.error(
                "Erreur de validation lors de la régénération du PDF (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            # Erreur inattendue : logger avec trace complète
            logger.exception("Erreur inattendue lors de la régénération du PDF")
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route("/companies/<int:company_id>/invoices/<int:invoice_id>/cancel")
class CancelInvoice(Resource):
    """Endpoint pour annuler une facture."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id, invoice_id):  # noqa: PLR0911
        """Annuler une facture."""
        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
                cid = None
            except Exception:
                # Erreur inattendue : logger et utiliser None
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Non autorisé",
                    logger_instance=logger,
                )

            # Utiliser le repository pour récupérer la facture (avec lines pour libérer les réservations)
            from repositories.invoice_repository import InvoiceRepository

            invoice_repo = InvoiceRepository()
            invoice = invoice_repo.find_model_by_id_with_eager_loading(
                invoice_id, company_id
            )
            if not invoice:
                return APIErrorHandler.handle_not_found(
                    "Facture",
                    invoice_id if "invoice_id" in locals() else None,
                    logger,
                )

            # ✅ DDD: Annuler la facture via use case
            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            from application.invoices import CancelInvoiceInput

            uc = CancelInvoiceUseCase(booking_repo=booking_repo)
            input_data = CancelInvoiceInput(invoice=invoice)
            cancel_result = uc.execute(input_data)

            if not cancel_result.success:
                return cancel_result.error, cancel_result.status_code or 400

            return {
                "message": "Facture annulée",
                "status": invoice.status.value,
            }

        except ValueError as e:
            # Erreur de validation métier
            logger.warning(
                "Erreur métier lors de l'annulation de la facture %s: %s", invoice_id, e
            )
            db.session.rollback()
            return APIErrorHandler.handle_validation_error(
                str(e),
                logger_instance=logger,
            )
        except (OperationalError, DBAPIError, IntegrityError) as e:
            # Erreurs DB attendues : connexion, contraintes, timeout
            logger.error(
                "Erreur DB lors de l'annulation de la facture (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)
        except Exception as e:
            # Erreur inattendue : logger avec trace complète
            logger.exception("Erreur inattendue lors de l'annulation de la facture")
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route("/companies/<int:company_id>/invoices/<int:invoice_id>/duplicate")
class DuplicateInvoice(Resource):
    """Endpoint pour dupliquer une facture existante en brouillon correctif."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id, invoice_id):  # noqa: PLR0911
        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
                cid = None
            except Exception:
                # Erreur inattendue : logger et utiliser None
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Non autorisé",
                    logger_instance=logger,
                )

            # Utiliser le repository pour récupérer la facture normale
            from repositories.invoice_repository import InvoiceRepository

            invoice_repo = InvoiceRepository()
            invoice = invoice_repo.find_model_by_id_and_company(invoice_id, company_id)

            # Si ce n'est pas une facture normale, vérifier si c'est une facture partenaire
            if not invoice:
                from models.booking_transfer import BookingTransfer
                from models.enums import TransferStatus
                from models.partner_invoice import (
                    PartnerInvoice,
                    PartnerInvoiceStatus,
                    partner_invoice_transfers,
                )

                # Chercher une facture partenaire
                partner_invoice = PartnerInvoice.query.get(invoice_id)
                if partner_invoice:
                    # Vérifier que l'entreprise est associée à cette facture partenaire
                    # Soit via les transferts où elle est executing_company_id,
                    # soit si elle est l'entreprise qui a généré la facture (executing_company_id)
                    transfers_count = (
                        db.session.query(BookingTransfer)
                        .join(
                            partner_invoice_transfers,
                            BookingTransfer.id
                            == partner_invoice_transfers.c.booking_transfer_id,
                        )
                        .filter(
                            partner_invoice_transfers.c.partner_invoice_id
                            == partner_invoice.id,
                            BookingTransfer.executing_company_id == company_id,
                            BookingTransfer.status == TransferStatus.COMPLETED,
                        )
                        .count()
                    )

                    # Vérifier aussi si l'entreprise est celle qui a généré la facture
                    is_executing_company = (
                        partner_invoice.executing_company_id == company_id
                    )

                    if transfers_count > 0 or is_executing_company:
                        # Vérifier que la facture peut être annulée
                        if partner_invoice.status == PartnerInvoiceStatus.CANCELLED:
                            return APIErrorHandler.handle_validation_error(
                                "La facture partenaire est déjà annulée.",
                                logger_instance=logger,
                            ), 400

                        # Vérifier que la facture n'est pas réellement payée
                        # (vérifier amount_paid > 0, pas seulement le statut)
                        if (
                            partner_invoice.status == PartnerInvoiceStatus.PAID
                            and partner_invoice.amount_paid > 0
                        ):
                            return APIErrorHandler.handle_validation_error(
                                "Impossible de dupliquer une facture partenaire déjà payée.",
                                logger_instance=logger,
                            ), 400

                        # Sauvegarder les valeurs nécessaires avant la suppression
                        partnership_id = partner_invoice.partnership_id
                        period_year = partner_invoice.period_year
                        period_month = partner_invoice.period_month
                        executing_company_id = partner_invoice.executing_company_id

                        # Supprimer la facture partenaire (au lieu de la marquer comme CANCELLED)
                        # car la contrainte unique ne permet pas deux factures pour la même période
                        # Les relations dans partner_invoice_transfers seront supprimées automatiquement
                        # grâce à ondelete="CASCADE"
                        db.session.delete(partner_invoice)
                        db.session.commit()

                        # Construire le contexte pour régénérer la facture partenaire
                        # Utiliser l'executing_company_id original de la facture
                        draft_context = {
                            "billing_type": "partner",
                            "partnership_id": partnership_id,
                            "period_year": period_year,
                            "period_month": period_month,
                            "executing_company_id": executing_company_id,
                        }

                        return {
                            "message": (
                                "La facture partenaire a été annulée. "
                                "Vous pouvez régénérer une nouvelle facture "
                                "avec les mêmes paramètres."
                            ),
                            "draft": draft_context,
                        }, 200

                return APIErrorHandler.handle_not_found(
                    "Facture",
                    invoice_id if "invoice_id" in locals() else None,
                    logger,
                )

            # ✅ DDD: Dupliquer la facture normale via use case
            uc = DuplicateInvoiceUseCase()
            duplicate_result = uc.execute(DuplicateInvoiceInput(invoice=invoice))

            if not duplicate_result.success:
                return duplicate_result.error, duplicate_result.status_code or 400

            return {
                "message": (
                    "Les transports ont été libérés. "
                    "Veuillez corriger le brouillon puis générer "
                    "une nouvelle facture."
                ),
                "draft": duplicate_result.draft_context,
            }, 200

        except ValueError as e:
            # Erreur de validation métier
            logger.warning(
                "Erreur métier lors de la duplication de la facture %s: %s",
                invoice_id,
                e,
            )
            db.session.rollback()
            return APIErrorHandler.handle_validation_error(
                str(e),
                logger_instance=logger,
            )
        except (OperationalError, DBAPIError, IntegrityError) as e:
            # Erreurs DB attendues : connexion, contraintes, timeout
            logger.error(
                "Erreur DB lors de la duplication de la facture (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)
        except Exception as e:
            # Erreur inattendue : logger avec trace complète
            logger.exception("Erreur inattendue lors de la duplication de la facture")
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route("/companies/<int:company_id>/clients/institutions")
class InstitutionsList(Resource):
    """Endpoint pour récupérer la liste des institutions (cliniques)."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, company_id):
        """Liste les institutions (cliniques) pour la facturation tierce."""
        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
                cid = None
            except Exception:
                # Erreur inattendue : logger et utiliser None
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Non autorisé",
                    logger_instance=logger,
                )

            # Récupérer toutes les institutions actives de l'entreprise
            # Imports locaux pour éviter dépendances circulaires
            from repositories.client_repository import ClientRepository

            client_repo = ClientRepository()
            institutions = client_repo.find_models_by_company_and_institution_status(
                company_id, is_institution=True, is_active=True
            )

            return {
                "institutions": [
                    {
                        "id": inst.id,
                        "institution_name": inst.institution_name
                        or "Institution sans nom",
                        "clinic_company_id": inst.default_billed_to_company_id,
                        "contact_email": inst.contact_email,
                        "contact_phone": inst.contact_phone,
                        "billing_address": inst.billing_address,
                        "user": {
                            "first_name": inst.user.first_name if inst.user else "",
                            "last_name": inst.user.last_name if inst.user else "",
                            "username": inst.user.username if inst.user else "",
                        }
                        if inst.user
                        else None,
                    }
                    for inst in institutions
                ]
            }

        except (OperationalError, DBAPIError) as e:
            # Erreurs DB attendues : connexion, timeout
            logger.error(
                "Erreur DB lors de la récupération des institutions (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : données invalides, attributs manquants
            logger.error(
                "Erreur de validation lors de la récupération des institutions (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            # Erreur inattendue : logger avec trace complète
            logger.exception(
                "Erreur inattendue lors de la récupération des institutions"
            )
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route(
    "/companies/<int:company_id>/clients/<int:client_id>/toggle-institution"
)
class ToggleInstitution(Resource):
    """Endpoint pour marquer/démarquer un client comme institution."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id, client_id):  # noqa: PLR0911
        """Bascule le statut d'institution d'un client."""
        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
                cid = None
            except Exception:
                # Erreur inattendue : logger et utiliser None
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Non autorisé",
                    logger_instance=logger,
                )

            # Utiliser le repository pour récupérer le client
            from repositories.client_repository import ClientRepository

            client_repo = ClientRepository()
            client = client_repo.find_model_by_id_and_company(client_id, company_id)
            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client",
                    client_id if "client_id" in locals() else None,
                    logger,
                )

            data = request.get_json() or {}
            is_institution = data.get("is_institution", not bool(client.is_institution))
            institution_name = data.get("institution_name")

            client.is_institution = is_institution
            if is_institution and institution_name:
                client.institution_name = institution_name
            elif not is_institution:
                client.institution_name = None

            db.session.commit()

            return {
                "message": (
                    f"Client "
                    f"{'marqué comme' if is_institution else 'démarqué en tant que'} "
                    "institution"
                ),
                "client": client.serialize,
            }

        except (OperationalError, DBAPIError, IntegrityError) as e:
            # Erreurs DB attendues : connexion, contraintes, timeout
            logger.error(
                "Erreur DB lors de la modification du statut d'institution (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : données invalides, attributs manquants
            logger.error(
                "Erreur de validation lors de la modification du statut d'institution (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            db.session.rollback()
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            # Erreur inattendue : logger avec trace complète
            logger.exception(
                "Erreur inattendue lors de la modification du statut d'institution"
            )
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route(
    "/companies/<int:company_id>/clients/<int:client_id>/unbilled-reservations"
)
class UnbilledReservations(Resource):
    """Endpoint pour récupérer les réservations non encore facturées d'un client."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, company_id, client_id):
        """Récupère les réservations non facturées avec filtres optionnels
        Query params:
        - year: Année (ex: 2025)
        - month: Mois (ex: 10)
        - billed_to_type: Type de facturation ('patient', 'clinic', 'insurance').
        """
        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
                cid = None
            except Exception:
                # Erreur inattendue : logger et utiliser None
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Non autorisé",
                    logger_instance=logger,
                )

            # Récupérer les paramètres
            period_year = request.args.get("year", type=int)
            period_month = request.args.get("month", type=int)
            billed_to_filter = request.args.get("billed_to_type", type=str)

            # ✅ Utilisation du repository pour la requête avec filtres dynamiques
            from repositories.booking_repository import BookingRepository

            # 🔍 LOG : Debug pour voir ce qui est trouvé
            logger.warning(
                (
                    "🔍 [Unbilled] Recherche pour client_id=%s, "
                    "company_id=%s, year=%s, month=%s, "
                    "billed_to_filter=%s"
                ),
                client_id,
                company_id,
                period_year,
                period_month,
                billed_to_filter,
            )

            booking_repo = BookingRepository()
            reservations = booking_repo.find_models_unbilled_by_company_and_client(
                company_id=company_id,
                client_id=client_id,
                period_year=period_year,
                period_month=period_month,
            )

            # 🔍 LOG : Compter AVANT filtre billed_to_type
            count_before_filter = len(reservations)
            logger.warning(
                "🔍 [Unbilled] Avant filtre billed_to_type: %s bookings trouvés",
                count_before_filter,
            )

            # ⚠️ NE PAS filtrer par billed_to_type :
            # on veut TOUS les transports non facturés du client
            # Même si le type de facturation ne correspond pas, on affiche tout
            # Le dispatcher pourra choisir ce qu'il veut facturer
            # if billed_to_filter and billed_to_filter in [
            #     'patient', 'clinic', 'insurance'
            # ]:
            #     reservations = [r for r in reservations if r.billed_to_type == billed_to_filter]
            #     logger.warning(
            #         "🔍 [Unbilled] Filtre appliqué: billed_to_type=%s",
            #         billed_to_filter
            #     )

            # 🔍 LOG : Afficher les résultats trouvés
            logger.warning(
                "🔍 [Unbilled] FINAL: Trouvé %s réservations non facturées",
                len(reservations),
            )
            for r in reservations:
                logger.warning(
                    (
                        "   - Booking #%s: %s, %s, status=%s, "
                        "billed_to_type=%s billed_to_company_id=%s billing_party_id=%s "
                        "invoice_line_id=%s"
                    ),
                    r.id,
                    r.customer_name,
                    r.scheduled_time,
                    r.status,
                    r.billed_to_type,
                    getattr(r, "billed_to_company_id", None),
                    getattr(r, "billing_party_id", None),
                    r.invoice_line_id,
                )

            return {
                "reservations": [
                    {
                        "id": r.id,
                        "date": r.scheduled_time.isoformat()
                        if r.scheduled_time
                        else None,
                        "scheduled_time": r.scheduled_time.isoformat()
                        if r.scheduled_time
                        else None,  # ✅ Alias pour compatibilité frontend
                        "pickup_location": r.pickup_location,
                        "pickup_address": r.pickup_location,  # ✅ Alias pour compatibilité frontend
                        "dropoff_location": r.dropoff_location,
                        "dropoff_address": r.dropoff_location,  # ✅ Alias pour compatibilité frontend
                        "amount": float(r.amount or 0),
                        "billed_to_type": r.billed_to_type,
                        "billed_to_company_id": r.billed_to_company_id,
                        "billed_to_contact": r.billed_to_contact,
                        "customer_name": r.customer_name,
                        "status": r.status.value,
                        "is_urgent": r.is_urgent or False,
                        "is_return": r.is_return or False,
                        "medical_facility": r.medical_facility,
                        # ✅ Ajouter billing_source et billing_source_ref pour traçabilité
                        # Note: billing_source est un enum str, .value retourne la valeur snake_case
                        "billing_source": (
                            r.billing_source.value if r.billing_source else None
                        ),
                        "billing_source_ref": r.billing_source_ref,
                        # ✅ Ajouter invoice_line_id et billing_review_status pour le modal S2
                        "invoice_line_id": r.invoice_line_id,
                        # ✅ Normaliser billing_review_status en lowercase pour cohérence frontend
                        "billing_review_status": (
                            r.billing_review_status.value.lower() if r.billing_review_status else None
                        ),
                    }
                    for r in reservations
                ],
                "total_amount": sum(float(r.amount or 0) for r in reservations),
                "count": len(reservations),
                "summary_by_type": {
                    "patient": sum(
                        1 for r in reservations if bool(r.billed_to_type == "patient")
                    ),
                    "clinic": sum(
                        1 for r in reservations if bool(r.billed_to_type == "clinic")
                    ),
                    "insurance": sum(
                        1 for r in reservations if bool(r.billed_to_type == "insurance")
                    ),
                },
            }

        except (OperationalError, DBAPIError) as e:
            # Erreurs DB attendues : connexion, timeout
            logger.error(
                "Erreur DB lors de la récupération des réservations non facturées (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : données invalides, attributs manquants
            logger.error(
                "Erreur de validation lors de la récupération des réservations non facturées (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            # Erreur inattendue : logger avec trace complète
            logger.exception(
                "Erreur inattendue lors de la récupération des réservations non facturées"
            )
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route(
    "/companies/<int:company_id>/clients/<int:client_id>/unbilled-reservations/ids"
)
class UnbilledReservationIds(Resource):
    """Endpoint pour récupérer uniquement les IDs des réservations non encore facturées d'un client."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, company_id, client_id):
        """Récupère uniquement les IDs des réservations non facturées avec filtres optionnels
        Query params:
        - year: Année (ex: 2025)
        - month: Mois (ex: 10)
        - billed_to_type: Type de facturation ("patient", "clinic", "insurance").

        Retour: { reservation_ids: [1, 2, 3, ...] }
        """
        try:
            # ✅ DDD: Utilise use-case au lieu de service directement
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                cid = None
            except Exception:
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Non autorisé",
                    logger_instance=logger,
                )

            # Récupérer les paramètres
            period_year = request.args.get("year", type=int)
            period_month = request.args.get("month", type=int)
            billed_to_filter = request.args.get("billed_to_type", type=str)

            # ✅ Utilisation du repository pour la requête optimisée (IDs uniquement)
            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            reservation_ids = booking_repo.find_unbilled_ids_by_company_and_client(
                company_id=company_id,
                client_id=client_id,
                period_year=period_year,
                period_month=period_month,
                billed_to_type=billed_to_filter,
            )

            return {
                "reservation_ids": reservation_ids,
            }

        except (OperationalError, DBAPIError) as e:
            logger.error(
                "Erreur DB lors de la récupération des IDs de réservations non facturées (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(
                "Erreur de validation lors de la récupération des IDs de réservations non facturées (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            logger.exception(
                "Erreur inattendue lors de la récupération des IDs de réservations non facturées"
            )
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route("/companies/<int:company_id>/reservations/<int:reservation_id>")
class SingleReservation(Resource):
    """Endpoint pour récupérer une réservation spécifique par ID."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, company_id, reservation_id):  # noqa: PLR0911
        """Récupère les détails d'une réservation spécifique.

        Utilisé pour hydrater les objets minimaux {id} en objets complets
        sans dépendre de la période/status de l'endpoint unbilled-reservations.
        """
        try:
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            # Vérifier que l'ID de l'entreprise correspond
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except (ValueError, TypeError, OverflowError):
                cid = None
            except Exception:
                logger.debug(
                    "Unexpected error converting company.id to int: %s", cid_obj
                )
                cid = None
            if cid != company_id:
                return APIErrorHandler.handle_permission_error(
                    "Non autorisé",
                    logger_instance=logger,
                )

            # Validation
            if reservation_id <= 0:
                return APIErrorHandler.handle_validation_error(
                    "reservation_id must be positive",
                    field="reservation_id",
                    logger_instance=logger,
                )

            # Récupérer la réservation
            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            booking = booking_repo.find_model_by_id_with_visibility(
                reservation_id, company_id
            )

            if not booking:
                return APIErrorHandler.handle_not_found(
                    "Réservation",
                    resource_id=reservation_id,
                    logger_instance=logger,
                )

            # Formater la réponse dans le même format que unbilled-reservations
            return {
                "reservation": {
                    "id": booking.id,
                    "date": booking.scheduled_time.isoformat()
                    if booking.scheduled_time
                    else None,
                    "scheduled_time": booking.scheduled_time.isoformat()
                    if booking.scheduled_time
                    else None,
                    "pickup_location": booking.pickup_location,
                    "pickup_address": booking.pickup_location,
                    "dropoff_location": booking.dropoff_location,
                    "dropoff_address": booking.dropoff_location,
                    "amount": float(booking.amount or 0),
                    "billed_to_type": booking.billed_to_type,
                    "billed_to_company_id": booking.billed_to_company_id,
                    "billed_to_contact": booking.billed_to_contact,
                    "customer_name": booking.customer_name,
                    "status": booking.status.value,
                    "is_urgent": booking.is_urgent or False,
                    "is_return": booking.is_return or False,
                    "medical_facility": booking.medical_facility,
                    "billing_source": (
                        booking.billing_source.value if booking.billing_source else None
                    ),
                    "billing_source_ref": booking.billing_source_ref,
                    "invoice_line_id": booking.invoice_line_id,
                }
            }

        except (OperationalError, DBAPIError) as e:
            logger.error(
                "Erreur DB lors de la récupération de la réservation (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(
                "Erreur de validation lors de la récupération de la réservation (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            logger.exception(
                "Erreur inattendue lors de la récupération de la réservation"
            )
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route("/companies/<int:company_id>/partners/billable")
class BillablePartners(Resource):
    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, company_id: int):  # noqa: ARG002
        """Récupère les partenaires actifs avec leurs transferts facturables."""
        # company_id est requis par Flask-RESTX pour le routage mais n'est pas utilisé
        # car l'entreprise est récupérée via _get_current_company_via_use_case()
        try:
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            logger.info(
                (
                    "🔍 [BillablePartners] After _get_current_company - "
                    "has_company=%s, "
                    "company_id=%s, "
                    "has_error=%s, "
                    "status_code=%s"
                ),
                company is not None,
                company.id if company else None,
                error_response is not None,
                status_code,
            )
            if error_response or not company:
                return error_response, status_code

            from models.booking_transfer import BookingTransfer
            from models.enums import PartnershipStatus, TransferStatus
            from models.partnership import Partnership

            # Récupérer les partenariats actifs où l'entreprise est l'exécutante
            # (elle peut facturer l'entreprise propriétaire)
            # Seule l'entreprise partenaire (exécutante) peut facturer l'entreprise propriétaire
            # Vérifier aussi si l'entreprise est owner (pour debug)
            all_partnerships = (
                db.session.query(Partnership)
                .filter(
                    (
                        (Partnership.owner_company_id == company.id)
                        | (Partnership.partner_company_id == company.id)
                    ),
                    Partnership.status == PartnershipStatus.ACCEPTED,
                )
                .all()
            )

            # Log détaillé pour debug
            partnerships_as_owner = [
                p for p in all_partnerships if p.owner_company_id == company.id
            ]
            partnerships_as_partner = [
                p for p in all_partnerships if p.partner_company_id == company.id
            ]

            logger.info(
                (
                    "🔍 [BillablePartners] Partnerships analysis - "
                    "all_count=%s, "
                    "as_owner_count=%s, "
                    "as_partner_count=%s, "
                    "company_id=%s"
                ),
                len(all_partnerships),
                len(partnerships_as_owner),
                len(partnerships_as_partner),
                company.id if company else None,
            )

            # Log détaillé de chaque partenariat
            for p in all_partnerships:
                logger.info(
                    (
                        "🔍 [BillablePartners] Partnership ID %s - "
                        "owner_company_id=%s, "
                        "partner_company_id=%s, "
                        "status=%s"
                    ),
                    p.id,
                    p.owner_company_id,
                    p.partner_company_id,
                    p.status.value if hasattr(p.status, "value") else p.status,
                )

            # Filtrer ceux où l'entreprise est partenaire (exécutante)
            partnerships = partnerships_as_partner

            # Si l'entreprise est owner mais pas partner, vérifier si elle est executing_company_id
            # dans les transferts (cas où le partenariat a été créé dans le mauvais sens)
            if not partnerships and partnerships_as_owner:
                logger.info(
                    (
                        "🔍 [BillablePartners] Company %s est owner dans %s partenariats. "
                        "Vérification si elle est executing_company_id dans les transferts..."
                    ),
                    company.id if company else None,
                    len(partnerships_as_owner),
                )
                # Vérifier si l'entreprise est executing_company_id dans les transferts
                # de ces partenariats (indique que le partenariat a été créé dans le mauvais sens)
                for p in partnerships_as_owner:
                    transfers_as_executing = (
                        db.session.query(BookingTransfer)
                        .filter(
                            BookingTransfer.partnership_id == p.id,
                            BookingTransfer.executing_company_id == company.id,
                            BookingTransfer.status == TransferStatus.COMPLETED,
                        )
                        .count()
                    )
                    logger.info(
                        (
                            "🔍 [BillablePartners] Partnership %s - "
                            "transfers avec executing_company_id=%s: %s"
                        ),
                        p.id,
                        company.id if company else None,
                        transfers_as_executing,
                    )
                    if transfers_as_executing > 0:
                        # L'entreprise est owner dans le partenariat mais executing dans les transferts
                        # Cela indique que le partenariat a été créé dans le mauvais sens
                        # On peut quand même permettre la facturation en utilisant ce partenariat
                        logger.warning(
                            (
                                "⚠️ [BillablePartners] Partnership %s créé dans le mauvais sens: "
                                "company %s est owner mais executing dans les transferts. "
                                "Permettant la facturation quand même."
                            ),
                            p.id,
                            company.id if company else None,
                        )
                        partnerships.append(p)

            # Si aucun partenariat trouvé, retourner une liste vide avec un log
            if not partnerships:
                if partnerships_as_owner:
                    logger.warning(
                        (
                            "⚠️ [BillablePartners] Company %s est owner dans %s partenariats "
                            "mais n'est pas partner et n'est pas executing dans les transferts. "
                            "Elle ne peut pas facturer, seulement être facturée."
                        ),
                        company.id if company else None,
                        len(partnerships_as_owner),
                    )
                else:
                    logger.warning(
                        (
                            "⚠️ [BillablePartners] Aucun partenariat trouvé où "
                            "company_id=%s est partenaire (partner_company_id) ou owner. "
                            "Vérifiez que les partenariats existent et sont ACCEPTED."
                        ),
                        company.id if company else None,
                    )
                return success_response(data=[])

            result = []
            for partnership in partnerships:
                # Déterminer quelle entreprise est facturée
                # Si l'entreprise actuelle est partner dans le partenariat, elle facture l'owner
                # Si l'entreprise actuelle est owner mais executing dans les transferts,
                # elle facture le partner (partenariat créé dans le mauvais sens)
                if partnership.partner_company_id == company.id:
                    # Cas normal : l'entreprise est partner, elle facture l'owner
                    partner_company = partnership.owner_company
                else:
                    # Cas où l'entreprise est owner mais executing dans les transferts
                    # Elle facture le partner
                    partner_company = partnership.partner_company

                # Récupérer les transferts validés et non facturés
                from models.partner_invoice import (
                    PartnerInvoice,
                    partner_invoice_transfers,
                )

                # Log pour debug - compter les transferts déjà facturés
                billed_count = (
                    db.session.query(partner_invoice_transfers.c.booking_transfer_id)
                    .join(PartnerInvoice)
                    .filter(PartnerInvoice.partnership_id == partnership.id)
                    .count()
                )
                logger.info(
                    "🔍 [BillablePartners] Partnership %s - billed_transfers_count=%s",
                    partnership.id,
                    billed_count,
                )

                # Vérifier tous les transferts pour ce partenariat (pour debug)
                all_transfers = (
                    db.session.query(BookingTransfer)
                    .filter(BookingTransfer.partnership_id == partnership.id)
                    .all()
                )

                completed_transfers = (
                    db.session.query(BookingTransfer)
                    .filter(
                        BookingTransfer.partnership_id == partnership.id,
                        BookingTransfer.status == TransferStatus.COMPLETED,
                    )
                    .all()
                )

                accepted_transfers = (
                    db.session.query(BookingTransfer)
                    .filter(
                        BookingTransfer.partnership_id == partnership.id,
                        BookingTransfer.status == TransferStatus.ACCEPTED,
                    )
                    .all()
                )

                validated_transfers = (
                    db.session.query(BookingTransfer)
                    .filter(
                        BookingTransfer.partnership_id == partnership.id,
                        BookingTransfer.status == TransferStatus.COMPLETED,
                        BookingTransfer.is_validated == True,  # noqa: E712
                    )
                    .all()
                )

                # Log détaillé de tous les transferts pour debug
                logger.info(
                    (
                        "🔍 [BillablePartners] Partnership %s - "
                        "all_transfers=%s, "
                        "accepted=%s, "
                        "completed=%s, "
                        "validated=%s"
                    ),
                    partnership.id,
                    len(all_transfers),
                    len(accepted_transfers),
                    len(completed_transfers),
                    len(validated_transfers),
                )
                for t in all_transfers:
                    logger.info(
                        (
                            "🔍 [BillablePartners] Transfer %s - "
                            "status=%s, "
                            "is_validated=%s, "
                            "executing_company_id=%s, "
                            "owner_company_id=%s"
                        ),
                        t.id,
                        t.status.value if hasattr(t.status, "value") else t.status,
                        t.is_validated,
                        t.executing_company_id,
                        t.owner_company_id,
                    )

                # Chercher les transferts facturables :
                # - Statut COMPLETED (les transferts ACCEPTED ne sont pas encore facturables)
                # - executing_company_id == company.id (l'entreprise actuelle est l'exécutante)
                # - Non facturés
                # Note: On inclut les transferts COMPLETED même s'ils ne sont pas encore validés,
                # pour que l'entreprise puisse voir ce qui sera facturable une fois validé.
                # Le service de génération de facture vérifiera que seuls les transferts validés
                # sont inclus dans la facture.

                # D'abord, chercher tous les transferts pour ce partenariat avec executing_company_id
                all_transfers_for_billing = (
                    db.session.query(BookingTransfer)
                    .filter(
                        BookingTransfer.partnership_id == partnership.id,
                        BookingTransfer.executing_company_id
                        == company.id,  # L'entreprise actuelle est l'exécutante
                        BookingTransfer.status
                        == TransferStatus.COMPLETED,  # ✅ Seulement COMPLETED
                    )
                    .all()
                )

                # Récupérer les IDs des transferts déjà facturés (non annulés)
                from models.partner_invoice import PartnerInvoiceStatus

                billed_ids_list = [
                    row[0]
                    for row in db.session.query(
                        partner_invoice_transfers.c.booking_transfer_id
                    )
                    .join(PartnerInvoice)
                    .filter(
                        PartnerInvoice.partnership_id == partnership.id,
                        PartnerInvoice.status != PartnerInvoiceStatus.CANCELLED,
                    )
                    .all()
                ]

                # Filtrer pour exclure ceux déjà facturés
                unbilled_transfers = [
                    t for t in all_transfers_for_billing if t.id not in billed_ids_list
                ]

                logger.info(
                    (
                        "🔍 [BillablePartners] Partnership %s - "
                        "all_transfers_for_billing=%s, "
                        "billed_ids=%s, "
                        "unbilled=%s"
                    ),
                    partnership.id,
                    len(all_transfers_for_billing),
                    len(billed_ids_list),
                    len(unbilled_transfers),
                )

                logger.info(
                    (
                        "🔍 [BillablePartners] Partnership %s (%s) - "
                        "all_transfers=%s, "
                        "accepted=%s, "
                        "completed=%s, "
                        "validated=%s, "
                        "unbilled=%s"
                    ),
                    partnership.id,
                    partner_company.name,
                    len(all_transfers),
                    len(accepted_transfers),
                    len(completed_transfers),
                    len(validated_transfers),
                    len(unbilled_transfers),
                )

                # Log détaillé si aucun transfert facturable
                if not unbilled_transfers and all_transfers_for_billing:
                    logger.warning(
                        (
                            "⚠️ [BillablePartners] Partnership %s - "
                            "Transferts trouvés mais tous déjà facturés: "
                            "all_for_billing=%s, billed=%s"
                        ),
                        partnership.id,
                        len(all_transfers_for_billing),
                        len(billed_ids_list),
                    )
                elif not all_transfers_for_billing:
                    logger.warning(
                        (
                            "⚠️ [BillablePartners] Partnership %s - "
                            "Aucun transfert COMPLETED et validé trouvé "
                            "avec executing_company_id=%s. "
                            "Total transfers: %s, Completed: %s, Validated: %s"
                        ),
                        partnership.id,
                        company.id,
                        len(all_transfers),
                        len(completed_transfers),
                        len(validated_transfers),
                    )

                # Filtrer les transferts validés pour le calcul du montant facturable
                validated_unbilled_transfers = [
                    t for t in unbilled_transfers if t.is_validated
                ]

                if unbilled_transfers:
                    # Montant total des transferts validés et non facturés
                    total_amount = sum(
                        float(t.partner_cost or 0) for t in validated_unbilled_transfers
                    )
                    result.append(
                        {
                            "partnership_id": partnership.id,
                            "partner_company_id": partner_company.id,
                            "partner_company_name": partner_company.name,
                            "unbilled_transfers_count": len(
                                unbilled_transfers
                            ),  # Total (validés + non validés)
                            "validated_unbilled_transfers_count": len(
                                validated_unbilled_transfers
                            ),  # Seulement validés (facturables)
                            "total_amount": total_amount,  # Montant des transferts validés uniquement
                            "currency": unbilled_transfers[0].currency
                            if unbilled_transfers
                            else "CHF",
                        }
                    )

            logger.info(
                "🔍 [BillablePartners] Returning result - count=%s, company_id=%s",
                len(result),
                company.id if company else None,
            )
            return success_response(data=result)
        except Exception as e:
            logger.exception(
                "Erreur lors de la récupération des partenaires facturables"
            )
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route("/companies/<int:company_id>/partners/invoices/generate")
class GeneratePartnerInvoice(Resource):
    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    @invoices_ns.param(
        "partnership_id", "ID du partenariat", type="integer", required=True
    )
    @invoices_ns.param(
        "period_year", "Année de la période", type="integer", required=True
    )
    @invoices_ns.param(
        "period_month", "Mois de la période (1-12)", type="integer", required=True
    )
    def post(self, company_id: int):  # noqa: ARG002
        """Génère une facture partenaire pour un partenariat."""
        try:
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            data = request.get_json() or {}
            partnership_id = data.get("partnership_id")
            period_year = data.get("period_year")
            period_month = data.get("period_month")

            # Validation des paramètres requis
            if not partnership_id or not period_year or not period_month:
                return APIErrorHandler.handle_validation_error(
                    "partnership_id, period_year et period_month sont requis",
                    logger_instance=logger,
                )

            # Vérifier que l'entreprise est bien partenaire du partenariat
            from models.booking_transfer import BookingTransfer
            from models.enums import TransferStatus
            from models.partnership import Partnership
            from services.partnerships.invoices import PartnerInvoiceService

            partnership = Partnership.query.get(int(partnership_id))
            if not partnership:
                return APIErrorHandler.handle_validation_error(
                    f"Partenariat {partnership_id} introuvable",
                    logger_instance=logger,
                )

            # Seule l'entreprise partenaire (exécutante) peut générer une facture
            # OU l'entreprise owner mais executing dans les transferts (partenariat créé dans le mauvais sens)
            is_partner = company.id == partnership.partner_company_id
            is_owner_executing = company.id == partnership.owner_company_id

            # Vérifier si l'entreprise est owner mais executing dans les transferts
            if is_owner_executing:
                transfers_as_executing = BookingTransfer.query.filter(
                    BookingTransfer.partnership_id == partnership.id,
                    BookingTransfer.executing_company_id == company.id,
                    BookingTransfer.status == TransferStatus.COMPLETED,
                ).count()
                if transfers_as_executing == 0:
                    is_owner_executing = False

            if not is_partner and not is_owner_executing:
                return APIErrorHandler.handle_validation_error(
                    "Seule l'entreprise partenaire peut générer une facture pour ce partenariat",
                    logger_instance=logger,
                )

            service = PartnerInvoiceService()

            # #region agent log
            import json
            from pathlib import Path

            try:
                with Path(r"c:\Users\jasiq\atmr\.cursor\debug.log").open(
                    "a", encoding="utf-8"
                ) as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "D",
                                "location": "routes/invoices.py:GeneratePartnerInvoice.post",
                                "message": "Appel generate_monthly_invoice",
                                "data": {
                                    "company_id": company.id,
                                    "partnership_id": int(partnership_id),
                                    "year": int(period_year),
                                    "month": int(period_month),
                                    "executing_company_id": company.id,
                                },
                                "timestamp": int(__import__("time").time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion

            service = PartnerInvoiceService()
            partner_invoice = service.generate_monthly_invoice(
                partnership_id=int(partnership_id),
                year=int(period_year),
                month=int(period_month),
                executing_company_id=company.id,  # ✅ Passer l'ID de l'entreprise exécutante
            )

            return success_response(
                data={
                    "id": partner_invoice.id,
                    "invoice_number": partner_invoice.invoice_number,
                    "pdf_url": partner_invoice.pdf_url,
                    "total_amount": float(partner_invoice.total_amount),
                    "currency": partner_invoice.currency,
                },
                message="Facture partenaire générée avec succès",
            )
        except (ValueError, Exception) as e:
            error_response = (
                APIErrorHandler.handle_validation_error(str(e), logger_instance=logger)
                if isinstance(e, ValueError)
                else None
            )
            if not error_response:
                logger.exception(
                    "Erreur lors de la génération de la facture partenaire"
                )
                error_response = APIErrorHandler.handle_exception(e, logger)
            return error_response


@invoices_ns.route("/companies/<int:company_id>/partners/debug")
class BillablePartnersDebug(Resource):
    """Endpoint de debug temporaire pour vérifier les partenariats et transferts."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, company_id: int):  # noqa: ARG002
        """Debug: Vérifie les partenariats et transferts pour l'entreprise connectée."""
        try:
            from routes.companies import _get_current_company_via_use_case

            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response, status_code

            from models.booking_transfer import BookingTransfer
            from models.enums import PartnershipStatus, TransferStatus
            from models.partner_invoice import PartnerInvoice, partner_invoice_transfers
            from models.partnership import Partnership

            # Récupérer tous les partenariats
            all_partnerships = Partnership.query.filter(
                Partnership.status == PartnershipStatus.ACCEPTED
            ).all()

            # Partenariats où l'entreprise est partenaire
            partnerships_as_partner = [
                p for p in all_partnerships if p.partner_company_id == company.id
            ]

            # Partenariats où l'entreprise est owner
            partnerships_as_owner = [
                p for p in all_partnerships if p.owner_company_id == company.id
            ]

            debug_data = {
                "company_id": company.id,
                "company_name": company.name,
                "all_partnerships_count": len(all_partnerships),
                "partnerships_as_partner_count": len(partnerships_as_partner),
                "partnerships_as_owner_count": len(partnerships_as_owner),
                "partnerships_as_partner": [
                    {
                        "id": p.id,
                        "owner_company_id": p.owner_company_id,
                        "owner_company_name": p.owner_company.name
                        if p.owner_company
                        else None,
                        "partner_company_id": p.partner_company_id,
                        "partner_company_name": p.partner_company.name
                        if p.partner_company
                        else None,
                        "status": p.status.value
                        if hasattr(p.status, "value")
                        else str(p.status),
                    }
                    for p in partnerships_as_partner
                ],
            }

            # Pour chaque partenariat où l'entreprise est partenaire, vérifier les transferts
            for p in partnerships_as_partner:
                all_transfers = BookingTransfer.query.filter(
                    BookingTransfer.partnership_id == p.id
                ).all()

                transfers_executing = [
                    t for t in all_transfers if t.executing_company_id == company.id
                ]

                transfers_completed = [
                    t
                    for t in transfers_executing
                    if t.status == TransferStatus.COMPLETED
                ]

                transfers_validated = [t for t in transfers_completed if t.is_validated]

                # Transferts déjà facturés
                billed_ids = [
                    row[0]
                    for row in db.session.query(
                        partner_invoice_transfers.c.booking_transfer_id
                    )
                    .join(PartnerInvoice)
                    .filter(PartnerInvoice.partnership_id == p.id)
                    .all()
                ]

                transfers_billable = [
                    t for t in transfers_validated if t.id not in billed_ids
                ]

                partnership_debug = {
                    "partnership_id": p.id,
                    "total_transfers": len(all_transfers),
                    "transfers_executing_company": len(transfers_executing),
                    "transfers_completed": len(transfers_completed),
                    "transfers_validated": len(transfers_validated),
                    "transfers_billed": len(billed_ids),
                    "transfers_billable": len(transfers_billable),
                    "transfers_details": [
                        {
                            "id": t.id,
                            "status": t.status.value
                            if hasattr(t.status, "value")
                            else str(t.status),
                            "is_validated": t.is_validated,
                            "validated_at": t.validated_at.isoformat()
                            if t.validated_at
                            else None,
                            "executing_company_id": t.executing_company_id,
                            "owner_company_id": t.owner_company_id,
                            "partner_cost": float(t.partner_cost)
                            if t.partner_cost
                            else 0,
                            "is_billed": t.id in billed_ids,
                        }
                        for t in transfers_executing[
                            :10
                        ]  # Limiter à 10 pour la lisibilité
                    ],
                }

                # Trouver l'index du partenariat dans la liste
                for idx, pd in enumerate(debug_data["partnerships_as_partner"]):
                    if pd["id"] == p.id:
                        debug_data["partnerships_as_partner"][idx].update(
                            partnership_debug
                        )
                        break

            return success_response(data=debug_data)
        except Exception as e:
            logger.exception("Erreur lors du debug des partenaires facturables")
            return APIErrorHandler.handle_exception(e, logger)


@invoices_ns.route("/companies/<int:company_id>/exports/payments.csv")
class ExportPaymentsCSV(Resource):
    """Export CSV des paiements encaissés pour la comptabilité suisse."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    @invoices_ns.param("year", "Année (YYYY)", required=True, type=int)
    @invoices_ns.param("month", "Mois (1-12)", required=True, type=int)
    @invoices_ns.param(
        "decimal",
        "Séparateur décimal (dot|comma). dot=150.50, comma=150,50",
        required=False,
        type=str,
        default="dot",
    )
    @invoices_ns.param(
        "with_meta",
        "Inclure métadonnées en commentaire (0|1)",
        required=False,
        type=int,
        default=0,
    )
    @invoices_ns.response(200, "CSV généré avec succès")
    @invoices_ns.response(400, "Paramètres invalides", validation_error_model)
    @invoices_ns.response(401, "Non authentifié", permission_error_model)
    @invoices_ns.response(403, "Non autorisé", permission_error_model)
    def get(self, company_id):
        """Exporte un CSV mensuel des paiements encaissés.

        Format CSV (séparateur `;`, UTF-8):
        Date paiement; Numéro facture; Client; Montant encaissé (CHF); Moyen de paiement; Référence; Devise; Type; ID paiement

        Règles:
        - Basé sur la DATE DE PAIEMENT (paid_at), pas la date de facture
        - Inclut uniquement les paiements validés (paid_at != NULL)
        - 1 ligne = 1 paiement (paiements partiels possibles)
        - Factures annulées exclues
        - Même CSV pour clients (S1) et cliniques (S2)

        Paramètres optionnels:
        - decimal=dot|comma : Format décimal (dot=150.50, comma=150,50). Défaut: dot
        """
        from datetime import datetime
        from io import StringIO

        from flask import make_response

        from models.enums import (
            InvoiceBillingStrategy,
            InvoiceStatus,
            PaymentMethod,
        )

        try:
            # Récupérer et valider les paramètres
            year = request.args.get("year", type=int)
            month = request.args.get("month", type=int)
            decimal_separator = request.args.get("decimal", "dot", type=str)

            # Validation des paramètres
            MONTH_MIN = 1
            MONTH_MAX = 12
            validation_error = None
            if not year or not month:
                validation_error = APIErrorHandler.handle_validation_error(
                    "Les paramètres 'year' et 'month' sont requis",
                    logger_instance=logger,
                )
            elif month < MONTH_MIN or month > MONTH_MAX:
                validation_error = APIErrorHandler.handle_validation_error(
                    f"Le paramètre 'month' doit être entre {MONTH_MIN} et {MONTH_MAX}",
                    logger_instance=logger,
                )
            elif decimal_separator not in ("dot", "comma"):
                validation_error = APIErrorHandler.handle_validation_error(
                    "Le paramètre 'decimal' doit être 'dot' ou 'comma'",
                    logger_instance=logger,
                )

            if validation_error:
                return validation_error

            # À ce stade, year et month sont garantis non-None
            assert year is not None
            assert month is not None

            # Vérifier que l'entreprise existe et que l'utilisateur y a accès
            company = Company.query.get(company_id)
            if not company:
                return APIErrorHandler.handle_not_found("Company", company_id, logger)

            # Calculer les dates de début et fin du mois
            DECEMBER = 12
            start_date = datetime(year, month, 1, tzinfo=UTC)
            if month == DECEMBER:
                end_date = datetime(year + 1, 1, 1, tzinfo=UTC)
            else:
                end_date = datetime(year, month + 1, 1, tzinfo=UTC)

            # Récupérer les paiements encaissés dans le mois
            # Filtrer par:
            # - company_id (factures de l'entreprise)
            # - paid_at dans le mois
            # - invoice.status != CANCELLED
            # Précharger les relations nécessaires avec joinedload
            payments = (
                db.session.query(InvoicePayment)
                .join(Invoice, InvoicePayment.invoice_id == Invoice.id)
                .options(
                    joinedload(InvoicePayment.invoice).joinedload(Invoice.client).joinedload(
                        Client.user
                    ),
                    joinedload(InvoicePayment.invoice).joinedload(Invoice.billed_to_company),
                )
                .filter(Invoice.company_id == company_id)
                .filter(Invoice.status != InvoiceStatus.CANCELLED)
                .filter(InvoicePayment.paid_at >= start_date)
                .filter(InvoicePayment.paid_at < end_date)
                .order_by(InvoicePayment.paid_at)
                .all()
            )

            # Récupérer paramètre with_meta
            with_meta = request.args.get("with_meta", 0, type=int)

            # Générer le CSV
            output = StringIO()
            output.write("\ufeff")  # BOM UTF-8 pour Excel

            # Métadonnées en commentaire (si activées)
            if with_meta:
                export_date = datetime.now(UTC).strftime("%d.%m.%Y %H:%M:%S")
                period_str = f"{month:02d}.{year}"
                output.write(f"# Période: {period_str}\n")
                output.write(f"# Entreprise ID: {company_id}\n")
                output.write(f"# Date d'export: {export_date}\n")
                output.write("#\n")  # Ligne vide pour séparer

            # En-têtes
            headers = [
                "Date paiement",
                "Numéro facture",
                "Client",
                "Montant encaissé (CHF)",
                "Moyen de paiement",
                "Référence",
                "Devise",
                "Type",
                "ID paiement",
            ]
            output.write(";".join(headers) + "\n")

            # Lignes de données
            for payment in payments:
                invoice = payment.invoice
                if not invoice:
                    continue

                # Date paiement (format DD.MM.YYYY pour comptabilité suisse)
                _paid_at = getattr(payment, "paid_at", None)
                paid_date = (
                    _paid_at.strftime("%d.%m.%Y") if _paid_at is not None else ""
                )

                # Numéro facture
                invoice_number = invoice.invoice_number or ""

                # Client : S1 = nom client, S2 = raison sociale clinique
                client_name = ""
                if invoice.billing_strategy == InvoiceBillingStrategy.S2_CLINIC_MONTHLY:
                    # S2 : utiliser le nom de la clinique (billed_to_company)
                    if invoice.billed_to_company:
                        client_name = invoice.billed_to_company.name or ""
                elif invoice.client and invoice.client.user:
                    # S1 : utiliser le nom du client
                    first_name = invoice.client.user.first_name or ""
                    last_name = invoice.client.user.last_name or ""
                    client_name = f"{first_name} {last_name}".strip()
                    if not client_name:
                        client_name = invoice.client.user.username or ""

                # Montant encaissé (2 décimales) avec séparateur configurable
                # ✅ FIX: Si payment.amount est 0 mais invoice est PAID, utiliser le montant payé calculé
                amount_float = float(payment.amount)
                if amount_float == 0.0 and invoice.status == InvoiceStatus.PAID:
                    # Garde-fou : calculer le montant depuis invoice.amount_paid
                    # (pour les anciens paiements créés avec le bug)
                    calculated_paid = float(invoice.amount_paid or 0)
                    if calculated_paid > 0:
                        logger.warning(
                            "Paiement ID %s a amount=0 mais facture PAID, utilisation de amount_paid=%s",
                            payment.id,
                            calculated_paid,
                        )
                        amount_float = calculated_paid
                if decimal_separator == "comma":
                    amount = f"{amount_float:.2f}".replace(".", ",")
                else:
                    amount = f"{amount_float:.2f}"

                # Moyen de paiement (traduire enum en français)
                method_map = {
                    PaymentMethod.BANK_TRANSFER: "Virement bancaire",
                    PaymentMethod.CASH: "Espèces",
                    PaymentMethod.CARD: "Carte",
                    PaymentMethod.ADJUSTMENT: "Ajustement",
                }
                payment_method = method_map.get(payment.method, payment.method.value)

                # Référence (qr_reference de la facture ou référence du paiement)
                reference = payment.reference or invoice.qr_reference or ""

                # Devise (toujours CHF)
                currency = "CHF"

                # Type : Client (S1) ou Clinique (S2)
                invoice_type = (
                    "Clinique"
                    if invoice.billing_strategy == InvoiceBillingStrategy.S2_CLINIC_MONTHLY
                    else "Client"
                )

                # ID paiement
                payment_id = str(payment.id)

                # Écrire la ligne (échapper les `;` dans les valeurs)
                row = [
                    paid_date,
                    invoice_number,
                    client_name.replace(";", ","),  # Remplacer ; par , pour éviter les problèmes
                    amount,
                    payment_method,
                    reference.replace(";", ","),
                    currency,
                    invoice_type,
                    payment_id,
                ]
                output.write(";".join(row) + "\n")

            # Créer la réponse CSV
            response = make_response(output.getvalue())
            response.headers["Content-Disposition"] = (
                f"attachment; filename=paiements_{year}_{month:02d}.csv"
            )
            response.headers["Content-Type"] = "text/csv; charset=utf-8"

            logger.info(
                "Export CSV paiements généré: company_id=%s, year=%s, month=%s, count=%s",
                company_id,
                year,
                month,
                len(payments),
            )

            return response

        except ValueError as e:
            return APIErrorHandler.handle_validation_error(str(e), logger_instance=logger)
        except Exception as e:
            logger.exception("Erreur lors de l'export CSV des paiements")
            return APIErrorHandler.handle_exception(e, logger)
