import logging
from datetime import datetime
from decimal import Decimal

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource, fields, reqparse
from sqlalchemy.orm import joinedload

from ext import limiter, role_required
from models import (
    Booking,
    BookingStatus,
    Client,
    Company,
    CompanyBillingSettings,
    Invoice,
    InvoicePayment,
    InvoiceStatus,
    PaymentMethod,
    User,
    db,
)
from services.invoice_service import InvoiceService
from services.pdf_service import PDFService

# Constantes pour éviter les valeurs magiques
SOLDE_ZERO = 0
BALANCE_DUE_ZERO = 0
REMINDER_LEVEL_ZERO = 0
AMOUNT_ZERO = 0
CURRENT_REMINDER_FEE_ZERO = 0
AMOUNT_PAID_ZERO = 0
PERIOD_MONTH_THRESHOLD = 12

# Configuration du logger
app_logger = logging.getLogger("invoices")

# Namespace pour les factures
invoices_ns = Namespace("invoices", description="Gestion des factures")

# Modèles de sérialisation
invoice_model = invoices_ns.model("Invoice", {
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
    "client": fields.Nested(invoices_ns.model("Client", {
        "id": fields.Integer(required=True),
        "first_name": fields.String(required=False),
        "last_name": fields.String(required=False),
        "username": fields.String(required=False),
    })),
})

invoice_line_model = invoices_ns.model("InvoiceLine", {
    "id": fields.Integer(required=True),
    "type": fields.String(required=True),
    "description": fields.String(required=True),
    "qty": fields.Float(required=True),
    "unit_price": fields.Float(required=True),
    "line_total": fields.Float(required=True),
    "reservation_id": fields.Integer(required=False),
})

payment_model = invoices_ns.model("Payment", {
    "amount": fields.Float(required=True),
    "paid_at": fields.DateTime(required=True),
    "method": fields.String(required=True),
    "reference": fields.String(required=False),
})

reminder_model = invoices_ns.model("Reminder", {
    "level": fields.Integer(required=True),
})

billing_settings_model = invoices_ns.model("BillingSettings", {
    "payment_terms_days": fields.Integer(required=False, allow_null=True, minimum=0, maximum=365, description="Jours de paiement (0-365)"),
    "overdue_fee": fields.Float(required=False, allow_null=True, minimum=0, description="Frais de retard (>= 0)"),
    "reminder1fee": fields.Float(required=False, allow_null=True, minimum=0, description="Frais rappel 1 (>= 0)"),
    "reminder2fee": fields.Float(required=False, allow_null=True, minimum=0, description="Frais rappel 2 (>= 0)"),
    "reminder3fee": fields.Float(required=False, allow_null=True, minimum=0, description="Frais rappel 3 (>= 0)"),
    "reminder_schedule_days": fields.Raw(required=False, description="Planification des rappels (liste de jours)"),
    "auto_reminders_enabled": fields.Boolean(required=False, description="Activer rappels automatiques"),
    "email_sender": fields.String(required=False, allow_null=True, max_length=254, description="Email expéditeur"),
    "invoice_number_format": fields.String(required=False, max_length=50, description="Format numéro facture"),
    "invoice_prefix": fields.String(required=False, max_length=20, description="Préfixe numéro facture"),
    "iban": fields.String(required=False, allow_null=True, pattern="^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$", description="IBAN (format: CH9300762011623852957)"),
    "qr_iban": fields.String(required=False, allow_null=True, pattern="^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$", description="QR IBAN"),
    "esr_ref_base": fields.String(required=False, allow_null=True, max_length=26, description="Référence ESR"),
    "invoice_message_template": fields.String(required=False, allow_null=True, max_length=1000, description="Template message facture"),
    "reminder1template": fields.String(required=False, allow_null=True, max_length=1000, description="Template rappel 1"),
    "reminder2template": fields.String(required=False, allow_null=True, max_length=1000, description="Template rappel 2"),
    "reminder3template": fields.String(required=False, allow_null=True, max_length=1000, description="Template rappel 3"),
    "legal_footer": fields.String(required=False, allow_null=True, max_length=2000, description="Pied de page légal"),
    "pdf_template_variant": fields.String(required=False, max_length=50, description="Variant template PDF"),
})

# Modèle Swagger pour génération de facture
invoice_generate_model = invoices_ns.model("InvoiceGenerate", {
    "client_id": fields.Integer(description="ID client unique (optionnel si client_ids utilisé)", minimum=1),
    "client_ids": fields.List(fields.Integer(description="ID client", minimum=1), description="Liste d'IDs clients (au moins 1 élément)"),
    "bill_to_client_id": fields.Integer(description="ID client payeur (facturation tierce)", minimum=1, allow_null=True),
    "period_year": fields.Integer(required=True, minimum=2000, maximum=2100, description="Année (2000-2100)"),
    "period_month": fields.Integer(required=True, minimum=1, maximum=12, description="Mois (1-12)"),
    "client_reservations": fields.Raw(description="Sélection manuelle: {client_id: [reservation_ids]}"),
    "reservation_ids": fields.List(fields.Integer(description="ID réservation"), description="Liste d'IDs réservations (optionnel)"),
})

# Parser pour les filtres
filter_parser = reqparse.RequestParser()
filter_parser.add_argument("status", type=str, help="Statut de la facture")
filter_parser.add_argument("client_id", type=int, help="ID du client")
filter_parser.add_argument("year", type=int, help="Année")
filter_parser.add_argument("month", type=int, help="Mois")
filter_parser.add_argument("q", type=str, help="Recherche textuelle")
filter_parser.add_argument("page", type=int, default=1, help="Page")
filter_parser.add_argument(
    "per_page",
    type=int,
    default=20,
    help="Éléments par page")
filter_parser.add_argument(
    "with_balance",
    type=bool,
    help="Avec solde > SOLDE_ZERO")
filter_parser.add_argument(
    "with_reminders",
    type=bool,
    help="Avec rappels en cours")


@invoices_ns.route("/companies/<int:company_id>/invoices")
class InvoicesList(Resource):
    def get(self, company_id):
        """Récupère la liste des factures avec filtres et pagination."""
        app_logger.info("🚀 InvoicesList.get() company_id=%s", company_id)

        args = request.args
        status_raw = (args.get("status") or "").strip().lower()
        client_id = args.get("client_id", type=int)
        year = args.get("year", type=int)
        month = args.get("month", type=int)
        q = (args.get("q") or "").strip()
        with_balance = args.get("with_balance") in ("1", "true", "True", "on")
        with_reminders = args.get("with_reminders") in (
            "1", "true", "True", "on")
        page = args.get("page", default=1, type=int)
        per_page = args.get("per_page", default=20, type=int)

        # ✅ Eager load client + lines pour éviter N+1
        query = Invoice.query.options(
            joinedload(Invoice.client).joinedload(Client.user),
            joinedload(Invoice.bill_to_client).joinedload(Client.user),
            joinedload(Invoice.lines),
            joinedload(Invoice.payments)
        ).filter(Invoice.company_id == company_id)

        # Status mapping frontend -> enum value
        status_map = {
            "draft": InvoiceStatus.DRAFT,
            "sent": InvoiceStatus.SENT,
            "partially_paid": InvoiceStatus.PARTIALLY_PAID,
            "paid": InvoiceStatus.PAID,
            "overdue": InvoiceStatus.OVERDUE,
            "cancelled": InvoiceStatus.CANCELLED,
        }
        if status_raw in status_map:
            query = query.filter_by(status=status_map[status_raw])

        if client_id:
            query = query.filter(Invoice.client_id == client_id)

        if year:
            query = query.filter(Invoice.period_year == year)

        if month:
            query = query.filter(Invoice.period_month == month)

        if with_balance:
            # balance_due > BALANCE_DUE_ZERO
            query = query.filter(Invoice.balance_due > BALANCE_DUE_ZERO)

        if with_reminders:
            # reminder_level > REMINDER_LEVEL_ZERO
            query = query.filter(Invoice.reminder_level > REMINDER_LEVEL_ZERO)

        if q:
            # Recherche sur numéro de facture + nom client/patient + nom
            # institution
            from sqlalchemy import or_
            from sqlalchemy.orm import aliased

            from models import User as MUser

            # Alias pour distinguer client (patient) et institution (payeur)
            PatientClient = aliased(Client)
            BillToClient = aliased(Client)
            PatientUser = aliased(MUser)

            # Jointure avec le client (patient)
            query = query.join(
                PatientClient,
                Invoice.client_id == PatientClient.id)
            query = query.join(PatientUser,
                               PatientClient.user_id == PatientUser.id)

            # Jointure OPTIONNELLE avec l'institution payeuse (bill_to_client)
            query = query.outerjoin(
                BillToClient, Invoice.bill_to_client_id == BillToClient.id)

            like = f"%{q}%"
            query = query.filter(or_(
                Invoice.invoice_number.ilike(like),
                PatientUser.first_name.ilike(like),
                PatientUser.last_name.ilike(like),
                PatientUser.username.ilike(like),
                # ✅ Rechercher aussi dans le nom de l'institution
                BillToClient.institution_name.ilike(like),
            ))

        # Tri par émission récente
        from sqlalchemy import desc
        query = query.order_by(desc(Invoice.issued_at))

        pagination = query.paginate(
            page=page, per_page=per_page, error_out=False)
        items = pagination.items

        result_invoices = [inv.to_dict() for inv in items]

        # Stats sur l'ensemble filtré (pas seulement la page)
        # Pour efficience on calcule sur la page si volumineux; ici simple sum
        # sur items
        total_issued = sum(float(inv["total_amount"])
                           for inv in result_invoices)
        total_paid = sum(float(inv["amount_paid"]) for inv in result_invoices)
        total_balance = sum(float(inv["balance_due"])
                            for inv in result_invoices)
        overdue_count = len(
            [inv for inv in result_invoices if inv["status"] == "OVERDUE"])

        return {
            "invoices": result_invoices,
            "pagination": {
                "page": pagination.page,
                "per_page": pagination.per_page,
                "total": pagination.total,
                "pages": pagination.pages,
                "has_next": pagination.has_next,
                "has_prev": pagination.has_prev,
            },
            "stats": {
                "total_issued": total_issued,
                "total_paid": total_paid,
                "total_balance": total_balance,
                "overdue_count": overdue_count,
            }
        }


@invoices_ns.route("/companies/<int:company_id>/billing-settings")
class CompanyBillingSettingsResource(Resource):
    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, company_id):
        """Récupère les paramètres de facturation d'une entreprise."""
        user_public_id = get_jwt_identity()
        user = User.query.filter_by(public_id=user_public_id).first()
        if not user:
            return {"error": "Utilisateur non trouvé"}, 404

        company = Company.query.filter_by(id=company_id).first()
        if not company or company.user_id != user.id:
            return {"error": "Entreprise non trouvée ou accès refusé"}, 404

        billing_settings = CompanyBillingSettings.query.filter_by(
            company_id=company_id).first()
        if not billing_settings:
            # Créer des paramètres par défaut si non existants
            billing_settings = CompanyBillingSettings()
            billing_settings.company_id = company_id
            db.session.add(billing_settings)
            db.session.commit()
        return billing_settings.to_dict()

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    @invoices_ns.expect(billing_settings_model, validate=False)
    def put(self, company_id):
        """Met à jour les paramètres de facturation d'une entreprise."""
        try:
            user_public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user:
                return {"error": "Utilisateur non trouvé"}, 404

            company = Company.query.filter_by(id=company_id).first()
            if not company or company.user_id != user.id:
                return {"error": "Entreprise non trouvée ou accès refusé"}, 404

            billing_settings = CompanyBillingSettings.query.filter_by(
                company_id=company_id).first()
            if not billing_settings:
                return {"error": "Paramètres de facturation non trouvés"}, 404

            data = request.get_json() or {}
            
            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import ValidationError

            from schemas.invoice_schemas import BillingSettingsUpdateSchema
            from schemas.validation_utils import handle_validation_error, validate_request
            
            try:
                validated_data = validate_request(BillingSettingsUpdateSchema(), data, strict=False)
            except ValidationError as e:
                return handle_validation_error(e)
            
            app_logger.info(
                "Données reçues pour les paramètres de facturation: %s", validated_data)

            # Convertir les chaînes vides en None pour les champs numériques
            for field in [
                "payment_terms_days",
                "overdue_fee",
                "reminder1fee",
                "reminder2fee",
                    "reminder3fee"]:
                if field in validated_data:
                    value = validated_data[field]
                    if value == "" or value is None:
                        setattr(billing_settings, field, None)
                    else:
                        try:
                            setattr(billing_settings, field, float(value)
                                    if "." in str(value) else int(value))
                        except (ValueError, TypeError):
                            app_logger.warning(
                                "Valeur invalide pour %s: %s", field, value)
                            setattr(billing_settings, field, None)

            # Mettre à jour les autres champs - utilise données validées
            # email d'envoi des factures
            if "billing_email" in validated_data or "email_sender" in validated_data:
                billing_settings.email_sender = validated_data.get(
                    "billing_email", validated_data.get(
                        "email_sender", billing_settings.email_sender))
            if "invoice_prefix" in validated_data:
                billing_settings.invoice_prefix = validated_data["invoice_prefix"]
            if "invoice_number_format" in validated_data:
                billing_settings.invoice_number_format = validated_data["invoice_number_format"]
            if "iban" in validated_data:
                billing_settings.iban = validated_data["iban"]
            if "qr_iban" in validated_data:
                billing_settings.qr_iban = validated_data["qr_iban"]
            # esr_ref_base dans le schéma, colonne esr_ref_base dans le modèle
            if "esr_ref_base" in validated_data:
                billing_settings.esr_ref_base = validated_data.get("esr_ref_base") or None

            # planning des rappels: accepter dict ou string JSON, ou tableau
            # ordonné
            if "reminder_schedule_days" in validated_data:
                sched = validated_data["reminder_schedule_days"]
                try:
                    if isinstance(sched, str):
                        import json
                        sched = json.loads(sched)
                    # Normaliser en dict str->int ex: {"1": 30, "2": 10, "3":
                    # 5}
                    if isinstance(sched, list):
                        # ex [30,10,5]
                        sched = {str(i + 1): int(v)
                                 for i, v in enumerate(sched)}
                    elif isinstance(sched, dict):
                        sched = {str(k): int(v) for k, v in sched.items()}
                    billing_settings.reminder_schedule_days = sched
                except Exception:
                    app_logger.warning(
                        "reminder_schedule_days invalide, valeur ignorée")

            # auto_reminders_enabled si fourni - utilise données validées
            if "auto_reminders_enabled" in validated_data:
                billing_settings.auto_reminders_enabled = validated_data["auto_reminders_enabled"]

            app_logger.info("Paramètres mis à jour avec succès")
            db.session.commit()
            return billing_settings.to_dict()

        except Exception as e:
            app_logger.error(
                "Erreur lors de la mise à jour des paramètres: %s", str(e))
            db.session.rollback()
            return {"error": f"Erreur interne du serveur: {e!s}"}, 500


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
            user_public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user or not user.company or user.company.id != company_id:
                result = {"error": "Entreprise non trouvée ou accès refusé"}
                status_code = 404
            else:
                data = request.get_json() or {}
                
                # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
                from marshmallow import ValidationError

                from schemas.invoice_schemas import InvoiceGenerateSchema
                from schemas.validation_utils import handle_validation_error, validate_request
                
                try:
                    validated_data = validate_request(InvoiceGenerateSchema(), data, strict=False)
                except ValidationError as e:
                    result = handle_validation_error(e)
                    status_code = 400
                    return result, status_code
                
                client_id = validated_data.get("client_id")
                # NOUVEAU: pour facturation groupée
                client_ids = validated_data.get("client_ids", [])
                # NOUVEAU: support facturation tierce
                bill_to_client_id = validated_data.get("bill_to_client_id")
                period_year = validated_data["period_year"]
                period_month = validated_data["period_month"]
                
                # period_year et period_month sont déjà validés par le schema, donc toujours présents
                invoice_service = InvoiceService()

                # Cas 1: Facturation groupée de plusieurs clients vers une
                # institution
                if client_ids and bill_to_client_id:
                    app_logger.info(
                        "Génération factures consolidées: %s clients vers institution %s",
                        len(client_ids),
                        bill_to_client_id)

                    # Vérifier que l'institution existe et appartient à
                    # l'entreprise
                    institution = Client.query.filter_by(
                        id=bill_to_client_id, company_id=company_id).first()
                    if not institution:
                        result = {"error": "Institution non trouvée"}
                        status_code = 404
                    elif not institution.is_institution:
                        result = {"error": "Le client sélectionné n'est pas une institution"}
                        status_code = 400
                    else:
                        # NOUVEAU: Support de la sélection manuelle de réservations
                        # Format: { client_id: [reservation_ids] }
                        client_reservations = validated_data.get("client_reservations")

                        # Générer les factures
                        invoice_result = invoice_service.generate_consolidated_invoice(
                            company_id,
                            client_ids,
                            period_year,
                            period_month,
                            bill_to_client_id,
                            client_reservations  # NOUVEAU
                        )

                        result = {
                            "message": f"{invoice_result['success_count']} facture(s) générée(s), {invoice_result['error_count']} erreur(s)",
                            "invoices": [inv.to_dict() for inv in invoice_result["invoices"]],
                            "errors": invoice_result["errors"],
                            "success_count": invoice_result["success_count"],
                            "error_count": invoice_result["error_count"]
                        }
                        status_code = 201

                # Cas 2: Facturation simple (avec ou sans tierce)
                elif client_id:
                    # Vérifier que le client appartient à l'entreprise
                    client = Client.query.filter_by(
                        id=client_id, company_id=company_id).first()
                    if not client:
                        result = {"error": "Client non trouvé"}
                        status_code = 404
                    # Si facturation tierce, vérifier l'institution
                    elif bill_to_client_id:
                        institution = Client.query.filter_by(
                            id=bill_to_client_id, company_id=company_id).first()
                        if not institution:
                            result = {"error": "Institution payeuse non trouvée"}
                            status_code = 404
                        elif not institution.is_institution:
                            result = {"error": "Le client sélectionné n'est pas une institution"}
                            status_code = 400
                        else:
                            # Générer la facture
                            invoice = invoice_service.generate_invoice(
                                company_id,
                                client_id,
                                period_year,
                                period_month,
                                bill_to_client_id,
                                validated_data.get("reservation_ids")
                            )
                            result = invoice.to_dict()
                            status_code = 201
                    else:
                        # Générer la facture sans facturation tierce
                        invoice = invoice_service.generate_invoice(
                            company_id,
                            client_id,
                            period_year,
                            period_month,
                            bill_to_client_id,
                            validated_data.get("reservation_ids")
                        )
                        result = invoice.to_dict()
                        status_code = 201
                else:
                    result = {"error": "client_id ou client_ids requis"}
                    status_code = 400

        except ValueError as e:
            app_logger.error("Erreur de validation: %s", str(e))
            result = {"error": str(e)}
            status_code = 400
        except Exception as e:
            app_logger.error(
                "Erreur lors de la génération de facture: %s", str(e))
            result = {"error": "Erreur interne du serveur"}
            status_code = 500

        return result, status_code


@invoices_ns.route("/companies/<int:company_id>/invoices/<int:invoice_id>")
class InvoiceDetail(Resource):
    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, company_id, invoice_id):
        """Récupère les détails d'une facture."""
        try:
            user_id = get_jwt_identity()
            company = Company.query.filter_by(id=company_id).first()
            if not company or company.user_id != user_id:
                return {"error": "Entreprise non trouvée ou accès refusé"}, 404

            invoice = Invoice.query.filter_by(
                id=invoice_id, company_id=company_id
            ).options(
                joinedload(Invoice.client),
                joinedload(Invoice.lines),
                joinedload(Invoice.payments),
                joinedload(Invoice.reminders)
            ).first()

            if not invoice:
                return {"error": "Facture non trouvée"}, 404

            return invoice.to_dict()

        except Exception as e:
            app_logger.error(
                "Erreur lors de la récupération de la facture: %s", str(e))
            return {"error": "Erreur interne du serveur"}, 500


@invoices_ns.route("/companies/<int:company_id>/invoices/<int:invoice_id>/send")
class SendInvoice(Resource):
    """Endpoint pour marquer une facture comme envoyée."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id, invoice_id):
        """Marquer une facture comme envoyée."""
        try:
            # Vérifier l'autorisation
            user_public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user or not user.company or user.company.id != company_id:
                return {"error": "Non autorisé"}, 403

            # Récupérer la facture
            invoice = Invoice.query.filter_by(
                id=invoice_id, company_id=company_id).first()
            if not invoice:
                return {"error": "Facture non trouvée"}, 404

            # Marquer comme envoyée
            invoice.status = InvoiceStatus.SENT
            invoice.sent_at = datetime.now(datetime.timezone.utc)
            db.session.commit()

            return {
                "message": "Facture marquée comme envoyée",
                "status": invoice.status.value}

        except Exception as e:
            app_logger.error(
                "Erreur lors de l'envoi de la facture: %s", str(e))
            db.session.rollback()
            return {"error": "Erreur interne du serveur"}, 500


@invoices_ns.route("/companies/<int:company_id>/invoices/<int:invoice_id>/payments")
class InvoicePayments(Resource):
    """Endpoint pour enregistrer un paiement."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id, invoice_id):
        """Enregistrer un paiement pour une facture."""
        try:
            # Vérifier l'autorisation
            user_public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user or not user.company or user.company.id != company_id:
                return {"error": "Non autorisé"}, 403

            # Récupérer la facture
            invoice = Invoice.query.filter_by(
                id=invoice_id, company_id=company_id).first()
            if not invoice:
                return {"error": "Facture non trouvée"}, 404

            data = request.get_json()
            raw_amount = data.get("amount", 0)
            try:
                amount = Decimal(str(raw_amount))
            except Exception:
                return {"error": "Montant invalide"}, 400
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
            raw_method = str(method).strip(
            ) if method is not None else "bank_transfer"
            method_norm = method_map.get(
                raw_method.lower(), raw_method.lower())
            if method_norm not in PaymentMethod.choices():
                method_norm = "bank_transfer"
            payment_method = PaymentMethod(method_norm)
            method_value = payment_method.value
            app_logger.info(
                "Paiement: method reçu='%s', normalisé='%s', value='%s'",
                method,
                method_norm,
                method_value)
            waive_reminder_fees = bool(data.get("waive_reminder_fees", False))

            if amount <= AMOUNT_ZERO:
                return {"error": "Le montant doit être positif"}, 400

            # Optionnel: annuler les frais de rappel avant d'appliquer le
            # paiement
            if waive_reminder_fees and hasattr(invoice, "reminder_fee_amount"):
                try:
                    current_reminder_fee = Decimal(
                        str(invoice.reminder_fee_amount or 0))
                except Exception:
                    current_reminder_fee = Decimal("0")
                if current_reminder_fee > CURRENT_REMINDER_FEE_ZERO:
                    invoice.reminder_fee_amount = Decimal("0")
                    # Recalcule le total si un champ total existe
                    if hasattr(
                        invoice,
                        "subtotal_amount") and hasattr(
                        invoice,
                        "late_fee_amount") and hasattr(
                        invoice,
                            "total_amount"):
                        subtotal = Decimal(str(invoice.subtotal_amount or 0))
                        late_fee = Decimal(str(invoice.late_fee_amount or 0))
                        invoice.total_amount = subtotal + late_fee

            # Créer le paiement
            payment = InvoicePayment()
            payment.invoice_id = invoice.id
            payment.amount = 0.0
            payment.method = method_value  # passer la valeur ENUM attendue par la colonne SAEnum
            payment.paid_at = datetime.now(datetime.timezone.utc)
            db.session.add(payment)

            # Mettre à jour le montant payé de la facture
            current_paid = Decimal(str(invoice.amount_paid or 0))
            total_amount = Decimal(str(invoice.total_amount or 0))
            invoice.amount_paid = current_paid + amount
            invoice.balance_due = total_amount - invoice.amount_paid

            # Mettre à jour le statut
            if invoice.balance_due <= BALANCE_DUE_ZERO:
                invoice.status = InvoiceStatus.PAID
                invoice.paid_at = datetime.now(datetime.timezone.utc)
            elif invoice.amount_paid > AMOUNT_PAID_ZERO:
                invoice.status = InvoiceStatus.PARTIALLY_PAID

            db.session.commit()

            return {
                "message": "Paiement enregistré",
                "balance_due": float(invoice.balance_due),
                "amount_paid": float(invoice.amount_paid),
                "status": invoice.status.value
            }

        except Exception as e:
            app_logger.error(
                "Erreur lors de l'enregistrement du paiement: %s", str(e))
            db.session.rollback()
            return {"error": "Erreur interne du serveur"}, 500


@invoices_ns.route("/companies/<int:company_id>/invoices/<int:invoice_id>/reminders")
class InvoiceReminders(Resource):
    """Endpoint pour générer un rappel."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id, invoice_id):
        """Générer un rappel pour une facture."""
        try:
            # Vérifier l'autorisation
            user_public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user or not user.company or user.company.id != company_id:
                return {"error": "Non autorisé"}, 403

            # Récupérer la facture
            invoice = Invoice.query.filter_by(
                id=invoice_id, company_id=company_id).first()
            if not invoice:
                return {"error": "Facture non trouvée"}, 404

            data = request.get_json()
            level = int(data.get("level", 1))

            # Générer le rappel
            invoice_service = InvoiceService()
            reminder = invoice_service.generate_reminder(invoice_id, level)

            if reminder:
                # Régénérer le PDF pour inclure les frais de rappel
                pdf_service = PDFService()
                pdf_url = pdf_service.generate_invoice_pdf(invoice)

                if pdf_url:
                    invoice.pdf_url = pdf_url
                    db.session.commit()

                return {
                    "message": f"Rappel niveau {level} généré et PDF mis à jour",
                    "reminder_level": invoice.reminder_level,
                    "pdf_url": pdf_url}
            return {"error": "Erreur lors de la génération du rappel"}, 500

        except Exception as e:
            app_logger.error(
                "Erreur lors de la génération du rappel: %s", str(e))
            return {"error": "Erreur interne du serveur"}, 500


@invoices_ns.route("/companies/<int:company_id>/invoices/<int:invoice_id>/regenerate-pdf")
class RegenerateInvoicePdf(Resource):
    """Endpoint pour régénérer le PDF d'une facture."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id, invoice_id):
        """Régénérer le PDF d'une facture."""
        try:
            # Vérifier l'autorisation
            user_public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user or not user.company or user.company.id != company_id:
                return {"error": "Non autorisé"}, 403

            # Récupérer la facture
            invoice = Invoice.query.filter_by(
                id=invoice_id, company_id=company_id).first()
            if not invoice:
                return {"error": "Facture non trouvée"}, 404

            # Régénérer le PDF
            pdf_service = PDFService()
            pdf_url = pdf_service.generate_invoice_pdf(invoice)

            if pdf_url:
                invoice.pdf_url = pdf_url
                db.session.commit()
                return {"message": "PDF régénéré", "pdf_url": pdf_url}
            return {"error": "Erreur lors de la génération du PDF"}, 500

        except Exception as e:
            app_logger.error(
                "Erreur lors de la régénération du PDF: %s", str(e))
            return {"error": "Erreur interne du serveur"}, 500


@invoices_ns.route("/companies/<int:company_id>/invoices/<int:invoice_id>/cancel")
class CancelInvoice(Resource):
    """Endpoint pour annuler une facture."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id, invoice_id):
        """Annuler une facture."""
        try:
            # Vérifier l'autorisation
            user_public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user or not user.company or user.company.id != company_id:
                return {"error": "Non autorisé"}, 403

            # Récupérer la facture
            invoice = Invoice.query.filter_by(
                id=invoice_id, company_id=company_id).first()
            if not invoice:
                return {"error": "Facture non trouvée"}, 404

            # Annuler la facture
            invoice.status = InvoiceStatus.CANCELLED
            invoice.cancelled_at = datetime.now(datetime.timezone.utc)
            db.session.commit()

            return {
                "message": "Facture annulée",
                "status": invoice.status.value}

        except Exception as e:
            app_logger.error(
                "Erreur lors de l'annulation de la facture: %s", str(e))
            db.session.rollback()
            return {"error": "Erreur interne du serveur"}, 500


@invoices_ns.route("/companies/<int:company_id>/clients/institutions")
class InstitutionsList(Resource):
    """Endpoint pour récupérer la liste des institutions (cliniques)."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, company_id):
        """Liste les institutions (cliniques) pour la facturation tierce."""
        try:
            user_public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user or not user.company or user.company.id != company_id:
                return {"error": "Non autorisé"}, 403

            # Récupérer toutes les institutions actives de l'entreprise
            institutions = Client.query.filter_by(
                company_id=company_id,
                is_institution=True,
                is_active=True
            ).all()

            return {
                "institutions": [
                    {
                        "id": inst.id,
                        "institution_name": inst.institution_name or "Institution sans nom",
                        "contact_email": inst.contact_email,
                        "contact_phone": inst.contact_phone,
                        "billing_address": inst.billing_address,
                        "user": {
                            "first_name": inst.user.first_name if inst.user else "",
                            "last_name": inst.user.last_name if inst.user else "",
                            "username": inst.user.username if inst.user else "",
                        } if inst.user else None
                    }
                    for inst in institutions
                ]
            }

        except Exception as e:
            app_logger.error(
                "Erreur lors de la récupération des institutions: %s", str(e))
            return {"error": "Erreur interne du serveur"}, 500


@invoices_ns.route("/companies/<int:company_id>/clients/<int:client_id>/toggle-institution")
class ToggleInstitution(Resource):
    """Endpoint pour marquer/démarquer un client comme institution."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def post(self, company_id, client_id):
        """Bascule le statut d'institution d'un client."""
        try:
            user_public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user or not user.company or user.company.id != company_id:
                return {"error": "Non autorisé"}, 403

            client = Client.query.filter_by(
                id=client_id, company_id=company_id).first()
            if not client:
                return {"error": "Client non trouvé"}, 404

            data = request.get_json() or {}
            is_institution = data.get(
                "is_institution", not client.is_institution)
            institution_name = data.get("institution_name")

            client.is_institution = is_institution
            if is_institution and institution_name:
                client.institution_name = institution_name
            elif not is_institution:
                client.institution_name = None

            db.session.commit()

            return {
                "message": f"Client {'marqué comme' if is_institution else 'démarqué en tant que'} institution",
                "client": client.serialize}

        except Exception as e:
            app_logger.error(
                "Erreur lors de la modification du statut d'institution: %s", str(e))
            db.session.rollback()
            return {"error": "Erreur interne du serveur"}, 500


@invoices_ns.route("/companies/<int:company_id>/clients/<int:client_id>/unbilled-reservations")
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
            user_public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user or not user.company or user.company.id != company_id:
                return {"error": "Non autorisé"}, 403

            # Récupérer les paramètres
            period_year = request.args.get("year", type=int)
            period_month = request.args.get("month", type=int)
            billed_to_filter = request.args.get("billed_to_type", type=str)

            # Query de base : réservations TERMINÉES non facturées (seulement
            # COMPLETED et RETURN_COMPLETED)
            query = Booking.query.filter(
                Booking.company_id == company_id,
                Booking.client_id == client_id,
                # ✅ Seulement les courses TERMINÉES
                Booking.status.in_([
                    BookingStatus.COMPLETED,
                    BookingStatus.RETURN_COMPLETED
                ]),
                Booking.invoice_line_id.is_(None)  # Pas encore facturé
            )

            # 🔍 LOG : Debug pour voir ce qui est trouvé
            app_logger.warning(
                "🔍 [Unbilled] Recherche pour client_id=%s, company_id=%s, year=%s, month=%s, billed_to_filter=%s",
                client_id, company_id, period_year, period_month, billed_to_filter)

            # Filtrer par période si fournie
            if period_year and period_month:
                start_date = datetime(period_year, period_month, 1)
                if period_month == PERIOD_MONTH_THRESHOLD:
                    end_date = datetime(period_year + 1, 1, 1)
                else:
                    end_date = datetime(period_year, period_month + 1, 1)

                query = query.filter(
                    Booking.scheduled_time >= start_date,
                    Booking.scheduled_time < end_date
                )

            # 🔍 LOG : Compter AVANT filtre billed_to_type
            count_before_filter = query.count()
            app_logger.warning(
                "🔍 [Unbilled] Avant filtre billed_to_type: %s bookings trouvés",
                count_before_filter)

            # ⚠️ NE PAS filtrer par billed_to_type : on veut TOUS les transports non facturés du client
            # Même si le type de facturation ne correspond pas, on affiche tout
            # Le dispatcher pourra choisir ce qu'il veut facturer
            # if billed_to_filter and billed_to_filter in ['patient', 'clinic', 'insurance']:
            #     query = query.filter(Booking.billed_to_type == billed_to_filter)
            #     app_logger.warning("🔍 [Unbilled] Filtre appliqué: billed_to_type=%s", billed_to_filter)

            # Trier par date
            from sqlalchemy import asc
            query = query.order_by(asc(Booking.scheduled_time))

            reservations = query.all()

            # 🔍 LOG : Afficher les résultats trouvés
            app_logger.warning(
                "🔍 [Unbilled] FINAL: Trouvé %s réservations non facturées",
                len(reservations))
            for r in reservations:
                app_logger.warning(
                    "   - Booking #%s: %s, %s, status=%s, billed_to_type=%s, invoice_line_id=%s",
                    r.id, r.customer_name, r.scheduled_time, r.status, r.billed_to_type, r.invoice_line_id)

            return {
                "reservations": [
                    {
                        "id": r.id,
                        "date": r.scheduled_time.isoformat() if r.scheduled_time else None,
                        "pickup_location": r.pickup_location,
                        "dropoff_location": r.dropoff_location,
                        "amount": float(r.amount or 0),
                        "billed_to_type": r.billed_to_type,
                        "billed_to_company_id": r.billed_to_company_id,
                        "billed_to_contact": r.billed_to_contact,
                        "customer_name": r.customer_name,
                        "status": r.status.value,
                        "is_urgent": r.is_urgent or False,
                        "is_return": r.is_return or False,
                        "medical_facility": r.medical_facility,
                    }
                    for r in reservations
                ],
                "total_amount": sum(float(r.amount or 0) for r in reservations),
                "count": len(reservations),
                "summary_by_type": {
                    "patient": sum(1 for r in reservations if r.billed_to_type == "patient"),
                    "clinic": sum(1 for r in reservations if r.billed_to_type == "clinic"),
                    "insurance": sum(1 for r in reservations if r.billed_to_type == "insurance"),
                }
            }

        except Exception as e:
            app_logger.error(
                "Erreur lors de la récupération des réservations non facturées: %s",
                str(e))
            return {"error": "Erreur interne du serveur"}, 500
