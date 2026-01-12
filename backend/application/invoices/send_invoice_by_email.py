"""Use-case: envoyer une facture par email.

Ce use case gère l'envoi d'une facture par email à un client,
incluant la validation de l'email, la génération du PDF si nécessaire,
et le marquage de la facture comme envoyée.

✅ Refactoré pour utiliser Brevo (service transactionnel) au lieu de SMTP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ext import db
from models import Client, Company, CompanyBillingSettings, Invoice
from services.documents.pdf import PDFService
from services.email.brevo_provider import BrevoEmailProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SendInvoiceByEmailInput:
    """Input pour l'envoi d'une facture par email."""

    invoice_id: int
    recipient_email: str | None = None  # Si None, utilise client.contact_email
    force_regenerate_pdf: bool = False  # Regénérer le PDF même s'il existe


@dataclass(frozen=True, slots=True)
class SendInvoiceByEmailResult:
    """Résultat de l'envoi d'une facture par email."""

    success: bool
    invoice_id: int
    recipient: str | None = None
    sent_at: datetime | None = None
    error: str | None = None
    status_code: int = 200


class SendInvoiceByEmailUseCase:
    """Use-case Application: envoyer une facture par email via Brevo."""

    def __init__(self) -> None:  # type: ignore[reportMissingSuperCall]
        self.brevo_provider = BrevoEmailProvider()
        self.pdf_service = PDFService()

    def execute(self, input_data: SendInvoiceByEmailInput) -> SendInvoiceByEmailResult:  # noqa: PLR0911
        """
        Envoie une facture par email.

        Étapes:
        1. Valider que la facture existe
        2. Charger le client et l'entreprise
        3. Déterminer l'email du destinataire
        4. Générer le PDF si nécessaire
        5. Envoyer l'email
        6. Marquer la facture comme envoyée
        7. Persister les changements

        Args:
            input_data: Données d'entrée avec invoice_id et recipient_email

        Returns:
            SendInvoiceByEmailResult avec le statut d'envoi
        """
        try:
            # 1. Valider que la facture existe
            invoice = Invoice.query.get(input_data.invoice_id)
            if not invoice:
                return SendInvoiceByEmailResult(
                    success=False,
                    invoice_id=input_data.invoice_id,
                    error=f"Facture #{input_data.invoice_id} introuvable",
                    status_code=404,
                )

            # 2. Charger le client et l'entreprise
            client = Client.query.get(invoice.client_id)
            if not client:
                return SendInvoiceByEmailResult(
                    success=False,
                    invoice_id=input_data.invoice_id,
                    error=f"Client #{invoice.client_id} introuvable",
                    status_code=404,
                )

            company = Company.query.get(invoice.company_id)
            if not company:
                return SendInvoiceByEmailResult(
                    success=False,
                    invoice_id=input_data.invoice_id,
                    error=f"Entreprise #{invoice.company_id} introuvable",
                    status_code=404,
                )

            # 3. Déterminer l'email du destinataire
            recipient_email = input_data.recipient_email or client.contact_email
            if not recipient_email:
                return SendInvoiceByEmailResult(
                    success=False,
                    invoice_id=input_data.invoice_id,
                    error=(
                        "Aucune adresse email disponible pour ce client. "
                        "Veuillez ajouter un email de contact ou spécifier "
                        "un destinataire."
                    ),
                    status_code=400,
                )

            # 4. Générer le PDF si nécessaire
            pdf_path = None
            logger.info(
                "[INVOICE EMAIL] invoice.pdf_url=%s, force_regenerate=%s",
                invoice.pdf_url,
                input_data.force_regenerate_pdf,
            )
            if not invoice.pdf_url or input_data.force_regenerate_pdf:
                logger.info(
                    "Génération du PDF pour la facture %s", invoice.invoice_number
                )
                pdf_url = self.pdf_service.generate_invoice_pdf(invoice)
                if pdf_url:
                    invoice.pdf_url = pdf_url
                    db.session.commit()
                    # Convertir l'URL en chemin système si nécessaire
                    from flask import current_app

                    uploads_dir = Path(
                        current_app.config.get("UPLOAD_FOLDER", "/app/uploads")
                    )

                    # Extraire le chemin relatif depuis l'URL
                    if pdf_url.startswith("/uploads/"):
                        relative_path = pdf_url.removeprefix("/uploads/")
                    elif "/uploads/" in pdf_url:
                        relative_path = pdf_url.split("/uploads/", 1)[1]
                    else:
                        relative_path = None

                    if relative_path:
                        pdf_path = str(uploads_dir / relative_path)
                        logger.info(
                            "[INVOICE EMAIL] PDF généré: %s -> %s",
                            pdf_url,
                            pdf_path,
                        )
                else:
                    logger.warning(
                        "Impossible de générer le PDF pour la facture %s",
                        invoice.invoice_number,
                    )
            elif invoice.pdf_url:
                # Utiliser le PDF existant
                from flask import current_app

                uploads_dir = Path(
                    current_app.config.get("UPLOAD_FOLDER", "/app/uploads")
                )

                # Extraire le chemin relatif depuis l'URL (gérer les URLs complètes et relatives)
                if invoice.pdf_url.startswith("/uploads/"):
                    # URL relative : /uploads/invoices/...
                    relative_path = invoice.pdf_url.removeprefix("/uploads/")
                elif "/uploads/" in invoice.pdf_url:
                    # URL complète : http://localhost:5000/uploads/invoices/...
                    relative_path = invoice.pdf_url.split("/uploads/", 1)[1]
                else:
                    logger.warning(
                        "[INVOICE EMAIL] Format d'URL inattendu pour invoice.pdf_url: %s",
                        invoice.pdf_url,
                    )
                    relative_path = None

                if relative_path:
                    pdf_path = str(uploads_dir / relative_path)
                    logger.info(
                        "[INVOICE EMAIL] PDF extrait de l'URL: %s -> %s",
                        invoice.pdf_url,
                        pdf_path,
                    )

            logger.info(
                "[INVOICE EMAIL] Après traitement PDF: pdf_path=%s",
                pdf_path,
            )

            # 5. Charger les paramètres de facturation (pour config Brevo)
            billing_settings = CompanyBillingSettings.query.filter_by(
                company_id=company.id
            ).first()

            if not billing_settings:
                return SendInvoiceByEmailResult(
                    success=False,
                    invoice_id=input_data.invoice_id,
                    recipient=recipient_email,
                    error=(
                        "Paramètres de facturation non configurés. "
                        "Veuillez configurer l'email d'envoi dans les paramètres."
                    ),
                    status_code=400,
                )

            # 6. Vérifier que le domaine email est validé dans Brevo
            from_email = (
                billing_settings.smtp_username
            )  # Champ existant utilisé pour from_email
            from_name = billing_settings.from_name or company.name

            if not from_email:
                return SendInvoiceByEmailResult(
                    success=False,
                    invoice_id=input_data.invoice_id,
                    recipient=recipient_email,
                    error=(
                        "Adresse email d'envoi non configurée. "
                        "Veuillez configurer l'email d'envoi dans Paramètres > Facturation."
                    ),
                    status_code=400,
                )

            if not billing_settings.domain_verified:
                return SendInvoiceByEmailResult(
                    success=False,
                    invoice_id=input_data.invoice_id,
                    recipient=recipient_email,
                    error=(
                        f"Le domaine email ({from_email.split('@')[1]}) n'est pas vérifié. "
                        "Veuillez configurer les enregistrements DNS (SPF/DKIM) "
                        "dans Paramètres > Facturation > Configuration Email."
                    ),
                    status_code=403,
                )

            # 7. Lire le PDF en bytes pour l'attachement
            pdf_bytes = None
            logger.info(
                "[INVOICE EMAIL] pdf_path=%s, exists=%s",
                pdf_path,
                Path(pdf_path).exists() if pdf_path else False,
            )
            if pdf_path and Path(pdf_path).exists():
                try:
                    with Path(pdf_path).open("rb") as f:
                        pdf_bytes = f.read()
                    logger.info(
                        "[INVOICE EMAIL] PDF lu avec succès, taille=%s bytes",
                        len(pdf_bytes),
                    )
                except Exception as e:
                    logger.warning(
                        "Impossible de lire le PDF %s: %s",
                        pdf_path,
                        e,
                    )
            else:
                logger.warning(
                    "[INVOICE EMAIL] Aucun PDF disponible pour la facture %s (pdf_path=%s)",
                    invoice.invoice_number,
                    pdf_path,
                )

            # 8. Générer le contenu HTML de l'email
            # Récupérer le nom du client (utilisé dans template et message par défaut)
            client_name = "Client"
            if client.user:
                first_name = getattr(client.user, "first_name", "")
                last_name = getattr(client.user, "last_name", "")
                client_name = (
                    f"{first_name} {last_name}".strip()
                    or client.user.username
                    or "Client"
                )

            # Utiliser le template de message s'il existe, sinon message par défaut
            template = (
                billing_settings.invoice_message_template
                if billing_settings and billing_settings.invoice_message_template
                else None
            )

            if template:
                # Remplacer les variables du template
                html_content = template.replace("{client_name}", client_name)
                html_content = html_content.replace(
                    "{invoice_number}", invoice.invoice_number or ""
                )
                html_content = html_content.replace(
                    "{amount}",
                    f"{invoice.total_amount:.2f}" if invoice.total_amount else "0.00",
                )
                html_content = html_content.replace(
                    "{due_date}",
                    invoice.due_date.strftime("%d/%m/%Y")
                    if invoice.due_date
                    else "À définir",
                )
                # Convertir les sauts de ligne en <br>
                html_content = html_content.replace("\n", "<br>")
            else:
                # Message par défaut

                html_content = f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                    <p>Bonjour {client_name},</p>
                    <p>Veuillez trouver ci-joint la facture <strong>{invoice.invoice_number}</strong>
                    d'un montant de <strong>{invoice.total_amount:.2f} CHF</strong>.</p>
                    <p>Date d'échéance : <strong>{invoice.due_date.strftime("%d/%m/%Y") if invoice.due_date else "À définir"}</strong></p>
                    <p>Merci de procéder au paiement dans les délais.</p>
                    <br>
                    <p>Cordialement,<br><strong>{company.name}</strong></p>
                </body>
                </html>
                """

            # 9. Préparer l'attachement PDF
            attachments = []
            if pdf_bytes:
                attachments.append(
                    {
                        "filename": f"facture_{invoice.invoice_number}.pdf",
                        "content": pdf_bytes,
                    }
                )

            # 10. Envoyer l'email via Brevo
            logger.info(
                "[INVOICE EMAIL] Envoi de la facture %s par email via Brevo à %s (depuis %s) avec %d attachement(s)",
                invoice.invoice_number,
                recipient_email,
                from_email,
                len(attachments),
            )

            email_result = self.brevo_provider.send_invoice_email(
                from_email=from_email,
                from_name=from_name,
                to_email=recipient_email,
                to_name=client_name,
                subject=f"Facture {invoice.invoice_number} - {company.name}",
                html_content=html_content,
                attachments=attachments,
            )

            if not email_result.success:
                logger.error(
                    "Échec de l'envoi de la facture %s via Brevo: %s",
                    invoice.invoice_number,
                    email_result.error,
                )
                return SendInvoiceByEmailResult(
                    success=False,
                    invoice_id=input_data.invoice_id,
                    recipient=recipient_email,
                    error=f"Erreur Brevo: {email_result.error}",
                    status_code=500,
                )

            # 11. Marquer la facture comme envoyée
            invoice.mark_as_sent()
            db.session.commit()

            logger.info(
                "✅ Facture %s envoyée avec succès via Brevo à %s (message_id: %s)",
                invoice.invoice_number,
                recipient_email,
                email_result.message_id or "N/A",
            )

            return SendInvoiceByEmailResult(
                success=True,
                invoice_id=input_data.invoice_id,
                recipient=recipient_email,
                sent_at=datetime.now(),
            )

        except Exception as e:
            logger.exception(
                "Erreur lors de l'envoi de la facture %s par email",
                input_data.invoice_id,
            )
            db.session.rollback()
            return SendInvoiceByEmailResult(
                success=False,
                invoice_id=input_data.invoice_id,
                error=f"Erreur inattendue: {e!s}",
                status_code=500,
            )
