"""Use-case: envoyer un rappel de paiement par email.

Ce use case gère l'envoi d'un rappel de paiement par email,
incluant la validation, la génération du PDF du rappel si nécessaire,
et le marquage du rappel comme envoyé.

✅ Refactoré pour utiliser Brevo (service transactionnel) au lieu de SMTP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ext import db
from models import Client, Company, CompanyBillingSettings, Invoice, InvoiceReminder
from services.documents.pdf import PDFService
from services.email.brevo_provider import BrevoEmailProvider

logger = logging.getLogger(__name__)

# Constants pour les niveaux de rappel
REMINDER_LEVEL_1 = 1
REMINDER_LEVEL_2 = 2
REMINDER_LEVEL_3 = 3


@dataclass(frozen=True, slots=True)
class SendReminderByEmailInput:
    """Input pour l'envoi d'un rappel par email."""

    reminder_id: int
    recipient_email: str | None = None  # Si None, utilise client.contact_email
    force_regenerate_pdf: bool = False  # Regénérer le PDF même s'il existe


@dataclass(frozen=True, slots=True)
class SendReminderByEmailResult:
    """Résultat de l'envoi d'un rappel par email."""

    success: bool
    reminder_id: int
    invoice_id: int | None = None
    recipient: str | None = None
    sent_at: datetime | None = None
    error: str | None = None
    status_code: int = 200


class SendReminderByEmailUseCase:
    """Use-case Application: envoyer un rappel de paiement par email via Brevo."""

    def __init__(self) -> None:  # type: ignore[reportMissingSuperCall]
        self.brevo_provider = BrevoEmailProvider()
        self.pdf_service = PDFService()

    def execute(  # noqa: PLR0911
        self, input_data: SendReminderByEmailInput
    ) -> SendReminderByEmailResult:
        """
        Envoie un rappel de paiement par email.

        Étapes:
        1. Valider que le rappel existe
        2. Charger la facture, le client et l'entreprise
        3. Déterminer l'email du destinataire
        4. Générer le PDF du rappel si nécessaire
        5. Envoyer l'email
        6. Marquer le rappel comme envoyé
        7. Persister les changements

        Args:
            input_data: Données d'entrée avec reminder_id et recipient_email

        Returns:
            SendReminderByEmailResult avec le statut d'envoi
        """
        try:
            # 1. Valider que le rappel existe
            reminder = InvoiceReminder.query.get(input_data.reminder_id)
            if not reminder:
                return SendReminderByEmailResult(
                    success=False,
                    reminder_id=input_data.reminder_id,
                    error=f"Rappel #{input_data.reminder_id} introuvable",
                    status_code=404,
                )

            # 2. Charger la facture, le client et l'entreprise
            invoice = Invoice.query.get(reminder.invoice_id)
            if not invoice:
                return SendReminderByEmailResult(
                    success=False,
                    reminder_id=input_data.reminder_id,
                    invoice_id=reminder.invoice_id,
                    error=f"Facture #{reminder.invoice_id} introuvable",
                    status_code=404,
                )

            client = Client.query.get(invoice.client_id)
            if not client:
                return SendReminderByEmailResult(
                    success=False,
                    reminder_id=input_data.reminder_id,
                    invoice_id=invoice.id,
                    error=f"Client #{invoice.client_id} introuvable",
                    status_code=404,
                )

            company = Company.query.get(invoice.company_id)
            if not company:
                return SendReminderByEmailResult(
                    success=False,
                    reminder_id=input_data.reminder_id,
                    invoice_id=invoice.id,
                    error=f"Entreprise #{invoice.company_id} introuvable",
                    status_code=404,
                )

            # 3. Déterminer l'email du destinataire
            recipient_email = input_data.recipient_email or client.contact_email
            if not recipient_email:
                return SendReminderByEmailResult(
                    success=False,
                    reminder_id=input_data.reminder_id,
                    invoice_id=invoice.id,
                    error=(
                        "Aucune adresse email disponible pour ce client. "
                        "Veuillez ajouter un email de contact ou spécifier "
                        "un destinataire."
                    ),
                    status_code=400,
                )

            # 4. Utiliser le PDF de la facture originale (pas un PDF de rappel spécifique)
            pdf_path = None
            logger.info(
                "[REMINDER EMAIL] invoice.pdf_url=%s, force_regenerate=%s",
                invoice.pdf_url,
                input_data.force_regenerate_pdf,
            )
            if not invoice.pdf_url or input_data.force_regenerate_pdf:
                logger.info(
                    "Génération du PDF pour la facture %s (pour rappel N°%s)",
                    invoice.invoice_number,
                    reminder.level,
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
                            "[REMINDER EMAIL] PDF généré: %s -> %s",
                            pdf_url,
                            pdf_path,
                        )
                else:
                    logger.warning(
                        "Impossible de générer le PDF pour la facture %s",
                        invoice.invoice_number,
                    )
            elif invoice.pdf_url:
                # Utiliser le PDF existant de la facture
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
                        "[REMINDER EMAIL] Format d'URL inattendu pour invoice.pdf_url: %s",
                        invoice.pdf_url,
                    )
                    relative_path = None

                if relative_path:
                    pdf_path = str(uploads_dir / relative_path)
                    logger.info(
                        "[REMINDER EMAIL] PDF de la facture extrait de l'URL: %s -> %s",
                        invoice.pdf_url,
                        pdf_path,
                    )

            logger.info(
                "[REMINDER EMAIL] Après traitement PDF: pdf_path=%s",
                pdf_path,
            )

            # 5. Charger les paramètres de facturation (pour config Brevo)
            billing_settings = CompanyBillingSettings.query.filter_by(
                company_id=company.id
            ).first()

            if not billing_settings:
                return SendReminderByEmailResult(
                    success=False,
                    reminder_id=input_data.reminder_id,
                    invoice_id=invoice.id,
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
                return SendReminderByEmailResult(
                    success=False,
                    reminder_id=input_data.reminder_id,
                    invoice_id=invoice.id,
                    recipient=recipient_email,
                    error=(
                        "Adresse email d'envoi non configurée. "
                        "Veuillez configurer l'email d'envoi dans Paramètres > Facturation."
                    ),
                    status_code=400,
                )

            if not billing_settings.domain_verified:
                return SendReminderByEmailResult(
                    success=False,
                    reminder_id=input_data.reminder_id,
                    invoice_id=invoice.id,
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
                "[REMINDER EMAIL] pdf_path=%s, exists=%s",
                pdf_path,
                Path(pdf_path).exists() if pdf_path else False,
            )
            if pdf_path and Path(pdf_path).exists():
                try:
                    with Path(pdf_path).open("rb") as f:
                        pdf_bytes = f.read()
                    logger.info(
                        "[REMINDER EMAIL] PDF lu avec succès, taille=%s bytes",
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
                    "[REMINDER EMAIL] Aucun PDF disponible pour le rappel N°%s de %s (pdf_path=%s)",
                    reminder.level,
                    invoice.invoice_number,
                    pdf_path,
                )

            # 8. Générer le contenu HTML de l'email selon le niveau de rappel
            # Utiliser le template approprié s'il existe
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

            template = None
            if (
                reminder.level == REMINDER_LEVEL_1
                and billing_settings.reminder1_template
            ):
                template = billing_settings.reminder1_template
            elif (
                reminder.level == REMINDER_LEVEL_2
                and billing_settings.reminder2_template
            ):
                template = billing_settings.reminder2_template
            elif (
                reminder.level == REMINDER_LEVEL_3
                and billing_settings.reminder3_template
            ):
                template = billing_settings.reminder3_template

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
                html_content = html_content.replace(
                    "{reminder_level}", str(reminder.level)
                )
                # Ajouter les frais de rappel si présents
                if reminder.added_fee and reminder.added_fee > 0:
                    html_content = html_content.replace(
                        "{reminder_fee}", f"{reminder.added_fee:.2f}"
                    )
                # Convertir les sauts de ligne en <br>
                html_content = html_content.replace("\n", "<br>")
            else:
                # Message par défaut selon le niveau
                if reminder.level == REMINDER_LEVEL_1:
                    subject_prefix = "Rappel"
                    message_intro = "Nous constatons que la facture ci-dessous n'a pas encore été réglée."
                elif reminder.level == REMINDER_LEVEL_2:
                    subject_prefix = "2e rappel"
                    message_intro = "Malgré notre précédent rappel, nous constatons que la facture ci-dessous reste impayée."
                else:
                    subject_prefix = "Mise en demeure"
                    message_intro = "Nous vous informons que la facture ci-dessous demeure impayée malgré nos rappels précédents."

                fee_text = ""
                if reminder.added_fee and reminder.added_fee > 0:
                    fee_text = f"<p><strong>Frais de rappel : {reminder.added_fee:.2f} CHF</strong></p>"

                html_content = f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                    <p>Bonjour {client_name},</p>
                    <p>{message_intro}</p>
                    <p>Facture : <strong>{invoice.invoice_number}</strong><br>
                    Montant : <strong>{invoice.total_amount:.2f} CHF</strong><br>
                    Date d'échéance : <strong>{invoice.due_date.strftime("%d/%m/%Y") if invoice.due_date else "À définir"}</strong></p>
                    {fee_text}
                    <p>Merci de procéder au règlement dans les plus brefs délais.</p>
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
                        "filename": f"rappel_{reminder.level}_facture_{invoice.invoice_number}.pdf",
                        "content": pdf_bytes,
                    }
                )

            # 10. Envoyer l'email via Brevo
            subject_prefix = {
                1: "Rappel",
                2: "2e rappel",
                3: "Mise en demeure",
            }.get(reminder.level, "Rappel")

            logger.info(
                "[REMINDER EMAIL] Envoi du rappel N°%s pour la facture %s par email via Brevo à %s (depuis %s) avec %d attachement(s)",
                reminder.level,
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
                subject=f"{subject_prefix} - Facture {invoice.invoice_number} - {company.name}",
                html_content=html_content,
                attachments=attachments,
            )

            if not email_result.success:
                logger.error(
                    "Échec de l'envoi du rappel N°%s pour %s via Brevo: %s",
                    reminder.level,
                    invoice.invoice_number,
                    email_result.error,
                )
                return SendReminderByEmailResult(
                    success=False,
                    reminder_id=input_data.reminder_id,
                    invoice_id=invoice.id,
                    recipient=recipient_email,
                    error=f"Erreur Brevo: {email_result.error}",
                    status_code=500,
                )

            # 11. Marquer le rappel comme envoyé
            reminder.sent_at = datetime.now()
            db.session.commit()

            logger.info(
                "✅ Rappel N°%s pour la facture %s envoyé avec succès via Brevo à %s (message_id: %s)",
                reminder.level,
                invoice.invoice_number,
                recipient_email,
                email_result.message_id or "N/A",
            )

            return SendReminderByEmailResult(
                success=True,
                reminder_id=input_data.reminder_id,
                invoice_id=invoice.id,
                recipient=recipient_email,
                sent_at=datetime.now(),
            )

        except Exception as e:
            logger.exception(
                "Erreur lors de l'envoi du rappel %s par email",
                input_data.reminder_id,
            )
            db.session.rollback()
            return SendReminderByEmailResult(
                success=False,
                reminder_id=input_data.reminder_id,
                error=f"Erreur inattendue: {e!s}",
                status_code=500,
            )
