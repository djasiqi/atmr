"""
Service d'envoi d'emails pour factures et rappels.

Ce service gère l'envoi d'emails transactionnels (factures, rappels, confirmations)
via Flask-Mail avec support de templates HTML Jinja2 et configuration multi-tenant SMTP.

Chaque entreprise peut avoir sa propre configuration SMTP (serveur, port, credentials).
Si non configurée, le service utilise la configuration globale du .env.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import current_app, render_template
from flask_mail import Mail, Message  # pyright: ignore[reportMissingImports]

from ext import mail

from .validators import EmailValidator

logger = logging.getLogger(__name__)


@dataclass
class EmailSendResult:
    """Résultat d'envoi d'email."""

    success: bool
    recipient: str | None
    error: str | None
    sent_at: datetime | None


class EmailService:
    """Service d'envoi d'emails transactionnels avec support multi-tenant."""

    def __init__(self) -> None:  # type: ignore[reportMissingSuperCall]
        self.validator = EmailValidator()
        self.max_retries = 3

    def _get_mail_instance(self, billing_settings: Any | None = None) -> Mail:
        """
        Retourne une instance Mail configurée pour l'entreprise ou avec config globale.

        Args:
            billing_settings: CompanyBillingSettings ou None pour config globale

        Returns:
            Instance Mail configurée
        """
        # Si l'entreprise a sa propre config SMTP activée, l'utiliser
        if (
            billing_settings
            and hasattr(billing_settings, "smtp_enabled")
            and billing_settings.smtp_enabled
            and billing_settings.smtp_server
        ):
            logger.info(
                "Utilisation de la configuration SMTP de l'entreprise (company_id=%s)",
                billing_settings.company_id,
            )

            # Créer une instance Mail temporaire avec la config de l'entreprise
            mail_instance = Mail()

            # Configuration temporaire
            current_app.config["MAIL_SERVER"] = billing_settings.smtp_server
            current_app.config["MAIL_PORT"] = billing_settings.smtp_port or 587
            current_app.config["MAIL_USE_TLS"] = billing_settings.smtp_use_tls
            current_app.config["MAIL_USE_SSL"] = billing_settings.smtp_use_ssl
            current_app.config["MAIL_USERNAME"] = billing_settings.smtp_username
            current_app.config["MAIL_PASSWORD"] = (
                billing_settings.smtp_password
            )  # Déchiffré automatiquement
            current_app.config["MAIL_DEFAULT_SENDER"] = (
                billing_settings.email_sender or billing_settings.smtp_username
            )

            mail_instance.init_app(current_app)
            return mail_instance

        # Sinon, utiliser la configuration globale
        logger.info("Utilisation de la configuration SMTP globale (.env)")
        return mail

    def send_invoice_email(
        self,
        invoice: Any,
        recipient_email: str,
        company: Any,
        client: Any,
        pdf_path: str | None = None,
        billing_settings: Any | None = None,
    ) -> EmailSendResult:
        """
        Envoie un email de facture à un client.

        Args:
            invoice: Instance de modèle Invoice
            recipient_email: Email du destinataire
            company: Instance de modèle Company
            client: Instance de modèle Client
            pdf_path: Chemin vers le PDF de la facture (optionnel)
            billing_settings: CompanyBillingSettings (optionnel, pour SMTP custom)

        Returns:
            EmailSendResult avec le statut d'envoi
        """
        # Validation de l'email
        validation = self.validator.validate(recipient_email)
        if not validation["valid"]:
            return EmailSendResult(
                success=False,
                recipient=recipient_email,
                error=validation["error"],
                sent_at=None,
            )

        normalized_email = validation["normalized"]

        try:
            # Préparer le sujet
            subject = (
                f"Facture {invoice.invoice_number} - "
                f"{company.name if hasattr(company, 'name') else 'Votre entreprise'}"
            )

            # Préparer les données pour le template
            template_data = {
                "company_name": getattr(company, "name", "Votre entreprise"),
                "invoice_number": invoice.invoice_number,
                "invoice_date": (
                    invoice.issued_at.strftime("%d.%m.%Y")
                    if invoice.issued_at
                    else "N/A"
                ),
                "due_date": (
                    invoice.due_date.strftime("%d.%m.%Y") if invoice.due_date else "N/A"
                ),
                "total_amount": float(invoice.total_amount),
                "currency": invoice.currency or "CHF",
                "client_name": self._format_client_name(client),
                "payment_terms_days": getattr(
                    invoice, "payment_terms_days", 30
                ),  # Fallback
            }

            # Rendu du template HTML
            html_body = render_template("emails/invoice_email.html", **template_data)

            # Obtenir l'instance Mail configurée (globale ou entreprise)
            mail_instance = self._get_mail_instance(billing_settings)

            # Déterminer l'expéditeur
            sender_email = (
                billing_settings.email_sender
                if billing_settings and billing_settings.email_sender
                else current_app.config.get("MAIL_DEFAULT_SENDER", "noreply@atmr.ch")
            )

            # Créer le message
            msg = Message(
                subject=subject,
                recipients=[normalized_email],
                html=html_body,
                sender=sender_email,
            )

            # Attacher le PDF si fourni
            if pdf_path and Path(pdf_path).exists():
                with Path(pdf_path).open("rb") as pdf_file:
                    msg.attach(
                        filename=f"Facture_{invoice.invoice_number}.pdf",
                        content_type="application/pdf",
                        data=pdf_file.read(),
                    )
            else:
                logger.warning(
                    "PDF non trouvé pour la facture %s: %s",
                    invoice.invoice_number,
                    pdf_path,
                )

            # Envoyer l'email
            mail_instance.send(msg)

            logger.info(
                "Email de facture envoyé avec succès: %s → %s",
                invoice.invoice_number,
                normalized_email,
            )

            return EmailSendResult(
                success=True,
                recipient=normalized_email,
                error=None,
                sent_at=datetime.now(UTC),
            )

        except Exception as e:
            logger.error(
                "Erreur lors de l'envoi de l'email de facture %s: %s",
                invoice.invoice_number,
                e,
            )
            return EmailSendResult(
                success=False,
                recipient=normalized_email,
                error=f"Erreur d'envoi: {e!s}",
                sent_at=None,
            )

    def send_reminder_email(
        self,
        invoice: Any,
        reminder: Any,
        recipient_email: str,
        company: Any,
        client: Any,
        pdf_path: str | None = None,
        billing_settings: Any | None = None,
    ) -> EmailSendResult:
        """
        Envoie un email de rappel de paiement.

        Args:
            invoice: Instance de modèle Invoice
            reminder: Instance de modèle InvoiceReminder
            recipient_email: Email du destinataire
            company: Instance de modèle Company
            client: Instance de modèle Client
            pdf_path: Chemin vers le PDF du rappel (optionnel)
            billing_settings: CompanyBillingSettings (optionnel, pour SMTP custom)

        Returns:
            EmailSendResult avec le statut d'envoi
        """
        # Validation de l'email
        validation = self.validator.validate(recipient_email)
        if not validation["valid"]:
            return EmailSendResult(
                success=False,
                recipient=recipient_email,
                error=validation["error"],
                sent_at=None,
            )

        normalized_email = validation["normalized"]

        try:
            # Préparer le sujet
            reminder_level = getattr(reminder, "level", 1)
            subject = (
                f"RAPPEL N°{reminder_level} - Facture {invoice.invoice_number} - "
                f"{company.name if hasattr(company, 'name') else 'Votre entreprise'}"
            )

            # Préparer les données pour le template
            template_data = {
                "company_name": getattr(company, "name", "Votre entreprise"),
                "invoice_number": invoice.invoice_number,
                "reminder_level": reminder_level,
                "invoice_date": (
                    invoice.issued_at.strftime("%d.%m.%Y")
                    if invoice.issued_at
                    else "N/A"
                ),
                "original_due_date": (
                    invoice.due_date.strftime("%d.%m.%Y") if invoice.due_date else "N/A"
                ),
                "balance_due": float(invoice.balance_due),
                "reminder_fee": float(getattr(reminder, "added_fee", 0)),
                "currency": invoice.currency or "CHF",
                "client_name": self._format_client_name(client),
                "days_overdue": self._calculate_days_overdue(invoice),
            }

            # Rendu du template HTML
            html_body = render_template("emails/reminder_email.html", **template_data)

            # Obtenir l'instance Mail configurée (globale ou entreprise)
            mail_instance = self._get_mail_instance(billing_settings)

            # Déterminer l'expéditeur
            sender_email = (
                billing_settings.email_sender
                if billing_settings and billing_settings.email_sender
                else current_app.config.get("MAIL_DEFAULT_SENDER", "noreply@atmr.ch")
            )

            # Créer le message
            msg = Message(
                subject=subject,
                recipients=[normalized_email],
                html=html_body,
                sender=sender_email,
            )

            # Attacher le PDF si fourni
            if pdf_path and Path(pdf_path).exists():
                with Path(pdf_path).open("rb") as pdf_file:
                    msg.attach(
                        filename=f"Rappel_{reminder_level}_{invoice.invoice_number}.pdf",
                        content_type="application/pdf",
                        data=pdf_file.read(),
                    )

            # Envoyer l'email
            mail_instance.send(msg)

            logger.info(
                "Email de rappel N°%s envoyé avec succès: %s → %s",
                reminder_level,
                invoice.invoice_number,
                normalized_email,
            )

            return EmailSendResult(
                success=True,
                recipient=normalized_email,
                error=None,
                sent_at=datetime.now(UTC),
            )

        except Exception as e:
            logger.error(
                "Erreur lors de l'envoi de l'email de rappel %s: %s",
                invoice.invoice_number,
                e,
            )
            return EmailSendResult(
                success=False,
                recipient=normalized_email,
                error=f"Erreur d'envoi: {e!s}",
                sent_at=None,
            )

    def send_payment_confirmation_email(
        self,
        invoice: Any,
        payment: Any,
        recipient_email: str,
        company: Any,
        client: Any,
        billing_settings: Any | None = None,
    ) -> EmailSendResult:
        """
        Envoie un email de confirmation de paiement.

        Args:
            invoice: Instance de modèle Invoice
            payment: Instance de modèle InvoicePayment
            recipient_email: Email du destinataire
            company: Instance de modèle Company
            client: Instance de modèle Client
            billing_settings: CompanyBillingSettings (optionnel, pour SMTP custom)

        Returns:
            EmailSendResult avec le statut d'envoi
        """
        # Validation de l'email
        validation = self.validator.validate(recipient_email)
        if not validation["valid"]:
            return EmailSendResult(
                success=False,
                recipient=recipient_email,
                error=validation["error"],
                sent_at=None,
            )

        normalized_email = validation["normalized"]

        try:
            # Préparer le sujet
            subject = (
                f"Confirmation de paiement - Facture {invoice.invoice_number} - "
                f"{company.name if hasattr(company, 'name') else 'Votre entreprise'}"
            )

            # Préparer les données pour le template
            template_data = {
                "company_name": getattr(company, "name", "Votre entreprise"),
                "invoice_number": invoice.invoice_number,
                "payment_amount": float(getattr(payment, "amount", 0)),
                "payment_date": (
                    payment.paid_at.strftime("%d.%m.%Y")
                    if hasattr(payment, "paid_at") and payment.paid_at
                    else datetime.now(UTC).strftime("%d.%m.%Y")
                ),
                "payment_method": getattr(payment, "method", "N/A"),
                "remaining_balance": float(invoice.balance_due),
                "currency": invoice.currency or "CHF",
                "client_name": self._format_client_name(client),
                "is_fully_paid": float(invoice.balance_due) <= 0,
            }

            # Rendu du template HTML
            html_body = render_template(
                "emails/payment_confirmation_email.html", **template_data
            )

            # Obtenir l'instance Mail configurée (globale ou entreprise)
            mail_instance = self._get_mail_instance(billing_settings)

            # Déterminer l'expéditeur
            sender_email = (
                billing_settings.email_sender
                if billing_settings and billing_settings.email_sender
                else current_app.config.get("MAIL_DEFAULT_SENDER", "noreply@atmr.ch")
            )

            # Créer le message
            msg = Message(
                subject=subject,
                recipients=[normalized_email],
                html=html_body,
                sender=sender_email,
            )

            # Envoyer l'email
            mail_instance.send(msg)

            logger.info(
                "Email de confirmation de paiement envoyé: %s → %s",
                invoice.invoice_number,
                normalized_email,
            )

            return EmailSendResult(
                success=True,
                recipient=normalized_email,
                error=None,
                sent_at=datetime.now(UTC),
            )

        except Exception as e:
            logger.error(
                "Erreur lors de l'envoi de la confirmation de paiement %s: %s",
                invoice.invoice_number,
                e,
            )
            return EmailSendResult(
                success=False,
                recipient=normalized_email,
                error=f"Erreur d'envoi: {e!s}",
                sent_at=None,
            )

    def send_test_email(
        self, recipient_email: str, billing_settings: Any | None = None
    ) -> EmailSendResult:
        """
        Envoie un email de test pour vérifier la configuration SMTP.

        Args:
            recipient_email: Email du destinataire
            billing_settings: CompanyBillingSettings (optionnel, pour tester SMTP custom)

        Returns:
            EmailSendResult avec le statut d'envoi
        """
        # Validation de l'email
        validation = self.validator.validate(recipient_email)
        if not validation["valid"]:
            return EmailSendResult(
                success=False,
                recipient=recipient_email,
                error=validation["error"],
                sent_at=None,
            )

        normalized_email = validation["normalized"]

        try:
            # Obtenir l'instance Mail configurée (globale ou entreprise)
            mail_instance = self._get_mail_instance(billing_settings)

            # Déterminer l'expéditeur
            sender_email = (
                billing_settings.email_sender
                if billing_settings and billing_settings.email_sender
                else current_app.config.get("MAIL_DEFAULT_SENDER", "noreply@atmr.ch")
            )

            # Créer le message de test
            msg = Message(
                subject="Test ATMR - Configuration Email",
                recipients=[normalized_email],
                html=render_template("emails/test_email.html"),
                sender=sender_email,
            )

            # Envoyer l'email
            mail_instance.send(msg)

            logger.info("Email de test envoyé avec succès à: %s", normalized_email)

            return EmailSendResult(
                success=True,
                recipient=normalized_email,
                error=None,
                sent_at=datetime.now(UTC),
            )

        except Exception as e:
            logger.error("Erreur lors de l'envoi de l'email de test: %s", e)
            return EmailSendResult(
                success=False,
                recipient=normalized_email,
                error=f"Erreur d'envoi: {e!s}",
                sent_at=None,
            )

    # ========== Helpers privés ==========

    @staticmethod
    def _format_client_name(client: Any) -> str:
        """Formate le nom du client."""
        if hasattr(client, "is_institution") and client.is_institution:
            return getattr(client, "institution_name", "Client")

        first_name = getattr(client, "first_name", "")
        last_name = getattr(client, "last_name", "")

        if first_name and last_name:
            return f"{first_name} {last_name}"
        return first_name or last_name or "Client"

    @staticmethod
    def _calculate_days_overdue(invoice: Any) -> int:
        """Calcule le nombre de jours de retard."""
        if not hasattr(invoice, "due_date") or not invoice.due_date:
            return 0

        today = datetime.now(UTC).date()
        due_date = (
            invoice.due_date.date()
            if hasattr(invoice.due_date, "date")
            else invoice.due_date
        )

        if today > due_date:
            return (today - due_date).days

        return 0
