"""Use-case: envoyer une facture par email.

Ce use case gère l'envoi d'une facture par email à un client,
incluant la validation de l'email, la génération du PDF si nécessaire,
et le marquage de la facture comme envoyée.

✅ Refactoré pour utiliser Brevo (service transactionnel) au lieu de SMTP.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ext import db
from models import Client, Company, CompanyBillingSettings, Invoice
from services.documents.pdf import PDFService
from services.email.brevo_provider import BrevoEmailProvider
from services.email.recipient_utils import normalize_relationship_label
from services.email.signature_utils import inject_signature_into_html

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
                pdf_url = self.pdf_service.generate_invoice_pdf(
                    invoice, force_regenerate=input_data.force_regenerate_pdf
                )
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

            # Déterminer le destinataire réel (clinique / tiers payeur / patient)
            recipient_name = client_name
            recipient_type = "patient"
            billing_party = getattr(invoice, "billing_party", None)
            billed_company = getattr(invoice, "billed_to_company", None)
            bill_to_client = getattr(invoice, "bill_to_client", None)
            relationship_label = None

            if billing_party and invoice.client_id:
                try:
                    from models.billing_party import ClientBillingParty

                    link = ClientBillingParty.query.filter_by(
                        client_id=invoice.client_id,
                        billing_party_id=billing_party.id,
                    ).first()
                    relationship_label = getattr(link, "role", None)
                except Exception:
                    relationship_label = None

            if billing_party:
                recipient_name = billing_party.display_name or client_name
                recipient_type = getattr(billing_party.type, "value", billing_party.type)
            elif billed_company:
                recipient_name = billed_company.name or client_name
                recipient_type = "clinic"
            elif bill_to_client and getattr(bill_to_client, "is_institution", False):
                institution_name = getattr(bill_to_client, "institution_name", None)
                recipient_name = institution_name or client_name
                recipient_type = "clinic"

            is_clinic_recipient = str(recipient_type).lower() in {
                "clinic",
                "hospital",
                "ems",
            }
            is_family_recipient = str(recipient_type).lower() == "family"
            is_curator_recipient = str(recipient_type).lower() == "curatorship"
            is_insurance_recipient = str(recipient_type).lower() == "insurance"
            relationship_display = normalize_relationship_label(relationship_label)

            # Utiliser le template de message s'il existe, sinon message par défaut
            template = (
                billing_settings.invoice_message_template
                if billing_settings and billing_settings.invoice_message_template
                else None
            )

            if template:
                # Remplacer les variables du template
                html_content = template.replace("{client_name}", client_name)
                html_content = html_content.replace("{recipient_name}", recipient_name)
                html_content = html_content.replace("{payer_name}", recipient_name)
                html_content = html_content.replace(
                    "{recipient_type}",
                    str(recipient_type),
                )
                html_content = html_content.replace("{patient_name}", client_name)
                html_content = html_content.replace(
                    "{relationship_label}",
                    relationship_display or "",
                )
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
                if is_clinic_recipient:
                    recipient_line = (
                        "Veuillez trouver ci-joint la facture "
                        f"<strong>{invoice.invoice_number}</strong> d'un montant de "
                        f"<strong>{invoice.total_amount:.2f} CHF</strong> pour "
                        "les transports des patients pris en charge."
                    )
                elif is_family_recipient:
                    if relationship_display:
                        recipient_line = (
                            "Veuillez trouver ci-joint la facture "
                            f"<strong>{invoice.invoice_number}</strong> d'un montant de "
                            f"<strong>{invoice.total_amount:.2f} CHF</strong> "
                            f"pour votre {relationship_display} <strong>{client_name}</strong>."
                        )
                    else:
                        recipient_line = (
                            "Veuillez trouver ci-joint la facture "
                            f"<strong>{invoice.invoice_number}</strong> d'un montant de "
                            f"<strong>{invoice.total_amount:.2f} CHF</strong> "
                            f"concernant <strong>{client_name}</strong>."
                        )
                elif is_curator_recipient:
                    recipient_line = (
                        "Veuillez trouver ci-joint la facture "
                        f"<strong>{invoice.invoice_number}</strong> d'un montant de "
                        f"<strong>{invoice.total_amount:.2f} CHF</strong> "
                        f"pour la personne protégée <strong>{client_name}</strong>."
                    )
                elif is_insurance_recipient:
                    recipient_line = (
                        "Veuillez trouver ci-joint la facture "
                        f"<strong>{invoice.invoice_number}</strong> d'un montant de "
                        f"<strong>{invoice.total_amount:.2f} CHF</strong> "
                        f"pour l'assuré <strong>{client_name}</strong>."
                    )
                else:
                    recipient_line = (
                        "Veuillez trouver ci-joint la facture "
                        f"<strong>{invoice.invoice_number}</strong> d'un montant de "
                        f"<strong>{invoice.total_amount:.2f} CHF</strong>."
                    )

                html_content = f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                    <p>Bonjour {recipient_name},</p>
                    <p>{recipient_line}</p>
                    <p>Date d'échéance : <strong>{invoice.due_date.strftime("%d/%m/%Y") if invoice.due_date else "À définir"}</strong></p>
                    <p>Merci de procéder au paiement dans les délais.</p>
                    <br>
                    <p>Cordialement,<br><strong>{company.name}</strong></p>
                </body>
                </html>
                """

            # 8.5. Injecter la signature email si configurée
            provider_mode = (
                (os.getenv("EMAIL_PROVIDER_MODE", "brevo_api") or "brevo_api")
                .strip()
                .lower()
            )
            logo_mode = "url" if provider_mode == "brevo_api" else "cid"
            cache_bust = str(invoice.id) if logo_mode == "url" else None

            logo_info: dict[str, Any] | None = None
            if billing_settings:
                html_content, logo_info = inject_signature_into_html(
                    html_content,
                    company=company,
                    billing_settings=billing_settings,
                    logo_mode=logo_mode,
                    cache_bust=cache_bust,
                )

            # 9. Préparer les attachements (PDF + logo inline si mode CID)
            EMAIL_SIGNATURE_DEBUG = os.getenv("EMAIL_SIGNATURE_DEBUG", "0") == "1"
            if EMAIL_SIGNATURE_DEBUG:
                logger.info(
                    "[EMAIL_SIGNATURE_DEBUG] send_invoice_by_email: provider_mode=%s, logo_mode=%s",
                    provider_mode,
                    logo_mode,
                )
            attachments = []
            if pdf_bytes:
                attachments.append(
                    {
                        "filename": f"facture_{invoice.invoice_number}.pdf",
                        "content": pdf_bytes,
                    }
                )
            # Logo inline pour signature (CID)
            if logo_info:
                # Vérifier que logo_info est valide avant d'ajouter
                if not logo_info.get("bytes") or len(logo_info.get("bytes", b"")) == 0:
                    logger.warning(
                        "[INVOICE EMAIL] Logo bytes vides pour facture %s - logo inline ignoré",
                        invoice.invoice_number,
                    )
                elif logo_info.get("cid") != "company_logo":
                    logger.warning(
                        "[INVOICE EMAIL] CID inattendu: %s (attendu: company_logo) - logo inline ignoré",
                        logo_info.get("cid"),
                    )
                else:
                    attachments.append(
                        {
                            "filename": logo_info["filename"],
                            "content": logo_info["bytes"],
                            "cid": logo_info["cid"],  # Doit être "company_logo"
                            "mime_type": logo_info["mime_type"],
                        }
                    )
                    if EMAIL_SIGNATURE_DEBUG:
                        logger.info(
                            (
                                "[EMAIL_SIGNATURE_DEBUG] send_invoice_by_email: logo inline ajouté - "
                                "cid=%s, filename=%s, mime_type=%s, bytes_len=%d"
                            ),
                            logo_info["cid"],
                            logo_info["filename"],
                            logo_info["mime_type"],
                            len(logo_info["bytes"]),
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
                to_name=recipient_name,
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
