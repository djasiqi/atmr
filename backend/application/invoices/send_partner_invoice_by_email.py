"""Use-case: envoyer une facture partenaire par email.

Ce use case gère l'envoi d'une facture partenaire par email à l'entreprise
partenaire, incluant la validation de l'email, la régénération du PDF si demandé,
et le marquage de la facture comme envoyée.

Utilise Brevo (service transactionnel) pour l'envoi.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ext import db
from models import Company, CompanyBillingSettings
from models.partner_invoice import PartnerInvoice, PartnerInvoiceStatus
from services.email.brevo_provider import BrevoEmailProvider
from services.email.signature_utils import inject_signature_into_html
from services.partnerships.invoices import PartnerInvoiceService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SendPartnerInvoiceByEmailInput:
    """Input pour l'envoi d'une facture partenaire par email."""

    partner_invoice_id: int
    company_id: int  # Entreprise qui envoie (executing_company)
    recipient_email: str | None = None  # Si None, utilise partner_company.contact_email
    force_regenerate_pdf: bool = False  # Regénérer le PDF même s'il existe


@dataclass(frozen=True, slots=True)
class SendPartnerInvoiceByEmailResult:
    """Résultat de l'envoi d'une facture partenaire par email."""

    success: bool
    partner_invoice_id: int
    recipient: str | None = None
    sent_at: datetime | None = None
    error: str | None = None
    status_code: int = 200


class SendPartnerInvoiceByEmailUseCase:
    """Use-case Application: envoyer une facture partenaire par email via Brevo."""

    def __init__(self) -> None:  # type: ignore[reportMissingSuperCall]
        self.brevo_provider = BrevoEmailProvider()
        self.partner_invoice_service = PartnerInvoiceService()

    def execute(  # noqa: PLR0911 - Multiple early returns for validation steps
        self, input_data: SendPartnerInvoiceByEmailInput
    ) -> SendPartnerInvoiceByEmailResult:
        """
        Envoie une facture partenaire par email.

        Étapes:
        1. Valider que la facture partenaire existe
        2. Vérifier que l'entreprise est autorisée (executing_company)
        3. Charger le partenariat et déterminer le destinataire
        4. Déterminer l'email du destinataire
        5. Régénérer le PDF si demandé
        6. Charger les paramètres de facturation (config Brevo)
        7. Générer le contenu HTML de l'email
        8. Envoyer l'email
        9. Marquer la facture comme envoyée
        10. Persister les changements

        Args:
            input_data: Données d'entrée avec partner_invoice_id et recipient_email

        Returns:
            SendPartnerInvoiceByEmailResult avec le statut d'envoi
        """
        try:
            # 1. Valider que la facture partenaire existe
            partner_invoice = PartnerInvoice.query.get(input_data.partner_invoice_id)
            if not partner_invoice:
                return SendPartnerInvoiceByEmailResult(
                    success=False,
                    partner_invoice_id=input_data.partner_invoice_id,
                    error=f"Facture partenaire #{input_data.partner_invoice_id} introuvable",
                    status_code=404,
                )

            # 2. Vérifier que l'entreprise est autorisée (executing_company)
            if partner_invoice.executing_company_id != input_data.company_id:
                return SendPartnerInvoiceByEmailResult(
                    success=False,
                    partner_invoice_id=input_data.partner_invoice_id,
                    error="Seule l'entreprise exécutante peut envoyer cette facture",
                    status_code=403,
                )

            # 3. Charger le partenariat et les entreprises
            partnership = partner_invoice.partnership
            if not partnership:
                return SendPartnerInvoiceByEmailResult(
                    success=False,
                    partner_invoice_id=input_data.partner_invoice_id,
                    error="Partenariat introuvable pour cette facture",
                    status_code=404,
                )

            # Déterminer l'entreprise destinataire (l'autre entreprise du partenariat)
            executing_company = Company.query.get(partner_invoice.executing_company_id)
            if not executing_company:
                return SendPartnerInvoiceByEmailResult(
                    success=False,
                    partner_invoice_id=input_data.partner_invoice_id,
                    error="Entreprise exécutante introuvable",
                    status_code=404,
                )

            # L'entreprise destinataire est celle qui doit payer (pas celle qui exécute)
            # Si executing_company == owner_company -> destinataire = partner_company
            # Si executing_company == partner_company -> destinataire = owner_company
            if partner_invoice.executing_company_id == partnership.owner_company_id:
                billed_company = partnership.partner_company
            else:
                billed_company = partnership.owner_company

            if not billed_company:
                return SendPartnerInvoiceByEmailResult(
                    success=False,
                    partner_invoice_id=input_data.partner_invoice_id,
                    error="Entreprise destinataire (facturée) introuvable",
                    status_code=404,
                )

            # 4. Déterminer l'email du destinataire
            recipient_email = (
                input_data.recipient_email
                or billed_company.billing_email
                or billed_company.contact_email
            )
            if not recipient_email:
                return SendPartnerInvoiceByEmailResult(
                    success=False,
                    partner_invoice_id=input_data.partner_invoice_id,
                    error=(
                        f"Aucune adresse email disponible pour {billed_company.name}. "
                        "Veuillez ajouter un email de contact ou de facturation, "
                        "ou spécifier un destinataire."
                    ),
                    status_code=400,
                )

            # 5. Régénérer le PDF si demandé ou s'il n'existe pas
            pdf_path = None
            logger.info(
                "[PARTNER INVOICE EMAIL] partner_invoice.pdf_url=%s, force_regenerate=%s",
                partner_invoice.pdf_url,
                input_data.force_regenerate_pdf,
            )

            if not partner_invoice.pdf_url or input_data.force_regenerate_pdf:
                logger.info(
                    "Régénération du PDF pour la facture partenaire %s",
                    partner_invoice.invoice_number,
                )
                try:
                    pdf_url = self.partner_invoice_service.regenerate_pdf(
                        partner_invoice.id
                    )
                    if pdf_url:
                        partner_invoice.pdf_url = pdf_url
                        db.session.commit()
                        # Recharger l'objet après régénération
                        db.session.refresh(partner_invoice)
                except Exception as e:
                    logger.warning(
                        "Impossible de régénérer le PDF pour la facture partenaire %s: %s",
                        partner_invoice.invoice_number,
                        e,
                    )

            # Convertir l'URL en chemin système
            if partner_invoice.pdf_url:
                from flask import current_app

                uploads_dir = Path(
                    current_app.config.get("UPLOAD_FOLDER", "/app/uploads")
                )

                # Extraire le chemin relatif depuis l'URL
                if partner_invoice.pdf_url.startswith("/uploads/"):
                    relative_path = partner_invoice.pdf_url.removeprefix("/uploads/")
                elif "/uploads/" in partner_invoice.pdf_url:
                    relative_path = partner_invoice.pdf_url.split("/uploads/", 1)[1]
                else:
                    logger.warning(
                        "[PARTNER INVOICE EMAIL] Format d'URL inattendu: %s",
                        partner_invoice.pdf_url,
                    )
                    relative_path = None

                if relative_path:
                    pdf_path = str(uploads_dir / relative_path)
                    logger.info(
                        "[PARTNER INVOICE EMAIL] PDF path: %s -> %s",
                        partner_invoice.pdf_url,
                        pdf_path,
                    )

            # 6. Charger les paramètres de facturation (pour config Brevo)
            billing_settings = CompanyBillingSettings.query.filter_by(
                company_id=executing_company.id
            ).first()

            if not billing_settings:
                return SendPartnerInvoiceByEmailResult(
                    success=False,
                    partner_invoice_id=input_data.partner_invoice_id,
                    recipient=recipient_email,
                    error=(
                        "Paramètres de facturation non configurés. "
                        "Veuillez configurer l'email d'envoi dans les paramètres."
                    ),
                    status_code=400,
                )

            # Vérifier que le domaine email est validé dans Brevo
            from_email = billing_settings.smtp_username
            from_name = billing_settings.from_name or executing_company.name

            if not from_email:
                return SendPartnerInvoiceByEmailResult(
                    success=False,
                    partner_invoice_id=input_data.partner_invoice_id,
                    recipient=recipient_email,
                    error=(
                        "Adresse email d'envoi non configurée. "
                        "Veuillez configurer l'email d'envoi dans Paramètres > Facturation."
                    ),
                    status_code=400,
                )

            if not billing_settings.domain_verified:
                return SendPartnerInvoiceByEmailResult(
                    success=False,
                    partner_invoice_id=input_data.partner_invoice_id,
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
                "[PARTNER INVOICE EMAIL] pdf_path=%s, exists=%s",
                pdf_path,
                Path(pdf_path).exists() if pdf_path else False,
            )
            if pdf_path and Path(pdf_path).exists():
                try:
                    with Path(pdf_path).open("rb") as f:
                        pdf_bytes = f.read()
                    logger.info(
                        "[PARTNER INVOICE EMAIL] PDF lu avec succès, taille=%s bytes",
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
                    "[PARTNER INVOICE EMAIL] Aucun PDF disponible pour la facture %s (pdf_path=%s)",
                    partner_invoice.invoice_number,
                    pdf_path,
                )

            # 8. Générer le contenu HTML de l'email
            recipient_name = billed_company.name or "Partenaire"

            # Formater la période (ex: "janvier 2026")
            MONTHS_FR = [
                "",
                "janvier",
                "février",
                "mars",
                "avril",
                "mai",
                "juin",
                "juillet",
                "août",
                "septembre",
                "octobre",
                "novembre",
                "décembre",
            ]
            period_str = (
                f"{MONTHS_FR[partner_invoice.period_month]} {partner_invoice.period_year}"
            )

            # Utiliser le template de message s'il existe, sinon message par défaut
            template = (
                billing_settings.invoice_message_template
                if billing_settings and billing_settings.invoice_message_template
                else None
            )

            if template:
                # Remplacer les variables du template
                html_content = template.replace("{partner_name}", recipient_name)
                html_content = html_content.replace("{recipient_name}", recipient_name)
                html_content = html_content.replace(
                    "{invoice_number}", partner_invoice.invoice_number or ""
                )
                html_content = html_content.replace("{period}", period_str)
                html_content = html_content.replace(
                    "{amount}",
                    f"{partner_invoice.total_amount:.2f}"
                    if partner_invoice.total_amount
                    else "0.00",
                )
                html_content = html_content.replace(
                    "{due_date}",
                    partner_invoice.due_date.strftime("%d/%m/%Y")
                    if partner_invoice.due_date
                    else "À définir",
                )
                # Convertir les sauts de ligne en <br>
                html_content = html_content.replace("\n", "<br>")
            else:
                # Message par défaut pour factures partenaires
                html_content = f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                    <p>Bonjour,</p>
                    <p>Veuillez trouver ci-joint la facture partenaire
                    <strong>{partner_invoice.invoice_number}</strong>
                    pour la période <strong>{period_str}</strong>,
                    d'un montant de <strong>{partner_invoice.total_amount:.2f} CHF</strong>.</p>
                    <p>Date d'échéance : <strong>{partner_invoice.due_date.strftime("%d/%m/%Y") if partner_invoice.due_date else "À définir"}</strong></p>
                    <p>Merci de procéder au paiement dans les délais convenus.</p>
                    <br>
                    <p>Cordialement,<br><strong>{executing_company.name}</strong></p>
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
            cache_bust = str(partner_invoice.id) if logo_mode == "url" else None

            logo_info: dict[str, Any] | None = None
            if billing_settings:
                html_content, logo_info = inject_signature_into_html(
                    html_content,
                    company=executing_company,
                    billing_settings=billing_settings,
                    logo_mode=logo_mode,
                    cache_bust=cache_bust,
                )

            # 9. Préparer les attachements (PDF + logo inline si mode CID)
            EMAIL_SIGNATURE_DEBUG = os.getenv("EMAIL_SIGNATURE_DEBUG", "0") == "1"
            if EMAIL_SIGNATURE_DEBUG:
                logger.info(
                    "[EMAIL_SIGNATURE_DEBUG] send_partner_invoice_by_email: provider_mode=%s, logo_mode=%s",
                    provider_mode,
                    logo_mode,
                )
            attachments = []
            if pdf_bytes:
                attachments.append(
                    {
                        "filename": f"facture_{partner_invoice.invoice_number}.pdf",
                        "content": pdf_bytes,
                    }
                )
            # Logo inline pour signature (CID)
            if logo_info:
                if not logo_info.get("bytes") or len(logo_info.get("bytes", b"")) == 0:
                    logger.warning(
                        "[PARTNER INVOICE EMAIL] Logo bytes vides - logo inline ignoré",
                    )
                elif logo_info.get("cid") != "company_logo":
                    logger.warning(
                        "[PARTNER INVOICE EMAIL] CID inattendu: %s - logo inline ignoré",
                        logo_info.get("cid"),
                    )
                else:
                    attachments.append(
                        {
                            "filename": logo_info["filename"],
                            "content": logo_info["bytes"],
                            "cid": logo_info["cid"],
                            "mime_type": logo_info["mime_type"],
                        }
                    )

            # 10. Envoyer l'email via Brevo
            logger.info(
                "[PARTNER INVOICE EMAIL] Envoi de la facture %s par email via Brevo à %s (depuis %s) avec %d attachement(s)",
                partner_invoice.invoice_number,
                recipient_email,
                from_email,
                len(attachments),
            )

            email_result = self.brevo_provider.send_invoice_email(
                from_email=from_email,
                from_name=from_name,
                to_email=recipient_email,
                to_name=recipient_name,
                subject=f"Facture partenaire {partner_invoice.invoice_number} - {executing_company.name}",
                html_content=html_content,
                attachments=attachments,
            )

            if not email_result.success:
                logger.error(
                    "Échec de l'envoi de la facture partenaire %s via Brevo: %s",
                    partner_invoice.invoice_number,
                    email_result.error,
                )
                return SendPartnerInvoiceByEmailResult(
                    success=False,
                    partner_invoice_id=input_data.partner_invoice_id,
                    recipient=recipient_email,
                    error=f"Erreur Brevo: {email_result.error}",
                    status_code=500,
                )

            # 11. Marquer la facture comme envoyée
            if partner_invoice.status == PartnerInvoiceStatus.DRAFT:
                partner_invoice.status = PartnerInvoiceStatus.SENT
                partner_invoice.sent_at = datetime.now()
            db.session.commit()

            logger.info(
                "✅ Facture partenaire %s envoyée avec succès via Brevo à %s (message_id: %s)",
                partner_invoice.invoice_number,
                recipient_email,
                email_result.message_id or "N/A",
            )

            return SendPartnerInvoiceByEmailResult(
                success=True,
                partner_invoice_id=input_data.partner_invoice_id,
                recipient=recipient_email,
                sent_at=datetime.now(),
            )

        except Exception as e:
            logger.exception(
                "Erreur lors de l'envoi de la facture partenaire %s par email",
                input_data.partner_invoice_id,
            )
            db.session.rollback()
            return SendPartnerInvoiceByEmailResult(
                success=False,
                partner_invoice_id=input_data.partner_invoice_id,
                error=f"Erreur inattendue: {e!s}",
                status_code=500,
            )
