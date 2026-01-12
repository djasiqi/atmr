"""
Provider d'email transactionnel Brevo (ex-Sendinblue).

Architecture simple :
- Un seul provider centralisé
- Validation de domaine (SPF/DKIM)
- Envoi avec From personnalisé par entreprise

Usage :
    provider = BrevoEmailProvider()
    result = provider.send_invoice_email(
        from_email="noreply@entreprise.ch",
        from_name="Mon Entreprise",
        to_email="client@example.com",
        to_name="Client",
        subject="Facture #2024-001",
        html_content="<html>...</html>",
        attachments=[{"filename": "facture.pdf", "content": pdf_bytes}]
    )
"""

import base64
import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

# HTTP Status Codes constants
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_NOT_FOUND = 404


@dataclass
class EmailResult:
    """Résultat d'envoi d'email."""

    success: bool
    message_id: str | None = None
    error: str | None = None
    provider_response: dict[str, Any] | None = None


@dataclass
class DomainVerificationResult:
    """Résultat de vérification de domaine."""

    verified: bool
    domain: str
    spf_record: str | None = None
    dkim_record: str | None = None
    error: str | None = None


class BrevoEmailProvider:
    """
    Provider d'email transactionnel Brevo.

    Fonctionnalités :
    - Envoi d'emails transactionnels avec pièces jointes
    - Validation de domaine (SPF/DKIM)
    - Support multi-tenant (from_email par entreprise)
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialise le provider Brevo.

        Args:
            api_key: Clé API Brevo (si None, utilise env var BREVO_API_KEY)
        """
        super().__init__()

        self.api_key = api_key or os.getenv("BREVO_API_KEY")
        if not self.api_key:
            error_msg = (
                "Brevo API key manquante. "
                "Définir BREVO_API_KEY dans les variables d'environnement."
            )
            raise ValueError(error_msg)

        self.base_url = "https://api.brevo.com/v3"
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "api-key": self.api_key,
        }

    def send_invoice_email(
        self,
        from_email: str,
        from_name: str,
        to_email: str,
        to_name: str,
        subject: str,
        html_content: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> EmailResult:
        """
        Envoie un email de facture via Brevo.

        Args:
            from_email: Adresse d'envoi (ex: noreply@entreprise.ch)
            from_name: Nom d'expéditeur (ex: "Mon Entreprise")
            to_email: Destinataire
            to_name: Nom du destinataire
            subject: Sujet de l'email
            html_content: Contenu HTML de l'email
            attachments: Liste de pièces jointes
                Format: [{"filename": "facture.pdf", "content": bytes}]

        Returns:
            EmailResult avec succès/erreur
        """
        try:
            # Préparer les pièces jointes pour Brevo
            brevo_attachments = []
            if attachments:
                for attachment in attachments:
                    filename = attachment.get("filename", "attachment.pdf")
                    content = attachment.get("content")

                    if isinstance(content, bytes):
                        # Encoder en base64
                        content_b64 = base64.b64encode(content).decode("utf-8")
                    elif isinstance(content, str):
                        # Déjà encodé en base64
                        content_b64 = content
                    else:
                        logger.warning(
                            "Type de contenu pièce jointe non supporté : %s",
                            type(content),
                        )
                        continue

                    brevo_attachments.append({"name": filename, "content": content_b64})

            # Construire la requête Brevo
            payload = {
                "sender": {"email": from_email, "name": from_name},
                "to": [{"email": to_email, "name": to_name}],
                "subject": subject,
                "htmlContent": html_content,
            }

            if brevo_attachments:
                payload["attachment"] = brevo_attachments

            # Envoi via API Brevo
            logger.info(
                "Envoi email via Brevo : %s -> %s (sujet: %s)",
                from_email,
                to_email,
                subject,
            )

            response = requests.post(
                f"{self.base_url}/smtp/email",
                json=payload,
                headers=self.headers,
                timeout=30,
            )

            if response.status_code == HTTP_CREATED:
                data = response.json()
                message_id = data.get("messageId")
                logger.info("✅ Email envoyé avec succès : messageId=%s", message_id)
                return EmailResult(
                    success=True,
                    message_id=message_id,
                    provider_response=data,
                )

            error_msg = f"Erreur Brevo {response.status_code}: {response.text}"
            logger.error("❌ Échec envoi email : %s", error_msg)
            return EmailResult(
                success=False,
                error=error_msg,
                provider_response=response.json() if response.text else None,
            )

        except requests.exceptions.RequestException as e:
            error_msg = f"Erreur réseau Brevo : {e!s}"
            logger.error("❌ Échec envoi email : %s", error_msg)
            return EmailResult(success=False, error=error_msg)
        except Exception as e:
            error_msg = f"Erreur inattendue : {e!s}"
            logger.exception("❌ Échec envoi email : %s", error_msg)
            return EmailResult(success=False, error=error_msg)

    def verify_domain(self, domain: str) -> DomainVerificationResult:
        """
        Vérifie si un domaine est validé dans Brevo.

        Note: Cette méthode interroge l'API Brevo pour vérifier
        si le domaine a été ajouté et validé (SPF/DKIM configurés).

        Args:
            domain: Domaine à vérifier (ex: "entreprise.ch")

        Returns:
            DomainVerificationResult avec statut et enregistrements DNS
        """
        # #region agent log
        logger.info("🔍 [DEBUG] verify_domain: START - domain=%s", domain)
        # #endregion

        try:
            # #region agent log
            logger.info(
                "🔍 [DEBUG] verify_domain: Sending GET request to %s/senders/domains/%s",
                self.base_url,
                domain,
            )
            # #endregion

            # Essayer de récupérer les détails du domaine directement
            response = requests.get(
                f"{self.base_url}/senders/domains/{domain}",
                headers=self.headers,
                timeout=10,
            )

            # #region agent log
            logger.info(
                "🔍 [DEBUG] verify_domain: Got response status=%s", response.status_code
            )
            # #endregion

            if response.status_code == HTTP_OK:
                # Domaine trouvé
                data = response.json()

                # #region agent log
                logger.info("🔍 [DEBUG] verify_domain: Got 200 response from Brevo")
                logger.info(
                    "🔍 [DEBUG] verify_domain: Response data keys: %s", data.keys()
                )
                # #endregion

                verified = data.get("verified", False) or data.get(
                    "authenticated", False
                )

                # Extraire les enregistrements DNS
                dns_records = data.get("dns_records", {})

                # #region agent log
                logger.info(
                    "🔍 [DEBUG] verify_domain: dns_records from API: %s", dns_records
                )
                logger.info(
                    "🔍 [DEBUG] verify_domain: dns_records keys: %s",
                    dns_records.keys() if dns_records else "None",
                )
                # #endregion

                # Formater les enregistrements DNS pour affichage
                dkim1 = dns_records.get("dkim1Record", {})
                dkim2 = dns_records.get("dkim2Record", {})
                brevo_code = dns_records.get("brevo_code", {})

                # #region agent log
                logger.info(
                    "🔍 [DEBUG] verify_domain: dkim1=%s, dkim2=%s, brevo_code=%s",
                    dkim1,
                    dkim2,
                    brevo_code,
                )
                # #endregion

                # Formater pour affichage
                dkim_instructions = []
                if dkim1:
                    dkim_instructions.append(
                        f"Type: CNAME\nHôte: {dkim1.get('host_name')}\nValeur: {dkim1.get('value')}"
                    )
                if dkim2:
                    dkim_instructions.append(
                        f"Type: CNAME\nHôte: {dkim2.get('host_name')}\nValeur: {dkim2.get('value')}"
                    )

                spf_txt = (
                    f"Type: TXT\nHôte: {brevo_code.get('host_name', '@')}\nValeur: {brevo_code.get('value')}"
                    if brevo_code
                    else None
                )
                dkim_txt = "\n\n".join(dkim_instructions) if dkim_instructions else None

                # #region agent log
                logger.info(
                    "🔍 [DEBUG] verify_domain: Final spf_txt=%s, dkim_txt=%s",
                    bool(spf_txt),
                    bool(dkim_txt),
                )
                # #endregion

                logger.info(
                    "Domaine %s : verified=%s, SPF=%s, DKIM=%s",
                    domain,
                    verified,
                    bool(spf_txt),
                    bool(dkim_txt),
                )

                return DomainVerificationResult(
                    verified=verified,
                    domain=domain,
                    spf_record=spf_txt,
                    dkim_record=dkim_txt,
                )

            if response.status_code == HTTP_NOT_FOUND:
                # Domaine non trouvé - essayer de l'ajouter automatiquement
                # #region agent log
                logger.info(
                    "🔍 [DEBUG] verify_domain: Got 404, trying to add domain automatically"
                )
                # #endregion

                logger.info(
                    "Domaine %s non trouvé, tentative d'ajout automatique", domain
                )
                add_result = self._add_domain_to_brevo(domain)

                if add_result:
                    # #region agent log
                    logger.info(
                        "🔍 [DEBUG] verify_domain: Domain added successfully, returning result"
                    )
                    # #endregion
                    return add_result

                # Si l'ajout a échoué
                # #region agent log
                logger.warning(
                    "🔍 [DEBUG] verify_domain: Domain addition failed, returning error result"
                )
                # #endregion

                return DomainVerificationResult(
                    verified=False,
                    domain=domain,
                    error="Domaine non configuré dans Brevo et ajout automatique échoué",
                )

            # #region agent log
            logger.error(
                "🔍 [DEBUG] verify_domain: Got unexpected status code %s, response: %s",
                response.status_code,
                response.text[:200],
            )
            # #endregion

            return DomainVerificationResult(
                verified=False,
                domain=domain,
                error=f"Erreur API Brevo : {response.status_code}",
            )

        except Exception as e:
            # #region agent log
            logger.error("🔍 [DEBUG] verify_domain: Exception occurred: %s", str(e))
            # #endregion

            logger.exception("Erreur vérification domaine %s : %s", domain, e)
            return DomainVerificationResult(verified=False, domain=domain, error=str(e))

    def _add_domain_to_brevo(self, domain: str) -> DomainVerificationResult | None:
        """
        Ajoute un domaine dans Brevo pour obtenir les enregistrements DNS.

        Args:
            domain: Domaine à ajouter

        Returns:
            DomainVerificationResult avec les enregistrements DNS, ou None si échec
        """
        try:
            payload = {"name": domain}
            logger.info("Tentative d'ajout du domaine %s dans Brevo", domain)

            response = requests.post(
                f"{self.base_url}/senders/domains",
                headers=self.headers,
                json=payload,
                timeout=10,
            )

            logger.info("Réponse Brevo : status=%s", response.status_code)

            if response.status_code == HTTP_OK:  # Brevo retourne 200, pas 201
                # Domaine ajouté avec succès, récupérer les infos
                data = response.json()
                dns_records = data.get("dns_records", {})

                # Extraire les enregistrements
                dkim1 = dns_records.get("dkim1Record", {})
                dkim2 = dns_records.get("dkim2Record", {})
                brevo_code = dns_records.get("brevo_code", {})

                # Formater pour affichage
                dkim_instructions = []
                if dkim1:
                    dkim_instructions.append(
                        f"Type: CNAME\nHôte: {dkim1.get('host_name')}\nValeur: {dkim1.get('value')}"
                    )
                if dkim2:
                    dkim_instructions.append(
                        f"Type: CNAME\nHôte: {dkim2.get('host_name')}\nValeur: {dkim2.get('value')}"
                    )

                spf_txt = (
                    f"Type: TXT\nHôte: {brevo_code.get('host_name', '@')}\nValeur: {brevo_code.get('value')}"
                    if brevo_code
                    else None
                )
                dkim_txt = "\n\n".join(dkim_instructions) if dkim_instructions else None

                logger.info(
                    "✅ Domaine %s ajouté dans Brevo",
                    domain,
                )

                return DomainVerificationResult(
                    verified=False,  # Pas encore vérifié
                    domain=domain,
                    spf_record=spf_txt,
                    dkim_record=dkim_txt,
                )

            logger.warning(
                "Échec ajout domaine %s dans Brevo : %s - %s",
                domain,
                response.status_code,
                response.text,
            )
            return None

        except Exception as e:
            logger.exception("Erreur ajout domaine %s dans Brevo : %s", domain, e)
            return None

    def get_domain_dns_records(self, domain: str) -> dict[str, str] | None:
        """
        Récupère les enregistrements DNS à configurer pour un domaine.

        Args:
            domain: Domaine à configurer

        Returns:
            Dict avec 'spf' et 'dkim' à copier, ou None si erreur
        """
        # #region agent log
        logger.info("🔍 [DEBUG] get_domain_dns_records called for domain: %s", domain)
        # #endregion

        result = self.verify_domain(domain)

        # #region agent log
        logger.info(
            "🔍 [DEBUG] verify_domain returned - verified=%s, spf_exists=%s, dkim_exists=%s",
            result.verified,
            bool(result.spf_record),
            bool(result.dkim_record),
        )
        logger.info("🔍 [DEBUG] SPF value: %s", result.spf_record)
        logger.info("🔍 [DEBUG] DKIM value: %s", result.dkim_record)
        # #endregion

        if result.spf_record or result.dkim_record:
            dns_dict = {
                "spf": result.spf_record or "Aucun enregistrement SPF trouvé",
                "dkim": result.dkim_record or "Aucun enregistrement DKIM trouvé",
            }

            # #region agent log
            logger.info("🔍 [DEBUG] Returning DNS dict: %s", dns_dict)
            # #endregion

            return dns_dict

        # #region agent log
        logger.warning("🔍 [DEBUG] Returning None - no DNS records found!")
        # #endregion

        return None

    def test_connection(self) -> bool:
        """
        Test la connexion à l'API Brevo.

        Returns:
            True si la connexion fonctionne, False sinon
        """
        try:
            response = requests.get(
                f"{self.base_url}/account",
                headers=self.headers,
                timeout=10,
            )
            return response.status_code == HTTP_OK
        except Exception as e:
            logger.error("Test connexion Brevo échoué : %s", e)
            return False
