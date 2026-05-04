"""
Routes API pour la configuration des emails transactionnels (Brevo).

Ces routes permettent de :
1. Configurer l'adresse email d'envoi (from_email + from_name)
2. Récupérer les enregistrements DNS à configurer (SPF + DKIM)
3. Vérifier que le domaine est validé dans Brevo
"""

import logging
import os
import re
import socket
from http import HTTPStatus

from flask import current_app, request
from flask_jwt_extended import jwt_required  # pyright: ignore
from flask_restx import Namespace, Resource, fields  # pyright: ignore

from ext import db, role_required
from models import UserRole
from models.invoice import CompanyBillingSettings
from routes.companies import get_company_from_token
from services.email.brevo_provider import BrevoEmailProvider

logger = logging.getLogger(__name__)

# Constants
MAX_DOMAIN_LENGTH = 100

email_ns = Namespace("email", description="Configuration des emails transactionnels")

# --- Modèles API ---

domain_setup_model = email_ns.model(
    "EmailDomainSetup",
    {
        "from_email": fields.String(
            required=True,
            description="Adresse email d'envoi (ex: noreply@entreprise.ch)",
            example="noreply@entreprise.ch",
        ),
        "from_name": fields.String(
            required=True,
            description="Nom d'expéditeur (ex: 'Lirie Transports')",
            example="Lirie Transports",
        ),
    },
)

dns_records_model = email_ns.model(
    "DNSRecords",
    {
        "spf": fields.String(description="Enregistrement SPF TXT à ajouter au domaine"),
        "dkim": fields.String(
            description="Enregistrement DKIM TXT à ajouter au domaine"
        ),
    },
)

domain_setup_response = email_ns.model(
    "EmailDomainSetupResponse",
    {
        "success": fields.Boolean(description="Succès de la configuration"),
        "domain": fields.String(description="Domaine configuré"),
        "from_email": fields.String(description="Adresse email d'envoi"),
        "from_name": fields.String(description="Nom d'expéditeur"),
        "verified": fields.Boolean(description="Domaine déjà vérifié dans Brevo"),
        "dns_records": fields.Nested(
            dns_records_model, description="Enregistrements DNS à configurer"
        ),
        "message": fields.String(description="Message d'information"),
    },
)

domain_verify_response = email_ns.model(
    "EmailDomainVerifyResponse",
    {
        "success": fields.Boolean(description="Succès de la vérification"),
        "verified": fields.Boolean(description="Domaine validé dans Brevo"),
        "domain": fields.String(description="Domaine vérifié"),
        "message": fields.String(description="Message d'information ou d'erreur"),
    },
)


def extract_domain(email: str) -> str:
    """Extrait le domaine d'une adresse email."""
    if "@" not in email:
        raise ValueError("Format d'email invalide")
    return email.split("@")[1].lower()


def validate_email(email: str) -> bool:
    """Valide le format d'une adresse email."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


@email_ns.route("/domain/setup")
class EmailDomainSetup(Resource):
    """Configure l'adresse email d'envoi et récupère les DNS à configurer."""

    @jwt_required()
    @role_required(UserRole.company)
    @email_ns.expect(domain_setup_model)
    @email_ns.response(200, "Configuration réussie", domain_setup_response)
    @email_ns.response(400, "Requête invalide")
    @email_ns.response(403, "Permission refusée")
    @email_ns.response(500, "Erreur serveur")
    def post(self):
        """
        Configure l'adresse email d'envoi pour l'entreprise.

        Cette route :
        1. Valide l'adresse email et le nom d'expéditeur
        2. Extrait le domaine et vérifie son statut dans Brevo
        3. Récupère les enregistrements DNS (SPF + DKIM) à configurer
        4. Sauvegarde la configuration dans la base de données

        **Important** : Après cette configuration, le client doit :
        - Ajouter les enregistrements SPF et DKIM dans son DNS
        - Attendre la propagation DNS (15 min - 24h)
        - Appeler `/email/domain/verify` pour vérifier
        """
        data = request.json

        # 1. Validation des inputs
        from_email = data.get("from_email", "").strip()
        from_name = data.get("from_name", "").strip()

        if not from_email or not from_name:
            return {"error": "from_email et from_name sont requis"}, 400

        if not validate_email(from_email):
            return {"error": "Format d'email invalide"}, 400

        if len(from_name) > MAX_DOMAIN_LENGTH:
            return {"error": "from_name trop long (max 100 caractères)"}, 400

        # 2. Récupérer l'entreprise de l'utilisateur
        company, err, code = get_company_from_token()
        if err:
            return err, code

        if not company:
            return {"error": "Entreprise non trouvée"}, 404

        # 3. Récupérer ou créer les paramètres de facturation
        billing_settings = CompanyBillingSettings.query.filter_by(
            company_id=company.id
        ).first()

        if not billing_settings:
            billing_settings = CompanyBillingSettings(company_id=company.id)
            db.session.add(billing_settings)

        # 4. Extraire le domaine et vérifier dans Brevo
        domain = extract_domain(from_email)
        provider = BrevoEmailProvider()

        verification_result = provider.verify_domain(domain)
        dns_records = provider.get_domain_dns_records(domain)
        # 5. Sauvegarder la configuration
        billing_settings.smtp_username = (
            from_email  # Champ existant (sera renommé plus tard)
        )
        billing_settings.from_name = from_name
        billing_settings.domain_verified = (
            verification_result.verified if verification_result else False
        )
        billing_settings.domain_dns_records = dns_records if dns_records else None

        db.session.commit()

        # 6. Préparer le message de réponse
        if verification_result and verification_result.verified:
            message = (
                f"✅ Domaine {domain} déjà vérifié ! Vous pouvez envoyer des emails."
            )
        else:
            message = (
                f"⚠️ Domaine {domain} pas encore vérifié. "
                f"Ajoutez les enregistrements DNS ci-dessous, puis cliquez sur 'Vérifier'."
            )

        response_data = {
            "success": True,
            "domain": domain,
            "from_email": from_email,
            "from_name": from_name,
            "verified": verification_result.verified if verification_result else False,
            "dns_records": dns_records if dns_records else {},
            "message": message,
        }
        return response_data, 200


@email_ns.route("/domain/verify")
class EmailDomainVerify(Resource):
    """Vérifie que le domaine est validé dans Brevo."""

    @jwt_required()
    @role_required(UserRole.company)
    @email_ns.response(200, "Vérification réussie", domain_verify_response)
    @email_ns.response(400, "Aucun domaine configuré")
    @email_ns.response(403, "Permission refusée")
    @email_ns.response(500, "Erreur serveur")
    def post(self):
        """
        Vérifie que le domaine est validé dans Brevo (SPF + DKIM configurés).

        Cette route :
        1. Lit l'adresse email d'envoi depuis la base de données
        2. Extrait le domaine
        3. Vérifie le statut dans Brevo
        4. Met à jour le champ `domain_verified` en base

        **Note** : Cette vérification peut prendre jusqu'à 24h après la
        configuration DNS en fonction de la propagation.
        """
        # 1. Récupérer l'entreprise
        company, err, code = get_company_from_token()
        if err:
            return err, code

        if not company:
            return {"error": "Entreprise non trouvée"}, 404

        # 2. Récupérer les paramètres de facturation
        billing_settings = CompanyBillingSettings.query.filter_by(
            company_id=company.id
        ).first()

        if not billing_settings or not billing_settings.smtp_username:
            return {
                "error": "Aucun domaine configuré. Appelez d'abord /email/domain/setup"
            }, 400

        # 3. Extraire le domaine et vérifier
        from_email = billing_settings.smtp_username
        domain = extract_domain(from_email)

        provider = BrevoEmailProvider()
        verification_result = provider.verify_domain(domain)

        # 4. Mettre à jour le statut en base
        billing_settings.domain_verified = verification_result.verified
        db.session.commit()

        # 5. Préparer le message de réponse
        if verification_result.verified:
            message = f"✅ Domaine {domain} vérifié avec succès ! Vous pouvez maintenant envoyer des emails."
        else:
            message = (
                f"❌ Domaine {domain} pas encore vérifié. "
                f"Vérifiez que les enregistrements DNS ont bien été ajoutés et réessayez dans quelques heures. "
                f"Erreur Brevo : {verification_result.error or 'En attente de validation'}"
            )

        return {
            "success": True,
            "verified": verification_result.verified,
            "domain": domain,
            "message": message,
        }, 200


@email_ns.route("/config")
class EmailConfig(Resource):
    """Récupère la configuration email actuelle."""

    @jwt_required()
    @role_required(UserRole.company)
    @email_ns.response(200, "Configuration récupérée")
    @email_ns.response(403, "Permission refusée")
    def get(self):
        """
        Récupère la configuration email actuelle de l'entreprise.

        Retourne :
        - from_email : Adresse email d'envoi
        - from_name : Nom d'expéditeur
        - domain_verified : Domaine validé ou non
        - dns_records : Enregistrements DNS à configurer (si pas encore vérifié)
        """
        # 1. Récupérer l'entreprise
        company, err, code = get_company_from_token()
        if err:
            return err, code

        if not company:
            return {"error": "Entreprise non trouvée"}, 404

        # 2. Récupérer les paramètres de facturation
        billing_settings = CompanyBillingSettings.query.filter_by(
            company_id=company.id
        ).first()

        if not billing_settings:
            return {
                "configured": False,
                "message": "Aucune configuration email. Appelez /email/domain/setup pour configurer.",
            }, 200

        # 3. Extraire les informations
        from_email = billing_settings.smtp_username
        from_name = billing_settings.from_name
        domain_verified = billing_settings.domain_verified
        dns_records = billing_settings.domain_dns_records

        return {
            "configured": bool(from_email),
            "from_email": from_email,
            "from_name": from_name,
            "domain_verified": domain_verified,
            "dns_records": dns_records,
        }, 200


@email_ns.route("/domain/diagnostic")
class EmailDomainDiagnostic(Resource):
    """Diagnostic complet du domaine Brevo (mode debug)."""

    @jwt_required()
    @role_required(UserRole.company)
    @email_ns.response(200, "Diagnostic réussi")
    @email_ns.response(400, "Aucun domaine configuré")
    @email_ns.response(403, "Permission refusée")
    def post(self):
        """
        Effectue un diagnostic complet du domaine dans Brevo.

        Retourne la réponse complète de l'API Brevo pour debug.
        """
        # 1. Récupérer l'entreprise
        company, err, code = get_company_from_token()
        if err:
            return err, code

        if not company:
            return {"error": "Entreprise non trouvée"}, 404

        # 2. Récupérer les paramètres de facturation
        billing_settings = CompanyBillingSettings.query.filter_by(
            company_id=company.id
        ).first()

        if not billing_settings or not billing_settings.smtp_username:
            return {
                "error": "Aucun domaine configuré. Appelez d'abord /email/domain/setup"
            }, 400

        # 3. Extraire le domaine
        from_email = billing_settings.smtp_username
        domain = extract_domain(from_email)

        # 4. Effectuer le diagnostic via Brevo API
        provider = BrevoEmailProvider()

        logger.info("🔍 [DIAGNOSTIC] Début diagnostic pour domaine: %s", domain)

        try:
            import requests

            # Appel direct à l'API Brevo pour récupérer les détails
            response = requests.get(
                f"https://api.brevo.com/v3/senders/domains/{domain}",
                headers={
                    "accept": "application/json",
                    "api-key": provider.api_key,
                },
                timeout=10,
            )

            logger.info("🔍 [DIAGNOSTIC] Réponse Brevo status=%s", response.status_code)

            if response.status_code == HTTPStatus.OK:
                data = response.json()

                # Extraire les détails
                verified = data.get("verified", False)
                authenticated = data.get("authenticated", False)
                dns_records = data.get("dns_records", {})

                # Vérifier chaque enregistrement
                brevo_code = dns_records.get("brevo_code", {})
                dkim1 = dns_records.get("dkim1Record", {})
                dkim2 = dns_records.get("dkim2Record", {})

                brevo_valid = brevo_code.get("is_valid", False)
                dkim1_valid = dkim1.get("is_valid", False)
                dkim2_valid = dkim2.get("is_valid", False)

                diagnostic = {
                    "success": True,
                    "domain": domain,
                    "brevo_status": {
                        "verified": verified,
                        "authenticated": authenticated,
                    },
                    "dns_validation": {
                        "brevo_code_valid": brevo_valid,
                        "dkim1_valid": dkim1_valid,
                        "dkim2_valid": dkim2_valid,
                        "all_valid": brevo_valid and dkim1_valid and dkim2_valid,
                    },
                    "dns_records": {
                        "brevo_code": {
                            "host": brevo_code.get("host_name", "@"),
                            "value": brevo_code.get("value", "N/A"),
                            "is_valid": brevo_valid,
                        },
                        "dkim1": {
                            "host": dkim1.get("host_name", "N/A"),
                            "value": dkim1.get("value", "N/A"),
                            "is_valid": dkim1_valid,
                        },
                        "dkim2": {
                            "host": dkim2.get("host_name", "N/A"),
                            "value": dkim2.get("value", "N/A"),
                            "is_valid": dkim2_valid,
                        },
                    },
                    "message": (
                        "✅ Tous les enregistrements DNS sont valides ! "
                        "Brevo devrait valider le domaine sous peu."
                        if brevo_valid and dkim1_valid and dkim2_valid
                        else "⚠️ Certains enregistrements DNS ne sont pas encore validés par Brevo."
                    ),
                    "raw_response": data,  # Réponse complète pour debug
                }

                logger.info(
                    "🔍 [DIAGNOSTIC] Statut: verified=%s, authenticated=%s, all_dns_valid=%s",
                    verified,
                    authenticated,
                    brevo_valid and dkim1_valid and dkim2_valid,
                )

                return diagnostic, 200

            if response.status_code == HTTPStatus.NOT_FOUND:
                return {
                    "success": False,
                    "error": f"Domaine {domain} non trouvé dans Brevo",
                    "message": (
                        "Le domaine n'a pas été ajouté dans Brevo. "
                        "Connectez-vous à app.brevo.com et ajoutez le domaine manuellement."
                    ),
                }, 200

            return {
                "success": False,
                "error": f"Erreur API Brevo: {response.status_code}",
                "details": response.text,
            }, 200

        except Exception as e:
            logger.exception("🔍 [DIAGNOSTIC] Erreur: %s", e)
            return {
                "success": False,
                "error": f"Erreur lors du diagnostic: {e!s}",
            }, 500


@email_ns.route("/health")
class EmailHealth(Resource):
    """Healthcheck de la configuration email (SMTP/Brevo)."""

    @jwt_required()
    @role_required(UserRole.company)
    @email_ns.response(200, "Healthcheck email")
    @email_ns.response(403, "Permission refusée")
    def get(self):
        """Retourne l'état de la configuration email active.

        - N'envoie aucun email
        - Vérifie la joignabilité SMTP si provider SMTP
        - Expose un diagnostic lisible pour environnement local
        """
        provider = (os.getenv("EMAIL_PROVIDER", "smtp") or "smtp").strip().lower()
        notifications_enabled = (
            os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "false") or "false"
        ).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        smtp_host = (
            os.getenv("SMTP_HOST") or os.getenv("MAIL_SERVER") or "smtp.gmail.com"
        ).strip()
        smtp_port_raw = (
            os.getenv("SMTP_PORT") or os.getenv("MAIL_PORT") or "587"
        ).strip()
        try:
            smtp_port = int(smtp_port_raw)
        except ValueError:
            smtp_port = 587

        smtp_user = (os.getenv("SMTP_USER") or os.getenv("SMTP_USERNAME") or "").strip()
        smtp_password_set = bool((os.getenv("SMTP_PASSWORD") or "").strip())
        brevo_api_key_set = bool((os.getenv("BREVO_API_KEY") or "").strip())

        is_localhost = smtp_host in {"localhost", "127.0.0.1", "::1"}

        smtp_connect_ok = None
        smtp_connect_error = None
        if provider == "smtp":
            try:
                with socket.create_connection((smtp_host, smtp_port), timeout=2.5):
                    smtp_connect_ok = True
            except Exception as e:
                smtp_connect_ok = False
                smtp_connect_error = str(e)

        return {
            "success": True,
            "email_notifications_enabled": notifications_enabled,
            "provider": provider,
            "brevo": {
                "api_key_configured": brevo_api_key_set,
            },
            "smtp": {
                "host": smtp_host,
                "port": smtp_port,
                "is_localhost": is_localhost,
                "username_configured": bool(smtp_user),
                "password_configured": smtp_password_set,
                "connectivity_ok": smtp_connect_ok,
                "connectivity_error": smtp_connect_error,
            },
            "message": (
                "Provider Brevo actif."
                if provider == "brevo"
                else "Provider SMTP actif."
            ),
        }, 200


@email_ns.route("/health/public")
class EmailHealthPublic(Resource):
    """Healthcheck email public en local (sans JWT)."""

    @email_ns.response(200, "Healthcheck email public")
    @email_ns.response(403, "Accès refusé")
    def get(self):
        """Version publique du healthcheck, limitée aux environnements dev/test."""
        env = str(current_app.config.get("ENVIRONMENT", "")).strip().lower()
        is_testing = bool(current_app.config.get("TESTING"))
        is_dev = env in {"development", "dev", "local"}
        if not (is_dev or is_testing):
            return {
                "error": "Endpoint disponible uniquement en développement/test."
            }, 403

        remote = (request.remote_addr or "").strip()
        if remote not in {"127.0.0.1", "::1", ""}:
            return {"error": "Endpoint public autorisé uniquement en localhost."}, 403

        provider = (os.getenv("EMAIL_PROVIDER", "smtp") or "smtp").strip().lower()
        notifications_enabled = (
            os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "false") or "false"
        ).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        smtp_host = (
            os.getenv("SMTP_HOST") or os.getenv("MAIL_SERVER") or "smtp.gmail.com"
        ).strip()
        smtp_port_raw = (
            os.getenv("SMTP_PORT") or os.getenv("MAIL_PORT") or "587"
        ).strip()
        try:
            smtp_port = int(smtp_port_raw)
        except ValueError:
            smtp_port = 587

        smtp_connect_ok = None
        smtp_connect_error = None
        if provider == "smtp":
            try:
                with socket.create_connection((smtp_host, smtp_port), timeout=2.5):
                    smtp_connect_ok = True
            except Exception as e:
                smtp_connect_ok = False
                smtp_connect_error = str(e)

        return {
            "success": True,
            "scope": "public_local_only",
            "provider": provider,
            "email_notifications_enabled": notifications_enabled,
            "smtp": {
                "host": smtp_host,
                "port": smtp_port,
                "connectivity_ok": smtp_connect_ok,
                "connectivity_error": smtp_connect_error,
            },
        }, 200
