#!/usr/bin/env python3
"""
Script de test pour le service d'envoi d'emails.

Ce script permet de tester :
1. La configuration SMTP
2. L'envoi d'un email de test
3. La validation d'adresses email

Usage:
    python scripts/test_email_service.py --email votre-email@example.com
    python scripts/test_email_service.py --email votre-email@example.com --check-dns
"""

import argparse
import logging
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_email_validation(email: str, check_dns: bool) -> bool:
    """
    Teste la validation d'une adresse email.

    Args:
        email: Adresse email à valider
        check_dns: Si True, vérifie l'existence du domaine

    Returns:
        True si l'email est valide, False sinon
    """
    from services.email.validators import EmailValidator

    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 1 : Validation d'email")
    logger.info("=" * 80)
    logger.info("Email à valider : %s", email)
    logger.info("Vérification DNS : %s", "Oui" if check_dns else "Non")

    # Validation
    result = EmailValidator.validate(email, check_dns=check_dns)

    logger.info("")
    logger.info("Résultat de la validation :")
    logger.info("  - Valide : %s", "✅ Oui" if result["valid"] else "❌ Non")
    if result["normalized"]:
        logger.info("  - Email normalisé : %s", result["normalized"])
    if result["error"]:
        logger.error("  - Erreur : %s", result["error"])

    logger.info("")
    return result["valid"]


def test_smtp_configuration() -> bool:
    """
    Teste la configuration SMTP de Flask-Mail.

    Returns:
        True si la configuration est valide, False sinon
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 2 : Configuration SMTP")
    logger.info("=" * 80)

    try:
        from flask import current_app

        config_keys = [
            "MAIL_SERVER",
            "MAIL_PORT",
            "MAIL_USE_TLS",
            "MAIL_USE_SSL",
            "MAIL_USERNAME",
            "MAIL_DEFAULT_SENDER",
        ]

        all_configured = True
        for key in config_keys:
            value = current_app.config.get(key)
            if key == "MAIL_PASSWORD":
                # Ne pas afficher le mot de passe
                display_value = "***" if value else "Non configuré"
            else:
                display_value = value if value else "Non configuré"

            logger.info("  - %s : %s", key, display_value)

            if key in ["MAIL_SERVER", "MAIL_USERNAME"] and not value:
                all_configured = False

        logger.info("")
        if all_configured:
            logger.info("✅ Configuration SMTP complète")
        else:
            logger.warning("⚠️ Configuration SMTP incomplète")
            logger.warning(
                "Veuillez configurer les variables MAIL_* dans votre fichier .env"
            )

        return all_configured

    except Exception as e:
        logger.error("❌ Erreur lors de la vérification de la configuration : %s", e)
        return False


def test_send_email(recipient_email: str) -> bool:
    """
    Teste l'envoi d'un email de test.

    Args:
        recipient_email: Adresse email du destinataire

    Returns:
        True si l'envoi a réussi, False sinon
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 3 : Envoi d'email de test")
    logger.info("=" * 80)
    logger.info("Destinataire : %s", recipient_email)

    try:
        from services.email.email_service import EmailService

        email_service = EmailService()

        # Envoi de l'email de test
        result = email_service.send_test_email(recipient_email)

        logger.info("")
        if result.success:
            logger.info("✅ Email envoyé avec succès !")
            logger.info("  - Destinataire : %s", result.recipient)
            logger.info("  - Envoyé à : %s", result.sent_at)
            logger.info("")
            logger.info(
                "👉 Vérifiez votre boîte de réception (et les spams) : %s",
                result.recipient,
            )
        else:
            logger.error("❌ Échec de l'envoi de l'email")
            logger.error("  - Destinataire : %s", result.recipient)
            logger.error("  - Erreur : %s", result.error)

        logger.info("")
        return result.success

    except Exception as e:
        logger.error("❌ Erreur lors de l'envoi de l'email : %s", e)
        logger.exception("Détails de l'erreur :")
        return False


def main() -> int:
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(
        description="Test du service d'envoi d'emails ATMR"
    )
    parser.add_argument("--email", required=True, help="Adresse email pour les tests")
    parser.add_argument(
        "--check-dns",
        action="store_true",
        help="Vérifier l'existence du domaine (nécessite dnspython)",
    )
    parser.add_argument(
        "--skip-send",
        action="store_true",
        help="Ne pas envoyer d'email (validation uniquement)",
    )

    args = parser.parse_args()

    # Initialiser Flask app
    try:
        from app import create_app

        logger.info("Initialisation de l'application Flask...")
        app = create_app()

        with app.app_context():
            logger.info("✅ Application Flask initialisée")
            logger.info("")

            # Test 1 : Validation d'email
            email_valid = test_email_validation(args.email, args.check_dns)

            if not email_valid:
                logger.error("")
                logger.error("❌ Email invalide, arrêt des tests")
                return 1

            # Test 2 : Configuration SMTP
            smtp_configured = test_smtp_configuration()

            if not smtp_configured:
                logger.warning("")
                logger.warning("⚠️ Configuration SMTP incomplète")
                logger.warning(
                    "Certains tests peuvent échouer. Configurez les variables MAIL_* "
                    "dans .env"
                )

            # Test 3 : Envoi d'email (si non skip)
            if not args.skip_send:
                send_success = test_send_email(args.email)

                if not send_success:
                    logger.error("")
                    logger.error("❌ Échec de l'envoi d'email")
                    logger.error("")
                    logger.error("💡 Vérifications suggérées :")
                    logger.error(
                        "  1. Vérifiez que les variables MAIL_* sont correctement "
                        "configurées dans .env"
                    )
                    logger.error(
                        "  2. Vérifiez que le serveur SMTP est accessible depuis "
                        "votre réseau"
                    )
                    logger.error(
                        "  3. Vérifiez les identifiants (username/password) SMTP"
                    )
                    logger.error(
                        "  4. Activez MAIL_DEBUG=True dans .env pour plus de détails"
                    )
                    return 1

            # Résumé final
            logger.info("")
            logger.info("=" * 80)
            logger.info("RÉSUMÉ DES TESTS")
            logger.info("=" * 80)
            logger.info("  - Validation email : %s", "✅" if email_valid else "❌")
            logger.info("  - Configuration SMTP : %s", "✅" if smtp_configured else "⚠️")
            if not args.skip_send:
                logger.info("  - Envoi d'email : %s", "✅" if send_success else "❌")
            else:
                logger.info("  - Envoi d'email : ⏭️ Ignoré")
            logger.info("=" * 80)
            logger.info("")

            if email_valid and (args.skip_send or send_success):
                logger.info("🎉 Tous les tests ont réussi !")
                logger.info("")
                logger.info("Prochaines étapes :")
                logger.info("  1. Configurer les routes API pour l'envoi de factures")
                logger.info(
                    "  2. Intégrer le service dans les use cases de facturation"
                )
                logger.info("  3. Créer l'interface UI pour l'envoi manuel de factures")
                return 0

            return 1

    except Exception as e:
        logger.error("❌ Erreur fatale : %s", e)
        logger.exception("Détails de l'erreur :")
        return 1


if __name__ == "__main__":
    sys.exit(main())
