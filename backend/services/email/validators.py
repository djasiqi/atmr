"""
Validateurs pour les adresses email.

Ce module fournit des fonctions de validation d'adresses email,
incluant la validation syntaxique (regex) et la vérification DNS (optionnelle).
"""

import logging
import re
from typing import TypedDict

logger = logging.getLogger(__name__)

# Regex RFC 5322 simplifiée (couvre 99% des cas d'usage)
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
)

# Constantes pour validation
EMAIL_MAX_LENGTH = 254  # RFC 5321
EMAIL_DOMAIN_PARTS = 2  # user@domain


class EmailValidationResult(TypedDict):
    """Résultat de validation d'email."""

    valid: bool
    email: str | None
    error: str | None
    normalized: str | None


class EmailValidator:
    """Validateur d'adresses email avec normalisation et vérification DNS."""

    @staticmethod
    def is_valid_format(email: str | None) -> bool:
        """
        Vérifie si l'email a un format valide (syntaxe uniquement).

        Args:
            email: Adresse email à valider

        Returns:
            True si le format est valide, False sinon
        """
        if not email:
            return False

        email = email.strip()

        # Vérifications basiques
        if len(email) > EMAIL_MAX_LENGTH:
            return False

        if email.count("@") != 1:
            return False

        # Validation regex
        return bool(EMAIL_REGEX.match(email))

    @staticmethod
    def normalize(email: str | None) -> str | None:
        """
        Normalise une adresse email (lowercase, strip).

        Args:
            email: Adresse email à normaliser

        Returns:
            Email normalisé ou None si invalide
        """
        if not email:
            return None

        email = email.strip().lower()

        if not EmailValidator.is_valid_format(email):
            return None

        return email

    @staticmethod
    def validate(email: str | None, check_dns: bool = False) -> EmailValidationResult:
        """
        Valide une adresse email complète.

        Args:
            email: Adresse email à valider
            check_dns: Si True, vérifie l'existence du domaine (MX records)

        Returns:
            Dictionnaire avec le résultat de validation
        """
        result: EmailValidationResult = {
            "valid": False,
            "email": email,
            "error": None,
            "normalized": None,
        }

        # Validation de base
        if not email:
            result["error"] = "Email vide ou invalide"
            return result

        # Normalisation
        normalized = EmailValidator.normalize(email)
        if not normalized:
            result["error"] = "Format d'email invalide"
            return result

        result["normalized"] = normalized

        # Validation DNS (optionnelle)
        if check_dns:
            dns_error = EmailValidator._check_dns(normalized)
            if dns_error:
                result["error"] = dns_error
                return result

        # Validation réussie
        result["valid"] = True
        return result

    @staticmethod
    def _check_dns(normalized_email: str) -> str | None:  # noqa: PLR0911
        """
        Vérifie l'existence du domaine via DNS MX records.

        Args:
            normalized_email: Email normalisé

        Returns:
            Message d'erreur ou None si valide
        """
        domain = normalized_email.split("@")[1]

        try:
            # Vérification DNS MX records
            import dns.resolver  # pyright: ignore[reportMissingImports]

            mx_records = dns.resolver.resolve(domain, "MX")
            if not mx_records:
                return "Domaine sans serveur de messagerie (MX)"
        except ImportError:
            logger.warning("Module dnspython non installé, validation DNS désactivée")
            return None
        except AttributeError:  # dns.resolver not available
            return f"Domaine inexistant: {domain}"
        except Exception as e:
            logger.error("Erreur lors de la validation DNS: %s", e)
            error_name = type(e).__name__
            if "NXDOMAIN" in error_name:
                return f"Domaine inexistant: {domain}"
            if "NoAnswer" in error_name:
                return f"Pas de MX record pour: {domain}"
            return f"Erreur de validation DNS: {e!s}"

        return None

    @staticmethod
    def extract_domain(email: str | None) -> str | None:
        """
        Extrait le domaine d'une adresse email.

        Args:
            email: Adresse email

        Returns:
            Domaine ou None si invalide
        """
        normalized = EmailValidator.normalize(email)
        if not normalized:
            return None

        parts = normalized.split("@")
        return parts[1] if len(parts) == EMAIL_DOMAIN_PARTS else None
