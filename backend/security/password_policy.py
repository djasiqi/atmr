"""✅ S3: Service de validation de mots de passe renforcée.

Implémente une politique de mots de passe stricte avec :
- Validation de complexité renforcée (caractères spéciaux, longueur minimale)
- Vérification contre Have I Been Pwned (mots de passe compromis)
- Vérification de l'historique (empêcher réutilisation)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Configuration
MIN_PASSWORD_LENGTH = int(os.getenv("MIN_PASSWORD_LENGTH", "12"))
MAX_PASSWORD_LENGTH = int(os.getenv("MAX_PASSWORD_LENGTH", "128"))
PASSWORD_HISTORY_COUNT = int(
    os.getenv("PASSWORD_HISTORY_COUNT", "5")
)  # Empêcher réutilisation des 5 derniers
PASSWORD_EXPIRATION_DAYS = int(
    os.getenv("PASSWORD_EXPIRATION_DAYS", "0")
)  # 0 = désactivé
HIBP_API_TIMEOUT = int(
    os.getenv("HIBP_API_TIMEOUT", "5")
)  # Timeout pour Have I Been Pwned
HIBP_ENABLED = os.getenv("HIBP_ENABLED", "true").lower() == "true"


class PasswordPolicyError(Exception):
    """Exception levée lors de la validation de mot de passe."""

    def __init__(self, message: str, code: str = "INVALID_PASSWORD"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class PasswordPolicyService:
    """Service pour valider les mots de passe selon une politique stricte."""

    @staticmethod
    def validate_complexity(password: str) -> tuple[bool, str | None]:
        """✅ S3: Valide la complexité d'un mot de passe.

        Critères renforcés:
        - Au moins 12 caractères (configurable via MIN_PASSWORD_LENGTH)
        - Au moins une majuscule
        - Au moins une minuscule
        - Au moins un chiffre
        - Au moins un caractère spécial (!@#$%^&*()_+-=[]{}|;:,.<>?)

        Args:
            password: Mot de passe à valider

        Returns:
            Tuple (is_valid, error_message)
        """
        if not password:
            return False, "Le mot de passe ne peut pas être vide"

        if len(password) < MIN_PASSWORD_LENGTH:
            return (
                False,
                f"Le mot de passe doit contenir au moins {MIN_PASSWORD_LENGTH} caractères",
            )

        if len(password) > MAX_PASSWORD_LENGTH:
            return (
                False,
                f"Le mot de passe ne peut pas dépasser {MAX_PASSWORD_LENGTH} caractères",
            )

        # Vérifier les critères de complexité
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = bool(re.search(r"[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]", password))

        errors: list[str] = []
        if not has_upper:
            errors.append("au moins une majuscule")
        if not has_lower:
            errors.append("au moins une minuscule")
        if not has_digit:
            errors.append("au moins un chiffre")
        if not has_special:
            errors.append("au moins un caractère spécial (!@#$%^&*()_+-=[]{}|;:,.<>?)")

        if errors:
            return (
                False,
                f"Le mot de passe doit contenir {', '.join(errors)}",
            )

        return True, None

    @staticmethod
    def check_hibp(password: str) -> tuple[bool, str | None]:
        """✅ S3: Vérifie si un mot de passe a été compromis via Have I Been Pwned.

        Utilise l'API k-anonymity de Have I Been Pwned pour vérifier
        si le mot de passe apparaît dans des fuites de données.

        Args:
            password: Mot de passe à vérifier

        Returns:
            Tuple (is_safe, error_message)
            - is_safe: True si le mot de passe n'est pas compromis, False sinon
            - error_message: Message d'erreur si compromis, None sinon
        """
        if not HIBP_ENABLED:
            logger.debug("[PasswordPolicy] HIBP désactivé, vérification ignorée")
            return True, None

        try:
            # Calculer le hash SHA-1 du mot de passe
            # SHA-1 utilisé pour l'API HIBP (k-anonymity), pas pour la sécurité cryptographique
            password_hash = (
                hashlib.sha1(password.encode("utf-8"), usedforsecurity=False)
                .hexdigest()
                .upper()
            )  # nosec B324
            prefix = password_hash[:5]
            suffix = password_hash[5:]

            # Appeler l'API Have I Been Pwned (k-anonymity)
            # On envoie seulement les 5 premiers caractères du hash
            url = f"https://api.pwnedpasswords.com/range/{prefix}"
            response = requests.get(url, timeout=HIBP_API_TIMEOUT)

            HTTP_OK = 200
            if response.status_code == HTTP_OK:
                # Chercher le suffixe dans la réponse
                hashes = response.text.splitlines()
                for hash_line in hashes:
                    if hash_line.startswith(suffix):
                        # Le hash complet a été trouvé, le mot de passe est compromis
                        count = hash_line.split(":")[1] if ":" in hash_line else "0"
                        logger.warning(
                            "[PasswordPolicy] ⚠️ Mot de passe compromis détecté via HIBP (count: %s)",
                            count,
                        )
                        return (
                            False,
                            (
                                "Ce mot de passe a été compromis dans des fuites de données. "
                                "Veuillez choisir un mot de passe différent."
                            ),
                        )

            # Le mot de passe n'a pas été trouvé, il est sûr
            logger.debug(
                "[PasswordPolicy] ✅ Mot de passe vérifié via HIBP, non compromis"
            )
            return True, None

        except requests.Timeout:
            logger.warning(
                "[PasswordPolicy] ⚠️ Timeout lors de la vérification HIBP, on accepte le mot de passe"
            )
            # En cas de timeout, on accepte le mot de passe (fail-open)
            return True, None
        except requests.RequestException as e:
            logger.warning(
                "[PasswordPolicy] ⚠️ Erreur lors de la vérification HIBP: %s, on accepte le mot de passe",
                e,
            )
            # En cas d'erreur réseau, on accepte le mot de passe (fail-open)
            return True, None
        except Exception as e:
            logger.error(
                "[PasswordPolicy] ❌ Erreur inattendue lors de la vérification HIBP: %s",
                e,
            )
            # En cas d'erreur inattendue, on accepte le mot de passe (fail-open)
            return True, None

    @staticmethod
    def validate_password(
        password: str, user_id: int | None = None, check_history: bool = True
    ) -> None:
        """✅ S3: Valide un mot de passe selon la politique stricte.

        Effectue toutes les validations :
        1. Complexité (longueur, caractères)
        2. Have I Been Pwned (si activé)
        3. Historique (si user_id fourni et check_history=True)

        Args:
            password: Mot de passe à valider
            user_id: ID de l'utilisateur (optionnel, pour vérification historique)
            check_history: Si True, vérifie l'historique des mots de passe

        Raises:
            PasswordPolicyError: Si le mot de passe ne respecte pas la politique
        """
        # 1. Validation de complexité
        is_valid, error_msg = PasswordPolicyService.validate_complexity(password)
        if not is_valid:
            raise PasswordPolicyError(
                error_msg or "Mot de passe invalide", "COMPLEXITY"
            )

        # 2. Vérification Have I Been Pwned
        is_safe, hibp_error = PasswordPolicyService.check_hibp(password)
        if not is_safe:
            raise PasswordPolicyError(hibp_error or "Mot de passe compromis", "HIBP")

        # 3. Vérification historique (si user_id fourni)
        if user_id is not None and check_history:
            from security.password_history import PasswordHistoryService

            is_not_reused, history_error = (
                PasswordHistoryService.check_password_history(user_id, password)
            )
            if not is_not_reused:
                raise PasswordPolicyError(
                    history_error or "Ce mot de passe a déjà été utilisé récemment",
                    "HISTORY",
                )

    @staticmethod
    def get_password_requirements() -> dict[str, Any]:
        """Retourne les exigences de mot de passe pour affichage au client.

        Returns:
            Dict avec les exigences de mot de passe
        """
        return {
            "min_length": MIN_PASSWORD_LENGTH,
            "max_length": MAX_PASSWORD_LENGTH,
            "requirements": [
                f"Au moins {MIN_PASSWORD_LENGTH} caractères",
                "Au moins une majuscule",
                "Au moins une minuscule",
                "Au moins un chiffre",
                "Au moins un caractère spécial (!@#$%^&*()_+-=[]{}|;:,.<>?)",
            ],
            "history_count": PASSWORD_HISTORY_COUNT,
            "expiration_days": PASSWORD_EXPIRATION_DAYS
            if PASSWORD_EXPIRATION_DAYS > 0
            else None,
            "hibp_enabled": HIBP_ENABLED,
        }
