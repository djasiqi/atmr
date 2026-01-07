# services/version_check.py
"""Service pour la vérification et comparaison de versions d'application mobile.

Ce service compare les versions semver et détermine si une mise à jour est
requise, recommandée, ou si tout est à jour.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from models.app_version_config import AppVersionConfig

logger = logging.getLogger(__name__)

# Statuts possibles pour la réponse de version check
UpdateStatus = Literal["OK", "UPDATE_RECOMMENDED", "UPDATE_REQUIRED"]


def compare_semver(version_a: str, version_b: str) -> int:
    """Compare deux versions semver (format: MAJOR.MINOR.PATCH).

    Args:
        version_a: Première version à comparer (ex: "1.2.3")
        version_b: Deuxième version à comparer (ex: "1.3.0")

    Returns:
        -1 si version_a < version_b
        0 si version_a == version_b
        1 si version_a > version_b

    Raises:
        ValueError: Si les versions ne sont pas au format semver valide

    Examples:
        >>> compare_semver("1.2.3", "1.3.0")
        -1
        >>> compare_semver("2.0.0", "1.9.9")
        1
        >>> compare_semver("1.0.0", "1.0.0")
        0
    """
    # Pattern semver: MAJOR.MINOR.PATCH (optionnellement avec préfixe "v")
    semver_pattern = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-.*)?$")

    def parse_version(version: str) -> tuple[int, int, int]:
        """Parse une version semver en tuple (major, minor, patch)."""
        version = version.strip()
        match = semver_pattern.match(version)
        if not match:
            raise ValueError(
                f"Version invalide: {version}. Format attendu: MAJOR.MINOR.PATCH"
            )
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))

    try:
        a_parts = parse_version(version_a)
        b_parts = parse_version(version_b)

        if a_parts < b_parts:
            return -1
        if a_parts > b_parts:
            return 1
        return 0
    except ValueError as e:
        logger.error(
            "Erreur comparaison versions %s vs %s: %s", version_a, version_b, e
        )
        raise


def check_version_status(
    current_version: str,
    min_required_version: str,
    latest_version: str,
) -> UpdateStatus:
    """Détermine le statut de mise à jour en comparant les versions.

    Args:
        current_version: Version actuelle de l'application
        min_required_version: Version minimale requise
        latest_version: Dernière version disponible

    Returns:
        "UPDATE_REQUIRED" si current < min_required
        "UPDATE_RECOMMENDED" si current >= min_required mais < latest
        "OK" si current >= latest

    Raises:
        ValueError: Si les versions ne sont pas au format semver valide
    """
    try:
        # Vérifier si la version actuelle est inférieure à la version minimale requise
        if compare_semver(current_version, min_required_version) < 0:
            return "UPDATE_REQUIRED"

        # Vérifier si la version actuelle est inférieure à la dernière version
        if compare_semver(current_version, latest_version) < 0:
            return "UPDATE_RECOMMENDED"

        # Tout est à jour
        return "OK"
    except ValueError as e:
        logger.error("Erreur lors de la vérification de version: %s", e)
        raise


def get_version_config(platform: str) -> AppVersionConfig | None:
    """Récupère la configuration de version pour une plateforme.

    Args:
        platform: "android" ou "ios"

    Returns:
        AppVersionConfig si trouvé, None sinon
    """
    if platform not in ("android", "ios"):
        raise ValueError(
            f"Plateforme invalide: {platform}. Attendu: 'android' ou 'ios'"
        )

    return AppVersionConfig.query.filter_by(platform=platform.lower()).first()


def check_app_version(platform: str, current_version: str) -> dict[str, str | None]:
    """Vérifie la version de l'application et retourne le statut de mise à jour.

    Args:
        platform: "android" ou "ios"
        current_version: Version actuelle de l'application (format semver)

    Returns:
        Dictionnaire avec:
        - platform: Plateforme
        - current_version: Version actuelle
        - latest_version: Dernière version disponible
        - min_required_version: Version minimale requise
        - status: "OK" | "UPDATE_RECOMMENDED" | "UPDATE_REQUIRED"
        - store_url: URL du store pour la mise à jour
        - message: Message personnalisé pour la mise à jour

    Raises:
        ValueError: Si la plateforme est invalide ou si la config n'existe pas
    """
    config = get_version_config(platform)
    if not config:
        # Si aucune config n'existe, on considère que tout est OK (comportement par défaut)
        logger.warning(
            "Aucune configuration de version trouvée pour %s. Comportement par défaut: OK",
            platform,
        )
        return {
            "platform": platform,
            "current_version": current_version,
            "latest_version": current_version,
            "min_required_version": current_version,
            "status": "OK",
            "store_url": None,
            "message": None,
        }

    status = check_version_status(
        current_version, config.min_required_version, config.latest_version
    )

    # Message par défaut si aucun message personnalisé
    message = config.update_message
    if not message:
        if status == "UPDATE_REQUIRED":
            message = (
                "Une nouvelle version de l'application est nécessaire pour continuer. "
                "Veuillez mettre à jour depuis le store."
            )
        elif status == "UPDATE_RECOMMENDED":
            message = (
                "Une nouvelle version de l'application est disponible. "
                "Nous vous recommandons de mettre à jour pour bénéficier des dernières améliorations."
            )

    return {
        "platform": platform,
        "current_version": current_version,
        "latest_version": config.latest_version,
        "min_required_version": config.min_required_version,
        "status": status,
        "store_url": config.store_url,
        "message": message,
    }
