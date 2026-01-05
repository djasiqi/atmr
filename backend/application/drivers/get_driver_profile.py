from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GetDriverProfileInput:
    """Input pour récupérer le profil d'un chauffeur.

    Attributes:
        driver: Le chauffeur dont on veut le profil
    """

    driver: Any


@dataclass(frozen=True, slots=True)
class GetDriverProfileOutput:
    """Output pour récupérer le profil d'un chauffeur.

    Attributes:
        success: True si l'opération a réussi
        response: Réponse avec le profil (si succès)
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP
    """

    success: bool
    response: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    status_code: int = 200


class GetDriverProfileUseCase:
    """Use-case Application: récupérer le profil du chauffeur (données sérialisées)."""

    def execute(self, input_data: GetDriverProfileInput) -> GetDriverProfileOutput:
        """Récupère le profil d'un chauffeur.

        Args:
            input_data: Input avec driver

        Returns:
            GetDriverProfileOutput avec le profil sérialisé
        """
        try:
            return GetDriverProfileOutput(
                success=True,
                response={"profile": input_data.driver.serialize},
                status_code=200,
            )
        except Exception:
            import logging

            logger = logging.getLogger(__name__)
            logger.exception("Erreur lors de la récupération du profil du chauffeur")
            return GetDriverProfileOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
