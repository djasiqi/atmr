"""Use-case: créer un paiement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from domain.payment_dto import PaymentDTO
from models.enums import PaymentStatus

logger = logging.getLogger(__name__)


class _PaymentWriterPort(Protocol):
    def create_and_commit(
        self,
        *,
        amount: float,
        method: str,
        user_id: int,
        client_id: int,
        booking_id: int,
        status: PaymentStatus = PaymentStatus.PENDING,
        reference: str | None = None,
    ) -> PaymentDTO: ...


@dataclass(frozen=True, slots=True)
class CreatePaymentInput:
    """Input pour le use case de création de paiement.

    Attributes:
        amount: Montant du paiement
        method: Méthode de paiement
        user_id: ID de l'utilisateur
        client_id: ID du client
        booking_id: ID de la réservation
        reference: Référence du paiement (optionnel)
    """

    amount: float
    method: str
    user_id: int
    client_id: int
    booking_id: int
    reference: str | None = None


@dataclass(frozen=True, slots=True)
class CreatePaymentOutput:
    """Output du use case de création de paiement.

    Attributes:
        success: True si l'opération a réussi
        payment_id: ID du paiement créé (si succès)
        payment: Paiement créé (si succès)
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    payment_id: int | None = None
    payment: PaymentDTO | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class CreatePaymentUseCase:
    """Use-case Application: créer un paiement pour une réservation."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, *, payment_writer: _PaymentWriterPort
    ) -> None:
        """Initialise le use case avec ses dépendances.

        Args:
            payment_writer: Port pour créer un paiement
        """
        self._payment_writer = payment_writer

    def execute(self, input_data: CreatePaymentInput) -> CreatePaymentOutput:
        """Crée un paiement pour une réservation.

        Args:
            input_data: Données d'entrée du use case

        Returns:
            CreatePaymentOutput avec le paiement créé
        """
        # Validation
        validation_error = self._validate_input(input_data)
        if validation_error:
            return CreatePaymentOutput(
                success=False,
                error=validation_error,
                status_code=400,
            )

        # Création
        try:
            payment = self._payment_writer.create_and_commit(
                amount=input_data.amount,
                method=input_data.method,
                user_id=input_data.user_id,
                client_id=input_data.client_id,
                booking_id=input_data.booking_id,
                status=PaymentStatus.PENDING,
                reference=input_data.reference,
            )
            return CreatePaymentOutput(
                success=True, payment_id=payment.id, payment=payment
            )
        except Exception:
            logger.exception("Erreur lors de la création du paiement")
            return CreatePaymentOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )

    def _validate_input(self, input_data: CreatePaymentInput) -> dict[str, str] | None:
        """Valide les inputs du use case.

        Args:
            input_data: Input à valider

        Returns:
            None si valide, dict d'erreurs sinon
        """
        errors: dict[str, str] = {}

        if input_data.amount <= 0:
            errors["amount"] = "Le montant doit être positif"

        if not input_data.method or len(input_data.method.strip()) == 0:
            errors["method"] = "La méthode de paiement est requise"

        if input_data.user_id <= 0:
            errors["user_id"] = "L'ID utilisateur doit être positif"

        if input_data.client_id <= 0:
            errors["client_id"] = "L'ID client doit être positif"

        if input_data.booking_id <= 0:
            errors["booking_id"] = "L'ID réservation doit être positif"

        return errors if errors else None
