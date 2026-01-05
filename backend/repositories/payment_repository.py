"""Repository pour l'accès aux données Payment."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, cast

from domain.payment_dto import PaymentDTO
from models import Payment
from models.enums import PaymentStatus

logger = __import__("logging").getLogger(__name__)


class PaymentRepositoryPort(Protocol):
    """Port (interface) pour le repository Payment.

    Cette interface définit le contrat que doit respecter toute implémentation
    du repository. Elle permet de découpler la couche Application de l'implémentation
    concrète (SQLAlchemy, MongoDB, etc.).
    """

    def find_by_id(self, payment_id: int) -> PaymentDTO | None:
        """Trouve un paiement par son ID.

        Args:
            payment_id: ID du paiement

        Returns:
            PaymentDTO ou None si non trouvé
        """
        ...

    def find_by_client_id(self, client_id: int) -> list[PaymentDTO]:
        """Trouve tous les paiements d'un client.

        Args:
            client_id: ID du client

        Returns:
            Liste de PaymentDTO
        """
        ...

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
    ) -> PaymentDTO:
        """Crée un paiement et le commit en base.

        Args:
            amount: Montant du paiement
            method: Méthode de paiement
            user_id: ID de l'utilisateur
            client_id: ID du client
            booking_id: ID de la réservation
            status: Statut du paiement (défaut: PENDING)
            reference: Référence du paiement (optionnel)

        Returns:
            PaymentDTO créé

        Side-effects:
            - DB: Crée Payment et commit
        """
        ...

    def update_status(
        self, payment_id: int, status: PaymentStatus
    ) -> PaymentDTO | None:
        """Met à jour le statut d'un paiement.

        Args:
            payment_id: ID du paiement
            status: Nouveau statut

        Returns:
            PaymentDTO mis à jour ou None si non trouvé

        Side-effects:
            - DB: Met à jour Payment.status et commit
        """
        ...


class PaymentRepository:
    """Repository SQLAlchemy pour Payment.

    Implémentation concrète du port PaymentRepositoryPort utilisant SQLAlchemy.
    Cette classe convertit les modèles SQLAlchemy en DTOs pour maintenir
    le découplage avec la couche Application.
    """

    def _to_dto(self, payment: Payment) -> PaymentDTO:
        """Convertit un modèle SQLAlchemy Payment en DTO.

        Args:
            payment: Modèle SQLAlchemy Payment

        Returns:
            PaymentDTO correspondant
        """
        return PaymentDTO(
            id=payment.id,
            user_id=cast(int, payment.user_id),
            client_id=cast(int, payment.client_id),
            booking_id=cast(int, payment.booking_id),
            amount=float(payment.amount),
            method=str(payment.method),
            status=payment.status,
            reference=getattr(payment, "reference", None),
            date=cast(datetime | None, payment.date),
            updated_at=cast(datetime | None, payment.updated_at),
        )

    def find_by_id(self, payment_id: int) -> PaymentDTO | None:
        """Trouve un paiement par son ID.

        Args:
            payment_id: ID du paiement

        Returns:
            PaymentDTO ou None si non trouvé
        """
        payment = Payment.query.get(payment_id)
        if payment is None:
            return None
        return self._to_dto(payment)

    def find_by_client_id(self, client_id: int) -> list[PaymentDTO]:
        """Trouve tous les paiements d'un client.

        Args:
            client_id: ID du client

        Returns:
            Liste de PaymentDTO
        """
        payments = Payment.query.filter_by(client_id=client_id).all()
        return [self._to_dto(p) for p in payments]

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
    ) -> PaymentDTO:
        """Crée un paiement et le commit en base.

        Args:
            amount: Montant du paiement
            method: Méthode de paiement
            user_id: ID de l'utilisateur
            client_id: ID du client
            booking_id: ID de la réservation
            status: Statut du paiement (défaut: PENDING)
            reference: Référence du paiement (optionnel)

        Returns:
            PaymentDTO créé

        Side-effects:
            - DB: Crée Payment et commit
        """
        from models import db

        # Créer le paiement en utilisant les attributs du modèle
        payment = Payment()
        payment.amount = amount
        payment.method = method
        payment.user_id = user_id
        payment.client_id = client_id
        payment.booking_id = booking_id
        payment.status = status
        # Si le modèle Payment a un champ reference, l'ajouter
        if reference and hasattr(payment, "reference"):
            payment.reference = reference

        db.session.add(payment)
        db.session.commit()
        return self._to_dto(payment)

    def update_status(
        self, payment_id: int, status: PaymentStatus
    ) -> PaymentDTO | None:
        """Met à jour le statut d'un paiement.

        Args:
            payment_id: ID du paiement
            status: Nouveau statut

        Returns:
            PaymentDTO mis à jour ou None si non trouvé

        Side-effects:
            - DB: Met à jour Payment.status et commit
        """
        from models import db

        payment = Payment.query.get(payment_id)
        if payment is None:
            return None

        payment.status = status
        db.session.commit()
        return self._to_dto(payment)
