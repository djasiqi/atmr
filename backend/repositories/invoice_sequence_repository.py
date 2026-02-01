"""Repository pour l'accès aux données InvoiceSequence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from models import InvoiceSequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InvoiceSequenceDTO:
    """DTO pour InvoiceSequence."""

    id: int
    company_id: int
    year: int
    month: int
    sequence: int


class InvoiceSequenceRepositoryPort(Protocol):
    """Port (interface) pour le repository InvoiceSequence."""

    def find_or_create(
        self, company_id: int, year: int, month: int
    ) -> InvoiceSequenceDTO:
        """Trouve ou crée une séquence pour un mois donné.

        Args:
            company_id: ID de l'entreprise
            year: Année
            month: Mois (1-12)

        Returns:
            InvoiceSequenceDTO trouvée ou créée
        """
        ...

    def increment_sequence(self, sequence_id: int) -> InvoiceSequenceDTO:
        """Incrémente la séquence et retourne la nouvelle valeur.

        Args:
            sequence_id: ID de la séquence

        Returns:
            InvoiceSequenceDTO avec séquence incrémentée
        """
        ...


class InvoiceSequenceRepository:
    """Repository SQLAlchemy pour InvoiceSequence."""

    def _to_dto(self, sequence: InvoiceSequence) -> InvoiceSequenceDTO:
        """Convertit un modèle SQLAlchemy InvoiceSequence en DTO."""
        return InvoiceSequenceDTO(
            id=sequence.id,
            company_id=sequence.company_id,
            year=sequence.year,
            month=sequence.month,
            sequence=sequence.sequence,
        )

    def find_or_create(
        self, company_id: int, year: int, month: int
    ) -> InvoiceSequenceDTO:
        """Trouve ou crée une séquence pour un mois donné.

        Args:
            company_id: ID de l'entreprise
            year: Année
            month: Mois (1-12)

        Returns:
            InvoiceSequenceDTO trouvée ou créée

        Side-effects:
            - DB: Crée InvoiceSequence si inexistante et commit
        """
        from ext import db

        sequence = InvoiceSequence.query.filter_by(
            company_id=company_id, year=year, month=month
        ).first()

        if not sequence:
            sequence = InvoiceSequence()
            sequence.company_id = company_id
            sequence.year = year
            sequence.month = month
            sequence.sequence = 0
            db.session.add(sequence)
            db.session.commit()

        return self._to_dto(sequence)

    def increment_sequence(self, sequence_id: int) -> InvoiceSequenceDTO:
        """Incrémente la séquence et retourne la nouvelle valeur.

        Utilise SELECT ... FOR UPDATE pour éviter les conditions de concurrence
        où deux requêtes obtiendraient le même numéro de facture.

        Args:
            sequence_id: ID de la séquence

        Returns:
            InvoiceSequenceDTO avec séquence incrémentée

        Side-effects:
            - DB: Met à jour InvoiceSequence et commit
        """
        from ext import db

        # ✅ Verrouiller la ligne pour éviter les doublons de numéro (race condition)
        sequence = (
            InvoiceSequence.query.filter_by(id=sequence_id).with_for_update().first()
        )
        if not sequence:
            msg = f"Séquence {sequence_id} non trouvée"
            raise ValueError(msg)

        sequence.sequence += 1
        db.session.commit()

        return self._to_dto(sequence)
