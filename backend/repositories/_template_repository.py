"""Template pour créer un nouveau repository conforme au pattern standardisé.

Copier ce fichier et adapter selon les besoins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ===== DTO (Data Transfer Object) =====


@dataclass(frozen=True, slots=True)
class EntityDTO:
    """DTO pour Entity.

    Un DTO est une structure de données immuable qui représente une entité
    sans dépendances à l'ORM. Il facilite le découplage entre la couche
    Infrastructure et la couche Application.

    Attributes:
        id: ID de l'entité
        field1: Description du champ 1
        field2: Description du champ 2
    """

    id: int
    field1: str
    field2: int


# ===== Protocol (Interface) =====


class EntityRepositoryPort(Protocol):
    """Port (interface) pour le repository Entity.

    Cette interface définit le contrat que doit respecter toute implémentation
    du repository. Elle permet de découpler la couche Application de l'implémentation
    concrète (SQLAlchemy, MongoDB, etc.).

    Note: Les use cases utilisent ce Protocol, pas l'implémentation concrète.
    """

    def find_by_id(self, entity_id: int) -> EntityDTO | None:
        """Trouve une entité par son ID.

        Args:
            entity_id: ID de l'entité

        Returns:
            EntityDTO ou None si non trouvé
        """
        ...

    def find_all(self, company_id: int) -> list[EntityDTO]:
        """Trouve toutes les entités d'une entreprise.

        Args:
            company_id: ID de l'entreprise

        Returns:
            Liste de EntityDTO
        """
        ...

    def find_by_criteria(
        self, *, company_id: int, status: str | None = None
    ) -> list[EntityDTO]:
        """Trouve des entités selon des critères.

        Args:
            company_id: ID de l'entreprise
            status: Filtre par statut (optionnel)

        Returns:
            Liste de EntityDTO
        """
        ...

    def create(self, *, field1: str, field2: int, company_id: int) -> EntityDTO:
        """Crée une nouvelle entité.

        Args:
            field1: Description du champ 1
            field2: Description du champ 2
            company_id: ID de l'entreprise

        Returns:
            EntityDTO créé

        Side-effects:
            - DB: Crée Entity et commit
        """
        ...

    def update(self, entity_id: int, *, field1: str | None = None) -> EntityDTO | None:
        """Met à jour une entité existante.

        Args:
            entity_id: ID de l'entité à mettre à jour
            field1: Nouvelle valeur pour field1 (optionnel)

        Returns:
            EntityDTO mis à jour ou None si non trouvé

        Side-effects:
            - DB: Met à jour Entity et commit
        """
        ...

    def delete(self, entity_id: int) -> bool:
        """Supprime une entité.

        Args:
            entity_id: ID de l'entité à supprimer

        Returns:
            True si supprimé, False si non trouvé

        Side-effects:
            - DB: Supprime Entity et commit
        """
        ...


# ===== Implémentation concrète (SQLAlchemy) =====


class EntityRepository:
    """Repository SQLAlchemy pour Entity.

    Implémentation concrète du port EntityRepositoryPort utilisant SQLAlchemy.
    Cette classe convertit les modèles SQLAlchemy en DTOs pour maintenir
    le découplage avec la couche Application.
    """

    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialise le repository."""
        # Import ici pour éviter les dépendances circulaires
        from models import Entity

        self._model = Entity

    def _to_dto(self, entity: Any) -> EntityDTO:
        """Convertit un modèle SQLAlchemy en DTO.

        Args:
            entity: Modèle SQLAlchemy Entity

        Returns:
            EntityDTO correspondant
        """
        return EntityDTO(
            id=entity.id,
            field1=entity.field1,
            field2=entity.field2,
        )

    def find_by_id(self, entity_id: int) -> EntityDTO | None:
        """Trouve une entité par son ID.

        Args:
            entity_id: ID de l'entité

        Returns:
            EntityDTO ou None si non trouvé
        """
        entity = self._model.query.get(entity_id)
        if entity is None:
            return None
        return self._to_dto(entity)

    def find_all(self, company_id: int) -> list[EntityDTO]:
        """Trouve toutes les entités d'une entreprise.

        Args:
            company_id: ID de l'entreprise

        Returns:
            Liste de EntityDTO
        """
        entities = self._model.query.filter_by(company_id=company_id).all()
        return [self._to_dto(e) for e in entities]

    def find_by_criteria(
        self, *, company_id: int, status: str | None = None
    ) -> list[EntityDTO]:
        """Trouve des entités selon des critères.

        Args:
            company_id: ID de l'entreprise
            status: Filtre par statut (optionnel)

        Returns:
            Liste de EntityDTO
        """
        query = self._model.query.filter_by(company_id=company_id)
        if status:
            query = query.filter_by(status=status)
        entities = query.all()
        return [self._to_dto(e) for e in entities]

    def create(self, *, field1: str, field2: int, company_id: int) -> EntityDTO:
        """Crée une nouvelle entité.

        Args:
            field1: Description du champ 1
            field2: Description du champ 2
            company_id: ID de l'entreprise

        Returns:
            EntityDTO créé

        Side-effects:
            - DB: Crée Entity et commit
        """
        from models import db

        entity = self._model()
        entity.field1 = field1
        entity.field2 = field2
        entity.company_id = company_id

        db.session.add(entity)
        db.session.commit()
        return self._to_dto(entity)

    def update(self, entity_id: int, *, field1: str | None = None) -> EntityDTO | None:
        """Met à jour une entité existante.

        Args:
            entity_id: ID de l'entité à mettre à jour
            field1: Nouvelle valeur pour field1 (optionnel)

        Returns:
            EntityDTO mis à jour ou None si non trouvé

        Side-effects:
            - DB: Met à jour Entity et commit
        """
        from models import db

        entity = self._model.query.get(entity_id)
        if entity is None:
            return None

        if field1 is not None:
            entity.field1 = field1

        db.session.commit()
        return self._to_dto(entity)

    def delete(self, entity_id: int) -> bool:
        """Supprime une entité.

        Args:
            entity_id: ID de l'entité à supprimer

        Returns:
            True si supprimé, False si non trouvé

        Side-effects:
            - DB: Supprime Entity et commit
        """
        from models import db

        entity = self._model.query.get(entity_id)
        if entity is None:
            return False

        db.session.delete(entity)
        db.session.commit()
        return True
