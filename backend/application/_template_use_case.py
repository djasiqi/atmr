"""Template pour créer un nouveau use case conforme au pattern standardisé.

Copier ce fichier et adapter selon les besoins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ===== Protocols (Ports) =====


class _RepositoryPort(Protocol):
    """Port pour le repository (adapter depuis infrastructure)."""

    def find_by_id(self, entity_id: int) -> Any | None:
        """Trouve une entité par ID."""
        ...

    def create_and_commit(self, *, field1: str, field2: int) -> Any:
        """Crée et commit une entité."""
        ...


# ===== Input/Output Dataclasses =====


@dataclass(frozen=True, slots=True)
class CreateEntityInput:
    """Input pour le use case de création d'entité.

    Attributes:
        field1: Description du champ 1
        field2: Description du champ 2
    """

    field1: str
    field2: int


@dataclass(frozen=True, slots=True)
class CreateEntityOutput:
    """Output du use case de création d'entité.

    Attributes:
        success: True si l'opération a réussi
        entity_id: ID de l'entité créée (si succès)
        entity: Entité créée (si succès)
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    entity_id: int | None = None
    entity: Any | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


# ===== Use Case =====


class CreateEntityUseCase:
    """Use-case Application: créer une entité.

    Responsabilités:
        - Validation des inputs
        - Orchestration de la logique métier
        - Appels aux repositories/services
        - Gestion des erreurs

    Exemple:
        >>> repo = EntityRepository()
        >>> uc = CreateEntityUseCase(entity_repo=repo)
        >>> result = uc.execute(CreateEntityInput(field1="value", field2=42))
        >>> if result.success:
        ...     print(f"Entité créée: {result.entity_id}")
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        *,
        entity_repo: _RepositoryPort,
    ) -> None:
        """Initialise le use case avec ses dépendances.

        Args:
            entity_repo: Repository pour les entités
        """
        self._entity_repo = entity_repo

    def execute(self, input_data: CreateEntityInput) -> CreateEntityOutput:
        """Exécute le use case.

        Args:
            input_data: Données d'entrée du use case

        Returns:
            CreateEntityOutput avec le résultat
        """
        # 1. Validation des inputs
        validation_error = self._validate_input(input_data)
        if validation_error:
            return CreateEntityOutput(
                success=False,
                error=validation_error,
                status_code=400,
            )

        # 2. Logique métier et persistance
        try:
            entity = self._entity_repo.create_and_commit(
                field1=input_data.field1,
                field2=input_data.field2,
            )

            return CreateEntityOutput(
                success=True,
                entity_id=entity.id,
                entity=entity,
            )
        except ValueError as e:
            # Erreurs de validation métier
            logger.warning("Erreur de validation métier: %s", e)
            return CreateEntityOutput(
                success=False,
                error={"error": str(e)},
                status_code=400,
            )
        except Exception:
            # Erreurs inattendues
            logger.exception("Erreur inattendue dans CreateEntityUseCase")
            return CreateEntityOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )

    def _validate_input(self, input_data: CreateEntityInput) -> dict[str, str] | None:
        """Valide les inputs du use case.

        Args:
            input_data: Input à valider

        Returns:
            None si valide, dict d'erreurs sinon
        """
        errors: dict[str, str] = {}

        # Validation des champs
        if not input_data.field1 or len(input_data.field1.strip()) == 0:
            errors["field1"] = "Le champ field1 est requis"

        if input_data.field2 <= 0:
            errors["field2"] = "Le champ field2 doit être positif"

        # Validation de dépendances (ex: entité existe)
        # existing = self._entity_repo.find_by_id(input_data.field2)
        # if not existing:
        #     errors["field2"] = "Entité non trouvée"

        return errors if errors else None


# ===== Exemple: Get Use Case =====


@dataclass(frozen=True, slots=True)
class GetEntityInput:
    """Input pour récupérer une entité."""

    entity_id: int
    company_id: int  # Pour vérification d'autorisation


@dataclass(frozen=True, slots=True)
class GetEntityOutput:
    """Output pour récupérer une entité."""

    found: bool
    entity: Any | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class GetEntityUseCase:
    """Use-case Application: récupérer une entité par ID."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, *, entity_repo: _RepositoryPort
    ) -> None:
        self._entity_repo = entity_repo

    def execute(self, input_data: GetEntityInput) -> GetEntityOutput:
        """Récupère une entité par son ID.

        Args:
            input_data: Input avec entity_id et company_id

        Returns:
            GetEntityOutput avec l'entité si trouvée
        """
        entity = self._entity_repo.find_by_id(input_data.entity_id)

        if not entity:
            return GetEntityOutput(
                found=False,
                error={"error": "Entité non trouvée"},
                status_code=404,
            )

        # Vérification d'autorisation (ex: entity.company_id == input_data.company_id)
        # if entity.company_id != input_data.company_id:
        #     return GetEntityOutput(
        #         found=False,
        #         error={"error": "Accès refusé"},
        #         status_code=403,
        #     )

        return GetEntityOutput(found=True, entity=entity)


# ===== Exemple: List Use Case =====


@dataclass(frozen=True, slots=True)
class ListEntitiesInput:
    """Input pour lister des entités."""

    company_id: int
    page: int = 1
    per_page: int = 20
    status_filter: str | None = None


@dataclass(frozen=True, slots=True)
class ListEntitiesOutput:
    """Output pour lister des entités."""

    success: bool
    entities: list[Any] | None = None
    total: int | None = None
    page: int | None = None
    per_page: int | None = None
    total_pages: int | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class ListEntitiesUseCase:
    """Use-case Application: lister des entités avec pagination."""

    MAX_PER_PAGE = 100

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, *, entity_repo: _RepositoryPort
    ) -> None:
        self._entity_repo = entity_repo

    def execute(self, input_data: ListEntitiesInput) -> ListEntitiesOutput:
        """Liste des entités avec pagination.

        Args:
            input_data: Input avec filtres et pagination

        Returns:
            ListEntitiesOutput avec les entités paginées
        """
        # Validation
        if input_data.page < 1:
            return ListEntitiesOutput(
                success=False,
                error={"page": "Le numéro de page doit être >= 1"},
                status_code=400,
            )

        if input_data.per_page < 1 or input_data.per_page > self.MAX_PER_PAGE:
            return ListEntitiesOutput(
                success=False,
                error={
                    "per_page": f"Le nombre par page doit être entre 1 et {self.MAX_PER_PAGE}"
                },
                status_code=400,
            )

        try:
            # Récupération paginée (exemple avec SQLAlchemy)
            # query = self._entity_repo.find_by_company_id_query(
            #     company_id=input_data.company_id,
            #     status_filter=input_data.status_filter,
            # )
            # pagination = query.paginate(
            #     page=input_data.page, per_page=input_data.per_page, error_out=False
            # )
            # total = pagination.total or 0
            # entities = pagination.items or []
            # total_pages = (total + input_data.per_page - 1) // input_data.per_page if total > 0 else 0

            # Exemple simplifié
            entities = []  # self._entity_repo.find_by_company_id(...)
            total = 0
            total_pages = 0

            return ListEntitiesOutput(
                success=True,
                entities=entities,
                total=total,
                page=input_data.page,
                per_page=input_data.per_page,
                total_pages=total_pages,
            )
        except Exception:
            logger.exception("Erreur lors de la liste des entités")
            return ListEntitiesOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
