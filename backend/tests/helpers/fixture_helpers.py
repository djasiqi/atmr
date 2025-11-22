# backend/tests/helpers/fixture_helpers.py
"""Helpers pour créer des fixtures découplées et réutilisables.

Ce module fournit des fonctions utilitaires pour créer des fixtures
indépendantes qui peuvent être utilisées sans dépendances explicites.
"""

from __future__ import annotations

from typing import Any, Callable, Type

from tests.conftest import persisted_fixture


def create_independent_fixture(
    db_session: Any,
    factory_func: Callable[[], Any],
    model_class: Type[Any],
    *,
    count: int = 1,
    **factory_kwargs: Any,
) -> Any | list[Any]:
    """Crée une fixture indépendante (sans dépendances).

    Cette fonction crée un objet ou une liste d'objets persistés,
    sans dépendre d'autres fixtures.

    Args:
        db_session: Session SQLAlchemy
        factory_func: Fonction factory (ex: CompanyFactory)
        model_class: Classe du modèle SQLAlchemy
        count: Nombre d'objets à créer (1 = objet unique, >1 = liste)
        **factory_kwargs: Arguments à passer à la factory

    Returns:
        Objet unique si count=1, liste d'objets si count>1

    Example:
        ```python
        @pytest.fixture
        def company(db):
            return create_independent_fixture(db, CompanyFactory, Company)

        @pytest.fixture
        def drivers(db):
            # Crée sa propre company si nécessaire
            company = create_independent_fixture(db, CompanyFactory, Company)
            return create_independent_fixture(
                db,
                lambda: DriverFactory(company=company),
                Driver,
                count=3,
            )
        ```
    """
    if count == 1:
        instance = factory_func(**factory_kwargs)
        return persisted_fixture(db_session, instance, model_class)

    instances = [factory_func(**factory_kwargs) for _ in range(count)]
    return [
        persisted_fixture(db_session, instance, model_class) for instance in instances
    ]


def create_fixture_with_optional_dependency(
    db_session: Any,
    factory_func: Callable[[Any], Any],
    model_class: Type[Any],
    dependency: Any | None = None,
    dependency_factory: Callable[[], Any] | None = None,
    *,
    count: int = 1,
    **factory_kwargs: Any,
) -> Any | list[Any]:
    """Crée une fixture avec dépendance optionnelle.

    Si la dépendance n'est pas fournie, elle est créée automatiquement
    via la factory de dépendance.

    Args:
        db_session: Session SQLAlchemy
        factory_func: Fonction factory qui prend la dépendance en paramètre
        model_class: Classe du modèle SQLAlchemy
        dependency: Dépendance existante (optionnelle)
        dependency_factory: Factory pour créer la dépendance si None
        count: Nombre d'objets à créer
        **factory_kwargs: Arguments supplémentaires pour la factory

    Returns:
        Objet unique si count=1, liste d'objets si count>1

    Example:
        ```python
        @pytest.fixture
        def drivers(db, company=None):
            return create_fixture_with_optional_dependency(
                db,
                lambda c: DriverFactory(company=c),
                Driver,
                dependency=company,
                dependency_factory=lambda: CompanyFactory(),
                count=3,
            )
        ```
    """
    # Créer la dépendance si nécessaire
    if dependency is None and dependency_factory is not None:
        dependency = persisted_fixture(
            db_session, dependency_factory(), type(dependency)
        )

    # Créer les objets
    if count == 1:
        instance = factory_func(dependency, **factory_kwargs)
        return persisted_fixture(db_session, instance, model_class)

    instances = [factory_func(dependency, **factory_kwargs) for _ in range(count)]
    return [
        persisted_fixture(db_session, instance, model_class) for instance in instances
    ]
