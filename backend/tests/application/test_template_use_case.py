"""Couverture du template de use case ``application._template_use_case``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from application._template_use_case import (
    CreateEntityInput,
    CreateEntityUseCase,
    GetEntityInput,
    GetEntityUseCase,
    ListEntitiesInput,
    ListEntitiesOutput,
    ListEntitiesUseCase,
    _RepositoryPort,
)


class _FakeRepo:
    def __init__(
        self,
        *,
        entity: Any | None = None,
        create_error: Exception | None = None,
    ) -> None:
        self._entity = entity
        self._create_error = create_error
        self.created: dict[str, Any] | None = None

    def find_by_id(self, entity_id: int) -> Any | None:
        if self._entity is not None and getattr(self._entity, "id", None) == entity_id:
            return self._entity
        return self._entity if entity_id == 1 else None

    def create_and_commit(self, *, field1: str, field2: int) -> Any:
        if self._create_error:
            raise self._create_error
        self.created = {"field1": field1, "field2": field2}
        return SimpleNamespace(id=42, field1=field1, field2=field2)


def test_protocol_stubs_ellipsis():
    assert _RepositoryPort.find_by_id(None, 1) is None  # type: ignore[arg-type]
    assert (
        _RepositoryPort.create_and_commit(None, field1="a", field2=1) is None
    )  # type: ignore[arg-type]


def test_create_entity_validation_et_succes():
    uc = CreateEntityUseCase(entity_repo=_FakeRepo())

    empty = uc.execute(CreateEntityInput(field1="", field2=0))
    assert empty.success is False
    assert empty.status_code == 400
    assert empty.error is not None
    assert "field1" in empty.error
    assert "field2" in empty.error

    blank = uc.execute(CreateEntityInput(field1="   ", field2=3))
    assert blank.success is False
    assert blank.error is not None
    assert "field1" in blank.error

    ok = uc.execute(CreateEntityInput(field1="valeur", field2=7))
    assert ok.success is True
    assert ok.entity_id == 42
    assert ok.entity is not None


def test_create_entity_erreurs_repo():
    uc_val = CreateEntityUseCase(
        entity_repo=_FakeRepo(create_error=ValueError("doublon"))
    )
    bad = uc_val.execute(CreateEntityInput(field1="x", field2=1))
    assert bad.success is False
    assert bad.status_code == 400
    assert bad.error == {"error": "doublon"}

    uc_boom = CreateEntityUseCase(
        entity_repo=_FakeRepo(create_error=RuntimeError("db down"))
    )
    boom = uc_boom.execute(CreateEntityInput(field1="x", field2=1))
    assert boom.success is False
    assert boom.status_code == 500
    assert boom.error == {"error": "Erreur interne"}


def test_get_entity_trouve_et_404():
    entity = SimpleNamespace(id=1)
    uc = GetEntityUseCase(entity_repo=_FakeRepo(entity=entity))
    found = uc.execute(GetEntityInput(entity_id=1, company_id=9))
    assert found.found is True
    assert found.entity is entity

    missing = uc.execute(GetEntityInput(entity_id=99, company_id=9))
    assert missing.found is False
    assert missing.status_code == 404
    assert missing.error == {"error": "Entité non trouvée"}


def test_list_entities_pagination_et_succes():
    uc = ListEntitiesUseCase(entity_repo=_FakeRepo())

    bad_page = uc.execute(ListEntitiesInput(company_id=1, page=0))
    assert bad_page.success is False
    assert bad_page.status_code == 400
    assert bad_page.error is not None
    assert "page" in bad_page.error

    bad_low = uc.execute(ListEntitiesInput(company_id=1, per_page=0))
    assert bad_low.success is False
    assert "per_page" in (bad_low.error or {})

    bad_high = uc.execute(ListEntitiesInput(company_id=1, per_page=101))
    assert bad_high.success is False
    assert "per_page" in (bad_high.error or {})

    ok = uc.execute(ListEntitiesInput(company_id=1, page=2, per_page=10))
    assert ok.success is True
    assert ok.entities == []
    assert ok.total == 0
    assert ok.page == 2
    assert ok.per_page == 10
    assert ok.total_pages == 0


def test_list_entities_exception(monkeypatch):
    real = ListEntitiesOutput

    def _flaky(*args: Any, **kwargs: Any) -> ListEntitiesOutput:
        if kwargs.get("success") is True:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "application._template_use_case.ListEntitiesOutput",
        _flaky,
    )
    uc = ListEntitiesUseCase(entity_repo=_FakeRepo())
    out = uc.execute(ListEntitiesInput(company_id=1))
    assert out.success is False
    assert out.status_code == 500
    assert out.error == {"error": "Erreur interne"}
