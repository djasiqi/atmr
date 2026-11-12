"""Contrat de diffusion réseau : choix, zone structurée, besoins non modélisés."""

from application.institutions.network_eligibility import (
    area_covers,
    network_notice,
    resolve_dispatch_target,
    unmodeled_network_needs,
)


def test_default_target_follows_existing_preferences() -> None:
    assert resolve_dispatch_target(None, has_preferences=False) == "network"
    assert resolve_dispatch_target(None, has_preferences=True) == "partners"
    assert resolve_dispatch_target("network", has_preferences=True) == "network"
    assert resolve_dispatch_target("partners", has_preferences=False) == "partners"


def test_unknown_zone_does_not_cover_and_national_is_explicit() -> None:
    assert area_covers(
        mode="B_PICKUP_ONLY",
        area_geo_unit_id=9,
        pickup_ids={1, 2},
        drop_ids=set(),
        drop_known=False,
        same_canton=False,
    ) is False
    assert area_covers(
        mode="D_NATIONAL",
        area_geo_unit_id=9,
        pickup_ids={1},
        drop_ids=set(),
        drop_known=False,
        same_canton=False,
    ) is True
    assert area_covers(
        mode="C_INTRA_ONLY",
        area_geo_unit_id=1,
        pickup_ids={1},
        drop_ids=set(),
        drop_known=False,
        same_canton=False,
    ) is False


def test_personal_wheelchair_is_modeled_but_stretcher_is_not() -> None:
    assert unmodeled_network_needs({"wheelchair": True}) == []
    assert unmodeled_network_needs({"stretcher": True, "oxygen": True}) == [
        "brancard",
        "oxygène",
    ]
    assert unmodeled_network_needs({"vehicle_wheelchair": True}) == []
    assert unmodeled_network_needs({"needs_assistance": True}) == []
    assert unmodeled_network_needs(
        {"needs_assistance": True, "assistance_type": "soins"}
    ) == ["accompagnement spécialisé"]


def test_pickup_match_includes_company_and_miss_excludes_it(monkeypatch) -> None:
    from types import SimpleNamespace

    from application.institutions import network_eligibility as eligibility

    inside = SimpleNamespace(id=7)
    outside = SimpleNamespace(id=8)
    pickup = SimpleNamespace(id=1, name="Genève", type=None)

    class _Rows:
        def __init__(self, rows):
            self.rows = rows

        def filter(self, *_args, **_kwargs):
            return self

        def with_entities(self, *_args, **_kwargs):
            return self

        def all(self):
            return self.rows

    def fake_catalog(**_kwargs: object) -> list[SimpleNamespace]:
        return [inside, outside]

    monkeypatch.setattr(
        eligibility,
        "_local_unit",
        lambda text: pickup if text == "prise" else None,
    )
    monkeypatch.setattr(
        eligibility,
        "geo_chain",
        lambda unit: [unit] if unit is not None else [],
    )
    monkeypatch.setattr(
        "application.institutions.eligible_carriers.companies_for_marketplace_dispatch",
        fake_catalog,
    )
    class _Col:
        def in_(self, _values):
            return self

        def is_(self, _value):
            return self

    class _Area:
        company_id = _Col()
        is_active = _Col()
        query = _Rows(
            [
                SimpleNamespace(company_id=7, geo_unit_id=1, coverage_mode="B_PICKUP_ONLY"),
                SimpleNamespace(company_id=8, geo_unit_id=99, coverage_mode="B_PICKUP_ONLY"),
            ]
        )

    class _Vehicle:
        company_id = _Col()
        is_active = _Col()
        wheelchair_accessible = _Col()
        query = _Rows([])

    monkeypatch.setattr(eligibility, "ServiceArea", _Area)
    monkeypatch.setattr(eligibility, "Vehicle", _Vehicle)

    request = SimpleNamespace(
        pickup_location="prise",
        dropoff_location="depose",
        institution=object(),
        get_mobility=lambda: {"wheelchair": False},
    )
    selection = eligibility.select_network_companies(request)
    assert selection.error is None
    assert [company.id for company in selection.companies] == [7]

    request_chair = SimpleNamespace(
        pickup_location="prise",
        dropoff_location="depose",
        institution=object(),
        get_mobility=lambda: {"wheelchair": True},
    )
    _Vehicle.query = _Rows([(7,)])
    with_chair = eligibility.select_network_companies(request_chair)
    assert [company.id for company in with_chair.companies] == [7]

    _Vehicle.query = _Rows([])
    without_chair = eligibility.select_network_companies(request_chair)
    assert without_chair.companies == ()
    assert without_chair.error is not None

    _Area.query = _Rows([])
    empty_zone = eligibility.select_network_companies(request)
    assert empty_zone.companies == ()
    assert "Aucun transporteur du réseau" in (empty_zone.error or "")


def test_unmodeled_need_does_not_open_the_catalog(monkeypatch) -> None:
    from types import SimpleNamespace

    from application.institutions import network_eligibility as eligibility

    called = {"catalog": False}

    def fail_catalog(**_kwargs: object) -> list[object]:
        called["catalog"] = True
        return []

    monkeypatch.setattr(
        "application.institutions.eligible_carriers.companies_for_marketplace_dispatch",
        fail_catalog,
    )
    request = SimpleNamespace(
        pickup_location="Genève",
        dropoff_location="Lausanne",
        institution=object(),
        get_mobility=lambda: {"stretcher": True, "oxygen": True},
    )
    selection = eligibility.select_network_companies(request)
    assert called["catalog"] is False
    assert selection.companies == ()
    assert "brancard" in (selection.error or "")
    assert "oxygène" in (selection.error or "")


def test_declared_capabilities_filter_without_opening_specialized_needs(monkeypatch) -> None:
    from types import SimpleNamespace

    from application.institutions import network_eligibility as eligibility

    declared = SimpleNamespace(
        id=7,
        provides_standard_assistance=True,
        provides_manual_wheelchair=True,
        manual_wheelchair_count=1,
    )
    silent = SimpleNamespace(
        id=8,
        provides_standard_assistance=None,
        provides_manual_wheelchair=False,
        manual_wheelchair_count=None,
    )

    class _Rows:
        def __init__(self, rows):
            self.rows = rows

        def filter(self, *_args, **_kwargs):
            return self

        def with_entities(self, *_args, **_kwargs):
            return self

        def all(self):
            return self.rows

    class _Col:
        def in_(self, _values):
            return self

        def is_(self, _value):
            return self

    class _Area:
        company_id = _Col()
        is_active = _Col()
        query = _Rows(
            [
                SimpleNamespace(company_id=7, geo_unit_id=1, coverage_mode="B_PICKUP_ONLY"),
                SimpleNamespace(company_id=8, geo_unit_id=1, coverage_mode="B_PICKUP_ONLY"),
            ]
        )

    class _Vehicle:
        company_id = _Col()
        is_active = _Col()
        wheelchair_accessible = _Col()
        query = _Rows([])

    monkeypatch.setattr(
        "application.institutions.eligible_carriers.companies_for_marketplace_dispatch",
        lambda **_kwargs: [declared, silent],
    )
    monkeypatch.setattr(eligibility, "_local_unit", lambda _text: SimpleNamespace(id=1, name="Genève", type=None))
    monkeypatch.setattr(eligibility, "geo_chain", lambda unit: [unit])
    monkeypatch.setattr(eligibility, "ServiceArea", _Area)
    monkeypatch.setattr(eligibility, "Vehicle", _Vehicle)

    assistance = eligibility.select_network_companies(
        SimpleNamespace(
            pickup_location="prise",
            dropoff_location="depose",
            institution=object(),
            get_mobility=lambda: {"needs_assistance": True},
        )
    )
    assert [company.id for company in assistance.companies] == [7]

    chair = eligibility.select_network_companies(
        SimpleNamespace(
            pickup_location="prise",
            dropoff_location="depose",
            institution=object(),
            get_mobility=lambda: {"vehicle_wheelchair": True},
        )
    )
    assert [company.id for company in chair.companies] == [7]

    declared.manual_wheelchair_count = None
    without_stock = eligibility.select_network_companies(
        SimpleNamespace(
            pickup_location="prise",
            dropoff_location="depose",
            institution=object(),
            get_mobility=lambda: {"vehicle_wheelchair": True},
        )
    )
    assert without_stock.companies == ()
    assert "Aucun transporteur du réseau" in (without_stock.error or "")

    specialized = eligibility.select_network_companies(
        SimpleNamespace(
            pickup_location="prise",
            dropoff_location="depose",
            institution=object(),
            get_mobility=lambda: {
                "needs_assistance": True,
                "assistance_type": "accompagnement médical",
            },
        )
    )
    assert specialized.companies == ()
    assert "accompagnement spécialisé" in (specialized.error or "")


def test_network_notice_omits_patient_identity() -> None:
    notice = network_notice(place="Genève", when_label="21.10.2026 14:00")
    assert notice == "Nouvelle demande de transport — Genève, 21.10.2026 14:00"
    assert "Martin" not in notice
    assert "patient" not in notice.lower()
