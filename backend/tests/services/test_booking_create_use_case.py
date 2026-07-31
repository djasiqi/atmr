"""Tests de caractérisation — CreateBookingUseCase canonique + façade."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from marshmallow import ValidationError

from application.bookings.create_booking import CreateBookingUseCase
from domain.bookings.commands import CreateBookingCommand
from domain.events.events import BookingCreatedEvent
from models.enums import ClientType
from services.platform_exceptions import PlatformTenantSuspended
from shared.booking_company_resolution import (
    resolve_booking_owner_company_id_for_create,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent


class _FakeClientRepo:
    def __init__(
        self,
        company_id: int = 7,
        *,
        preferential_rate: float | None = None,
    ) -> None:
        self._company_id = company_id
        self._preferential_rate = preferential_rate

    def find_by_id(self, _client_id: int):  # type: ignore[no-untyped-def]
        ct = ClientType.TRANSPORT if (self._company_id or 0) > 0 else ClientType.PORTAL
        return SimpleNamespace(
            id=1,
            company_id=self._company_id,
            client_type=ct,
            preferential_rate=self._preferential_rate,
        )


class _FakeCompanyRepo:
    def find_model_by_id(self, _company_id: int):  # type: ignore[no-untyped-def]
        return None


class _FakeBookingWriter:
    def __init__(self, *, booking_id: int = 123) -> None:
        self.last_kwargs: dict[str, Any] = {}
        self.calls = 0
        self._booking_id = booking_id
        self.fail = False

    def create_and_commit(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.fail:
            raise RuntimeError("writer failed")
        self.last_kwargs = kwargs
        return SimpleNamespace(id=self._booking_id, company_id=7, status="pending")


class _FakeGeocoding:
    def __init__(self, *, succeed: bool = True) -> None:
        self.succeed = succeed
        self.calls: list[str] = []

    def geocode_address(  # type: ignore[no-untyped-def]
        self, address: str, *, country: str | None = None
    ):
        _ = country
        self.calls.append(address)
        if self.succeed:
            return {"lat": 46.2, "lon": 6.1}
        return None


def _admin_token(**_kwargs: Any) -> dict[str, str]:
    return {
        "token": "commune:6630",
        "canton_code": "GE",
        "source": "db",
        "confidence": "authoritative",
        "label": "Anieres (GE)",
    }


def _base_cmd(**overrides: Any) -> CreateBookingCommand:
    data: dict[str, Any] = {
        "customer_name": "A",
        "pickup_location": "Rue Pickup 1, Geneve",
        "dropoff_location": "Rue Dropoff 2, Geneve",
        "scheduled_time": datetime.now(UTC).isoformat(timespec="seconds"),
        "amount": 10.0,
        "is_round_trip": False,
    }
    data.update(overrides)
    return CreateBookingCommand(user_id=1, client_id=1, data=data)


def _build_uc(
    *,
    writer: _FakeBookingWriter | None = None,
    geocoding: _FakeGeocoding | None = None,
    client_repo: _FakeClientRepo | None = None,
    distance_fn: Any = None,
    gate_fn: Any = None,
    trigger_async: Any = None,
    distance_calls: list[tuple[str, str]] | None = None,
) -> CreateBookingUseCase:
    dist_log = distance_calls if distance_calls is not None else []

    def _distance(pickup: str, dropoff: str) -> tuple[int, int]:
        dist_log.append((pickup, dropoff))
        if distance_fn is not None:
            return distance_fn(pickup, dropoff)
        return (60, 1000)

    return CreateBookingUseCase(
        client_repo=client_repo or _FakeClientRepo(company_id=7),  # type: ignore[arg-type]
        company_lookup=_FakeCompanyRepo(),  # type: ignore[arg-type]
        booking_writer=writer or _FakeBookingWriter(),  # type: ignore[arg-type]
        geocoding_service=geocoding or _FakeGeocoding(),  # type: ignore[arg-type]
        distance_duration_fn=_distance,
        company_creation_gate_fn=gate_fn or (lambda _cid: None),
        fallback_coords_fn=lambda _company: (46.2044, 6.1432),
        trigger_async_geocoding_fn=trigger_async,
    )


_SCHEMA_SIDE_CHANNELS = frozenset({"amount_source", "bill_to_patient"})


def _allow_cmd_data_side_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    """amount_source / bill_to_patient sont lus sur cmd.data mais absents du schema.

    On les retire uniquement pour Marshmallow ; cmd.data reste intact pour le UC.
    """
    from schemas.booking_schemas import BookingCreateSchema

    original_load = BookingCreateSchema.load

    def _load(self, data, *args, **kwargs):  # type: ignore[no-untyped-def]
        filtered = {
            key: value
            for key, value in dict(data or {}).items()
            if key not in _SCHEMA_SIDE_CHANNELS
        }
        return original_load(self, filtered, *args, **kwargs)

    monkeypatch.setattr(BookingCreateSchema, "load", _load)


def _patch_common(monkeypatch: pytest.MonkeyPatch, mod: Any) -> None:
    monkeypatch.setattr(mod, "resolve_pickup_admin", _admin_token)
    monkeypatch.setattr(
        mod,
        "geo_unit_id_from_pickup_admin_token",
        lambda _token: None,
    )
    import services.billing.client_stay_resolver as stay_mod

    monkeypatch.setattr(stay_mod, "find_active_stay_for_client", lambda **_k: None)
    monkeypatch.setattr(stay_mod, "get_clinic_address_for_stay", lambda _s: None)


# ---------------------------------------------------------------------------
# Tests existants (gate injecté)
# ---------------------------------------------------------------------------


def test_create_booking_use_case_publishes_booking_created_event(monkeypatch) -> None:
    published: list[dict[str, Any]] = []

    import application.bookings.create_booking as mod

    def fake_publish_event(evt):  # type: ignore[no-untyped-def]
        published.append(evt.to_dict())

    monkeypatch.setattr(mod, "publish_event", fake_publish_event)
    _patch_common(monkeypatch, mod)

    writer = _FakeBookingWriter()
    uc = _build_uc(writer=writer)
    booking = uc.execute(_base_cmd())
    assert getattr(booking, "id", None) == 123

    assert published
    assert published[0]["event_type"] == "BookingCreatedEvent"
    assert published[0]["booking_id"] == 123
    assert published[0]["company_id"] == 7
    assert writer.last_kwargs["pickup_admin_source"] in {
        "db",
        "geoadmin",
        "photon",
        "unknown",
    }
    assert writer.last_kwargs["dropoff_admin_source"] in {
        "db",
        "geoadmin",
        "photon",
        "unknown",
    }
    assert "pickup_admin_resolved_at" in writer.last_kwargs
    assert "dropoff_admin_resolved_at" in writer.last_kwargs


def test_create_booking_use_case_invalid_scheduled_time_raises_value_error() -> None:
    uc = _build_uc()
    with pytest.raises(ValidationError, match="scheduled_time"):
        uc.execute(
            CreateBookingCommand(
                user_id=1,
                client_id=1,
                data={
                    "customer_name": "A",
                    "pickup_location": "X",
                    "dropoff_location": "Y",
                    "scheduled_time": "not-a-date",
                    "amount": 10.0,
                },
            )
        )


def test_resolve_booking_owner_company_id_order() -> None:
    assert (
        resolve_booking_owner_company_id_for_create(
            SimpleNamespace(
                company_id=3,
                default_billed_to_company_id=9,
                client_type=ClientType.TRANSPORT,
            )
        )
        == 3
    )
    assert (
        resolve_booking_owner_company_id_for_create(
            SimpleNamespace(
                company_id=None,
                default_billed_to_company_id=9,
                client_type=ClientType.PORTAL,
            )
        )
        is None
    )
    with pytest.raises(ValueError, match="Client TRANSPORT sans company_id"):
        resolve_booking_owner_company_id_for_create(
            SimpleNamespace(
                company_id=0,
                default_billed_to_company_id=None,
                client_type=ClientType.TRANSPORT,
            )
        )


def test_resolve_portal_without_attachment_is_open_market() -> None:
    assert (
        resolve_booking_owner_company_id_for_create(
            SimpleNamespace(
                company_id=None,
                default_billed_to_company_id=None,
                client_type=ClientType.PORTAL,
            )
        )
        is None
    )


def test_resolve_booking_owner_company_id_unknown_client_type_raises() -> None:
    with pytest.raises(ValueError, match="ClientType non géré"):
        resolve_booking_owner_company_id_for_create(
            SimpleNamespace(
                company_id=42,
                client_type="CORPORATE",
            )
        )


def test_resolve_portal_always_open_market() -> None:
    assert (
        resolve_booking_owner_company_id_for_create(
            SimpleNamespace(
                company_id=99,
                default_billed_to_company_id=88,
                client_type=ClientType.PORTAL,
            )
        )
        is None
    )


# ---------------------------------------------------------------------------
# Plan PR1 — caractérisation
# ---------------------------------------------------------------------------


def test_legacy_facade_exports_exact_canonical_class() -> None:
    from bookings.application.use_cases.create_booking import (
        CreateBookingUseCase as FacadeUC,
    )

    assert FacadeUC is CreateBookingUseCase


def test_suspended_company_fails_before_stay_pricing_distance_geocoding_and_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import application.bookings.create_booking as mod
    import services.billing.client_stay_resolver as stay_mod

    stay_calls: list[Any] = []
    gate_calls: list[int] = []
    published: list[Any] = []
    distance_calls: list[tuple[str, str]] = []
    async_calls: list[Any] = []

    def suspended_gate(company_id: int) -> None:
        gate_calls.append(company_id)
        raise PlatformTenantSuspended()

    monkeypatch.setattr(mod, "publish_event", lambda evt: published.append(evt))
    monkeypatch.setattr(mod, "resolve_pickup_admin", _admin_token)
    monkeypatch.setattr(
        stay_mod,
        "find_active_stay_for_client",
        lambda **kwargs: stay_calls.append(kwargs) or None,
    )

    geocoding = _FakeGeocoding()
    writer = _FakeBookingWriter()
    freeze_mock = MagicMock(return_value=(1, 2, 15.0, {"x": 1}))
    monkeypatch.setattr(
        CreateBookingUseCase, "_compute_pricing_freeze", freeze_mock
    )

    uc = _build_uc(
        writer=writer,
        geocoding=geocoding,
        gate_fn=suspended_gate,
        distance_calls=distance_calls,
        trigger_async=lambda *a: async_calls.append(a),
    )

    with pytest.raises(PlatformTenantSuspended):
        uc.execute(_base_cmd())

    assert gate_calls == [7]
    assert stay_calls == []
    assert distance_calls == []
    assert geocoding.calls == []
    assert freeze_mock.call_count == 0
    assert writer.calls == 0
    assert async_calls == []
    assert published == []


def test_active_stay_rewrites_pickup_before_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import application.bookings.create_booking as mod
    import services.billing.client_stay_resolver as stay_mod

    clinic_address = "Clinique Geneve, Avenue de la Clinique 1"
    monkeypatch.setattr(mod, "publish_event", lambda _evt: None)
    monkeypatch.setattr(mod, "resolve_pickup_admin", _admin_token)
    monkeypatch.setattr(
        stay_mod,
        "find_active_stay_for_client",
        lambda **_k: SimpleNamespace(id=1),
    )
    monkeypatch.setattr(
        stay_mod,
        "get_clinic_address_for_stay",
        lambda _stay: {
            "address": clinic_address,
            "clinic_name": "Clinique",
            "clinic_id": 9,
            "preferential_rate": None,
        },
    )

    distance_calls: list[tuple[str, str]] = []
    uc = _build_uc(distance_calls=distance_calls)
    uc.execute(_base_cmd(pickup_location="Adresse domicile"))
    assert distance_calls
    assert distance_calls[0][0] == clinic_address


def test_clinic_preferential_rate_wins_over_client_manual_and_computed_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import application.bookings.create_booking as mod
    import services.billing.client_stay_resolver as stay_mod

    _allow_cmd_data_side_channels(monkeypatch)
    monkeypatch.setattr(mod, "publish_event", lambda _evt: None)
    monkeypatch.setattr(mod, "resolve_pickup_admin", _admin_token)
    monkeypatch.setattr(
        stay_mod,
        "find_active_stay_for_client",
        lambda **_k: SimpleNamespace(id=1),
    )
    monkeypatch.setattr(
        stay_mod,
        "get_clinic_address_for_stay",
        lambda _stay: {
            "address": "Clinique X",
            "clinic_name": "X",
            "clinic_id": 1,
            "preferential_rate": 42.5,
        },
    )
    monkeypatch.setattr(
        CreateBookingUseCase,
        "_compute_pricing_freeze",
        lambda *a, **k: (9, 8, 99.0, {"computed": True}),
    )

    writer = _FakeBookingWriter()
    uc = _build_uc(
        writer=writer,
        client_repo=_FakeClientRepo(company_id=7, preferential_rate=30.0),
    )
    cmd = _base_cmd(amount=12.0)
    cmd.data["amount_source"] = "manual"
    uc.execute(cmd)
    assert writer.last_kwargs["amount"] == 42.5
    breakdown = writer.last_kwargs.get("price_breakdown_json") or {}
    assert breakdown.get("overridden_by_preferential") is True
    assert breakdown.get("preferential_source") == "clinic"


def test_bill_to_patient_excludes_clinic_rate_but_preserves_client_rate_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import application.bookings.create_booking as mod
    import services.billing.client_stay_resolver as stay_mod

    _allow_cmd_data_side_channels(monkeypatch)
    monkeypatch.setattr(mod, "publish_event", lambda _evt: None)
    monkeypatch.setattr(mod, "resolve_pickup_admin", _admin_token)
    monkeypatch.setattr(
        stay_mod,
        "find_active_stay_for_client",
        lambda **_k: SimpleNamespace(id=1),
    )
    monkeypatch.setattr(
        stay_mod,
        "get_clinic_address_for_stay",
        lambda _stay: {
            "address": "Clinique X",
            "clinic_name": "X",
            "clinic_id": 1,
            "preferential_rate": 42.5,
        },
    )
    monkeypatch.setattr(
        CreateBookingUseCase,
        "_compute_pricing_freeze",
        lambda *a, **k: (None, None, None, {}),
    )

    writer = _FakeBookingWriter()
    uc = _build_uc(
        writer=writer,
        client_repo=_FakeClientRepo(company_id=7, preferential_rate=33.0),
    )
    cmd = _base_cmd(amount=10.0)
    cmd.data["bill_to_patient"] = True
    uc.execute(cmd)
    assert writer.last_kwargs["amount"] == 33.0
    breakdown = writer.last_kwargs.get("price_breakdown_json") or {}
    assert breakdown.get("preferential_source") == "client"
    assert breakdown.get("overridden_by_preferential") is True


def test_price_freeze_persists_profile_version_amount_and_breakdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import application.bookings.create_booking as mod

    monkeypatch.setattr(mod, "publish_event", lambda _evt: None)
    _patch_common(monkeypatch, mod)
    monkeypatch.setattr(
        CreateBookingUseCase,
        "_compute_pricing_freeze",
        lambda *a, **k: (11, 22, 55.5, {"line": "frozen"}),
    )

    writer = _FakeBookingWriter()
    uc = _build_uc(writer=writer)
    uc.execute(_base_cmd(amount=1.0))
    assert writer.last_kwargs["pricing_profile_id"] == 11
    assert writer.last_kwargs["pricing_profile_version_id"] == 22
    assert writer.last_kwargs["price_amount"] == 55.5
    assert writer.last_kwargs["price_breakdown_json"] == {
        "line": "frozen",
        "pricing_amount_applied": True,
    }
    assert writer.last_kwargs["amount"] == 55.5


def test_publish_event_called_once_after_writer_returns_valid_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import application.bookings.create_booking as mod

    published: list[Any] = []
    monkeypatch.setattr(mod, "publish_event", lambda evt: published.append(evt))
    _patch_common(monkeypatch, mod)

    writer = _FakeBookingWriter(booking_id=456)
    uc = _build_uc(writer=writer)
    uc.execute(_base_cmd())
    assert len(published) == 1
    assert isinstance(published[0], BookingCreatedEvent)
    assert published[0].booking_id == 456
    assert writer.calls == 1


def test_publish_event_not_called_when_writer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import application.bookings.create_booking as mod

    published: list[Any] = []
    async_calls: list[Any] = []
    monkeypatch.setattr(mod, "publish_event", lambda evt: published.append(evt))
    _patch_common(monkeypatch, mod)

    writer = _FakeBookingWriter()
    writer.fail = True
    uc = _build_uc(
        writer=writer,
        trigger_async=lambda *a: async_calls.append(a),
    )
    with pytest.raises(RuntimeError, match="writer failed"):
        uc.execute(_base_cmd())
    assert published == []
    assert async_calls == []


def test_writer_result_without_valid_id_does_not_schedule_geo_or_publish_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import application.bookings.create_booking as mod

    published: list[Any] = []
    async_calls: list[Any] = []
    monkeypatch.setattr(mod, "publish_event", lambda evt: published.append(evt))
    _patch_common(monkeypatch, mod)

    # Force geocode_miss via failing geocode + fallback
    geocoding = _FakeGeocoding(succeed=False)
    writer = _FakeBookingWriter(booking_id=0)
    uc = _build_uc(
        writer=writer,
        geocoding=geocoding,
        trigger_async=lambda *a: async_calls.append(a),
    )
    with pytest.raises(RuntimeError, match="without a valid id"):
        uc.execute(_base_cmd())
    assert published == []
    assert async_calls == []


def _install_audit_order_spies(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    import middleware.trace_id as trace_mod
    import security.audit_log as audit_mod

    class _Audit:
        @staticmethod
        def log_action(**_kwargs: Any) -> None:
            calls.append("audit")

    monkeypatch.setattr(audit_mod, "AuditLogger", _Audit)
    monkeypatch.setattr(trace_mod, "get_trace_id", lambda: "t")


def test_geocode_miss_effect_order_is_writer_async_geo_event_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import application.bookings.create_booking as mod

    calls: list[str] = []
    monkeypatch.setattr(
        mod, "publish_event", lambda _evt: calls.append("event")
    )
    monkeypatch.setattr(mod, "resolve_pickup_admin", _admin_token)
    import services.billing.client_stay_resolver as stay_mod

    monkeypatch.setattr(stay_mod, "find_active_stay_for_client", lambda **_k: None)
    monkeypatch.setattr(stay_mod, "get_clinic_address_for_stay", lambda _s: None)
    _install_audit_order_spies(monkeypatch, calls)

    writer = _FakeBookingWriter()

    def _write(**kwargs):  # type: ignore[no-untyped-def]
        calls.append("writer")
        return writer.create_and_commit(**kwargs)

    wrapped = _FakeBookingWriter()
    wrapped.create_and_commit = _write  # type: ignore[method-assign]

    uc = _build_uc(
        writer=wrapped,
        geocoding=_FakeGeocoding(succeed=False),
        trigger_async=lambda *_a: calls.append("async_geo"),
    )
    uc.execute(_base_cmd())
    assert calls == ["writer", "async_geo", "event", "audit"]


def test_geocode_success_effect_order_is_writer_event_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import application.bookings.create_booking as mod

    calls: list[str] = []
    monkeypatch.setattr(
        mod, "publish_event", lambda _evt: calls.append("event")
    )
    monkeypatch.setattr(mod, "resolve_pickup_admin", _admin_token)
    import services.billing.client_stay_resolver as stay_mod

    monkeypatch.setattr(stay_mod, "find_active_stay_for_client", lambda **_k: None)
    monkeypatch.setattr(stay_mod, "get_clinic_address_for_stay", lambda _s: None)
    _install_audit_order_spies(monkeypatch, calls)

    writer = _FakeBookingWriter()

    def _write(**kwargs):  # type: ignore[no-untyped-def]
        calls.append("writer")
        return writer.create_and_commit(**kwargs)

    wrapped = _FakeBookingWriter()
    wrapped.create_and_commit = _write  # type: ignore[method-assign]

    uc = _build_uc(
        writer=wrapped,
        geocoding=_FakeGeocoding(succeed=True),
        trigger_async=lambda *_a: calls.append("async_geo"),
    )
    uc.execute(_base_cmd())
    assert calls == ["writer", "event", "audit"]


def test_async_geocoding_is_only_scheduled_after_writer_returns_booking_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import application.bookings.create_booking as mod

    order: list[str] = []
    monkeypatch.setattr(mod, "publish_event", lambda _e: None)
    _patch_common(monkeypatch, mod)

    writer = _FakeBookingWriter(booking_id=777)

    def _write(**kwargs):  # type: ignore[no-untyped-def]
        order.append("writer")
        return writer.create_and_commit(**kwargs)

    wrapped = _FakeBookingWriter(booking_id=777)
    wrapped.create_and_commit = _write  # type: ignore[method-assign]

    async_ids: list[int] = []

    def _async(booking_id: int, *_a: str) -> None:
        order.append("async_geo")
        async_ids.append(booking_id)

    uc = _build_uc(
        writer=wrapped,
        geocoding=_FakeGeocoding(succeed=False),
        trigger_async=_async,
    )
    uc.execute(_base_cmd())
    assert order.index("writer") < order.index("async_geo")
    assert async_ids == [777]


def test_adapter_injects_platform_suspension_gate() -> None:
    from bookings.infrastructure.adapters import booking_service_adapter as adapter_mod
    from services.platform_tenant_gates import assert_company_not_platform_suspended

    src = Path(adapter_mod.__file__).read_text(encoding="utf-8")
    assert "company_creation_gate_fn=assert_company_not_platform_suspended" in src
    assert adapter_mod.assert_company_not_platform_suspended is (
        assert_company_not_platform_suspended
    )


def test_known_client_post_routes_delegate_to_shared_helper() -> None:
    handlers = (
        ("routes/bookings.py", "CreateBooking", "post"),
        ("routes/clients.py", "ClientBookings", "post"),
        ("routes/clients.py", "ClientMyBookings", "post"),
    )
    for rel, class_name, method_name in handlers:
        path = BACKEND_ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    segment = ast.get_source_segment(
                        path.read_text(encoding="utf-8"), item
                    ) or ""
                    assert "execute_client_booking_creation" in segment, (
                        f"{rel} {class_name}.{method_name}"
                    )
                    found = True
        assert found, f"Handler manquant: {rel} {class_name}.{method_name}"
