"""Couverture critique de ``optimization.assignment_applier`` (seuil 95 %)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from models import BookingStatus
from services.unified_dispatch.optimization import assignment_applier as applier
from tests.factories import BookingFactory, CompanyFactory, DriverFactory


@pytest.fixture(autouse=True)
def _app_ctx(app):
    with app.app_context():
        yield


def _asg(booking_id, driver_id, score=1.0, **extra):
    data = {"booking_id": booking_id, "driver_id": driver_id, "score": score}
    data.update(extra)
    return data


class TestHelpers:
    def test_conflict_counter_singleton(self):
        applier.reset_db_conflict_counter()
        assert applier.get_db_conflict_count() == 0
        applier.increment_db_conflict_counter()
        applier.increment_db_conflict_counter()
        assert applier.get_db_conflict_count() == 2
        applier.reset_db_conflict_counter()
        assert applier.get_db_conflict_count() == 0
        a = applier.DBConflictCounter.get_instance()
        b = applier.DBConflictCounter.get_instance()
        assert a is b

    def test_driver_display_name(self):
        assert applier._driver_display_name(SimpleNamespace(user=None)) is None
        user = SimpleNamespace(first_name="Jean", last_name="Dupont", username="jd")
        assert applier._driver_display_name(SimpleNamespace(user=user)) == "Jean Dupont"
        empty = SimpleNamespace(first_name="  ", last_name="", username="alias")
        assert applier._driver_display_name(SimpleNamespace(user=empty)) == "alias"
        none = SimpleNamespace(first_name="", last_name="", username=None)
        assert applier._driver_display_name(SimpleNamespace(user=none)) is None

    def test_get_scoped_session_branches(self, monkeypatch):
        created = SimpleNamespace(session="main")

        def _ok():
            return "scoped-ok"

        created.create_scoped_session = _ok
        assert applier._get_scoped_session(created) == "scoped-ok"

        class BoomCreate:
            session = "main"
            engine = MagicMock()

            def create_scoped_session(self):
                raise AttributeError("gone")

        monkeypatch.setattr(
            applier, "sessionmaker", lambda **_k: MagicMock()
        )
        monkeypatch.setattr(applier, "scoped_session", lambda _f: "from-engine")
        assert applier._get_scoped_session(BoomCreate()) == "from-engine"

        class ViaGetEngine:
            session = "main"
            engine = None

            def get_engine(self):
                return MagicMock()

        assert applier._get_scoped_session(ViaGetEngine()) == "from-engine"

        class ViaBind:
            engine = None
            session = SimpleNamespace(get_bind=lambda: MagicMock())

        assert applier._get_scoped_session(ViaBind()) == "from-engine"

        class NoneBind:
            engine = None
            session = SimpleNamespace(get_bind=lambda: None)

        assert applier._get_scoped_session(NoneBind()) is NoneBind.session

        class NoEngine:
            engine = None
            session = SimpleNamespace()

        assert applier._get_scoped_session(NoEngine()) is NoEngine.session

        class DbErr:
            session = "main"
            engine = MagicMock()

            def create_scoped_session(self):
                raise AttributeError("x")

        monkeypatch.setattr(
            applier,
            "sessionmaker",
            lambda **_k: (_ for _ in ()).throw(
                OperationalError("s", {}, Exception("db"))
            ),
        )
        assert applier._get_scoped_session(DbErr()) == "main"

        monkeypatch.setattr(
            applier,
            "sessionmaker",
            lambda **_k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert applier._get_scoped_session(DbErr()) == "main"

    def test_scoped_session_context_close(self, monkeypatch):
        db_inst = SimpleNamespace(session="main")

        class Ok:
            def close(self):
                return None

        monkeypatch.setattr(applier, "_get_scoped_session", lambda _d: Ok())
        with applier.scoped_session_context(db_inst) as sess:
            assert sess is not None

        class CloseOp:
            def close(self):
                raise OperationalError("s", {}, Exception("x"))

        monkeypatch.setattr(applier, "_get_scoped_session", lambda _d: CloseOp())
        with applier.scoped_session_context(db_inst):
            pass

        class CloseBoom:
            def close(self):
                raise RuntimeError("close")

        monkeypatch.setattr(applier, "_get_scoped_session", lambda _d: CloseBoom())
        with applier.scoped_session_context(db_inst):
            pass

        monkeypatch.setattr(applier, "_get_scoped_session", lambda d: d.session)
        with applier.scoped_session_context(db_inst) as sess:
            assert sess == "main"

    def test_timeline_branches(self, monkeypatch):
        applier._record_driver_assigned_timeline(
            applied_pairs=[], booking_map={}, driver_map={}, company_id=1
        )

        class Col:
            def in_(self, _ids):
                return self

        class TR:
            booking_id = Col()
            query = MagicMock()

        TR.query.filter.return_value.all.return_value = []
        monkeypatch.setattr("models.TransportRequest", TR, raising=False)
        rec = MagicMock()
        monkeypatch.setattr(
            "services.institutions.transport_timeline_service.record_event",
            rec,
            raising=False,
        )
        monkeypatch.setattr(
            "services.institutions.transport_timeline_service.TimelineActor",
            lambda **k: k,
            raising=False,
        )
        applier._record_driver_assigned_timeline(
            applied_pairs=[(1, 10)], booking_map={}, driver_map={}, company_id=1
        )
        rec.assert_not_called()

        req = SimpleNamespace(booking_id=1, institution_id=9, id=44)
        TR.query.filter.return_value.all.return_value = [req]
        driver = SimpleNamespace(
            user=SimpleNamespace(first_name="A", last_name="B", username="ab"),
            company=SimpleNamespace(name="Cie"),
        )
        applier._record_driver_assigned_timeline(
            applied_pairs=[(1, 10), (2, 99)],
            booking_map={},
            driver_map={10: driver},
            company_id=7,
        )
        assert rec.called

        rec.side_effect = RuntimeError("timeline")
        applier._record_driver_assigned_timeline(
            applied_pairs=[(1, 10)],
            booking_map={},
            driver_map={10: driver},
            company_id=7,
        )


class TestApplyWrapper:
    def test_empty_assignments(self):
        result = applier.apply_assignments(1, [])
        assert result["applied"] == []
        assert result["skipped"] == {}

    def test_inner_errors_and_rollback(self, monkeypatch):
        monkeypatch.setattr(applier, "_in_tx", lambda: False)
        monkeypatch.setattr(
            applier,
            "_apply_assignments_inner",
            lambda **_k: (_ for _ in ()).throw(ValueError("bad")),
        )
        with pytest.raises(ValueError, match="bad"):
            applier.apply_assignments(1, [_asg(1, 1)])

        monkeypatch.setattr(
            applier,
            "_apply_assignments_inner",
            lambda **_k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(RuntimeError, match="boom"):
            applier.apply_assignments(1, [_asg(1, 1)])

        monkeypatch.setattr(applier, "_in_tx", lambda: True)
        monkeypatch.setattr(
            applier,
            "_apply_assignments_inner",
            lambda **_k: (_ for _ in ()).throw(
                OperationalError("s", {}, Exception("db"))
            ),
        )
        with pytest.raises(OperationalError):
            applier.apply_assignments(1, [_asg(1, 1)], dispatch_run_id=9)

    def test_aget_getattr_errors(self, monkeypatch):
        class Flaky:
            def __init__(self, exc: Exception):
                self._n = 0
                self._exc = exc

            def __getattribute__(self, name):
                if name == "booking_id":
                    n = object.__getattribute__(self, "_n")
                    object.__setattr__(self, "_n", n + 1)
                    if n == 0:
                        return 1
                    raise object.__getattribute__(self, "_exc")
                return object.__getattribute__(self, name)

        monkeypatch.setattr(applier, "_in_tx", lambda: False)
        with pytest.raises(TypeError):
            applier.apply_assignments(1, [Flaky(TypeError("attr"))])
        with pytest.raises(TypeError):
            applier.apply_assignments(1, [Flaky(RuntimeError("attr"))])


class TestEmitNotifications:
    def test_empty_pairs(self):
        applier._emit_notifications_after_commit([], 1)

    def test_booking_missing_and_publish_errors(self, db, monkeypatch):
        company = CompanyFactory()
        driver = DriverFactory(company=company, is_active=True, is_available=True)
        booking = BookingFactory(
            company=company, status=BookingStatus.ACCEPTED, driver_id=driver.id
        )
        db.session.commit()

        failed = MagicMock()
        emitted = MagicMock()
        latency = MagicMock()
        monkeypatch.setattr(applier, "NOTIF_FAILED", failed)
        monkeypatch.setattr(applier, "NOTIF_EMITTED", emitted)
        monkeypatch.setattr(applier, "NOTIF_LATENCY", latency)

        applier._emit_notifications_after_commit([(999999, driver.id)], company.id)

        monkeypatch.setattr(
            applier, "publish_event", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("pub"))
        )
        applier._emit_notifications_after_commit(
            [(booking.id, driver.id)], company.id
        )
        assert failed.labels.called

        monkeypatch.setattr(
            applier,
            "publish_event",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("pub")),
        )
        applier._emit_notifications_after_commit(
            [(booking.id, driver.id)], company.id
        )

        monkeypatch.setattr(
            applier,
            "scoped_session_context",
            lambda _db: (_ for _ in ()).throw(ValueError("ctx")),
        )
        applier._emit_notifications_after_commit([(booking.id, driver.id)], company.id)

        monkeypatch.setattr(
            applier,
            "scoped_session_context",
            lambda _db: (_ for _ in ()).throw(RuntimeError("ctx")),
        )
        applier._emit_notifications_after_commit([(booking.id, driver.id)], company.id)


class TestApplyInnerBranches:
    def test_score_dedup_reassign_and_eta(self, db, monkeypatch):
        company = CompanyFactory()
        d1 = DriverFactory(company=company, is_active=True, is_available=True)
        d2 = DriverFactory(company=company, is_active=True, is_available=True)
        booking = BookingFactory(company=company, status=BookingStatus.ACCEPTED)
        booking.driver_id = d1.id
        db.session.flush()
        booking.status = BookingStatus.ASSIGNED
        db.session.flush()

        blocked = applier.apply_assignments(
            company.id,
            [_asg(booking.id, d2.id, score=1.0)],
            allow_reassign=False,
            respect_existing=False,
        )
        assert booking.id in blocked["conflicts"]
        assert blocked["skipped"][booking.id] == "reassign_blocked"

        other = BookingFactory(company=company, status=BookingStatus.ACCEPTED)
        no_score = BookingFactory(company=company, status=BookingStatus.ACCEPTED)
        db.session.flush()
        result = applier.apply_assignments(
            company.id,
            [
                _asg(other.id, d1.id, score=1.0),
                _asg(other.id, d2.id, score=5.0, estimated_pickup_arrival="2030-01-01"),
                _asg(no_score.id, d1.id, score=3.0),
                {"booking_id": no_score.id, "driver_id": d2.id},
            ],
            dispatch_run_id=None,
            return_pairs=True,
        )
        assert other.id in result["applied"]
        applied_by_booking = dict(result["applied_pairs"])
        assert applied_by_booking[other.id] == d2.id
        assert applied_by_booking[no_score.id] == d2.id

        monkeypatch.setenv("UD_APPLY_SKIP_LOCKED", "true")
        again = applier.apply_assignments(
            company.id,
            [_asg(other.id, d2.id, score=1.0, estimated_dropoff_arrival="2030-01-02")],
            respect_existing=True,
        )
        assert other.id in again["skipped"] or other.id in again["applied"]

    def test_upsert_integrity_and_fallbacks(self, db, monkeypatch):
        company = CompanyFactory()
        driver = DriverFactory(company=company, is_active=True, is_available=True)
        booking = BookingFactory(company=company, status=BookingStatus.ACCEPTED)
        db.session.flush()

        class BoomStmt:
            def values(self, **_k):
                return self

            def on_conflict_do_nothing(self, **_k):
                return self

        monkeypatch.setattr(
            "sqlalchemy.dialects.postgresql.insert", lambda *_a, **_k: BoomStmt()
        )
        real_execute = db.session.execute

        def _integrity(stmt, *a, **k):
            if isinstance(stmt, BoomStmt):
                raise IntegrityError("s", {}, Exception("uq"))
            return real_execute(stmt, *a, **k)

        monkeypatch.setattr(db.session, "execute", _integrity)
        applier.reset_db_conflict_counter()
        result = applier.apply_assignments(company.id, [_asg(booking.id, driver.id)])
        assert booking.id in result["applied"]
        assert applier.get_db_conflict_count() >= 1

    def test_upsert_operational_valueerror_runtime(self, db, monkeypatch):
        company = CompanyFactory()
        driver = DriverFactory(company=company, is_active=True, is_available=True)

        class BoomStmt:
            def values(self, **_k):
                return self

            def on_conflict_do_nothing(self, **_k):
                return self

        monkeypatch.setattr(
            "sqlalchemy.dialects.postgresql.insert", lambda *_a, **_k: BoomStmt()
        )
        real_execute = db.session.execute

        def _op(stmt, *a, **k):
            if isinstance(stmt, BoomStmt):
                raise OperationalError("s", {}, Exception("db"))
            return real_execute(stmt, *a, **k)

        monkeypatch.setattr(db.session, "execute", _op)
        b1 = BookingFactory(company=company, status=BookingStatus.ACCEPTED)
        db.session.flush()
        applier.apply_assignments(company.id, [_asg(b1.id, driver.id)])

        def _val(stmt, *a, **k):
            if isinstance(stmt, BoomStmt):
                raise ValueError("sql")
            return real_execute(stmt, *a, **k)

        monkeypatch.setattr(db.session, "execute", _val)
        b2 = BookingFactory(company=company, status=BookingStatus.ACCEPTED)
        db.session.flush()
        applier.apply_assignments(company.id, [_asg(b2.id, driver.id)])

        def _rt(stmt, *a, **k):
            if isinstance(stmt, BoomStmt):
                raise RuntimeError("sql")
            return real_execute(stmt, *a, **k)

        monkeypatch.setattr(db.session, "execute", _rt)
        b3 = BookingFactory(company=company, status=BookingStatus.ACCEPTED)
        db.session.flush()
        applier.apply_assignments(company.id, [_asg(b3.id, driver.id)])

    def test_savepoint_validation_and_generic(self, db, monkeypatch):
        company = CompanyFactory()
        driver = DriverFactory(company=company, is_active=True, is_available=True)
        booking = BookingFactory(company=company, status=BookingStatus.ACCEPTED)
        db.session.flush()
        monkeypatch.setattr(
            db.session,
            "bulk_update_mappings",
            lambda *_a, **_k: (_ for _ in ()).throw(ValueError("upd")),
        )
        with pytest.raises(ValueError, match="upd"):
            applier.apply_assignments(company.id, [_asg(booking.id, driver.id)])

        booking2 = BookingFactory(company=company, status=BookingStatus.ACCEPTED)
        db.session.flush()
        monkeypatch.setattr(
            db.session,
            "bulk_update_mappings",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("upd")),
        )
        with pytest.raises(RuntimeError, match="upd"):
            applier.apply_assignments(company.id, [_asg(booking2.id, driver.id)])
