"""Gates P1 — replace atomique, provisional, cache post-commit, challenge snapshot."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from ext import db
from models import User
from models.enums import UserRole
from models.mobile_device_session import MobileDeviceSessionStatus
from security import mobile_device_session_service as svc


@pytest.fixture
def session_user(db_session):
    suffix = str(uuid.uuid4())[:8]
    user = User(
        username=f"mds_p1_{suffix}",
        email=f"mds_p1_{suffix}@test.local",
        public_id=str(uuid.uuid4()),
        role=UserRole.driver,
    )
    user.set_password("password123", force_change=False)
    db_session.session.add(user)
    db_session.session.commit()
    return user


def _fill_sessions(user_id: int, n: int, *, role: str = "driver") -> list:
    out = []
    for i in range(n):
        s, _r, _v, _ = svc.create_or_reuse_session(
            user_id=user_id,
            device_installation_id=f"install-{user_id}-{i}-{uuid.uuid4().hex[:6]}",
            role=role,
            meta=svc.DeviceSessionMetadata(
                device_model=f"Phone-{i}",
                platform="ios",
                app_version="1.0.11",
            ),
        )
        # Confirmer pour ne pas être provisional dans les tests de quota
        svc.mark_session_confirmed(s)
        out.append(s)
    db.session.commit()
    return out


def test_auth_capabilities_include_replace_and_provisional():
    caps = svc.auth_capabilities()["capabilities"]
    assert caps["device_session_management"] is True
    assert "device_session_replace" in caps
    assert "provisional_session_confirmation" in caps
    # Defaults fail-closed : activation runtime explicite
    assert caps["device_session_replace"] is False
    assert caps["provisional_session_confirmation"] is False


def test_create_returns_reaped_ids_for_post_commit_publish(
    app, session_user, monkeypatch
):
    monkeypatch.setattr(svc, "PROVISIONAL_CONFIRMATION_ENABLED", True)
    with app.app_context():
        with patch.object(svc, "get_device_session_limit", return_value=1):
            old, _, _, _ = svc.create_or_reuse_session(
                user_id=session_user.id,
                device_installation_id="old-slot",
                role="driver",
            )
            old.provisional_expires_at = datetime.now(UTC) - timedelta(minutes=1)
            db.session.commit()
            new_s, _, _, reaped = svc.create_or_reuse_session(
                user_id=session_user.id,
                device_installation_id="new-slot",
                role="driver",
            )
            db.session.commit()
            assert old.session_id in reaped
            assert new_s.is_active()
            fake_redis = MagicMock()
            with patch.object(svc, "_get_redis", return_value=fake_redis):
                for sid in reaped:
                    svc.publish_session_revoked(sid)
            assert fake_redis.setex.called
            assert not fake_redis.delete.called


def test_serialize_exposes_model_and_hides_installation(app, session_user):
    with app.app_context():
        session, _, _, _ = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id="secret-install",
            role="driver",
            meta=svc.DeviceSessionMetadata(
                device_name="iPhone de Test",
                device_model="iPhone 15 Pro",
                device_manufacturer="Apple",
                platform="ios",
                os_version="18.6",
                app_version="1.0.11",
                app_build="69",
            ),
        )
        db.session.commit()
        public = session.serialize()
        assert "device_installation_id" not in public
        assert public["device_model"] == "iPhone 15 Pro"
        assert public["device_name"] == "iPhone de Test"
        assert public["last_os_version"] == "18.6"
        assert public["last_app_build"] == "69"
        assert len(public["device_code"]) == 6


def test_new_session_is_provisional_when_enabled(app, session_user, monkeypatch):
    monkeypatch.setattr(svc, "PROVISIONAL_CONFIRMATION_ENABLED", True)
    with app.app_context():
        session, _, _, _ = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=f"prov-{uuid.uuid4()}",
            role="driver",
        )
        db.session.commit()
        assert session.confirmed_at is None
        assert session.provisional_expires_at is not None
        assert session.is_provisional() is True
        transitioned = svc.mark_session_confirmed(session)
        db.session.commit()
        assert transitioned is True
        assert session.confirmed_at is not None
        assert session.provisional_expires_at is None
        assert svc.mark_session_confirmed(session) is False


def test_refresh_implicitly_confirms_provisional(app, session_user, monkeypatch):
    monkeypatch.setattr(svc, "PROVISIONAL_CONFIRMATION_ENABLED", True)
    with app.app_context():
        session, _, _, _ = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=f"impl-{uuid.uuid4()}",
            role="driver",
        )
        db.session.commit()
        assert session.confirmed_at is None
        svc.bump_refresh_generation(session)
        db.session.commit()
        assert session.confirmed_at is not None


def test_reap_expired_provisional_before_count(app, session_user, monkeypatch):
    monkeypatch.setattr(svc, "PROVISIONAL_CONFIRMATION_ENABLED", True)
    monkeypatch.setenv("MAX_MOBILE_DEVICE_SESSIONS_DRIVER", "2")
    # reload limit via env already read at import — patch getter
    with app.app_context():
        with patch.object(svc, "get_device_session_limit", return_value=2):
            s1, _, _, _ = svc.create_or_reuse_session(
                user_id=session_user.id,
                device_installation_id="slot-a",
                role="driver",
            )
            s2, _, _, _ = svc.create_or_reuse_session(
                user_id=session_user.id,
                device_installation_id="slot-b",
                role="driver",
            )
            # expire both provisionals
            past = datetime.now(UTC) - timedelta(minutes=1)
            s1.provisional_expires_at = past
            s2.provisional_expires_at = past
            db.session.commit()

            # Nouveau device doit réussir après reap sync
            s3, _, _, _ = svc.create_or_reuse_session(
                user_id=session_user.id,
                device_installation_id="slot-c",
                role="driver",
            )
            db.session.commit()
            active = svc.list_active_sessions(session_user.id)
            assert len(active) == 1
            assert active[0].session_id == s3.session_id


def test_replace_keeps_exact_limit(app, session_user, monkeypatch):
    monkeypatch.setattr(svc, "DEVICE_SESSION_REPLACE_ENABLED", True)
    with app.app_context():
        with patch.object(svc, "get_device_session_limit", return_value=3):
            filled = _fill_sessions(session_user.id, 3)
            target = filled[1]
            allowed = [str(s.session_id) for s in filled]
            new_s, _, _, publish_ids = svc.replace_device_session(
                user_id=session_user.id,
                session_to_revoke=str(target.session_id),
                device_installation_id=f"new-phone-{uuid.uuid4()}",
                allowed_session_ids=allowed,
                role="driver",
                meta=svc.DeviceSessionMetadata(
                    device_model="Galaxy S24", platform="android"
                ),
            )
            db.session.commit()
            for _sid in publish_ids:
                svc.publish_session_revoked(_sid)
            active = svc.list_active_sessions(session_user.id)
            assert len(active) == 3
            assert target.session_id in publish_ids
            assert str(new_s.session_id) in {str(a.session_id) for a in active}
            db.session.refresh(target)
            assert target.status == MobileDeviceSessionStatus.revoked


def test_replace_rejects_session_not_in_challenge(app, session_user):
    with app.app_context():
        filled = _fill_sessions(session_user.id, 2)
        outsider, _, _, _ = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=f"outsider-{uuid.uuid4()}",
            role="driver",
        )
        svc.mark_session_confirmed(outsider)
        db.session.commit()
        with pytest.raises(svc.DeviceSessionResolutionError) as exc:
            svc.replace_device_session(
                user_id=session_user.id,
                session_to_revoke=str(outsider.session_id),
                device_installation_id=f"attacker-{uuid.uuid4()}",
                allowed_session_ids=[str(filled[0].session_id), str(filled[1].session_id)],
                role="driver",
            )
        assert exc.value.code == "session_not_in_challenge"


def test_replace_rollback_leaves_old_session_active(app, session_user):
    with app.app_context():
        filled = _fill_sessions(session_user.id, 2)
        target = filled[0]
        allowed = [str(s.session_id) for s in filled]
        # Simuler panne après revoke_state en forçant une erreur sur create locked
        with patch.object(
            svc,
            "_create_or_reuse_session_locked",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError):
                svc.replace_device_session(
                    user_id=session_user.id,
                    session_to_revoke=str(target.session_id),
                    device_installation_id=f"fail-{uuid.uuid4()}",
                    allowed_session_ids=allowed,
                    role="driver",
                )
            db.session.rollback()
        db.session.refresh(target)
        # Après rollback explicite, l'état ORM peut être détaché — recharger
        reloaded = svc.get_session_by_id(target.session_id)
        assert reloaded is not None
        assert reloaded.is_active()


def test_publish_session_revoked_sets_negative_cache_without_delete(app):
    sid = uuid.uuid4()
    fake_redis = MagicMock()
    with patch.object(svc, "_get_redis", return_value=fake_redis):
        svc.publish_session_revoked(sid)
    fake_redis.setex.assert_called_once()
    key, ttl, payload = fake_redis.setex.call_args[0]
    assert str(sid) in key
    assert ttl == svc.SESSION_NEGATIVE_CACHE_TTL_SECONDS
    data = json.loads(payload)
    assert data["status"] == "revoked"
    fake_redis.delete.assert_not_called()


def test_revoke_session_no_longer_invalidates_after_cache(app, session_user):
    with app.app_context():
        session, _, _, _ = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=f"cache-{uuid.uuid4()}",
            role="driver",
        )
        db.session.commit()
        fake_redis = MagicMock()
        with patch.object(svc, "_get_redis", return_value=fake_redis):
            svc.revoke_session(session, reason="test", publish_cache=True)
        # setex pour marqueur négatif, pas de delete contradictoire
        assert fake_redis.setex.called
        assert not fake_redis.delete.called


def test_challenge_issued_claimed_consumed_and_reclaim(app, session_user):
    store: dict[str, str] = {}

    class FakeRedis:
        def setex(self, key, ttl, value):
            store[key] = value
            return True

        def get(self, key):
            return store.get(key)

        def delete(self, key):
            store.pop(key, None)
            return 1

        def set(self, key, value, nx=False, ex=None):
            if nx and key in store:
                return False
            store[key] = value
            return True

        def ttl(self, key):
            return 120

    fake = FakeRedis()
    with app.app_context():
        filled = _fill_sessions(session_user.id, 2)
        with patch.object(svc, "_get_redis", return_value=fake):
            with patch.object(svc, "DEVICE_SESSION_REPLACE_ENABLED", True):
                token = svc.issue_device_session_resolution_token(
                    user_id=session_user.id,
                    requested_device_installation_id="phone-new",
                    allowed_sessions=filled,
                )
                assert token
                claimed = svc.claim_device_session_resolution_token(
                    token=token,
                    requested_device_installation_id="phone-new",
                )
                assert claimed["state"] == "claimed"
                assert str(filled[0].session_id) in claimed["allowed_session_ids"]

                # Second claim concurrent → in use
                with pytest.raises(svc.DeviceSessionResolutionError) as exc:
                    svc.claim_device_session_resolution_token(
                        token=token,
                        requested_device_installation_id="phone-new",
                    )
                assert exc.value.code == "resolution_token_in_use"

                svc.release_device_session_resolution_claim(token=token)
                # Après release, reclain possible
                again = svc.claim_device_session_resolution_token(
                    token=token,
                    requested_device_installation_id="phone-new",
                )
                assert again["operation_id"] == claimed["operation_id"]
                svc.consume_device_session_resolution_token(token=token)


def test_reuse_confirmed_installation_keeps_same_session_id(app, session_user, monkeypatch):
    monkeypatch.setattr(svc, "PROVISIONAL_CONFIRMATION_ENABLED", True)
    with app.app_context():
        install = f"stable-{uuid.uuid4()}"
        s1, _, _, _ = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=install,
            role="driver",
        )
        svc.mark_session_confirmed(s1)
        db.session.commit()
        confirmed_at = s1.confirmed_at
        s2, _, _, _ = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=install,
            role="driver",
        )
        db.session.commit()
        assert s1.session_id == s2.session_id
        assert s2.confirmed_at is not None
        assert s2.provisional_expires_at is None
        assert s2.confirmed_at == confirmed_at or s2.confirmed_at >= confirmed_at


def test_replace_then_validate_old_session_refused(app, session_user):
    with app.app_context():
        with patch.object(svc, "get_device_session_limit", return_value=2):
            filled = _fill_sessions(session_user.id, 2)
            old = filled[0]
            allowed = [str(s.session_id) for s in filled]
            fake_redis = MagicMock()
            store: dict[str, str] = {}

            def _setex(key, ttl, value):
                store[key] = value
                return True

            def _get(key):
                return store.get(key)

            fake_redis.setex.side_effect = _setex
            fake_redis.get.side_effect = _get

            new_s, _, _, publish_ids = svc.replace_device_session(
                user_id=session_user.id,
                session_to_revoke=str(old.session_id),
                device_installation_id=f"phone-{uuid.uuid4()}",
                allowed_session_ids=allowed,
                role="driver",
            )
            db.session.commit()
            with patch.object(svc, "_get_redis", return_value=fake_redis):
                for _sid in publish_ids:
                    svc.publish_session_revoked(_sid)
                err, _ = svc.validate_mobile_session(
                    session_id=str(old.session_id),
                    session_epoch=int(old.session_epoch or 1),
                    user_id=session_user.id,
                )
            assert err == "session_revoked"
            assert new_s.is_active()


def test_limit_invariant_after_two_creates_at_capacity(app, session_user):
    """INV-AUTH-DEVICE-01 : à la limite, un 2e create distinct échoue sans dépasser."""
    with app.app_context():
        with patch.object(svc, "get_device_session_limit", return_value=2):
            _fill_sessions(session_user.id, 2)
            with pytest.raises(svc.DeviceSessionLimitReached):
                svc.create_or_reuse_session(
                    user_id=session_user.id,
                    device_installation_id=f"overflow-{uuid.uuid4()}",
                    role="driver",
                )
            assert len(svc.list_active_sessions(session_user.id)) == 2


def test_replace_reaps_expired_provisional_under_lock(app, session_user, monkeypatch):
    monkeypatch.setattr(svc, "PROVISIONAL_CONFIRMATION_ENABLED", True)
    with app.app_context():
        with patch.object(svc, "get_device_session_limit", return_value=2):
            # 1 confirmed + 1 expired provisional = 2 actives avant reap
            a, _, _, _ = svc.create_or_reuse_session(
                user_id=session_user.id,
                device_installation_id="keep-me",
                role="driver",
            )
            svc.mark_session_confirmed(a)
            b, _, _, _ = svc.create_or_reuse_session(
                user_id=session_user.id,
                device_installation_id="expired-slot",
                role="driver",
            )
            b.provisional_expires_at = datetime.now(UTC) - timedelta(minutes=1)
            db.session.commit()

            # Replace de A : le reap doit libérer B avant create
            new_s, _, _, publish_ids = svc.replace_device_session(
                user_id=session_user.id,
                session_to_revoke=str(a.session_id),
                device_installation_id=f"incoming-{uuid.uuid4()}",
                allowed_session_ids=[str(a.session_id), str(b.session_id)],
                role="driver",
            )
            db.session.commit()
            active = svc.list_active_sessions(session_user.id)
            assert len(active) == 1
            assert active[0].session_id == new_s.session_id
            assert a.session_id in publish_ids
            db.session.refresh(b)
            assert b.status == MobileDeviceSessionStatus.revoked


def test_cross_user_replace_target_rejected(app, session_user, db_session):
    with app.app_context():
        other = User(
            username=f"other_{uuid.uuid4().hex[:6]}",
            email=f"other_{uuid.uuid4().hex[:6]}@test.local",
            public_id=str(uuid.uuid4()),
            role=UserRole.driver,
        )
        other.set_password("password123", force_change=False)
        db.session.add(other)
        db.session.commit()

        own = _fill_sessions(session_user.id, 1)[0]
        foreign = _fill_sessions(other.id, 1)[0]
        with pytest.raises(svc.DeviceSessionResolutionError) as exc:
            svc.replace_device_session(
                user_id=session_user.id,
                session_to_revoke=str(foreign.session_id),
                device_installation_id=f"attacker-{uuid.uuid4()}",
                allowed_session_ids=[str(own.session_id), str(foreign.session_id)],
                role="driver",
            )
        assert exc.value.code == "session_not_found"
        db.session.refresh(foreign)
        assert foreign.is_active()
