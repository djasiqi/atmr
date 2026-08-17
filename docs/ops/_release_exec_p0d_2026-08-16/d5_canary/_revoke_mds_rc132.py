from datetime import datetime, timezone
from app import create_app
app = create_app()
with app.app_context():
    from models import db
    from models.driver import Driver
    from models.mobile_device_session import MobileDeviceSession, MobileDeviceSessionStatus
    from security.mobile_device_session_service import (
        list_active_sessions,
        revoke_all_user_sessions,
        publish_session_revoked,
        get_device_session_limit,
    )
    from models.mobile_device_session import _device_code_from_installation

    DRIVER_ID = 20135
    driver = Driver.query.get(DRIVER_ID)
    if not driver or not driver.user_id:
        print("FAIL driver_or_user_missing", DRIVER_ID)
        raise SystemExit(2)
    user_id = int(driver.user_id)
    active = list_active_sessions(user_id)
    limit = get_device_session_limit("driver")
    print(f"BEFORE user_id={user_id} active={len(active)} limit={limit}")
    for s in active:
        code = _device_code_from_installation(s.device_installation_id)
        print(f"  {code} name={s.device_name!r} status={s.status.value} last_seen={s.last_seen_at}")
    to_revoke = [s.session_id for s in active]
    n = revoke_all_user_sessions(
        user_id,
        reason="ops_rc132_smoke_free_device_slots",
        status=MobileDeviceSessionStatus.revoked,
        except_session_id=None,
    )
    db.session.commit()
    for sid in to_revoke:
        publish_session_revoked(sid)
    after = list_active_sessions(user_id)
    print(f"AFTER revoked_count={n} active={len(after)}")
    print("OK")