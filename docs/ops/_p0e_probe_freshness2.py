from app import create_app
from datetime import UTC, datetime

app = create_app()
with app.app_context():
    from sqlalchemy import text
    from models import db
    from services.company_driver_location_freshness import (
        last_seen_seconds_from_location_fields,
        resolve_location_freshness_timestamp,
    )
    from services.geolocation.presence import compute_location_status
    from services.company_driver_locations import build_company_driver_locations_items
    from ext import redis_client

    DRIVER_ID = 20135
    MISSION_ID = 38243
    WITNESS_ID = 5846

    row = db.session.execute(
        text("SELECT * FROM driver_location_events WHERE id=:id"),
        {"id": WITNESS_ID},
    ).mappings().fetchone()
    print("WITNESS")
    print(f"  id={row['id']} driver_id={row['driver_id']} mission_id={row['mission_id']}")
    print(f"  tracking_session_id={row['tracking_session_id']}")
    print(f"  sequence_id={row['sequence_id']}")
    print(f"  location_event_id={row['location_event_id']}")
    print(f"  recorded_at={row['recorded_at']}")
    print(f"  created_at(persisted_at)={row['created_at']}")
    print(f"  location_mode={row['location_mode']}")
    print(f"  lat={row['raw_latitude']} lon={row['raw_longitude']}")

    drv = db.session.execute(
        text(
            "SELECT id, last_position_update, latitude, longitude, is_active "
            "FROM driver WHERE id=:d"
        ),
        {"d": DRIVER_ID},
    ).mappings().fetchone()
    now = datetime.now(UTC)
    print("DRIVER")
    for k, v in dict(drv).items():
        print(f"  {k}={v}")
    if drv["last_position_update"]:
        lpu = drv["last_position_update"]
        if lpu.tzinfo is None:
            lpu = lpu.replace(tzinfo=UTC)
        print(f"  age_lpu_s={int((now - lpu).total_seconds())}")

    booking = db.session.execute(
        text("SELECT id, status, driver_id FROM booking WHERE id=:m"),
        {"m": MISSION_ID},
    ).mappings().fetchone()
    print("BOOKING", dict(booking))

    key = f"driver:{DRIVER_ID}:loc:canonical"
    raw = redis_client.hgetall(key)
    print("REDIS_CANONICAL_KEY", key)
    if not raw:
        print("  EMPTY")
    else:
        decoded = {}
        for k, v in raw.items():
            kk = k.decode() if isinstance(k, bytes) else k
            vv = v.decode() if isinstance(v, bytes) else v
            decoded[kk] = vv
            print(f"  {kk}={vv}")
        ref = resolve_location_freshness_timestamp(decoded)
        age = last_seen_seconds_from_location_fields(decoded, now=now)
        mode = decoded.get("location_mode") or "mission_live"
        status = compute_location_status(mode=mode, last_seen_seconds=age)
        print("REDIS_FRESHNESS")
        print(f"  now_utc={now.isoformat()}")
        print(f"  ref_timestamp={ref}")
        print(f"  last_seen_seconds={age}")
        print(f"  location_status={status}")
        print(f"  thresholds_mission_live=live<=20 recent<=90 stale<=300 else offline")

    # Company REST projection (what dashboard GET uses)
    items = build_company_driver_locations_items(1, is_demo_company=False)
    hit = [i for i in items if int(i.get("driver_id") or i.get("id") or 0) == DRIVER_ID]
    if not hit:
        # try alternate id field
        hit = [i for i in items if str(i.get("id")) == str(DRIVER_ID) or str(i.get("driver_id")) == str(DRIVER_ID)]
    print("REST_PROJECTION_COUNT", len(items), "HIT", len(hit))
    if hit:
        d = hit[0]
        keys_of_interest = [
            "id", "driver_id", "latitude", "longitude", "lat", "lon",
            "recorded_at", "received_at", "timestamp", "ts",
            "last_seen_seconds", "location_status", "presence_status",
            "position_source", "tracking_display_status", "location_mode",
            "mission_id", "current_booking_id",
        ]
        print("REST_DRIVER_20135")
        for k in keys_of_interest:
            if k in d:
                print(f"  {k}={d.get(k)}")
        # dump extra keys briefly
        extra = sorted(set(d.keys()) - set(keys_of_interest))
        print("  extra_keys=", extra[:40])

    # TTL
    ttl = redis_client.ttl(key)
    print("REDIS_TTL_s", ttl)
