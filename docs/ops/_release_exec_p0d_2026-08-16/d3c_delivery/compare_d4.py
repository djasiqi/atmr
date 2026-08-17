#!/usr/bin/env python3
"""Compare one DLQ conflict message vs accepted ingest for same location_event_id."""
import json
import sys
from sqlalchemy import text
from app import create_app
from services.tracking.event_payload_hash import (
    PAYLOAD_SCHEMA_VERSION,
    compute_event_payload_hash_from_point,
)

picked_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d4_dlq_picked.json"
picked = json.load(open(picked_path, encoding="utf-8"))
om = picked.get("original_message") or {}
pl = om.get("payload") if isinstance(om.get("payload"), dict) else {}
eid = om.get("location_event_id") or pl.get("location_event_id")
print("EID", eid)

# Build point-like dict for hash (same fields client/HTTP typically send)
point = {
    "latitude": pl.get("latitude"),
    "longitude": pl.get("longitude"),
    "speed": pl.get("speed"),
    "heading": pl.get("heading"),
    "accuracy": pl.get("accuracy"),
    "recorded_at": pl.get("recorded_at"),
    "sent_at": pl.get("sent_at"),
    "location_mode": pl.get("location_mode"),
    "is_background": pl.get("is_background"),
    "mission_id": pl.get("mission_id"),
    "location_event_id": eid,
    "tracking_session_id": pl.get("tracking_session_id"),
    "sequence_id": pl.get("sequence_id"),
    "session_generation": pl.get("session_generation"),
}
dlq_hash, hashed_obj = compute_event_payload_hash_from_point(point)

app = create_app()
app.app_context().push()
from models import db

did = 20135
acc = db.session.execute(
    text(
        """
    SELECT location_event_id, event_payload_hash, payload_schema_version, source,
           recorded_at, received_at, tracking_session_id, sequence_id, session_generation,
           company_id, driver_id
    FROM tracking_ingest_events
    WHERE driver_id=:did AND location_event_id=:eid
    """
    ),
    {"did": did, "eid": eid},
).mappings().first()

loc = db.session.execute(
    text(
        """
    SELECT location_event_id, event_payload_hash, payload_schema_version, source,
           recorded_at, created_at, tracking_session_id, sequence_id, session_generation,
           raw_latitude, raw_longitude, accuracy_m, mission_id
    FROM driver_location_events
    WHERE driver_id=:did AND location_event_id=:eid
    """
    ),
    {"did": did, "eid": eid},
).mappings().first()

dlq_view = {
    "location_event_id": eid,
    "payload_hash_recomputed": dlq_hash,
    "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
    "driver_id": om.get("driver_id"),
    "company_id": om.get("company_id"),
    "mission_id": pl.get("mission_id"),
    "tracking_session_id": pl.get("tracking_session_id"),
    "session_generation": pl.get("session_generation"),
    "sequence_id": pl.get("sequence_id"),
    "queue_item_id": pl.get("queue_item_id") or pl.get("id"),
    "latitude": pl.get("latitude"),
    "longitude": pl.get("longitude"),
    "accuracy": pl.get("accuracy"),
    "speed": pl.get("speed"),
    "heading": pl.get("heading"),
    "recorded_at": pl.get("recorded_at"),
    "sent_at": pl.get("sent_at"),
    "location_mode": pl.get("location_mode"),
    "is_background": pl.get("is_background"),
    "hashed_object": hashed_obj,
}

out = {
    "dlq_meta": picked.get("dlq_meta"),
    "dlq": dlq_view,
    "accepted_ingest": dict(acc) if acc else None,
    "accepted_location": dict(loc) if loc else None,
}

if acc:
    same_seq = acc.get("sequence_id") == pl.get("sequence_id")
    same_sid = acc.get("tracking_session_id") == pl.get("tracking_session_id")
    same_gen = acc.get("session_generation") == pl.get("session_generation")
    hash_equal = str(acc.get("event_payload_hash")) == str(dlq_hash)
    coords_equal = False
    if loc is not None:
        coords_equal = (
            abs(float(loc["raw_latitude"]) - float(pl["latitude"])) < 1e-7
            and abs(float(loc["raw_longitude"]) - float(pl["longitude"])) < 1e-7
        )
    acc_rec = str(acc.get("recorded_at"))
    dlq_rec = str(pl.get("recorded_at"))
    recorded_equal = acc_rec.startswith(dlq_rec[:19]) if dlq_rec and acc_rec else False

    # Classify
    if same_seq and same_sid and (not coords_equal or not recorded_equal) and not hash_equal:
        # same identity + same sequence, mutable timestamps/coords -> D4-B (or A if coords differ a lot)
        if coords_equal and not recorded_equal:
            cas = "D4-B"
            note = "same event_id+sequence+coords; recorded_at/sent_at mutated on retry"
        elif not coords_equal and same_seq:
            cas = "D4-A_or_B"
            note = "same event_id+sequence but coords and/or timestamps differ"
        else:
            cas = "D4-B"
            note = "same event_id+sequence; payload hash differs"
    elif not same_seq and not hash_equal:
        cas = "D4-A"
        note = "same event_id reused across different sequences / fixes"
    elif hash_equal:
        cas = "UNEXPECTED_HASH_EQUAL"
        note = "hashes equal — conflict should have been soft-duplicate"
    else:
        cas = "D4-C_or_other"
        note = "needs manual inspection"

    out["diff"] = {
        "case": cas,
        "note": note,
        "same_event_id": True,
        "hash_equal": hash_equal,
        "accepted_hash": acc.get("event_payload_hash"),
        "dlq_hash_recomputed": dlq_hash,
        "same_sequence": same_seq,
        "accepted_sequence": acc.get("sequence_id"),
        "dlq_sequence": pl.get("sequence_id"),
        "same_session": same_sid,
        "same_generation": same_gen,
        "accepted_recorded_at": acc_rec,
        "dlq_recorded_at": dlq_rec,
        "dlq_sent_at": pl.get("sent_at"),
        "coords_equal": coords_equal,
        "accepted_latlon": [loc.get("raw_latitude"), loc.get("raw_longitude")] if loc else None,
        "dlq_latlon": [pl.get("latitude"), pl.get("longitude")],
        "accepted_accuracy": loc.get("accuracy_m") if loc else None,
        "dlq_accuracy": pl.get("accuracy"),
        "accepted_mission_id": loc.get("mission_id") if loc else None,
        "dlq_mission_id": pl.get("mission_id"),
    }
else:
    out["diff"] = {
        "case": "NO_ACCEPTED_ROW",
        "note": "DLQ event_id absent from tracking_ingest_events",
        "same_event_id": False,
    }

open("/tmp/d4_compare.json", "w", encoding="utf-8").write(
    json.dumps(out, indent=2, default=str)[:400000]
)
print(json.dumps(out, indent=2, default=str)[:15000])
