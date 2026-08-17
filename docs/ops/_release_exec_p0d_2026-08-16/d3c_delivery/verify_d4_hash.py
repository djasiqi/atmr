from services.tracking.event_payload_hash import compute_event_payload_hash_from_point
# same coords/seq/etc as DLQ but accepted recorded_at
p = {
  "location_event_id": "trk_1786888628909_kryu2j9y",
  "recorded_at": "2026-08-16T13:57:08.992849+00:00",
  "latitude": 46.2116156,
  "longitude": 6.1262053,
  "accuracy": 7.803999900817871,
  "heading": 0.0,
  "speed": 0.06219065189361572,
  "sequence_id": 10,
  "mission_id": 38224,
  "location_mode": "mission_live",
}
h, o = compute_event_payload_hash_from_point(p)
print("hash_with_accepted_recorded_at", h)
print("expected_accepted", "db6ef1eae59f3e175fd9da8ac77f8f7f8fa641d9e61291c7a82251e570decc6f")
print("match", h == "db6ef1eae59f3e175fd9da8ac77f8f7f8fa641d9e61291c7a82251e570decc6f")
print("canon", o)
