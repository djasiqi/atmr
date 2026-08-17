from services.tracking.event_payload_hash import compute_event_payload_hash_from_point
versions = [
 {"recorded_at":"2026-08-16T13:57:08.992849+00:00","sent_at":"2026-08-16T13:57:08.997096+00:00","off":4252},
 {"recorded_at":"2026-08-16T13:57:29.457798+00:00","sent_at":"2026-08-16T13:57:29.459316+00:00","off":4253},
 {"recorded_at":"2026-08-16T13:58:09.908303+00:00","sent_at":"2026-08-16T13:58:09.910255+00:00","off":4258},
]
base = {
 "latitude": 46.2116156,
 "longitude": 6.1262053,
 "speed": 0.06219065189361572,
 "heading": 0.0,
 "accuracy": 7.803999900817871,
 "location_mode": "mission_live",
 "mission_id": 38224,
 "location_event_id": "trk_1786888628909_kryu2j9y",
 "sequence_id": 10,
}
exp = "db6ef1eae59f3e175fd9da8ac77f8f7f8fa641d9e61291c7a82251e570decc6f"
for v in versions:
  p={**base, **{k:v[k] for k in ("recorded_at",)}}
  h,o=compute_event_payload_hash_from_point(p)
  print(v["off"], h, "EQ_ACCEPTED" if h==exp else "DIFF", "rec", o["recorded_at"], "speed_dms", o.get("speed_dms"))
