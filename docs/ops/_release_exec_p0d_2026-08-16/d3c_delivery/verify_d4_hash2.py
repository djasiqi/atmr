from services.tracking.event_payload_hash import compute_event_payload_hash
eid="trk_1786888628909_kryu2j9y"
rec="2026-08-16T13:57:08.992849+00:00"
base=dict(location_event_id=eid, recorded_at=rec, latitude=46.2116156, longitude=6.1262053, accuracy=7.803999900817871, sequence_id=10, mission_id=38224, location_mode="mission_live")
expected="db6ef1eae59f3e175fd9da8ac77f8f7f8fa641d9e61291c7a82251e570decc6f"
variants=[
 ("no_speed_heading", dict()),
 ("speed0_heading0", dict(speed=0.0, heading=0.0)),
 ("speed_dlq_heading0", dict(speed=0.06219065189361572, heading=0.0)),
 ("speed_dlq_no_heading", dict(speed=0.06219065189361572)),
 ("heading0_only", dict(heading=0.0)),
 ("acc_round78", dict(accuracy=7.8, speed=0.06219065189361572, heading=0.0)),
]
for name, extra in variants:
  h,_=compute_event_payload_hash(**{**base, **extra})
  print(name, h==expected, h[:16])
