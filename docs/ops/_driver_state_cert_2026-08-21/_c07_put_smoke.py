from app import create_app

app = create_app()
client = app.test_client()
r = client.post(
    "/api/v1/auth/login",
    json={"email": "atmr1@atmr.ch", "password": "Atmr1234"},
    headers={"X-Client-Platform": "android", "X-Requested-With": "Expo"},
)
print("LOGIN", r.status_code)
data = r.get_json() or {}
token = data.get("access_token")
print("token", bool(token))
payload = {
    "latitude": 46.170272,
    "longitude": 6.096074,
    "accuracy": 12.5,
    "recorded_at": "2026-08-21T19:58:13.748Z",
    "location_mode": "mission_live",
    "mission_id": 54,
    "event_id": "trk_c07_tc_1",
    "capture_id": "cap_c07_tc_1",
}
r2 = client.put(
    "/api/v1/driver/me/location",
    json=payload,
    headers={
        "Authorization": f"Bearer {token}",
        "X-Client-Platform": "android",
        "X-Requested-With": "Expo",
    },
)
print("PUT", r2.status_code, r2.get_data(as_text=True)[:1200])
