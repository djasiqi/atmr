from auth_claims import normalize_auth_payload


def test_uuid_sub():
    claims = normalize_auth_payload(
        {"sub": "550e8400-e29b-41d4-a716-446655440000", "role": "company", "company_id": 1}
    )
    assert claims is not None
    assert claims["user_id"] == "550e8400-e29b-41d4-a716-446655440000"
    assert claims["company_id"] == 1


def test_int_user_id():
    claims = normalize_auth_payload({"user_id": 42, "role": "driver", "driver_id": 7})
    assert claims is not None
    assert claims["user_id"] == "42"
