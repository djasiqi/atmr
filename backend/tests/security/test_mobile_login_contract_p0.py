"""P0 : contrat login mobile-device-session-v1 fail-closed."""

from __future__ import annotations

from unittest.mock import patch

CONTRACT_V1 = "mobile-device-session-v1"
MOBILE_HEADERS = {
    "X-Requested-With": "Expo",
    "X-Client-Platform": "android",
    "X-Auth-Contract-Version": CONTRACT_V1,
}


def _login_payload(sample_user):
    return {"email": sample_user.email, "password": "password123"}


class TestMobileLoginContractP0:
    def test_v1_without_device_id_returns_400_no_tokens(self, client, sample_user):
        resp = client.post(
            "/api/v1/auth/login",
            json=_login_payload(sample_user),
            headers=MOBILE_HEADERS,
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data.get("error_code") == "device_identity_required"
        assert "access_token" not in data
        assert "refresh_token" not in data
        assert "token" not in data
        assert "recovery_credential" not in data

    def test_v1_session_create_failed_returns_503(self, client, sample_user):
        with patch(
            "routes.auth.create_or_reuse_session",
            side_effect=RuntimeError("mds boom"),
        ):
            resp = client.post(
                "/api/v1/auth/login",
                json=_login_payload(sample_user),
                headers={
                    **MOBILE_HEADERS,
                    "X-Device-ID": "test-installation-id",
                    "X-Device-Name": "test-device",
                },
            )
        assert resp.status_code == 503
        data = resp.get_json()
        assert data.get("error_code") == "session_create_failed"
        assert "access_token" not in data
        assert "refresh_token" not in data

    def test_v1_incomplete_session_returns_503(self, client, sample_user):
        # Session créée mais recovery/revocation absents → contrat incomplet
        fake_session = type(
            "S",
            (),
            {
                "session_id": "00000000-0000-0000-0000-000000000001",
                "session_epoch": 1,
                "generation": 1,
                "credential_generation": 1,
                "refresh_generation": 1,
            },
        )()
        with patch(
            "routes.auth.create_or_reuse_session",
            return_value=(fake_session, None, None, []),
        ):
            resp = client.post(
                "/api/v1/auth/login",
                json=_login_payload(sample_user),
                headers={
                    **MOBILE_HEADERS,
                    "X-Device-ID": "test-installation-id",
                    "X-Device-Name": "test-device",
                },
            )
        assert resp.status_code == 503
        data = resp.get_json()
        assert data.get("error_code") == "mobile_session_contract_incomplete"
        assert "access_token" not in data
        assert "refresh_token" not in data

    def test_v1_complete_login_returns_required_fields(self, client, sample_user):
        resp = client.post(
            "/api/v1/auth/login",
            json=_login_payload(sample_user),
            headers={
                **MOBILE_HEADERS,
                "X-Device-ID": "test-installation-id-complete",
                "X-Device-Name": "test-device",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("access_token") or data.get("token")
        assert data.get("refresh_token")
        assert data.get("session_id")
        assert data.get("session_epoch") is not None
        assert data.get("refresh_generation") is not None
        assert isinstance(data.get("recovery_credential"), str)
        assert len(data["recovery_credential"]) > 0
        assert isinstance(data.get("revocation_secret"), str)
        assert len(data["revocation_secret"]) > 0

    def test_legacy_mobile_without_contract_still_200(self, client, sample_user):
        """Sans X-Auth-Contract-Version : anciens clients restent compatibles."""
        resp = client.post(
            "/api/v1/auth/login",
            json=_login_payload(sample_user),
            headers={"X-Requested-With": "Expo"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("access_token") or data.get("token")
        assert data.get("refresh_token")
