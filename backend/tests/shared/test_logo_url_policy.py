"""Politique des URLs de logo persistées."""

import pytest
from marshmallow import ValidationError

from schemas.company_schemas import CompanyUpdateSchema
from schemas.validation_utils import validate_request
from shared.logo_url_policy import InvalidLogoUrl, normalize_persisted_logo_url

DENIED = [
    "javascript:alert(1)",
    "JaVaScRiPt:alert(1)",
    " javascript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "data:text/javascript,alert(1)",
    "file:///etc/passwd",
    "vbscript:msgbox(1)",
    "//evil.example/logo.png",
    "http://evil.example/logo.png",
    "blob:https://example.com/123",
    "https://trusted.example@evil.example/logo.png",
    "/uploads/../secret.png",
]


@pytest.mark.parametrize("value", DENIED)
def test_denied_persisted_logo_urls(value):
    with pytest.raises(InvalidLogoUrl):
        normalize_persisted_logo_url(value)


def test_allowed_uploads_and_https():
    assert (
        normalize_persisted_logo_url("/uploads/company_logos/logo.png")
        == "/uploads/company_logos/logo.png"
    )
    assert (
        normalize_persisted_logo_url("https://cdn.example.com/logo.webp")
        == "https://cdn.example.com/logo.webp"
    )


def test_empty_is_none():
    assert normalize_persisted_logo_url(None) is None
    assert normalize_persisted_logo_url("   ") is None


@pytest.mark.parametrize("value", DENIED)
def test_company_update_schema_rejects_forbidden_logo_url(value):
    with pytest.raises(ValidationError) as exc:
        validate_request(CompanyUpdateSchema(), {"logo_url": value}, strict=False)
    assert "logo_url" in exc.value.messages.get("errors", {})


def test_company_update_schema_accepts_https_logo():
    result = validate_request(
        CompanyUpdateSchema(),
        {"logo_url": "https://cdn.example.com/logo.png"},
        strict=False,
    )
    assert result["logo_url"] == "https://cdn.example.com/logo.png"
