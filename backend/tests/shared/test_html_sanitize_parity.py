"""Parité de contrat Bleach (oracle EOL) → nh3 sur les signatures email.

Ne compare pas le HTML octet à octet : nh3 normalise le CSS.
On vérifie le contrat de sécurité et les tokens Outlook nécessaires.
"""

from __future__ import annotations

import bleach
from bleach.css_sanitizer import CSSSanitizer

from shared.html_sanitize import (
    SIGNATURE_ALLOWED_ATTRIBUTES,
    SIGNATURE_ALLOWED_CSS,
    SIGNATURE_ALLOWED_PROTOCOLS,
    SIGNATURE_ALLOWED_TAGS,
    sanitize_email_signature_html,
)

_BLEACH_CSS = CSSSanitizer(allowed_css_properties=list(SIGNATURE_ALLOWED_CSS))


def _bleach_reference(html: str) -> str:
    """Même allowlist que la primitive prod, via Bleach 6.4.0."""
    return bleach.clean(
        html,
        tags=SIGNATURE_ALLOWED_TAGS,
        attributes={
            key: list(values) for key, values in SIGNATURE_ALLOWED_ATTRIBUTES.items()
        },
        protocols=tuple(SIGNATURE_ALLOWED_PROTOCOLS),
        css_sanitizer=_BLEACH_CSS,
        strip=True,
    )


OUTLOOK_TABLE = (
    '<table cellpadding="0" cellspacing="0" border="0" width="520" align="left" '
    'style="width:520px; max-width:520px; font-family: Arial, sans-serif; '
    'font-size: 11px; color: #333; margin-top: 12px;">'
    '<tr><td style="vertical-align: top; padding-right: 12px; width: 50%;">'
    '<strong style="font-size: 12px;">Nom</strong><br>022 512 02 03'
    "</td>"
    '<td width="1" style="border-left: 2px solid #1b4b7a; padding-left: 12px;">'
    '<a href="mailto:a@b.c" style="color: #1b4b7a; text-decoration: none;">a@b.c</a>'
    "</td></tr></table>"
)

CID_IMG = (
    '<img src="cid:company_logo" alt="Logo" height="26" '
    'style="display:block;border:0;outline:none;text-decoration:none;'
    'height:26px;width:auto;max-width:100%;" />'
)

SAMPLES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "table_outlook": (
        OUTLOOK_TABLE,
        (
            "<table",
            "cellpadding",
            "cellspacing",
            "Nom",
            "mailto:a@b.c",
            "border-left",
            "#1b4b7a",
            "font-family",
            "<td",
        ),
        ("javascript:", "<script", "onerror", "onload"),
    ),
    "cid_img": (
        CID_IMG,
        ("cid:company_logo", "<img", "display"),
        ("javascript:", "onerror"),
    ),
    "https_img": (
        '<img src="https://example.com/logo.png" height="26" alt="Logo">',
        ("https://example.com/logo.png", "<img"),
        ("javascript:",),
    ),
    "script": (
        "ok<script>alert(1)</script><ScRiPt>alert(1)</ScRiPt>",
        ("ok",),
        ("<script",),
    ),
    "onerror": (
        '<img src="x" onerror=alert(1)>',
        ("<img",),
        ("onerror",),
    ),
    "svg": (
        "<svg onload=alert(1)>x</svg>",
        ("x",),
        ("<svg", "onload"),
    ),
    "js_href": (
        '<a href="javascript:alert(1)">x</a>',
        ("x",),
        ("javascript:",),
    ),
    "iframe": (
        '<div><iframe src="https://evil.test"><p>in</p></iframe></div>',
        ("in",),
        ("<iframe",),
    ),
    "css_js_url": (
        '<div style="background: url(javascript:alert(1)); color: red;">z</div>',
        ("z", "color"),
        ("javascript:",),
    ),
}


def _assert_contract(
    label: str, html: str, must: tuple[str, ...], must_not: tuple[str, ...]
) -> None:
    for token in must:
        assert token in html, f"{label} manque {token!r} dans {html}"
    lowered = html.lower()
    for token in must_not:
        assert token.lower() not in lowered, f"{label} fuit {token!r} dans {html}"


def test_bleach_and_nh3_share_security_contract() -> None:
    for name, (payload, must, must_not) in SAMPLES.items():
        bleach_html = _bleach_reference(payload)
        nh3_html = sanitize_email_signature_html(payload)
        _assert_contract(f"bleach:{name}", bleach_html, must, must_not)
        _assert_contract(f"nh3:{name}", nh3_html, must, must_not)


def test_outlook_table_attributes_survive_nh3() -> None:
    result = sanitize_email_signature_html(OUTLOOK_TABLE)
    for attr in ("cellpadding", "cellspacing", "border", "width", "align"):
        assert attr in result
    assert "mailto:a@b.c" in result
    assert "vertical-align" in result
