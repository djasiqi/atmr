"""Sanitization HTML à allowlist pour les signatures email.

Utilise bleach : pas de blacklist regex, uniquement des tags/attributs
et protocoles explicitement autorisés.
"""

from __future__ import annotations

import bleach
from bleach.css_sanitizer import CSSSanitizer

SIGNATURE_ALLOWED_TAGS = frozenset(
    {
        "table",
        "tbody",
        "thead",
        "tr",
        "td",
        "th",
        "p",
        "br",
        "a",
        "img",
        "strong",
        "b",
        "em",
        "i",
        "span",
        "div",
    }
)
SIGNATURE_ALLOWED_ATTRIBUTES = {
    "*": ["style"],
    "a": ["href"],
    "img": ["src", "alt", "height", "width"],
    "table": ["cellpadding", "cellspacing", "border", "width", "align"],
    "td": ["width", "align", "valign"],
    "th": ["width", "align", "valign"],
}
SIGNATURE_ALLOWED_PROTOCOLS = ("http", "https", "mailto", "cid")
SIGNATURE_ALLOWED_CSS = (
    "color",
    "background",
    "background-color",
    "font-family",
    "font-size",
    "font-weight",
    "font-style",
    "text-decoration",
    "text-align",
    "vertical-align",
    "line-height",
    "width",
    "max-width",
    "height",
    "min-height",
    "padding",
    "padding-top",
    "padding-right",
    "padding-bottom",
    "padding-left",
    "margin",
    "margin-top",
    "margin-right",
    "margin-bottom",
    "margin-left",
    "border",
    "border-top",
    "border-right",
    "border-bottom",
    "border-left",
    "border-collapse",
    "display",
    "outline",
)

_CSS_SANITIZER = CSSSanitizer(allowed_css_properties=list(SIGNATURE_ALLOWED_CSS))


def sanitize_email_signature_html(html: str) -> str:
    """Nettoie un HTML de signature avec une allowlist minimale."""
    if not html:
        return ""
    return bleach.clean(
        html,
        tags=SIGNATURE_ALLOWED_TAGS,
        attributes=SIGNATURE_ALLOWED_ATTRIBUTES,
        protocols=SIGNATURE_ALLOWED_PROTOCOLS,
        css_sanitizer=_CSS_SANITIZER,
        strip=True,
    )
