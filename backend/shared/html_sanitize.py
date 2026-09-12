"""Sanitization HTML à allowlist pour les signatures email.

Utilise nh3 (Ammonia) : pas de blacklist regex, uniquement des tags,
attributs, protocoles et propriétés CSS explicitement autorisés.
"""

from __future__ import annotations

import nh3

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
    "*": frozenset({"style"}),
    "a": frozenset({"href"}),
    "img": frozenset({"src", "alt", "height", "width"}),
    "table": frozenset({"cellpadding", "cellspacing", "border", "width", "align"}),
    "td": frozenset({"width", "align", "valign"}),
    "th": frozenset({"width", "align", "valign"}),
}
SIGNATURE_ALLOWED_PROTOCOLS = frozenset({"http", "https", "mailto", "cid"})
SIGNATURE_ALLOWED_CSS = frozenset(
    {
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
    }
)

_CLEANER = nh3.Cleaner(
    tags=set(SIGNATURE_ALLOWED_TAGS),
    attributes={
        key: set(values) for key, values in SIGNATURE_ALLOWED_ATTRIBUTES.items()
    },
    url_schemes=set(SIGNATURE_ALLOWED_PROTOCOLS),
    filter_style_properties=set(SIGNATURE_ALLOWED_CSS),
    link_rel=None,
)


def sanitize_email_signature_html(html: str) -> str:
    """Nettoie un HTML de signature avec une allowlist minimale."""
    if not html:
        return ""
    return _CLEANER.clean(html)
