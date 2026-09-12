"""Tests du sanitizer de signatures : allowlist nh3, pas de regex XSS."""

from html.parser import HTMLParser
from urllib.parse import urlsplit

from shared.html_sanitize import sanitize_email_signature_html


class _HrefSrcParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        found = {key: value for key, value in attrs if value is not None}
        if tag == "a" and "href" in found:
            self.hrefs.append(found["href"])
        if tag == "img" and "src" in found:
            self.srcs.append(found["src"])


class TestSanitizeEmailSignatureHtml:
    def test_keeps_allowed_table_markup(self):
        html = (
            '<table cellpadding="0" cellspacing="0" border="0" '
            'style="font-family: Arial, sans-serif;">'
            "<tr><td><strong>Nom</strong></td></tr></table>"
        )
        result = sanitize_email_signature_html(html)
        assert "<table" in result
        assert "<strong>" in result
        assert "Nom" in result
        assert "font-family" in result

    def test_strips_script_tags(self):
        result = sanitize_email_signature_html(
            "ok<script>alert(1)</script><ScRiPt>alert(1)</ScRiPt>"
        )
        assert "<script" not in result.lower()

    def test_strips_script_src(self):
        result = sanitize_email_signature_html("<script src=x></script>ok")
        assert "<script" not in result.lower()
        assert "ok" in result

    def test_strips_img_event_handler(self):
        result = sanitize_email_signature_html('<img src="x" onerror=alert(1)>')
        assert "onerror" not in result
        assert "<img" in result

    def test_strips_svg(self):
        result = sanitize_email_signature_html("<svg onload=alert(1)>x</svg>")
        assert "<svg" not in result.lower()
        assert "onload" not in result

    def test_strips_javascript_href(self):
        result = sanitize_email_signature_html('<a href="javascript:alert(1)">x</a>')
        assert "javascript:" not in result.lower()
        assert "x" in result

    def test_keeps_http_and_mailto_and_cid(self):
        html = (
            '<a href="https://example.com">site</a>'
            '<a href="mailto:a@b.c">mail</a>'
            '<img src="cid:company_logo" alt="Logo">'
        )
        result = sanitize_email_signature_html(html)
        parsed = _HrefSrcParser()
        parsed.feed(result)
        https_hrefs = [
            href for href in parsed.hrefs if urlsplit(href).scheme == "https"
        ]
        assert len(https_hrefs) == 1
        https_parts = urlsplit(https_hrefs[0])
        assert https_parts.hostname == "example.com"
        assert https_parts.scheme == "https"
        assert https_parts.username is None
        mailto_hrefs = [
            href for href in parsed.hrefs if urlsplit(href).scheme == "mailto"
        ]
        assert mailto_hrefs == ["mailto:a@b.c"]
        assert parsed.srcs == ["cid:company_logo"]

    def test_strips_iframe_and_malformed_nesting(self):
        result = sanitize_email_signature_html(
            '<div><iframe src="https://evil.test"><p>in</p></iframe></div>'
        )
        assert "<iframe" not in result.lower()
        assert "in" in result
