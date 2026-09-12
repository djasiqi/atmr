"""Tests du sanitizer de signatures : allowlist bleach, pas de regex XSS."""

from shared.html_sanitize import sanitize_email_signature_html


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
        assert "https://example.com" in result
        assert "mailto:a@b.c" in result
        assert "cid:company_logo" in result

    def test_strips_iframe_and_malformed_nesting(self):
        result = sanitize_email_signature_html(
            '<div><iframe src="https://evil.test"><p>in</p></iframe></div>'
        )
        assert "<iframe" not in result.lower()
        assert "in" in result
