"""Tests du contrat texte : extraction HTML via parseur, pas via blacklist."""

from shared.html_text import html_to_plain_text, strip_html_to_text


class TestStripHtmlToText:
    def test_keeps_inner_text_of_simple_tags(self):
        assert strip_html_to_text("<p>Hello World</p>") == "Hello World"

    def test_drops_script_content(self):
        result = strip_html_to_text("ok<script>alert(1)</script>suite")
        assert result == "oksuite"
        assert "alert" not in result

    def test_drops_script_case_and_whitespace_end_tag(self):
        result = strip_html_to_text("ok<ScRiPt>alert(1)</ScRiPt >suite")
        assert "alert" not in result
        assert "ok" in result
        assert "suite" in result

    def test_drops_img_and_event_handler_markup(self):
        result = strip_html_to_text("<img src=x onerror=alert(1)>texte")
        assert result == "texte"
        assert "onerror" not in result

    def test_decodes_entities_as_text(self):
        assert strip_html_to_text("A&amp;B") == "A&B"


class TestHtmlToPlainText:
    def test_block_tags_become_newlines(self):
        result = html_to_plain_text("<p>Bonjour</p><p>Monde</p>")
        assert "Bonjour" in result
        assert "Monde" in result
        assert "\n" in result

    def test_script_and_style_are_not_in_plain_text(self):
        html = "<style>body{color:red}</style><p>Visible</p><script>alert(1)</script>"
        result = html_to_plain_text(html)
        assert result == "Visible"
        assert "alert" not in result
        assert "color:red" not in result

    def test_unescapes_entities(self):
        assert html_to_plain_text("<p>A&amp;B</p>") == "A&B"
