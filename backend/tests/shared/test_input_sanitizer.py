"""✅ Tests unitaires pour l'utilitaire de sanitisation des inputs.

Teste que les fonctions de sanitisation échappent correctement HTML/JS
et valident les inputs utilisateur.
"""


from shared.input_sanitizer import (
    escape_html,
    escape_js,
    sanitize_email,
    sanitize_integer,
    sanitize_string,
    sanitize_url,
)


class TestEscapeHtml:
    """Tests pour escape_html()."""

    def test_escape_basic_html(self):
        """Test échappement HTML basique."""
        assert escape_html("<script>alert('xss')</script>") == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"

    def test_escape_special_chars(self):
        """Test échappement caractères spéciaux."""
        assert escape_html("Test & Test") == "Test &amp; Test"
        assert escape_html('Test "quote"') == "Test &quot;quote&quot;"

    def test_none_input(self):
        """Test avec input None."""
        assert escape_html(None) is None


class TestEscapeJs:
    """Tests pour escape_js()."""

    def test_escape_quotes(self):
        """Test échappement guillemets."""
        assert escape_js("test 'quote'") == "test \\'quote\\'"
        assert escape_js('test "quote"') == 'test \\"quote\\"'

    def test_escape_newlines(self):
        """Test échappement newlines."""
        assert escape_js("line1\nline2") == "line1\\nline2"
        assert escape_js("line1\rline2") == "line1\\rline2"

    def test_none_input(self):
        """Test avec input None."""
        assert escape_js(None) is None


class TestSanitizeString:
    """Tests pour sanitize_string()."""

    def test_strip_html(self):
        """Test suppression balises HTML."""
        result = sanitize_string("<script>alert('xss')</script>", strip_html=True)
        assert "<script>" not in result
        assert "alert" in result  # Le contenu reste

    def test_max_length(self):
        """Test limitation longueur."""
        long_string = "A" * 20000  # > MAX_STRING_LENGTH
        result = sanitize_string(long_string, max_length=100)
        assert len(result) <= 100

    def test_escape_html_chars(self):
        """Test échappement caractères HTML."""
        result = sanitize_string("<script>", escape_html_chars=True)
        assert "&lt;script&gt;" in result

    def test_none_input(self):
        """Test avec input None."""
        assert sanitize_string(None) is None


class TestSanitizeEmail:
    """Tests pour sanitize_email()."""

    def test_valid_email(self):
        """Test email valide."""
        assert sanitize_email("test@example.com") == "test@example.com"
        assert sanitize_email("TEST@EXAMPLE.COM") == "test@example.com"  # Lowercase

    def test_invalid_email_format(self):
        """Test email format invalide."""
        assert sanitize_email("invalid-email") is None
        assert sanitize_email("@example.com") is None

    def test_email_too_long(self):
        """Test email trop long."""
        long_email = "a" * 250 + "@example.com"  # > MAX_EMAIL_LENGTH
        assert sanitize_email(long_email) is None

    def test_none_input(self):
        """Test avec input None."""
        assert sanitize_email(None) is None


class TestSanitizeUrl:
    """Tests pour sanitize_url()."""

    def test_valid_http_url(self):
        """Test URL HTTP valide."""
        url = sanitize_url("http://example.com/path")
        assert url == "http://example.com/path"

    def test_valid_https_url(self):
        """Test URL HTTPS valide."""
        url = sanitize_url("https://example.com/path")
        assert url == "https://example.com/path"

    def test_invalid_scheme(self):
        """Test schéma invalide."""
        assert sanitize_url("javascript:alert('xss')") is None
        assert sanitize_url("ftp://example.com") is None  # Si http/https uniquement

    def test_url_too_long(self):
        """Test URL trop longue."""
        long_url = "https://example.com/" + "a" * 3000  # > MAX_URL_LENGTH
        assert sanitize_url(long_url) is None

    def test_none_input(self):
        """Test avec input None."""
        assert sanitize_url(None) is None


class TestSanitizeInteger:
    """Tests pour sanitize_integer()."""

    def test_valid_integer(self):
        """Test entier valide."""
        assert sanitize_integer(42) == 42
        assert sanitize_integer("42") == 42

    def test_integer_with_range(self):
        """Test entier avec limites."""
        assert sanitize_integer(50, min_val=1, max_val=100) == 50
        assert sanitize_integer(150, min_val=1, max_val=100) is None  # > max
        assert sanitize_integer(0, min_val=1, max_val=100) is None  # < min

    def test_invalid_input(self):
        """Test input invalide."""
        assert sanitize_integer("not_a_number") is None
        assert sanitize_integer(None) is None
