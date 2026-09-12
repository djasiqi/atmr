"""Conversion MIME HTML → text/plain : pas une barrière XSS."""

from services.notifications.email import _html_to_text


def test_html_to_text_keeps_visible_copy():
    result = _html_to_text("<p>Bonjour</p><div>Client</div>")
    assert "Bonjour" in result
    assert "Client" in result


def test_html_to_text_drops_script_and_style():
    result = _html_to_text(
        "<style>.x{color:red}</style><p>OK</p><script>alert(1)</script>"
    )
    assert result == "OK"
    assert "alert" not in result
    assert "color:red" not in result


def test_html_to_text_unescapes_entities():
    result = _html_to_text("<p>A&amp;C</p>")
    assert result == "A&C"
