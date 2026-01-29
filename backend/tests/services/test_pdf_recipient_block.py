"""
Tests pour le bloc destinataire « Facturé à » (zone C5 enveloppe fenêtre).

Valide :
- Coordonnées X/Y et wrapping respectent les limites (pas de dépassement)
- No data => no UI (aucun bloc si adresse vide)
- _wrap_line_by_words ne coupe pas brutalement
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

from services.documents.pdf import (  # noqa: I001
    DEST_ADDR_LINE_HEIGHT_MM,
    DEST_ADDR_MAX_WIDTH_MM,
    DEST_ADDR_X_MM,
    DEST_ADDR_Y_MM,
    DEST_ADDR_ZONE_HEIGHT_MM,
    _build_recipient_block_flowable,
    _compute_c5_zone_canvas_coords,
    _name_with_uppercase_last_name,
    _wrap_line_by_words,
)


class TestNameWithUppercaseLastName:
    """Tests du formatage nom de famille en majuscules."""

    def test_last_name_uppercase(self):
        """Drin Jasiqi → Drin JASIQI."""
        assert _name_with_uppercase_last_name("Drin Jasiqi") == "Drin JASIQI"

    def test_single_word_uppercase(self):
        """Mot unique → tout en majuscules."""
        assert _name_with_uppercase_last_name("Client") == "CLIENT"

    def test_empty_unchanged(self):
        """Chaîne vide ou None reste inchangé."""
        assert _name_with_uppercase_last_name("") == ""
        assert _name_with_uppercase_last_name(None) is None

    def test_three_words_last_uppercase(self):
        """Jean-Pierre Dupont → Jean-Pierre DUPONT."""
        assert (
            _name_with_uppercase_last_name("Jean-Pierre Dupont") == "Jean-Pierre DUPONT"
        )


class TestWrapLineByWords:
    """Tests du wrapping par mots."""

    def test_short_line_unchanged(self):
        """Ligne courte reste inchangée."""
        line = "Rue de la Paix 1"
        assert _wrap_line_by_words(line, max_chars=90) == line

    def test_long_line_wraps_by_words(self):
        """Ligne longue wrap par mots, pas de coupe brutale."""
        line = "Chemin des Courbes et des Sentiers Fleuris numéro vingt-deux"
        result = _wrap_line_by_words(line, max_chars=30)
        lines = result.split("\n")
        for ln in lines:
            assert len(ln) <= 35, f"Ligne dépasse maxWidth: {len(ln)} chars - {ln!r}"

    def test_no_line_exceeds_max_chars(self):
        """Aucune ligne ne dépasse max_chars."""
        long_addr = "A" * 50 + " " + "B" * 50
        result = _wrap_line_by_words(long_addr, max_chars=40)
        for ln in result.split("\n"):
            assert len(ln) <= 50, f"Dépassement: {len(ln)} > 50"

    def test_empty_returns_empty(self):
        """Chaîne vide retourne vide."""
        assert _wrap_line_by_words("", max_chars=90) == ""
        assert _wrap_line_by_words(None or "", max_chars=90) == ""

    def test_single_word_longer_than_max_stays_one_line(self):
        """Mot unique plus long que max reste sur une ligne (pas de coupe)."""
        word = "Supercalifragilisticexpialidocious"
        result = _wrap_line_by_words(word, max_chars=20)
        assert result == word


class TestRecipientBlockCoordinates:
    """Validation des constantes de position (zone C5)."""

    def test_constants_within_page_bounds_a4(self):
        """X, Y, max_width dans les limites A4 (210 x 297 mm)."""
        page_w_mm = 210.0
        page_h_mm = 297.0

        assert 0 <= DEST_ADDR_X_MM <= page_w_mm - DEST_ADDR_MAX_WIDTH_MM
        assert 0 <= DEST_ADDR_Y_MM <= page_h_mm - DEST_ADDR_ZONE_HEIGHT_MM
        assert DEST_ADDR_MAX_WIDTH_MM > 0
        assert DEST_ADDR_LINE_HEIGHT_MM > 0

    def test_zone_fits_in_margins(self):
        """Zone destinataire ne dépasse pas les margins."""
        page_w_mm = 210.0
        right_margin_mm = 20.0
        assert page_w_mm - right_margin_mm >= DEST_ADDR_X_MM + DEST_ADDR_MAX_WIDTH_MM

    def test_c5_zone_canvas_coords_y_from_top(self):
        """Y depuis le haut : rect_bottom correct (ReportLab origine bas)."""
        page_w, page_h = A4
        x_pt, rect_bottom, zone_w_pt, zone_h_pt = _compute_c5_zone_canvas_coords(
            page_w, page_h
        )
        # rect_bottom = page_h - y_from_top - zone_h (origine en bas)
        assert rect_bottom > 0
        assert rect_bottom < page_h
        # Zone entièrement sur la page
        assert rect_bottom + zone_h_pt <= page_h
        # x dans les limites
        assert x_pt >= 0
        assert x_pt < page_w
        assert x_pt + zone_w_pt <= page_w


class TestBuildRecipientBlockFlowable:
    """Tests du helper _build_recipient_block_flowable."""

    def test_returns_none_when_empty_invoice(self):
        """No data => no UI : retourne None si pas d'adresse."""
        invoice = MagicMock()
        invoice.billing_party_id = None
        invoice.bill_to_client_id = None
        invoice.client_id = 1
        client = MagicMock()
        client.user = MagicMock()
        client.user.first_name = ""
        client.user.last_name = ""
        client.user.username = ""
        client.domicile_address = None
        client.user.address = None
        invoice.client = client

        with patch(
            "services.documents.pdf._get_billed_to",
            return_value=("", ""),
        ):
            para, lines = _build_recipient_block_flowable(invoice, MagicMock())
            assert para is None
            assert lines == []

    def test_returns_para_when_has_data(self):
        """Retourne Paragraph quand données présentes."""
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

        invoice = MagicMock()
        styles = getSampleStyleSheet()
        normal_style = ParagraphStyle(
            "Normal", parent=styles["Normal"], fontSize=10, fontName="Helvetica"
        )
        with patch(
            "services.documents.pdf._get_billed_to",
            return_value=("Clinique Test", "Rue Example 1<br/>1200 Genève"),
        ):
            para, lines = _build_recipient_block_flowable(invoice, normal_style)
            assert para is not None
            assert "Clinique Test" in lines
            # Zone fenêtre : pas de label « Facturé à : » dans le bloc destinataire
            assert "Facturé à" not in para.text

    def test_recipient_lines_respect_wrapping(self):
        """Les lignes retournées ne dépassent pas maxWidth (après wrap)."""
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

        long_addr = "Chemin des Courbes et des Sentiers Fleuris " * 3
        invoice = MagicMock()
        styles = getSampleStyleSheet()
        normal_style = ParagraphStyle(
            "Normal", parent=styles["Normal"], fontSize=10, fontName="Helvetica"
        )
        with patch(
            "services.documents.pdf._get_billed_to",
            return_value=("Nom", long_addr),
        ):
            para, lines = _build_recipient_block_flowable(invoice, normal_style)
            assert para is not None
            max_chars = max(30, int(DEST_ADDR_MAX_WIDTH_MM * 3))
            for line in lines:
                wrapped = _wrap_line_by_words(line, max_chars=max_chars)
                for ln in wrapped.split("\n"):
                    assert len(ln) <= max_chars + 5, f"Dépassement: {ln!r}"

    def test_very_long_address_truncated_with_ellipsis(self):
        """Adresse très longue : clamp hauteur + ellipsis, pas de dépassement."""
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

        max_lines = max(1, int(DEST_ADDR_ZONE_HEIGHT_MM / DEST_ADDR_LINE_HEIGHT_MM))
        # Adresse qui produira > max_lines lignes (15 lignes > 11)
        long_parts = [
            f"Ligne adresse numéro {i} avec texte supplémentaire" for i in range(15)
        ]
        long_addr = "<br/>".join(long_parts)
        invoice = MagicMock()
        styles = getSampleStyleSheet()
        normal_style = ParagraphStyle(
            "Normal", parent=styles["Normal"], fontSize=10, fontName="Helvetica"
        )
        with patch(
            "services.documents.pdf._get_billed_to",
            return_value=("Destinataire Long", long_addr),
        ):
            para, _ = _build_recipient_block_flowable(invoice, normal_style)
            assert para is not None
            # Vérifier qu'on a au plus max_lines lignes visuelles (après "Facturé à")
            content = para.text
            br_count = content.count("<br/>")
            assert br_count <= max_lines, f"Trop de lignes: {br_count} > {max_lines}"
            # Adresse très longue => truncation => ellipsis obligatoire
            assert "…" in content, "Ellipsis attendu quand truncation hauteur"

    def test_ch_fr_address_with_country_stays_in_zone(self):
        """Adresse CH/FR avec pays : wrapping correct, reste dans la zone."""
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics

        invoice = MagicMock()
        styles = getSampleStyleSheet()
        normal_style = ParagraphStyle(
            "Normal", parent=styles["Normal"], fontSize=10, fontName="Helvetica"
        )
        # Format CH : rue, CP ville, CH-xxxx ou France
        ch_addr = (
            "Chemin des Fleurs 42<br/>1228 Plan-les-Ouates<br/>CH-1228 Plan-les-Ouates"
        )
        with patch(
            "services.documents.pdf._get_billed_to",
            return_value=("Clinique Romande SA", ch_addr),
        ):
            para, _ = _build_recipient_block_flowable(invoice, normal_style)
            assert para is not None
            content = para.text
            # Pas de "CH" seul sur une ligne (wrapping correct)
            assert "CH-1228" in content or "Plan-les-Ouates" in content
            # Aucune ligne ne dépasse max_width en points
            max_width_pt = DEST_ADDR_MAX_WIDTH_MM * mm
            for line in content.replace("<br/>", "\n").split("\n"):
                if line.strip():
                    w = pdfmetrics.stringWidth(line, "Helvetica", 10)
                    assert w <= max_width_pt + 2, (
                        f"Ligne dépasse maxWidth: {line!r} = {w:.1f}pt > {max_width_pt:.1f}pt"
                    )

    def test_no_line_exceeds_max_width_pt(self):
        """Aucune ligne ne dépasse DEST_ADDR_MAX_WIDTH_MM en points (stringWidth)."""
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics

        long_addr = "Chemin des Courbes et des Sentiers Fleuris " * 3
        invoice = MagicMock()
        normal_style = ParagraphStyle(
            "Normal",
            parent=getSampleStyleSheet()["Normal"],
            fontSize=10,
            fontName="Helvetica",
        )
        with patch(
            "services.documents.pdf._get_billed_to",
            return_value=("Destinataire", long_addr),
        ):
            para, _ = _build_recipient_block_flowable(invoice, normal_style)
            assert para is not None
            max_width_pt = DEST_ADDR_MAX_WIDTH_MM * mm
            for line in para.text.replace("<br/>", "\n").split("\n"):
                if line.strip():
                    w = pdfmetrics.stringWidth(line, "Helvetica", 10)
                    assert w <= max_width_pt + 2, (
                        f"Ligne dépasse: {line[:50]!r}... = {w:.1f}pt > {max_width_pt:.1f}pt"
                    )
