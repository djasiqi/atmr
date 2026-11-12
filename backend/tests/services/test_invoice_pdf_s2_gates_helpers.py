"""Helpers partagés pour STOP GATE PDF-S2-HEADER-01 et PDF-S2-LINES-01."""

from __future__ import annotations

import re
from io import BytesIO


def extract_text_per_page(pdf_content: bytes) -> list[str]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_content))
        return [page.extract_text() or "" for page in reader.pages]
    except ImportError:
        pytest = __import__("pytest")
        pytest.skip("pypdf requis pour STOP GATE PDF-S2-HEADER-01")


_PRESTATION_HEADER_RE = re.compile(
    r"Date\s+Description\s+Montant",
    re.IGNORECASE,
)
_SERVICE_DATE_RE = re.compile(r"\d{2}\.\s*\d{2}\.\s*\d{4}")
_TRAJET_RE = re.compile(r"Trajet\s*:", re.IGNORECASE)
_QR_PAGE_MARKERS = ("Récépissé", "Compte / Payable à", "QR-facture", "QR facture")


def page_has_prestation_lines(page_text: str) -> bool:
    """True si la page contient des lignes de prestation (hors QR seul)."""
    if any(m in page_text for m in _QR_PAGE_MARKERS) and not _SERVICE_DATE_RE.search(
        page_text
    ):
        return False
    if _SERVICE_DATE_RE.search(page_text):
        return True
    if _TRAJET_RE.search(page_text):
        return True
    return bool("Client :" in page_text and "CHF" in page_text)


def count_prestation_table_headers(page_text: str) -> int:
    """Compte les en-têtes « Date Description Montant » sur une page."""
    return len(_PRESTATION_HEADER_RE.findall(page_text))


def assert_pdf_s2_header_gate(pdf_bytes: bytes) -> None:
    """STOP GATE PDF-S2-HEADER-01 : max 1 en-tête par page avec prestations."""
    pages = extract_text_per_page(pdf_bytes)
    for idx, page_text in enumerate(pages):
        if not page_has_prestation_lines(page_text):
            continue
        header_count = count_prestation_table_headers(page_text)
        assert header_count == 1, (
            f"Page {idx + 1} : attendu 1 en-tête prestations, trouvé {header_count}"
        )


def count_html_br_lines(html: str) -> int:
    if not html:
        return 0
    return html.count("<br/>") + 1


def assert_pdf_contains_ar_tags_when_expected(pdf_bytes: bytes) -> None:
    """Vérifie présence [A/R] si la légende est affichée."""
    full = b"".join(extract_text_per_page(pdf_bytes))
    if "transport aller-retour" in full or "[A/R] =" in full:
        assert "[A/R]" in full, "Légende A/R sans tag inline sur les lignes"
