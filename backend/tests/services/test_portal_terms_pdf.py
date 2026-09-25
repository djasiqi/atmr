"""PDF officiel des conditions PORTAL."""

from services.legal.portal_terms_catalog import (
    portal_terms_v1,
    prepared_portal_terms_v2,
)
from services.legal.portal_terms_pdf import (
    build_portal_terms_pdf_bytes,
    portal_terms_pdf_filename,
    resolve_lirie_logo_path,
)


def test_portal_terms_pdf_contains_pdf_magic_and_version():
    cgu, cgt = prepared_portal_terms_v2()
    pdf = build_portal_terms_pdf_bytes(cgu)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000
    assert portal_terms_pdf_filename(cgu).endswith("_v2.0.pdf")
    assert "CGU" in portal_terms_pdf_filename(cgu)

    pdf_cgt = build_portal_terms_pdf_bytes(cgt)
    assert pdf_cgt.startswith(b"%PDF")
    assert "CGV" in portal_terms_pdf_filename(cgt)


def test_portal_terms_pdf_v1_still_builds():
    cgu, _ = portal_terms_v1()
    pdf = build_portal_terms_pdf_bytes(cgu)
    assert pdf.startswith(b"%PDF")


def test_lirie_logo_resolvable_in_container_or_repo():
    # En CI/Docker le logo doit être présent sous assets/lirie.
    path = resolve_lirie_logo_path()
    assert path is not None
    assert path.is_file()


def test_logo_flowable_preserves_native_aspect_ratio():
    from reportlab.lib.utils import ImageReader

    from services.legal.portal_terms_pdf import _logo_flowable

    logo = _logo_flowable()
    assert logo is not None
    path = resolve_lirie_logo_path()
    native_w, native_h = ImageReader(str(path)).getSize()
    expected = native_h / native_w
    actual = float(logo.drawHeight) / float(logo.drawWidth)
    assert abs(actual - expected) < 1e-6
