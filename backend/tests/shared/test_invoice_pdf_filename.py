"""Tests du nom de fichier PDF facture personnalisé."""

from types import SimpleNamespace

from shared.invoice_pdf_filename import (
    build_invoice_pdf_download_filename,
    format_invoice_amount_for_filename,
    slugify_invoice_filename_part,
)


def test_slugify_invoice_filename_part():
    assert slugify_invoice_filename_part("M. VUILLE Michel") == "M_VUILLE_Michel"
    assert slugify_invoice_filename_part("Février") == "Fevrier"


def test_format_invoice_amount_for_filename():
    assert format_invoice_amount_for_filename(155) == "155CHF"
    assert format_invoice_amount_for_filename(155.5) == "155_50CHF"


def test_build_invoice_pdf_download_filename():
    invoice = SimpleNamespace(
        id=12,
        period_month=7,
        period_year=2026,
        invoice_number="EM-2026-07-0005",
        total_amount=155,
        billing_party=None,
        billed_to_company=None,
        bill_to_client=SimpleNamespace(
            is_institution=False,
            institution_name=None,
            user=SimpleNamespace(
                first_name="Michel",
                last_name="VUILLE",
                username="M VUILLE Michel",
            ),
        ),
    )
    # Si prénom/nom présents → VUILLE_Michel (pas le username)
    assert (
        build_invoice_pdf_download_filename(invoice)
        == "Facture_Juillet_2026_EM-2026-07-0005_VUILLE_Michel_155CHF.pdf"
    )


def test_build_invoice_pdf_download_filename_from_username():
    invoice = SimpleNamespace(
        id=12,
        period_month=7,
        period_year=2026,
        invoice_number="EM-2026-07-0005",
        total_amount=155,
        billing_party=None,
        billed_to_company=None,
        bill_to_client=SimpleNamespace(
            is_institution=False,
            institution_name=None,
            user=SimpleNamespace(first_name="", last_name="", username="M VUILLE Michel"),
        ),
    )
    assert (
        build_invoice_pdf_download_filename(invoice)
        == "Facture_Juillet_2026_EM-2026-07-0005_M_VUILLE_Michel_155CHF.pdf"
    )
