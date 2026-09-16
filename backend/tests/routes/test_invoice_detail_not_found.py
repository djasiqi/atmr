"""GET détail / PDF facture : 404 réel, catalogues d'IDs séparés."""

from __future__ import annotations

from datetime import UTC, datetime

from tests.routes.test_invoices_list_lot6 import (
    _company_headers,
    _make_partner_invoice,
    lot6_partnership,
)


def test_missing_standard_invoice_is_not_found_not_validation_error(
    client, sample_user, sample_company
):
    headers = _company_headers(client, sample_user, sample_company.id)
    missing_id = 34

    resp = client.get(
        f"/api/v1/invoices/companies/{sample_company.id}/invoices/{missing_id}",
        headers=headers,
    )
    assert resp.status_code == 404
    body = resp.get_json()
    assert body["error_code"] == "not_found"
    assert body["error_code"] != "validation_error"
    assert "Erreur inconnue" not in (body.get("error") or "")
    assert body.get("details", {}).get("resource_type") == "Facture"
    assert body.get("details", {}).get("resource_id") == missing_id


def test_partner_invoice_id_does_not_resolve_as_standard_invoice(
    client,
    db,
    sample_user,
    sample_company,
    lot6_partnership,
):
    issued = datetime(2026, 8, 7, 10, 11, tzinfo=UTC)
    partner = _make_partner_invoice(
        db, sample_company, lot6_partnership, issued_at=issued
    )
    partner.pdf_url = f"/uploads/partner-invoices/{partner.id}.pdf"
    db.session.commit()

    headers = _company_headers(client, sample_user, sample_company.id)
    shared_id = partner.id

    detail = client.get(
        f"/api/v1/invoices/companies/{sample_company.id}/invoices/{shared_id}",
        headers=headers,
    )
    assert detail.status_code == 404
    detail_body = detail.get_json()
    assert detail_body["error_code"] == "not_found"
    assert detail_body["error_code"] != "validation_error"

    standard_pdf = client.get(
        f"/api/v1/invoices/companies/{sample_company.id}/invoices/{shared_id}/pdf",
        headers=headers,
    )
    assert standard_pdf.status_code == 404
    pdf_body = standard_pdf.get_json()
    assert pdf_body["error_code"] == "not_found"
    assert pdf_body["error_code"] != "validation_error"
