"""Le sélecteur patient refuse une entreprise qui n'est pas celle du jeton."""

from __future__ import annotations

from tests.routes.test_invoices_list_lot6 import _company_headers

pytest_plugins = ["tests.routes.test_invoices_list_lot6"]


def test_invoice_candidates_refuse_une_autre_entreprise(
    client, sample_user, sample_company
):
    headers = _company_headers(client, sample_user, sample_company.id)
    other_id = int(sample_company.id) + 999_999
    response = client.get(
        f"/api/v1/invoices/companies/{other_id}/invoices/invoice-candidates"
        "?payer_type=patient&period=2026-09",
        headers=headers,
    )
    assert response.status_code == 403


def test_invoice_candidates_de_sa_propre_entreprise(
    client, sample_user, sample_company
):
    headers = _company_headers(client, sample_user, sample_company.id)
    response = client.get(
        f"/api/v1/invoices/companies/{sample_company.id}/invoices/invoice-candidates"
        "?payer_type=patient&period=2026-09",
        headers=headers,
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["data"]["period"] == "2026-09"
    assert isinstance(body["data"]["patients"], list)
