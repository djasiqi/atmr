"""Tests Lot 0 P0 — accès uploads publics vs privés (SEC-06)."""

from __future__ import annotations

from pathlib import Path


class TestUploadsPublicVsPrivate:
    def test_private_prefixes_return_404(self, client, app, tmp_path):
        uploads = tmp_path / "uploads"
        for prefix in ("invoices", "chat", "transport_vouchers", "statements"):
            d = uploads / prefix
            d.mkdir(parents=True)
            (d / "secret.pdf").write_bytes(b"%PDF-1.4 secret")

        logos = uploads / "company_logos"
        logos.mkdir(parents=True)
        (logos / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\nlogo")

        app.config["UPLOADS_DIR"] = str(uploads)
        app.config["UPLOAD_FOLDER"] = str(uploads)

        for prefix in ("invoices", "chat", "transport_vouchers", "statements"):
            resp = client.get(f"/uploads/{prefix}/secret.pdf")
            assert resp.status_code == 404, prefix

        resp_logo = client.get("/uploads/company_logos/logo.png")
        assert resp_logo.status_code == 200

    def test_invoice_pdf_api_requires_auth(self, client):
        resp = client.get("/api/v1/invoices/companies/1/invoices/1/pdf")
        assert resp.status_code in (401, 422)
