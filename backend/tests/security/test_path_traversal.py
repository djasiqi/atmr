"""Tests unitaires pour la protection contre le path traversal."""

import tempfile
from pathlib import Path

import pytest
from werkzeug.exceptions import NotFound


@pytest.fixture
def uploads_dir(app):
    """Crée un répertoire uploads avec un fichier de test."""
    # Utiliser l'app fixture globale (session-scoped) et surcharger UPLOADS_DIR
    # avec un répertoire temporaire pour ce module.
    with tempfile.TemporaryDirectory() as tmpdir:
        app.config["UPLOADS_DIR"] = tmpdir
        app.config["UPLOAD_FOLDER"] = tmpdir
        uploads_dir = Path(tmpdir)
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Logos publics uniquement (Lot 0 SEC-06)
        logos = uploads_dir / "company_logos"
        logos.mkdir(parents=True, exist_ok=True)
        test_file = logos / "test.txt"
        test_file.write_text("test content")

        subdir = logos / "subdir"
        subdir.mkdir(exist_ok=True)
        subdir_file = subdir / "subfile.txt"
        subdir_file.write_text("subfile content")

        yield uploads_dir


class TestPathTraversalProtection:
    """Tests pour la protection contre le path traversal."""

    def test_valid_file_access(self, client, uploads_dir):
        """Test que l'accès à un logo public fonctionne."""
        response = client.get("/uploads/company_logos/test.txt")
        assert response.status_code == 200
        assert b"test content" in response.data

    def test_valid_subdirectory_file_access(self, client, uploads_dir):
        """Test que l'accès à un fichier logo dans un sous-répertoire fonctionne."""
        response = client.get("/uploads/company_logos/subdir/subfile.txt")
        assert response.status_code == 200
        assert b"subfile content" in response.data

    def test_non_public_prefix_returns_404(self, client, uploads_dir):
        """Lot 0 SEC-06: hors logos → 404 même si le fichier existe."""
        private = uploads_dir / "invoices"
        private.mkdir(parents=True, exist_ok=True)
        (private / "x.pdf").write_bytes(b"%PDF")
        response = client.get("/uploads/invoices/x.pdf")
        assert response.status_code == 404

    def test_path_traversal_dot_dot_slash(self, client, uploads_dir):
        """Test que ../ est bloqué."""
        response = client.get("/uploads/../app.py")
        assert response.status_code == 404

    def test_path_traversal_multiple_dot_dot(self, client, uploads_dir):
        """Test que ../../ est bloqué."""
        response = client.get("/uploads/../../etc/passwd")
        assert response.status_code == 404

    def test_path_traversal_encoded(self, client, uploads_dir):
        """Test que les encodages URL de ../ sont bloqués."""
        # %2e%2e%2f = ../
        response = client.get("/uploads/%2e%2e%2fapp.py")
        assert response.status_code == 404

    def test_path_traversal_double_encoded(self, client, uploads_dir):
        """Test que les double encodages sont bloqués."""
        # %252e%252e%252f = %2e%2e%2f = ../
        response = client.get("/uploads/%252e%252e%252fapp.py")
        assert response.status_code == 404

    def test_path_traversal_backslash(self, client, uploads_dir):
        """Test que les backslashes sont bloqués (Windows)."""
        # Flask normalise les backslashes, mais testons quand même
        response = client.get("/uploads/..\\app.py")
        assert response.status_code == 404

    def test_path_traversal_absolute_path(self, client, uploads_dir):
        """Test que les chemins absolus sont bloqués."""
        # Tenter d'accéder à un chemin absolu
        import os

        if os.name == "nt":  # Windows
            test_path = "C:\\Windows\\System32\\config\\sam"
        else:  # Unix
            test_path = "/etc/passwd"

        # Flask normalise généralement, mais testons
        response = client.get(f"/uploads/{test_path}")
        # Selon Flask/Werkzeug, un chemin absolu peut être normalisé via redirect (308)
        assert response.status_code in (308, 404)

    def test_path_traversal_null_byte(self, client, uploads_dir):
        """Test que les null bytes sont gérés correctement."""
        response = client.get("/uploads/company_logos/test.txt%00")
        assert response.status_code in (200, 404)

    def test_path_traversal_symlink(self, client, uploads_dir, tmp_path):
        """Test que les liens symboliques hors base sont bloqués."""
        external_dir = tmp_path / "external"
        external_dir.mkdir()
        external_file = external_dir / "secret.txt"
        external_file.write_text("secret content")

        symlink = uploads_dir / "company_logos" / "symlink"
        try:
            symlink.symlink_to(external_dir)
            response = client.get("/uploads/company_logos/symlink/secret.txt")
            assert response.status_code == 404
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

    def test_nonexistent_file(self, client, uploads_dir):
        """Test que les fichiers inexistants retournent 404."""
        response = client.get("/uploads/company_logos/nonexistent.txt")
        assert response.status_code == 404

    def test_empty_filename(self, client, uploads_dir):
        """Test qu'un nom de fichier vide est géré."""
        response = client.get("/uploads/")
        # Flask peut rediriger ou retourner 404
        assert response.status_code in (404, 405)

    def test_invoices_pdf_404_public_route_removed(self, client, uploads_dir):
        """Lot 0 SEC-06: /uploads/invoices/... n'est plus public (404)."""
        invoices_dir = uploads_dir / "invoices"
        invoices_dir.mkdir(parents=True, exist_ok=True)
        dummy_pdf = invoices_dir / "invoice_test_20260101_120000.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")

        response = client.get("/uploads/invoices/invoice_test_20260101_120000.pdf")
        assert response.status_code == 404

    def test_invoices_pdf_404_if_absent(self, client, uploads_dir):
        """Test que /uploads/invoices/<absent>.pdf retourne 404."""
        response = client.get("/uploads/invoices/nonexistent_invoice_999.pdf")
        assert response.status_code == 404
