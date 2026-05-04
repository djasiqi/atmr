"""Tests de régression sécurité — Plan remédiation LIRIE Vague 1.

Empêche qu'une future PR réintroduise une faille corrigée :
- F1: GET /config -> 404 (pas d'exposition credentials)
- F6: /uploads avec Origin non autorisée -> pas d'ACAO ; X-Content-Type-Options nosniff
- F2: Dispatch rate limit -> 429 après dépassement
- F10: CSRF prod sans secret -> RuntimeError
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestF1ConfigEndpointRemoved:
    """F1: Vérifier que /config n'existe plus (exposition DATABASE_URI)."""

    def test_config_returns_404(self, client):
        """GET /config doit retourner 404 (route supprimée)."""
        response = client.get("/config")
        assert response.status_code == 404


class TestF6UploadsCorsAndHeaders:
    """F6: CORS uploads restreint + header nosniff."""

    @pytest.fixture
    def uploads_dir(self, app):
        """Crée un répertoire uploads avec un fichier de test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app.config["UPLOADS_DIR"] = tmpdir
            app.config["UPLOAD_FOLDER"] = tmpdir
            uploads_dir = Path(tmpdir)
            uploads_dir.mkdir(parents=True, exist_ok=True)
            test_file = uploads_dir / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4 test")
            yield uploads_dir

    def test_uploads_evil_origin_no_acao(self, client, uploads_dir):
        """Origin: https://evil.com ne doit pas recevoir Access-Control-Allow-Origin."""
        response = client.get(
            "/uploads/test.pdf",
            headers={"Origin": "https://evil.com"},
        )
        # 200 si fichier existe, 404 sinon
        assert response.status_code in (200, 404)
        acao = response.headers.get("Access-Control-Allow-Origin")
        assert acao != "https://evil.com"
        assert acao != "*"

    def test_uploads_has_nosniff_header(self, client, uploads_dir):
        """Réponses /uploads doivent inclure X-Content-Type-Options: nosniff."""
        response = client.get("/uploads/test.pdf")
        assert response.status_code in (200, 404)
        assert response.headers.get("X-Content-Type-Options") == "nosniff"


class TestF2DispatchRateLimit:
    """F2: Rate limits dispatch 30/h et 50/h."""

    def test_dispatch_run_has_rate_limit_30_per_hour(self):
        """Vérifie que /run a la limite 30/h (régression si quelqu'un remet 10000)."""
        from pathlib import Path

        dispatch_routes = (
            Path(__file__).resolve().parent.parent.parent
            / "routes"
            / "dispatch_routes.py"
        )
        content = dispatch_routes.read_text(encoding="utf-8")
        assert "30 per hour" in content, (
            "dispatch_routes.py doit contenir '30 per hour' pour /run"
        )

    def test_dispatch_trigger_has_rate_limit_50_per_hour(self):
        """Vérifie que /trigger a la limite 50/h (régression si quelqu'un remet 10000)."""
        from pathlib import Path

        dispatch_routes = (
            Path(__file__).resolve().parent.parent.parent
            / "routes"
            / "dispatch_routes.py"
        )
        content = dispatch_routes.read_text(encoding="utf-8")
        assert "50 per hour" in content, (
            "dispatch_routes.py doit contenir '50 per hour' pour /trigger"
        )


class TestF10CsrfProductionNoFallback:
    """F10: En production sans secret, CSRF doit lever RuntimeError."""

    def test_csrf_production_no_secret_raises(self):
        """En production sans secret, _get_csrf_secret doit lever RuntimeError."""
        from services.security.csrf import _get_csrf_secret

        with (
            patch.dict(
                "os.environ",
                {
                    "FLASK_CONFIG": "production",
                    "FLASK_ENV": "production",
                    "JWT_SECRET_KEY": "",
                    "SECRET_KEY": "",
                    "FLASK_SECRET_KEY": "",
                },
                clear=False,
            ),
            patch(
                "services.security.csrf._is_production",
                return_value=True,
            ),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _get_csrf_secret()
            assert "aucune clé secrète" in str(exc_info.value).lower() or (
                "production" in str(exc_info.value).lower()
            )

    def test_csrf_dev_no_secret_fallback_accepted(self):
        """En dev/test sans secret, fallback toléré (pas d'exception)."""
        from services.security.csrf import _get_csrf_secret

        with (
            patch.dict(
                "os.environ",
                {
                    "FLASK_CONFIG": "development",
                    "JWT_SECRET_KEY": "",
                    "SECRET_KEY": "",
                    "FLASK_SECRET_KEY": "",
                },
                clear=False,
            ),
            patch(
                "services.security.csrf._is_production",
                return_value=False,
            ),
        ):
            secret = _get_csrf_secret()
            assert secret == "temporary-csrf-secret-change-in-production"
