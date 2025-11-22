"""Tests unitaires pour la centralisation des logs (ELK/Loki).

Valide le fonctionnement du handler de logging centralisé :
- Envoi vers Elasticsearch
- Envoi vers Loki
- Envoi vers endpoint générique
- Formatage des logs
- Gestion des erreurs
"""

import logging
from unittest.mock import MagicMock, patch

import requests

from shared.logging_centralized import CentralizedLogHandler, setup_centralized_logging


class TestHandlerElasticsearch:
    """Tests pour l'envoi vers Elasticsearch."""

    @patch("shared.logging_centralized.requests.post")
    @patch("shared.logging_centralized.os.getenv")
    def test_handler_elasticsearch_success(self, mock_getenv, mock_post):
        """Test envoi réussi vers Elasticsearch."""
        # Mock endpoint
        mock_getenv.return_value = "http://elasticsearch:9200/_bulk"

        # Mock réponse HTTP
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Créer handler
        handler = CentralizedLogHandler(
            endpoint="http://elasticsearch:9200/_bulk", log_type="elasticsearch"
        )

        # Créer un log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Émettre le log
        handler.emit(record)

        # Vérifier que la requête a été envoyée
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["headers"]["Content-Type"] == "application/x-ndjson"
        # Vérifier que les données sont au format bulk
        data = call_args[1]["data"]
        assert "index" in data
        assert "test-logs" in data or "atmr-logs" in data

    @patch("shared.logging_centralized.requests.post")
    @patch("shared.logging_centralized.os.getenv")
    def test_send_to_elasticsearch_bulk_format(self, mock_getenv, mock_post):
        """Test format bulk Elasticsearch."""
        # Mock endpoint
        mock_getenv.return_value = "http://elasticsearch:9200/_bulk"

        # Mock réponse HTTP
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Créer handler
        handler = CentralizedLogHandler(
            endpoint="http://elasticsearch:9200/_bulk", log_type="elasticsearch"
        )

        # Créer un log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Émettre le log
        handler.emit(record)

        # Vérifier le format bulk (2 lignes: action + document)
        call_args = mock_post.call_args
        data = call_args[1]["data"]
        lines = data.split("\n")
        assert len([line for line in lines if line.strip()]) == 2  # 2 lignes non vides


class TestHandlerLoki:
    """Tests pour l'envoi vers Loki."""

    @patch("shared.logging_centralized.requests.post")
    @patch("shared.logging_centralized.os.getenv")
    def test_handler_loki_success(self, mock_getenv, mock_post):
        """Test envoi réussi vers Loki."""
        # Mock endpoint
        mock_getenv.return_value = "http://loki:3100"

        # Mock réponse HTTP
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Créer handler
        handler = CentralizedLogHandler(endpoint="http://loki:3100", log_type="loki")

        # Créer un log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Émettre le log
        handler.emit(record)

        # Vérifier que la requête a été envoyée vers /loki/api/v1/push
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://loki:3100/loki/api/v1/push"
        assert call_args[1]["headers"]["Content-Type"] == "application/json"

        # Vérifier le format Loki
        loki_data = call_args[1]["json"]
        assert "streams" in loki_data
        assert len(loki_data["streams"]) == 1
        assert "stream" in loki_data["streams"][0]
        assert "values" in loki_data["streams"][0]

    @patch("shared.logging_centralized.requests.post")
    @patch("shared.logging_centralized.os.getenv")
    def test_send_to_loki_push_format(self, mock_getenv, mock_post):
        """Test format push Loki."""
        # Mock endpoint
        mock_getenv.return_value = "http://loki:3100"

        # Mock réponse HTTP
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Créer handler
        handler = CentralizedLogHandler(endpoint="http://loki:3100", log_type="loki")

        # Créer un log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Émettre le log
        handler.emit(record)

        # Vérifier le format Loki
        call_args = mock_post.call_args
        loki_data = call_args[1]["json"]
        stream = loki_data["streams"][0]
        assert "level" in stream["stream"]
        assert "logger" in stream["stream"]
        assert len(stream["values"]) == 1
        # Vérifier que la valeur contient un timestamp (nanosecondes) et le log JSON
        value = stream["values"][0]
        assert len(value) == 2
        assert value[0].isdigit()  # Timestamp en nanosecondes


class TestHandlerGeneric:
    """Tests pour l'envoi vers endpoint générique."""

    @patch("shared.logging_centralized.requests.post")
    @patch("shared.logging_centralized.os.getenv")
    def test_handler_generic_success(self, mock_getenv, mock_post):
        """Test envoi réussi vers endpoint générique."""
        # Mock endpoint
        mock_getenv.return_value = "http://log-aggregator:8080/logs"

        # Mock réponse HTTP
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Créer handler
        handler = CentralizedLogHandler(
            endpoint="http://log-aggregator:8080/logs", log_type="generic"
        )

        # Créer un log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Émettre le log
        handler.emit(record)

        # Vérifier que la requête a été envoyée
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://log-aggregator:8080/logs"
        assert call_args[1]["headers"]["Content-Type"] == "application/json"
        # Vérifier que les données sont en JSON
        json_data = call_args[1]["json"]
        assert "@timestamp" in json_data
        assert "level" in json_data
        assert "message" in json_data


class TestHandlerDisabled:
    """Tests pour handler désactivé."""

    @patch("shared.logging_centralized.requests.post")
    def test_handler_disabled_no_endpoint(self, mock_post):
        """Test handler désactivé si pas d'endpoint."""
        # Créer handler sans endpoint
        handler = CentralizedLogHandler(endpoint=None)

        # Créer un log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Émettre le log
        handler.emit(record)

        # Vérifier qu'aucune requête n'a été envoyée
        mock_post.assert_not_called()


class TestFormatLogRecord:
    """Tests pour le formatage des logs."""

    def test_format_log_record_complete(self):
        """Test formatage complet du log."""
        handler = CentralizedLogHandler(endpoint="http://test:8080/logs")

        # Créer un log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        log_data = handler._format_log_record(record)

        assert "@timestamp" in log_data
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test"
        assert log_data["function"] == "<module>"
        assert log_data["line"] == 42

    def test_format_log_record_with_exception(self):
        """Test formatage avec exception."""
        handler = CentralizedLogHandler(endpoint="http://test:8080/logs")

        # Créer un log record avec exception
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=True,
            )

            log_data = handler._format_log_record(record)

            assert "@timestamp" in log_data
            assert log_data["level"] == "ERROR"
            assert "exception" in log_data or "Traceback" in log_data["message"]

    def test_format_log_record_with_metadata(self):
        """Test formatage avec métadonnées."""
        handler = CentralizedLogHandler(endpoint="http://test:8080/logs")

        # Créer un log record avec métadonnées personnalisées
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        # Ajouter métadonnées personnalisées
        record.request_id = "req-123"
        record.user_id = "user-456"
        record.ip_address = "192.168.1.100"
        record.metadata = {"custom": "value"}

        log_data = handler._format_log_record(record)

        assert log_data["request_id"] == "req-123"
        assert log_data["user_id"] == "user-456"
        assert log_data["ip_address"] == "192.168.1.100"
        assert log_data["custom"] == "value"


class TestHandlerErrorHandling:
    """Tests pour la gestion des erreurs."""

    @patch("shared.logging_centralized.requests.post")
    @patch("shared.logging_centralized.os.getenv")
    def test_handler_error_handling(self, mock_getenv, mock_post):
        """Test gestion erreurs (ne bloque pas l'app)."""
        # Mock endpoint
        mock_getenv.return_value = "http://elasticsearch:9200/_bulk"

        # Mock erreur HTTP
        mock_post.side_effect = requests.RequestException("Connection error")

        # Créer handler
        handler = CentralizedLogHandler(
            endpoint="http://elasticsearch:9200/_bulk", log_type="elasticsearch"
        )

        # Créer un log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Émettre le log (ne doit pas lever d'exception)
        handler.emit(record)

        # Vérifier que handleError a été appelé (via le comportement de logging.Handler)
        # L'erreur est gérée silencieusement

    @patch("shared.logging_centralized.requests.post")
    @patch("shared.logging_centralized.os.getenv")
    def test_handler_timeout(self, mock_getenv, mock_post):
        """Test gestion timeout."""
        # Mock endpoint
        mock_getenv.return_value = "http://elasticsearch:9200/_bulk"

        # Mock timeout
        mock_post.side_effect = requests.Timeout("Request timeout")

        # Créer handler
        handler = CentralizedLogHandler(
            endpoint="http://elasticsearch:9200/_bulk",
            log_type="elasticsearch",
            timeout=5,
        )

        # Créer un log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Émettre le log (ne doit pas lever d'exception)
        handler.emit(record)


class TestSetupCentralizedLogging:
    """Tests pour la configuration du handler."""

    @patch("shared.logging_centralized.os.getenv")
    def test_setup_centralized_logging_enabled(self, mock_getenv):
        """Test configuration du handler activée."""
        # Mock variables d'environnement
        mock_getenv.side_effect = lambda key, default=None: {
            "LOG_CENTRALIZATION_ENDPOINT": "http://elasticsearch:9200/_bulk",
            "LOG_CENTRALIZATION_TYPE": "elasticsearch",
            "LOG_CENTRALIZATION_LEVEL": "INFO",
        }.get(key, default)

        # Créer app Flask
        from flask import Flask

        app = Flask(__name__)

        # Configurer le logging
        setup_centralized_logging(app)

        # Vérifier qu'un handler a été ajouté
        assert len(app.logger.handlers) > 0

    @patch("shared.logging_centralized.os.getenv")
    def test_setup_centralized_logging_disabled(self, mock_getenv):
        """Test configuration du handler désactivée."""
        # Mock pas d'endpoint
        mock_getenv.return_value = None

        # Créer app Flask
        from flask import Flask

        app = Flask(__name__)
        initial_handlers = len(app.logger.handlers)

        # Configurer le logging
        setup_centralized_logging(app)

        # Vérifier qu'aucun handler n'a été ajouté
        assert len(app.logger.handlers) == initial_handlers
