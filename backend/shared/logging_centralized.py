"""✅ Phase 3: Handler de logging pour centralisation des logs (ELK/Loki).

Permet d'envoyer les logs vers un système centralisé (Elasticsearch, Loki, etc.)
avec format JSON structuré.
"""
# pyright: reportImplicitOverride = false

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

try:
    from typing import override
except ImportError:  # pragma: no cover
    from typing_extensions import override

import requests


class CentralizedLogHandler(logging.Handler):
    """Handler de logging pour envoyer les logs vers un système centralisé.

    Supporte:
    - Elasticsearch (via HTTP API)
    - Loki (via HTTP API)
    - Format JSON structuré
    """

    def __init__(
        self,
        endpoint: str | None = None,
        log_type: str = "elasticsearch",
        level: int = logging.INFO,
        timeout: int = 5,
    ):
        """Initialise le handler.

        Args:
            endpoint: URL du endpoint de logging (ex: "http://elasticsearch:9200/_bulk")
            log_type: Type de système ("elasticsearch", "loki")
            level: Niveau de logging minimum
            timeout: Timeout en secondes pour les requêtes HTTP
        """
        super().__init__(level)
        self.endpoint = endpoint or os.getenv("LOG_CENTRALIZATION_ENDPOINT")
        self.log_type = (
            log_type or os.getenv("LOG_CENTRALIZATION_TYPE", "elasticsearch").lower()
        )
        self.timeout = timeout
        self.enabled = bool(self.endpoint)

    @override
    def emit(self, record: logging.LogRecord) -> None:
        """Envoie le log vers le système centralisé.

        Args:
            record: Enregistrement de log
        """
        if not self.enabled:
            return

        try:
            # Formater le log en JSON structuré
            log_data = self._format_log_record(record)

            # Envoyer selon le type de système
            if self.log_type == "elasticsearch":
                self._send_to_elasticsearch(log_data)
            elif self.log_type == "loki":
                self._send_to_loki(log_data)
            else:
                # Format générique JSON
                self._send_generic(log_data)

        except Exception:
            # Ne pas lever d'exception pour éviter de casser le logging
            self.handleError(record)

    def _format_log_record(self, record: logging.LogRecord) -> dict[str, Any]:
        """Formate un enregistrement de log en JSON structuré.

        Args:
            record: Enregistrement de log

        Returns:
            Dictionnaire JSON structuré
        """
        # Informations de base
        log_data = {
            "@timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Ajouter les informations de contexte si disponibles
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        if hasattr(record, "ip_address"):
            log_data["ip_address"] = record.ip_address

        # Ajouter les exceptions
        if record.exc_info:
            log_data["exception"] = self.format(record)

        # Ajouter les arguments supplémentaires
        if record.args:
            log_data["args"] = str(record.args)

        # Ajouter les métadonnées personnalisées
        if hasattr(record, "metadata"):
            log_data.update(record.metadata)

        return log_data

    def _send_to_elasticsearch(self, log_data: dict[str, Any]) -> None:
        """Envoie le log vers Elasticsearch.

        Args:
            log_data: Données de log formatées
        """
        if not self.endpoint:
            return

        # Format Elasticsearch bulk API
        index_name = os.getenv("LOG_ELASTICSEARCH_INDEX", "atmr-logs")
        timestamp = log_data["@timestamp"]
        date_suffix = timestamp.split("T")[0].replace("-", ".")

        # Index avec date (ex: atmr-logs-2025.01.20)
        index = f"{index_name}-{date_suffix}"

        # Format bulk API: action + document
        bulk_data = json.dumps({"index": {"_index": index}}) + "\n"
        bulk_data += json.dumps(log_data) + "\n"

        # Envoyer vers Elasticsearch
        response = requests.post(
            self.endpoint,
            data=bulk_data,
            headers={"Content-Type": "application/x-ndjson"},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def _send_to_loki(self, log_data: dict[str, Any]) -> None:
        """Envoie le log vers Loki.

        Args:
            log_data: Données de log formatées
        """
        if not self.endpoint:
            return

        # Format Loki push API
        labels = {
            "level": log_data["level"],
            "logger": log_data["logger"],
            "module": log_data["module"],
        }

        # Convertir timestamp en nanosecondes (Loki utilise nanosecondes)
        timestamp_ns = int(
            datetime.fromisoformat(
                log_data["@timestamp"].replace("Z", "+00:00")
            ).timestamp()
            * 1e9
        )

        # Format Loki push API
        loki_data = {
            "streams": [
                {
                    "stream": labels,
                    "values": [[str(timestamp_ns), json.dumps(log_data)]],
                }
            ]
        }

        # Envoyer vers Loki
        response = requests.post(
            f"{self.endpoint}/loki/api/v1/push",
            json=loki_data,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def _send_generic(self, log_data: dict[str, Any]) -> None:
        """Envoie le log vers un endpoint générique.

        Args:
            log_data: Données de log formatées
        """
        if not self.endpoint:
            return

        # Envoyer en JSON simple
        response = requests.post(
            self.endpoint,
            json=log_data,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()


def setup_centralized_logging(app) -> None:
    """Configure la centralisation des logs pour l'application Flask.

    Args:
        app: Instance Flask
    """
    endpoint = os.getenv("LOG_CENTRALIZATION_ENDPOINT")
    if not endpoint:
        app.logger.debug(
            "[Centralized Logging] Désactivé (LOG_CENTRALIZATION_ENDPOINT non configuré)"
        )
        return

    log_type = os.getenv("LOG_CENTRALIZATION_TYPE", "elasticsearch").lower()
    level_str = os.getenv("LOG_CENTRALIZATION_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    # Créer le handler
    handler = CentralizedLogHandler(endpoint=endpoint, log_type=log_type, level=level)

    # Ajouter au logger de l'application
    app.logger.addHandler(handler)
    app.logger.setLevel(level)

    app.logger.info(
        "[Centralized Logging] ✅ Activé - Endpoint: %s, Type: %s, Level: %s",
        endpoint,
        log_type,
        level_str,
    )
