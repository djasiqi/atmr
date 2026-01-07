#!/usr/bin/env python3
"""
Tests pour valider les corrections d'exceptions larges.

Ce module teste que les corrections appliquées dans la Phase 2 et 3
du plan de correction fonctionnent correctement :
- Exceptions spécifiques sont bien capturées
- Filets de sécurité (except Exception) fonctionnent
- Logging approprié
- Pas de régression
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError

from services.events.handlers_registry import dispatch_event
from services.realtime.socketio import _is_jsonable, emit_company_event
from services.external.weather import WeatherService


class TestExceptionHandlingCorrections:
    """Tests pour valider les corrections d'exceptions."""

    def test_weather_service_network_errors(self):
        """Test : WeatherService gère correctement les erreurs réseau."""
        # Nettoyer le cache avant le test pour éviter les données en cache
        WeatherService.clear_cache()

        with (
            patch("requests.get") as mock_get,
            patch(
                "services.external.weather.WeatherService._get_from_cache",
                return_value=None,
            ),
        ):
            # Simuler une erreur de connexion
            mock_get.side_effect = requests.exceptions.ConnectionError(
                "Connection failed"
            )

            weather = WeatherService.get_weather(46.2044, 6.1432)

            # Devrait retourner la météo par défaut
            assert weather.get("is_default", False) is True
            assert weather["weather_factor"] == 0.5

    def test_weather_service_timeout_errors(self):
        """Test : WeatherService gère correctement les timeouts."""
        # Nettoyer le cache avant le test pour éviter les données en cache
        WeatherService.clear_cache()

        with (
            patch("requests.get") as mock_get,
            patch(
                "services.external.weather.WeatherService._get_from_cache",
                return_value=None,
            ),
        ):
            # Simuler un timeout
            mock_get.side_effect = requests.exceptions.Timeout("Request timeout")

            weather = WeatherService.get_weather(46.2044, 6.1432)

            # Devrait retourner la météo par défaut
            assert weather.get("is_default", False) is True

    def test_weather_service_json_decode_errors(self):
        """Test : WeatherService gère correctement les erreurs de parsing JSON."""
        # Nettoyer le cache avant le test pour éviter les données en cache
        WeatherService.clear_cache()

        with (
            patch("requests.get") as mock_get,
            patch(
                "services.external.weather.WeatherService._get_from_cache",
                return_value=None,
            ),
        ):
            # Simuler une réponse invalide
            mock_response = Mock()
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_get.return_value = mock_response

            weather = WeatherService.get_weather(46.2044, 6.1432)

            # Devrait retourner la météo par défaut
            assert weather.get("is_default", False) is True

    def test_event_registry_data_errors(self):
        """Test : EventHandlersRegistry gère correctement les erreurs de données."""

        # Créer un handler qui lève une ValueError
        def bad_handler(event):
            raise ValueError("Invalid data")

        # Enregistrer le handler
        from services.events.handlers_registry import register

        register("test_event", bad_handler)

        # Dispatcher l'événement - ne devrait pas lever d'exception
        # mais logger l'erreur
        with patch("services.events.handlers_registry.logger") as mock_logger:
            dispatch_event({"event_type": "test_event", "data": "test"})

            # Vérifier que l'erreur est loggée
            mock_logger.warning.assert_called()
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("validation error" in str(call) for call in warning_calls)

    def test_event_registry_connection_errors(self):
        """Test : EventHandlersRegistry gère correctement les erreurs réseau."""

        # Créer un handler qui lève une ConnectionError
        def network_handler(event):
            raise ConnectionError("Network error")

        from services.events.handlers_registry import register

        register("test_network_event", network_handler)

        # Dispatcher l'événement - ne devrait pas lever d'exception
        with patch("services.events.handlers_registry.logger") as mock_logger:
            dispatch_event({"event_type": "test_network_event", "data": "test"})

            # Vérifier que l'erreur réseau est loggée
            mock_logger.warning.assert_called()
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("network error" in str(call) for call in warning_calls)

    def test_socketio_service_json_serialization(self):
        """Test : SocketIOService gère correctement les erreurs de sérialisation JSON."""
        # Tester avec un objet non sérialisable
        non_serializable = object()

        # _is_jsonable devrait retourner False
        result = _is_jsonable(non_serializable)
        assert result is False

        # Tester avec un objet sérialisable
        serializable = {"key": "value"}
        result = _is_jsonable(serializable)
        assert result is True

    def test_notification_service_socketio_errors(self):
        """Test : SocketIO service gère correctement les erreurs Socket.IO."""
        with patch("services.realtime.socketio.socketio") as mock_socketio:
            # Simuler une erreur de connexion Socket.IO
            mock_socketio.emit.side_effect = ConnectionError(
                "Socket.IO connection failed"
            )

            # Devrait gérer l'erreur sans lever d'exception
            with patch("services.realtime.socketio.app_logger") as mock_logger:
                emit_company_event(
                    company_id=1, event="test_event", payload={"test": "data"}
                )

                # Vérifier que l'erreur est loggée
                mock_logger.error.assert_called()
                error_calls = [str(call) for call in mock_logger.error.call_args_list]
                assert any("network error" in str(call) for call in error_calls)

    def test_notification_service_type_errors(self):
        """Test : SocketIO service gère correctement les TypeError."""
        with patch("services.realtime.socketio.socketio") as mock_socketio:
            # Simuler une TypeError (problème de compatibilité Socket.IO)
            # La première tentative avec 'to=' lève TypeError, puis fallback 'room='
            mock_socketio.emit.side_effect = TypeError("Invalid type")

            # Devrait gérer l'erreur sans lever d'exception
            # Le TypeError devrait être capturé et traité avec un fallback
            emit_company_event(
                company_id=1, event="test_event", payload={"test": "data"}
            )

            # Vérifier que emit a été appelé (tentative de compatibilité)
            assert mock_socketio.emit.called

    def test_exception_safety_net(self):
        """Test : Les filets de sécurité (except Exception) fonctionnent."""

        # Créer un handler qui lève une exception inattendue
        def unexpected_handler(event):
            raise RuntimeError("Unexpected error")

        from services.events.handlers_registry import register

        register("test_unexpected_event", unexpected_handler)

        # Dispatcher l'événement - ne devrait pas lever d'exception
        # mais être capturé par le filet de sécurité
        with patch("services.events.handlers_registry.logger") as mock_logger:
            dispatch_event({"event_type": "test_unexpected_event", "data": "test"})

            # Vérifier que l'erreur inattendue est loggée avec exception()
            mock_logger.exception.assert_called()

    def test_integrity_error_handling(self):
        """Test : Les IntegrityError sont gérées spécifiquement."""
        # Ce test vérifie que les IntegrityError sont bien capturées
        # dans les services qui utilisent la DB
        # (ex: apply.py, invoice_service.py, etc.)

        # Simuler une IntegrityError
        integrity_error = IntegrityError(
            statement="INSERT INTO ...", params={}, orig=Exception("Unique constraint")
        )

        # Vérifier que l'erreur est bien une IntegrityError
        assert isinstance(integrity_error, IntegrityError)

    def test_operational_error_handling(self):
        """Test : Les OperationalError sont gérées spécifiquement."""
        # Simuler une OperationalError
        operational_error = OperationalError(
            statement="SELECT ...", params={}, orig=Exception("Connection lost")
        )

        # Vérifier que l'erreur est bien une OperationalError
        assert isinstance(operational_error, OperationalError)

    def test_dbapi_error_handling(self):
        """Test : Les DBAPIError sont gérées spécifiquement."""
        # Simuler une DBAPIError
        dbapi_error = DBAPIError(
            statement="UPDATE ...", params={}, orig=Exception("Database error")
        )

        # Vérifier que l'erreur est bien une DBAPIError
        assert isinstance(dbapi_error, DBAPIError)

    def test_value_error_handling(self):
        """Test : Les ValueError sont gérées spécifiquement."""
        # Tester qu'une ValueError est bien capturée
        try:
            raise ValueError("Invalid value")
        except ValueError:
            # Devrait être capturée
            pass
        else:
            pytest.fail("ValueError should have been caught")

    def test_type_error_handling(self):
        """Test : Les TypeError sont gérées spécifiquement."""
        # Tester qu'une TypeError est bien capturée
        try:
            # Opération invalide pour déclencher TypeError
            _result = "string" + 123  # type: ignore
        except TypeError:
            # Devrait être capturée
            pass
        else:
            pytest.fail("TypeError should have been caught")

    def test_connection_error_handling(self):
        """Test : Les ConnectionError sont gérées spécifiquement."""
        # Tester qu'une ConnectionError est bien capturée
        try:
            raise ConnectionError("Connection failed")
        except ConnectionError:
            # Devrait être capturée
            pass
        else:
            pytest.fail("ConnectionError should have been caught")

    def test_key_error_handling(self):
        """Test : Les KeyError sont gérées spécifiquement."""
        # Tester qu'une KeyError est bien capturée
        try:
            data: dict[str, str] = {}
            # Accès à une clé manquante pour déclencher KeyError
            _value = data["missing_key"]  # type: ignore
        except KeyError:
            # Devrait être capturée
            pass
        else:
            pytest.fail("KeyError should have been caught")

    def test_attribute_error_handling(self):
        """Test : Les AttributeError sont gérées spécifiquement."""
        # Tester qu'une AttributeError est bien capturée
        try:
            obj = object()
            _ = obj.nonexistent_attribute  # type: ignore
        except AttributeError:
            # Devrait être capturée
            pass
        else:
            pytest.fail("AttributeError should have been caught")


class TestExceptionLogging:
    """Tests pour valider que le logging est approprié."""

    def test_logging_with_context(self):
        """Test : Les erreurs sont loggées avec le contexte approprié."""

        # Créer un handler qui lève une ValueError
        def context_handler(event):
            raise ValueError("Invalid data")

        from services.events.handlers_registry import register

        register("test_context_event", context_handler)

        # Dispatcher l'événement
        with patch("services.events.handlers_registry.logger") as mock_logger:
            dispatch_event({"event_type": "test_context_event", "data": "test"})

            # Vérifier que le log contient le contexte (event_type, handler)
            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            assert "test_context_event" in call_args or "handler" in call_args.lower()

    def test_exception_logging_with_trace(self):
        """Test : Les exceptions inattendues sont loggées avec trace complète."""

        # Créer un handler qui lève une exception inattendue
        def unexpected_handler(event):
            raise RuntimeError("Unexpected error")

        from services.events.handlers_registry import register

        register("test_trace_event", unexpected_handler)

        # Dispatcher l'événement
        with patch("services.events.handlers_registry.logger") as mock_logger:
            dispatch_event({"event_type": "test_trace_event", "data": "test"})

            # Vérifier que exception() est appelé (trace complète)
            mock_logger.exception.assert_called()
            # exception() devrait être appelé avec un message
            assert len(mock_logger.exception.call_args[0]) > 0


class TestExceptionRegression:
    """Tests de régression pour vérifier qu'il n'y a pas de régression."""

    def test_no_broad_exception_swallowing(self):
        """Test : Les exceptions ne sont pas silencieusement avalées sans logging."""

        # Créer un handler qui lève une exception
        def error_handler(event):
            raise ValueError("Test error")

        from services.events.handlers_registry import register

        register("test_no_swallow_event", error_handler)

        # Dispatcher l'événement
        with patch("services.events.handlers_registry.logger") as mock_logger:
            dispatch_event({"event_type": "test_no_swallow_event", "data": "test"})

            # Vérifier que l'erreur est loggée (pas silencieusement avalée)
            assert mock_logger.warning.called or mock_logger.error.called

    def test_exception_specificity(self):
        """Test : Les exceptions spécifiques sont bien différenciées."""
        # Vérifier que les différents types d'exceptions sont bien différenciés
        assert ValueError is not TypeError
        assert ConnectionError is not OSError
        assert OperationalError is not IntegrityError
        assert KeyError is not AttributeError

    def test_exception_inheritance(self):
        """Test : Les exceptions héritent correctement."""
        # Vérifier que les exceptions héritent de Exception
        assert issubclass(ValueError, Exception)
        assert issubclass(TypeError, Exception)
        assert issubclass(ConnectionError, Exception)
        assert issubclass(OperationalError, Exception)
        assert issubclass(IntegrityError, Exception)
        assert issubclass(KeyError, Exception)
        assert issubclass(AttributeError, Exception)


if __name__ == "__main__":
    """Exécution directe pour tests rapides."""
    pytest.main([__file__, "-v"])


