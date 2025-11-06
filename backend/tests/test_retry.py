"""Tests pour le module retry (retry uniformisé avec exponential backoff)."""

import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

from shared.retry import (
    calculate_backoff_delay,
    is_retryable_exception,
    retry_db_operation,
    retry_http_request,
    retry_with_backoff,
)


class TestBackoffCalculation(unittest.TestCase):
    """Tests pour le calcul de backoff."""
    
    def test_exponential_backoff(self):
        """Test que le backoff est exponentiel."""
        delays = [
            calculate_backoff_delay(i, base_delay_ms=250, use_jitter=False)
            for i in range(4)
        ]
        
        # Sans jitter: 250ms, 500ms, 1000ms, 2000ms
        self.assertAlmostEqual(delays[0], 0.250, places=2)
        self.assertAlmostEqual(delays[1], 0.500, places=2)
        self.assertAlmostEqual(delays[2], 1.000, places=2)
        self.assertAlmostEqual(delays[3], 2.000, places=2)
    
    def test_max_delay(self):
        """Test que le délai ne dépasse pas le maximum."""
        delay = calculate_backoff_delay(
            attempt=10,  # Très grand
            base_delay_ms=250,
            max_delay_ms=1000,
            use_jitter=False
        )
        self.assertLessEqual(delay, 1.0)  # 1000ms max
    
    def test_jitter_range(self):
        """Test que le jitter est dans une plage raisonnable."""
        delay = calculate_backoff_delay(
            attempt=1,
            base_delay_ms=1000,
            use_jitter=True
        )
        # Avec jitter [0.5, 1.5): entre 500ms et 1500ms
        self.assertGreaterEqual(delay, 0.5)
        self.assertLess(delay, 1.5)


class TestRetryableException(unittest.TestCase):
    """Tests pour la détection d'exceptions retryables."""
    
    def test_retryable_by_default(self):
        """Test que TimeoutError est retryable par défaut."""
        self.assertTrue(is_retryable_exception(TimeoutError()))
        self.assertTrue(is_retryable_exception(ConnectionError()))
    
    def test_non_retryable_by_default(self):
        """Test que ValueError n'est pas retryable par défaut."""
        self.assertFalse(is_retryable_exception(ValueError()))
        self.assertFalse(is_retryable_exception(KeyError()))
    
    def test_custom_retryable(self):
        """Test avec exceptions retryables personnalisées."""
        class CustomError(Exception):
            pass
        
        self.assertTrue(
            is_retryable_exception(
                CustomError(),
                retryable_exceptions=(CustomError,)
            )
        )
        self.assertFalse(
            is_retryable_exception(
                ValueError(),
                retryable_exceptions=(CustomError,)
            )
        )


class TestRetryWithBackoff(unittest.TestCase):
    """Tests pour retry_with_backoff."""
    
    def test_success_first_attempt(self):
        """Test qu'une fonction qui réussit du premier coup retourne le résultat."""
        result = retry_with_backoff(
            lambda: "success",
            max_retries=3
        )
        self.assertEqual(result, "success")
    
    def test_retry_on_retryable_exception(self):
        """Test qu'une fonction est retriée sur exception retryable."""
        attempts = [0]
        
        def failing_func():
            attempts[0] += 1
            if attempts[0] < 3:
                raise TimeoutError("Temporary error")
            return "success"
        
        with patch("time.sleep"):  # Pas d'attente réelle
            result = retry_with_backoff(
                failing_func,
                max_retries=3
            )
        
        self.assertEqual(result, "success")
        self.assertEqual(attempts[0], 3)
    
    def test_no_retry_on_non_retryable(self):
        """Test qu'une exception non retryable n'est pas retriée."""
        attempts = [0]
        
        def failing_func():
            attempts[0] += 1
            raise ValueError("Permanent error")
        
        with pytest.raises(ValueError, match="Permanent error"):
            retry_with_backoff(
                failing_func,
                max_retries=3
            )
        
        self.assertEqual(attempts[0], 1)  # Une seule tentative
    
    def test_max_retries_exceeded(self):
        """Test que l'exception est levée après max_retries."""
        attempts = [0]
        
        def always_failing():
            attempts[0] += 1
            raise TimeoutError("Always fails")
        
        with patch("time.sleep"), pytest.raises(TimeoutError):  # Pas d'attente réelle
            retry_with_backoff(
                always_failing,
                max_retries=2
            )
        
        # 1 tentative initiale + 2 retries = 3 tentatives
        self.assertEqual(attempts[0], 3)
    
    def test_on_retry_callback(self):
        """Test que le callback on_retry est appelé."""
        callback_calls = []
        
        def failing_func():
            raise TimeoutError("Error")
        
        def on_retry(attempt, exception, delay):
            callback_calls.append((attempt, type(exception).__name__, delay))
        
        with patch("time.sleep"), pytest.raises(TimeoutError):
            retry_with_backoff(
                failing_func,
                max_retries=2,
                on_retry=on_retry
            )
        
        # Callback appelé 2 fois (pour 2 retries)
        self.assertEqual(len(callback_calls), 2)
        self.assertEqual(callback_calls[0][0], 1)  # Première retry
        self.assertEqual(callback_calls[1][0], 2)  # Deuxième retry


class TestRetryHttpRequest(unittest.TestCase):
    """Tests pour retry_http_request."""
    
    def test_http_retryable_codes(self):
        """Test que les codes HTTP retryables déclenchent un retry."""
        import requests
        
        attempts = [0]
        
        def failing_request():
            attempts[0] += 1
            if attempts[0] < 2:
                response = Mock()
                response.status_code = 503
                response.raise_for_status = Mock(side_effect=requests.HTTPError(response=response))
                raise response.raise_for_status()
            return Mock(status_code=200)
        
        from contextlib import suppress
        
        with patch("time.sleep"), suppress(Exception):
            retry_http_request(
                failing_request,
                max_retries=3
            )
        
        # Devrait avoir fait au moins 2 tentatives
        self.assertGreaterEqual(attempts[0], 2)


class TestRetryDbOperation(unittest.TestCase):
    """Tests pour retry_db_operation."""
    
    def test_db_retry_on_operational_error(self):
        """Test que OperationalError déclenche un retry."""
        try:
            from sqlalchemy.exc import OperationalError
        except ImportError:
            self.skipTest("SQLAlchemy non disponible")
        
        attempts = [0]
        
        def failing_db_op():
            attempts[0] += 1
            if attempts[0] < 2:
                raise OperationalError("DB connection lost", None, None)
            return "success"
        
        with patch("time.sleep"):
            result = retry_db_operation(
                failing_db_op,
                max_retries=3
            )
        
        self.assertEqual(result, "success")
        self.assertEqual(attempts[0], 2)


class TestRetryDecorator(unittest.TestCase):
    """Tests pour l'utilisation comme décorateur."""
    
    def test_decorator_usage(self):
        """Test que le décorateur fonctionne."""
        attempts = [0]
        
        @retry_with_backoff(max_retries=2)
        def decorated_func():
            attempts[0] += 1
            if attempts[0] < 2:
                raise TimeoutError("Error")
            return "success"
        
        with patch("time.sleep"):
            result = decorated_func()
        
        self.assertEqual(result, "success")
        self.assertEqual(attempts[0], 2)


if __name__ == "__main__":
    unittest.main()

