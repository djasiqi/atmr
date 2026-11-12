"""
Helpers pour les tests d'intégration DDD.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

import pytest


def assert_response_status(response, expected_status: int) -> None:
    """Vérifie que le statut HTTP de la réponse correspond à celui attendu."""
    assert response.status_code == expected_status, (
        f"Expected status {expected_status}, got {response.status_code}. "
        f"Response: {response.get_data(as_text=True)}"
    )


def assert_response_json(
    response, expected_keys: list[str] | None = None
) -> dict[str, Any]:
    """Vérifie que la réponse contient du JSON et optionnellement certaines clés."""
    assert response.is_json, "Response should be JSON"
    data = response.get_json()
    assert data is not None, "Response JSON should not be None"

    if expected_keys:
        for key in expected_keys:
            assert key in data, (
                f"Response should contain key '{key}'. Got: {list(data.keys())}"
            )

    return data


def make_authenticated_request(
    client, method: str, url: str, headers: dict[str, str] | None = None, **kwargs
):
    """Fait une requête HTTP authentifiée."""
    if headers is None:
        headers = {}

    method_upper = method.upper()
    if method_upper == "GET":
        return client.get(url, headers=headers, **kwargs)
    if method_upper == "POST":
        return client.post(url, headers=headers, **kwargs)
    if method_upper == "PUT":
        return client.put(url, headers=headers, **kwargs)
    if method_upper == "DELETE":
        return client.delete(url, headers=headers, **kwargs)
    msg = f"Unsupported HTTP method: {method}"
    raise ValueError(msg)


def measure_performance(threshold_seconds: float):
    """Décorateur pour mesurer le temps d'exécution d'un test et valider un seuil."""

    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                if elapsed > threshold_seconds:
                    pytest.fail(
                        f"Performance threshold exceeded: {elapsed:.2f}s > "
                        f"{threshold_seconds}s"
                    )
                return result
            except Exception:
                elapsed = time.time() - start_time
                print(f"Test failed after {elapsed:.2f}s")
                raise

        # Marquer le test comme test de performance
        return pytest.mark.performance(wrapper)

    return decorator
