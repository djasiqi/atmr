"""
Tests pour le client OSRM (routing, matrice distances, cache).
"""
import pytest
import requests


def test_osrm_haversine_fallback():
    """Calcul de distance haversine pour fallback."""
    from services.osrm_client import _haversine_km
    
    # Lausanne (46.52, 6.63) -> Genève (46.20, 6.15)
    lausanne = (46.52, 6.63)
    geneva = (46.20, 6.15)
    
    distance = _haversine_km(lausanne, geneva)
    
    # Distance attendue ~50-60 km
    assert 40 < distance < 70


def test_osrm_fallback_matrix():
    """Matrice de fallback avec haversine."""
    from services.osrm_client import _fallback_matrix
    
    coords = [
        (46.52, 6.63),  # Lausanne
        (46.20, 6.15),  # Genève
        (47.37, 8.54),  # Zürich
    ]
    
    matrix = _fallback_matrix(coords, avg_kmh=60.0)
    
    # Vérifications basiques
    assert len(matrix) == 3
    assert len(matrix[0]) == 3
    # Diagonale devrait être 0
    assert matrix[0][0] == 0.0
    assert matrix[1][1] == 0.0
    # Distances symétriques (ou presque)
    assert matrix[0][1] > 0
    assert matrix[1][0] > 0


def test_osrm_table_mock_success(monkeypatch):
    """Mock OSRM table renvoie matrice de durées."""
    from services.osrm_client import _table
    
    def mock_requests_get(*args, **kwargs):
        class MockResponse:
            status_code = 200
            def json(self):
                return {
                    'code': 'Ok',
                    'durations': [[0, 600], [600, 0]]
                }
            def raise_for_status(self):
                pass
        return MockResponse()
    
    monkeypatch.setattr(requests, 'get', mock_requests_get)
    
    result = _table(
        base_url="http://localhost:5000",
        profile="driving",
        coords=[(46.52, 6.63), (46.20, 6.15)],
        sources=None,
        destinations=None,
        timeout=5
    )
    
    assert result['code'] == 'Ok'
    assert 'durations' in result
    assert len(result['durations']) == 2


def test_osrm_timeout_raises_exception(monkeypatch):
    """Timeout OSRM lève une exception après retries."""
    from services.osrm_client import _table
    
    def mock_requests_get(*args, **kwargs):
        raise requests.Timeout("Connection timeout")
    
    monkeypatch.setattr(requests, 'get', mock_requests_get)
    
    with pytest.raises((requests.Timeout, RuntimeError)):
        _table(
            base_url="http://localhost:5000",
            profile="driving",
            coords=[(46.52, 6.63), (46.20, 6.15)],
            sources=None,
            destinations=None,
            timeout=1
        )


def test_osrm_cache_key_generation():
    """Clés de cache sont stables et identiques pour mêmes coords."""
    from services.osrm_client import _canonical_key_table
    
    coords1 = [(46.52, 6.63), (46.20, 6.15)]
    coords2 = [(46.52, 6.63), (46.20, 6.15)]  # Identiques
    coords3 = [(46.20, 6.15), (46.52, 6.63)]  # Ordre inversé
    
    key1 = _canonical_key_table(coords1, None, None)
    key2 = _canonical_key_table(coords2, None, None)
    key3 = _canonical_key_table(coords3, None, None)
    
    # Mêmes coords = même clé
    assert key1 == key2
    # Ordre différent = clé différente
    assert key1 != key3
    # Clés sont des strings hexadécimales
    assert isinstance(key1, str)
    assert len(key1) == 40  # SHA-1 hex = 40 caractères


def test_osrm_eta_fallback():
    """ETA fallback avec haversine si OSRM échoue."""
    from services.osrm_client import _fallback_eta_seconds
    
    lausanne = (46.52, 6.63)
    geneva = (46.20, 6.15)
    
    eta = _fallback_eta_seconds(lausanne, geneva, avg_kmh=60.0)
    
    # Distance ~50 km à 60 km/h = ~50 minutes = ~3000 secondes
    assert 2000 < eta < 5000
    assert isinstance(eta, int)

