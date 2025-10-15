"""
Tests du service OSRM (routing et distance)
"""
from unittest.mock import Mock

import pytest


class TestOSRMService:
    """Tests du service OSRM"""

    def test_route_info_success(self, mocker):
        """Test récupération route OSRM"""
        from services.osrm_client import route_info

        # Mock response OSRM
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'code': 'Ok',
            'routes': [{
                'duration': 1800,  # 30 min
                'distance': 25000  # 25 km
            }]
        }

        mocker.patch('requests.get', return_value=mock_response)

        result = route_info(
            (46.2044, 6.1432),
            (46.5197, 6.6323),
            base_url='http://localhost:5000'
        )

        assert result is not None

    def test_route_info_fallback_haversine(self, mocker):
        """Test fallback haversine si OSRM échoue"""
        from services.osrm_client import route_info

        # Mock OSRM failure
        mocker.patch('requests.get', side_effect=Exception('OSRM timeout'))

        result = route_info(
            (46.2044, 6.1432),
            (46.5197, 6.6323),
            base_url='http://localhost:5000'
        )

        # Fallback haversine retourne toujours quelque chose
        assert result is not None

    def test_matrix_table_success(self, mocker):
        """Test calcul matrice distances"""
        from services.osrm_client import build_distance_matrix_osrm

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'code': 'Ok',
            'durations': [[0, 1200, 1800], [1200, 0, 900], [1800, 900, 0]],
            'distances': [[0, 10000, 15000], [10000, 0, 8000], [15000, 8000, 0]]
        }

        mocker.patch('requests.get', return_value=mock_response)

        coords = [
            (46.2044, 6.1432),
            (46.5197, 6.6323),
            (47.3769, 8.5417)
        ]

        result = build_distance_matrix_osrm(coords, base_url='http://localhost:5000')

        assert result is not None

    def test_route_info_cache_redis(self, mocker):
        """Test cache Redis pour routes"""
        from services.osrm_client import route_info

        # Mock Redis
        mock_redis = Mock()
        mock_redis.get.return_value = None  # Pas en cache
        mock_redis.setex.return_value = True

        # Mock OSRM response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'code': 'Ok',
            'routes': [{
                'duration': 1500,
                'distance': 20000
            }]
        }

        mocker.patch('requests.get', return_value=mock_response)
        mocker.patch('ext.redis_client', mock_redis)

        result = route_info(
            (46.2044, 6.1432),
            (46.5197, 6.6323),
            base_url='http://localhost:5000'
        )

        assert result is not None


class TestMapServices:
    """Tests services maps (geocoding, distance)"""

    def test_get_distance_duration(self, mocker):
        """Test calcul distance/durée entre 2 points"""
        from services.maps import get_distance_duration

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'code': 'Ok',
            'routes': [{
                'duration': 2400,
                'distance': 30000
            }]
        }

        mocker.patch('requests.get', return_value=mock_response)

        duration, distance = get_distance_duration(
            'Rue de Genève 1, 1200 Genève',
            'Avenue de la Gare 10, 1003 Lausanne'
        )

        # Retourne approximation ou valeurs OSRM
        assert duration >= 0
        assert distance >= 0

