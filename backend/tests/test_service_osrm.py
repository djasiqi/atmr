"""
Tests du service OSRM (routing et distance)
"""
import pytest
from unittest.mock import Mock, patch
from services.osrm_client import OSRMClient, route_info, matrix_table


class TestOSRMClient:
    """Tests du client OSRM"""
    
    @pytest.fixture
    def osrm_client(self):
        """Instance OSRMClient pour tests"""
        return OSRMClient(base_url='http://localhost:5000')
    
    def test_route_info_success(self, osrm_client, mocker):
        """Test récupération route OSRM"""
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
        
        duration, distance = osrm_client.route(
            start_lat=46.2044,
            start_lon=6.1432,
            end_lat=46.5197,
            end_lon=6.6323
        )
        
        assert duration == 1800
        assert distance == 25000
    
    def test_route_info_fallback_haversine(self, osrm_client, mocker):
        """Test fallback haversine si OSRM échoue"""
        # Mock OSRM failure
        mocker.patch('requests.get', side_effect=Exception('OSRM timeout'))
        
        duration, distance = osrm_client.route(
            start_lat=46.2044,
            start_lon=6.1432,
            end_lat=46.5197,
            end_lon=6.6323
        )
        
        # Fallback haversine retourne des valeurs approximatives
        assert duration is not None
        assert distance is not None
        assert distance > 0
    
    def test_matrix_table_success(self, osrm_client, mocker):
        """Test calcul matrice distances"""
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
        
        result = osrm_client.matrix(coords)
        
        assert result is not None
        assert 'durations' in result
        assert 'distances' in result
        assert len(result['durations']) == 3
    
    def test_route_info_cache_redis(self, osrm_client, mocker):
        """Test cache Redis pour routes"""
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
        
        duration, distance = osrm_client.route(
            start_lat=46.2044,
            start_lon=6.1432,
            end_lat=46.5197,
            end_lon=6.6323
        )
        
        assert duration == 1500
        assert distance == 20000


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

