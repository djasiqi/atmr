"""
Tests des routes bookings
"""
from datetime import datetime, timedelta, timezone
from flask_jwt_extended import create_access_token
from models import Booking, BookingStatus, db


class TestCreateBooking:
    """Tests création bookings"""
    
    def test_create_booking_success(self, test_client, client_user, app, mocker):
        """Test création booking standard"""
        client, user = client_user
        
        # Mock services externes
        mocker.patch('services.maps.get_distance_duration', return_value=(1800, 15000))
        mocker.patch('services.maps.geocode_address', return_value=(46.2044, 6.1432))
        mocker.patch('services.unified_dispatch.queue.trigger_on_booking_change')
        
        with app.app_context():
            token = create_access_token(identity=user.public_id)
        
        scheduled = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        
        response = test_client.post(
            f'/api/bookings/clients/{user.public_id}/bookings',
            json={
                'customer_name': 'Jean Dupont',
                'pickup_location': 'Rue de Genève 1, 1200 Genève',
                'dropoff_location': 'Hôpital Cantonal, 1211 Genève',
                'scheduled_time': scheduled,
                'amount': 45.50,
                'medical_facility': 'HUG',
                'doctor_name': 'Dr. Martin',
                'is_round_trip': False
            },
            headers={'Authorization': f'Bearer {token}'}
        )
        
        assert response.status_code == 201
        data = response.get_json()
        
        assert data['customer_name'] == 'Jean Dupont'
        assert data['status'] == 'PENDING'
        assert 'id' in data
    
    def test_create_booking_round_trip(self, test_client, client_user, app, mocker):
        """Test création booking aller-retour"""
        client, user = client_user
        
        mocker.patch('services.maps.get_distance_duration', return_value=(1800, 15000))
        mocker.patch('services.maps.geocode_address', return_value=(46.2044, 6.1432))
        mocker.patch('services.unified_dispatch.queue.trigger_on_booking_change')
        
        with app.app_context():
            token = create_access_token(identity=user.public_id)
        
        response = test_client.post(
            f'/api/bookings/clients/{user.public_id}/bookings',
            json={
                'customer_name': 'Marie Martin',
                'pickup_location': 'A',
                'dropoff_location': 'B',
                'scheduled_time': (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat(),
                'amount': 50.0,
                'is_round_trip': True
            },
            headers={'Authorization': f'Bearer {token}'}
        )
        
        assert response.status_code == 201
        data = response.get_json()
        
        # Vérifier booking aller
        assert data['is_round_trip'] is True
        assert data['is_return'] is False
    
    def test_create_booking_unauthorized(self, test_client, client_user):
        """Test création sans token → 401"""
        client, user = client_user
        
        response = test_client.post(
            f'/api/bookings/clients/{user.public_id}/bookings',
            json={
                'customer_name': 'Test',
                'pickup_location': 'A',
                'dropoff_location': 'B',
                'scheduled_time': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                'amount': 40.0
            }
            # Pas de header Authorization
        )
        
        assert response.status_code == 401


class TestAssignDriver:
    """Tests assignation chauffeur"""
    
    def test_assign_driver_success(self, test_client, company_user, client_user, driver_user, app, mocker):
        """Test assignation chauffeur à booking"""
        company, company_user_obj = company_user
        client, client_user_obj = client_user
        driver, driver_user_obj = driver_user
        
        mocker.patch('services.unified_dispatch.queue.trigger_on_booking_change')
        
        # Créer booking
        with app.app_context():
            booking = Booking()
            booking.customer_name = 'Test Patient'  # type: ignore
            booking.pickup_location = 'A'  # type: ignore
            booking.dropoff_location = 'B'  # type: ignore
            booking.scheduled_time = datetime.now(timezone.utc) + timedelta(hours=2)  # type: ignore
            booking.amount = 50.0  # type: ignore
            booking.status = BookingStatus.PENDING  # type: ignore
            booking.user_id = client_user_obj.id  # type: ignore
            booking.client_id = client.id  # type: ignore
            booking.company_id = company.id  # type: ignore
            db.session.add(booking)
            db.session.commit()
            booking_id = booking.id
        
        # Login as company
        with app.app_context():
            token = create_access_token(identity=company_user_obj.public_id)
        
        # Assign driver
        response = test_client.post(
            f'/api/bookings/{booking_id}/assign',
            json={'driver_id': driver.id},
            headers={'Authorization': f'Bearer {token}'}
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['driver_id'] == driver.id
        assert data['status'] == 'ASSIGNED'


class TestCancelBooking:
    """Tests annulation booking"""
    
    def test_cancel_booking_success(self, test_client, client_user, app, mocker):
        """Test annulation booking autorisée"""
        client, user = client_user
        
        mocker.patch('services.unified_dispatch.queue.trigger_on_booking_change')
        
        # Créer booking
        with app.app_context():
            booking = Booking()
            booking.customer_name = 'Test Cancel'  # type: ignore
            booking.pickup_location = 'A'  # type: ignore
            booking.dropoff_location = 'B'  # type: ignore
            booking.scheduled_time = datetime.now(timezone.utc) + timedelta(hours=2)  # type: ignore
            booking.amount = 40.0  # type: ignore
            booking.status = BookingStatus.PENDING  # type: ignore
            booking.user_id = user.id  # type: ignore
            booking.client_id = client.id  # type: ignore
            db.session.add(booking)
            db.session.commit()
            booking_id = booking.id
        
        with app.app_context():
            token = create_access_token(identity=user.public_id)
        
        response = test_client.delete(
            f'/api/bookings/{booking_id}',
            headers={'Authorization': f'Bearer {token}'}
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] in ['CANCELLED', 'CANCELED']

