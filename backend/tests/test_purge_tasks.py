"""✅ 3.3: Tests pour les tâches de purge RGPD."""

from datetime import UTC, datetime, timedelta

import pytest

from tasks.purge_tasks import (
    DEFAULT_RETENTION_DAYS,
    anonymize_old_user_data,
    purge_old_autonomous_actions,
    purge_old_bookings,
    purge_old_messages,
    purge_old_realtime_events,
    purge_old_task_failures,
)


class TestPurgeOldBookings:
    """Tests pour purge_old_bookings."""
    
    def test_purge_old_bookings_no_data(self, db):
        """Test purge quand aucune donnée ancienne."""
        result = purge_old_bookings(None)
        
        assert result["status"] == "success"
        assert result["model"] == "Booking"
        assert result["deleted_count"] == 0
        assert result["retention_days"] == DEFAULT_RETENTION_DAYS
    
    def test_purge_old_bookings_with_old_completed(self, db, factory_booking, sample_company):
        """Test purge avec bookings terminés anciens."""
        from models import BookingStatus
        
        # Créer booking terminé ancien (> 7 ans)
        old_date = datetime.now(UTC) - timedelta(days=DEFAULT_RETENTION_DAYS + 10)
        booking = factory_booking(
            company=sample_company,
            status=BookingStatus.COMPLETED,
        )
        booking.created_at = old_date
        db.session.commit()
        
        result = purge_old_bookings(None)
        
        assert result["status"] == "success"
        assert result["deleted_count"] == 1
        
        # Vérifier que le booking a été supprimé
        from models import Booking
        assert Booking.query.filter_by(id=booking.id).first() is None
    
    def test_purge_old_bookings_keeps_active(self, db, factory_booking, sample_company):
        """Test que les bookings actives ne sont pas purgées."""
        from models import BookingStatus
        
        # Créer booking actif ancien (ne doit pas être purgé)
        old_date = datetime.now(UTC) - timedelta(days=DEFAULT_RETENTION_DAYS + 10)
        booking = factory_booking(
            company=sample_company,
            status=BookingStatus.PENDING,
        )
        booking.created_at = old_date
        db.session.commit()
        
        result = purge_old_bookings(None)
        
        assert result["status"] == "success"
        assert result["deleted_count"] == 0
        
        # Vérifier que le booking existe encore
        from models import Booking
        assert Booking.query.filter_by(id=booking.id).first() is not None


class TestPurgeOldMessages:
    """Tests pour purge_old_messages."""
    
    def test_purge_old_messages_no_data(self, db):
        """Test purge quand aucune donnée ancienne."""
        result = purge_old_messages(None)
        
        assert result["status"] == "success"
        assert result["model"] == "Message"
        assert result["deleted_count"] == 0


class TestPurgeOldRealtimeEvents:
    """Tests pour purge_old_realtime_events."""
    
    def test_purge_old_realtime_events_no_data(self, db):
        """Test purge quand aucune donnée ancienne."""
        result = purge_old_realtime_events(None)
        
        assert result["status"] == "success"
        assert result["model"] == "RealtimeEvent"
        assert result["deleted_count"] == 0


class TestPurgeOldAutonomousActions:
    """Tests pour purge_old_autonomous_actions."""
    
    def test_purge_old_autonomous_actions_no_data(self, db):
        """Test purge quand aucune donnée ancienne."""
        result = purge_old_autonomous_actions(None)
        
        assert result["status"] == "success"
        assert result["model"] == "AutonomousAction"
        assert result["deleted_count"] == 0


class TestPurgeOldTaskFailures:
    """Tests pour purge_old_task_failures."""
    
    def test_purge_old_task_failures_no_data(self, db):
        """Test purge quand aucune donnée ancienne."""
        result = purge_old_task_failures(None)
        
        assert result["status"] == "success"
        assert result["model"] == "TaskFailure"
        assert result["deleted_count"] == 0


class TestAnonymizeOldUserData:
    """Tests pour anonymize_old_user_data."""
    
    def test_anonymize_old_user_data_no_data(self, db):
        """Test anonymisation quand aucune donnée ancienne."""
        result = anonymize_old_user_data(None)
        
        assert result["status"] == "success"
        assert result["model"] == "User"
        assert result["action"] == "anonymize"
        assert result["anonymized_count"] == 0
    
    def test_anonymize_old_user_data_with_old_user(self, db, sample_user):
        """Test anonymisation avec utilisateur ancien."""
        from models import UserRole
        
        # Modifier la date de création pour être > 7 ans
        old_date = datetime.now(UTC) - timedelta(days=DEFAULT_RETENTION_DAYS + 10)
        sample_user.created_at = old_date
        sample_user.role = UserRole.client  # Ne pas anonymiser admin
        sample_user.email = "old@example.com"
        sample_user.first_name = "John"
        sample_user.last_name = "Doe"
        db.session.commit()
        
        result = anonymize_old_user_data(None)
        
        assert result["status"] == "success"
        assert result["anonymized_count"] == 1
        
        # Vérifier que les données ont été anonymisées
        db.session.refresh(sample_user)
        assert sample_user.email == f"anonymized_{sample_user.id}@deleted.local"
        assert sample_user.username == f"anonymized_{sample_user.id}"
        assert sample_user.first_name == "Anonymized"
        assert sample_user.last_name == "User"
        assert sample_user.phone is None
        assert sample_user.address is None
    
    def test_anonymize_skips_admin(self, db, sample_user):
        """Test que les admins ne sont pas anonymisés."""
        from models import UserRole
        
        old_date = datetime.now(UTC) - timedelta(days=DEFAULT_RETENTION_DAYS + 10)
        sample_user.created_at = old_date
        sample_user.role = UserRole.admin
        sample_user.email = "admin@example.com"
        db.session.commit()
        
        result = anonymize_old_user_data(None)
        
        assert result["status"] == "success"
        assert result["anonymized_count"] == 0  # Admin skipped
        
        # Vérifier que l'email n'a pas changé
        db.session.refresh(sample_user)
        assert sample_user.email == "admin@example.com"

