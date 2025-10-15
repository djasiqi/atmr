"""
Tests unitaires des modèles (sans DB complète).
"""
import pytest
from models.enums import BookingStatus, UserRole, PaymentStatus


def test_booking_status_values():
    """BookingStatus contient toutes les valeurs attendues."""
    assert hasattr(BookingStatus, 'PENDING')
    assert hasattr(BookingStatus, 'ACCEPTED')
    assert hasattr(BookingStatus, 'COMPLETED')
    assert hasattr(BookingStatus, 'CANCELED')
    
    assert BookingStatus.PENDING.value == 'PENDING'
    assert BookingStatus.COMPLETED.value == 'COMPLETED'


def test_user_role_values():
    """UserRole contient tous les rôles."""
    assert hasattr(UserRole, 'ADMIN')
    assert hasattr(UserRole, 'CLIENT')
    assert hasattr(UserRole, 'DRIVER')
    assert hasattr(UserRole, 'COMPANY')
    
    roles = [UserRole.ADMIN, UserRole.CLIENT, UserRole.DRIVER, UserRole.COMPANY]
    assert len(roles) == 4


def test_payment_status_values():
    """PaymentStatus contient les statuts de paiement."""
    assert hasattr(PaymentStatus, 'PENDING')
    assert hasattr(PaymentStatus, 'COMPLETED')
    assert hasattr(PaymentStatus, 'FAILED')
    
    statuses = [PaymentStatus.PENDING, PaymentStatus.COMPLETED, PaymentStatus.FAILED]
    assert len(statuses) == 3


def test_booking_status_choices():
    """BookingStatus.choices() retourne toutes les valeurs."""
    choices = BookingStatus.choices()
    assert isinstance(choices, list)
    assert 'PENDING' in choices
    assert 'COMPLETED' in choices

