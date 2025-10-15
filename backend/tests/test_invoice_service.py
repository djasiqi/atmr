"""
Tests unitaires pour invoice_service.py

Teste les fonctionnalités principales :
- Génération de factures
- Calcul des montants
- Numérotation automatique
- Gestion des erreurs
"""
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from models import (
    Booking,
    BookingStatus,
    Client,
    ClientType,
    Company,
    CompanyBillingSettings,
    Invoice,
    InvoiceLineType,
    InvoiceStatus,
    User,
    db,
)

# Import du service et des models
from services.invoice_service import InvoiceService


@pytest.fixture
def app():
    """Créer une application Flask pour les tests."""
    from app import create_app
    app = create_app('testing')

    with app.app_context():
        # Créer les tables
        db.create_all()
        yield app
        # Nettoyer après les tests
        db.session.remove()
        db.drop_all()


@pytest.fixture
def company(app):
    """Créer une entreprise de test."""
    with app.app_context():
        company = Company(
            name="Test Transport SA",
            email="test@transport.ch",
            phone="+41123456789"
        )
        db.session.add(company)
        db.session.commit()
        return company


@pytest.fixture
def billing_settings(app, company):
    """Créer les paramètres de facturation."""
    with app.app_context():
        settings = CompanyBillingSettings(
            company_id=company.id,
            invoice_prefix="INV",
            invoice_start_number=1000,
            payment_terms_days=30,
            currency="CHF",
            vat_rate=7.7,
            vat_number="CHE-123.456.789"
        )
        db.session.add(settings)
        db.session.commit()
        return settings


@pytest.fixture
def client_user(app, company):
    """Créer un utilisateur client."""
    with app.app_context():
        user = User(
            email="patient@test.ch",
            first_name="Jean",
            last_name="Dupont",
            role="CLIENT"
        )
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()

        client = Client(
            company_id=company.id,
            user_id=user.id,
            client_type=ClientType.INDIVIDUAL,
            is_institution=False
        )
        db.session.add(client)
        db.session.commit()

        return client, user


@pytest.fixture
def institution_client(app, company):
    """Créer un client institution."""
    with app.app_context():
        user = User(
            email="hopital@test.ch",
            first_name="Hôpital",
            last_name="Cantonal",
            role="CLIENT"
        )
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()

        client = Client(
            company_id=company.id,
            user_id=user.id,
            client_type=ClientType.INSTITUTION,
            is_institution=True
        )
        db.session.add(client)
        db.session.commit()

        return client, user


@pytest.fixture
def completed_bookings(app, company, client_user):
    """Créer des réservations complétées pour facturation."""
    with app.app_context():
        client, user = client_user
        bookings = []

        # Créer 3 réservations en janvier 2025
        for i in range(3):
            booking = Booking(
                company_id=company.id,
                client_id=client.id,
                pickup_address=f"Rue Test {i+1}, Genève",
                dropoff_address=f"Avenue Test {i+1}, Genève",
                scheduled_time=datetime(2025, 1, 15 + i, 10, 0),
                status=BookingStatus.COMPLETED,
                amount=Decimal("50.00"),
                completed_at=datetime(2025, 1, 15 + i, 11, 0)
            )
            db.session.add(booking)
            bookings.append(booking)

        db.session.commit()
        return bookings


@pytest.fixture
def invoice_service(app):
    """Créer une instance du service."""
    with app.app_context():
        return InvoiceService()


# ============================================================
# Tests de génération de factures
# ============================================================

def test_generate_invoice_basic(app, invoice_service, company, client_user, completed_bookings, billing_settings):
    """Test de génération d'une facture simple."""
    with app.app_context():
        client, user = client_user

        # Générer la facture pour janvier 2025
        invoice = invoice_service.generate_invoice(
            company_id=company.id,
            client_id=client.id,
            period_year=2025,
            period_month=1
        )

        # Vérifications basiques
        assert invoice is not None
        assert invoice.company_id == company.id
        assert invoice.client_id == client.id
        assert invoice.status == InvoiceStatus.DRAFT
        assert invoice.currency == "CHF"

        # Vérifier le montant (3 réservations à 50 CHF)
        assert invoice.subtotal == Decimal("150.00")

        # Vérifier les lignes de facture
        assert len(invoice.lines) == 3
        for line in invoice.lines:
            assert line.line_type == InvoiceLineType.RESERVATION
            assert line.unit_price == Decimal("50.00")


def test_generate_invoice_with_specific_bookings(app, invoice_service, company, client_user, completed_bookings, billing_settings):
    """Test avec sélection manuelle de réservations."""
    with app.app_context():
        client, user = client_user

        # Générer facture pour seulement 2 réservations
        selected_ids = [completed_bookings[0].id, completed_bookings[2].id]

        invoice = invoice_service.generate_invoice(
            company_id=company.id,
            client_id=client.id,
            period_year=2025,
            period_month=1,
            reservation_ids=selected_ids
        )

        # Vérifications
        assert len(invoice.lines) == 2
        assert invoice.subtotal == Decimal("100.00")

        # Vérifier que les réservations sont marquées comme facturées
        for booking in completed_bookings[:3:2]:  # [0] et [2]
            db.session.refresh(booking)
            assert booking.invoice_line_id is not None


def test_generate_invoice_to_institution(app, invoice_service, company, client_user, institution_client, completed_bookings, billing_settings):
    """Test de facturation à une institution (tiers payant)."""
    with app.app_context():
        client, user = client_user
        institution, inst_user = institution_client

        # Générer facture adressée à l'institution
        invoice = invoice_service.generate_invoice(
            company_id=company.id,
            client_id=client.id,
            period_year=2025,
            period_month=1,
            bill_to_client_id=institution.id  # Facturer à l'institution
        )

        # Vérifications
        assert invoice.client_id == client.id  # Bénéficiaire
        assert invoice.bill_to_client_id == institution.id  # Payeur


def test_generate_invoice_number_increment(app, invoice_service, company, client_user, completed_bookings, billing_settings):
    """Test de l'incrémentation automatique du numéro de facture."""
    with app.app_context():
        client, user = client_user

        # Générer première facture
        invoice1 = invoice_service.generate_invoice(
            company_id=company.id,
            client_id=client.id,
            period_year=2025,
            period_month=1,
            reservation_ids=[completed_bookings[0].id]
        )

        # Créer une nouvelle réservation pour la deuxième facture
        booking2 = Booking(
            company_id=company.id,
            client_id=client.id,
            pickup_address="Test 2",
            dropoff_address="Test 2",
            scheduled_time=datetime(2025, 2, 1, 10, 0),
            status=BookingStatus.COMPLETED,
            amount=Decimal("60.00"),
            completed_at=datetime(2025, 2, 1, 11, 0)
        )
        db.session.add(booking2)
        db.session.commit()

        # Générer deuxième facture
        invoice2 = invoice_service.generate_invoice(
            company_id=company.id,
            client_id=client.id,
            period_year=2025,
            period_month=2,
            reservation_ids=[booking2.id]
        )

        # Vérifier l'incrémentation
        assert invoice1.number.startswith("INV")
        assert invoice2.number.startswith("INV")

        # Extraire les numéros et vérifier l'ordre
        num1 = int(invoice1.number.split('-')[-1])
        num2 = int(invoice2.number.split('-')[-1])
        assert num2 == num1 + 1


def test_generate_invoice_no_bookings(app, invoice_service, company, client_user, billing_settings):
    """Test d'erreur si aucune réservation trouvée."""
    with app.app_context():
        client, user = client_user

        # Essayer de générer facture sans réservations
        with pytest.raises(ValueError, match="Aucune réservation trouvée"):
            invoice_service.generate_invoice(
                company_id=company.id,
                client_id=client.id,
                period_year=2025,
                period_month=12  # Aucune réservation en décembre
            )


def test_generate_invoice_invalid_institution(app, invoice_service, company, client_user, completed_bookings, billing_settings):
    """Test d'erreur si l'institution n'existe pas."""
    with app.app_context():
        client, user = client_user

        # Essayer avec un ID institution invalide
        with pytest.raises(ValueError, match="Client payeur non trouvé"):
            invoice_service.generate_invoice(
                company_id=company.id,
                client_id=client.id,
                period_year=2025,
                period_month=1,
                bill_to_client_id=99999  # ID inexistant
            )


def test_generate_invoice_with_vat(app, invoice_service, company, client_user, completed_bookings, billing_settings):
    """Test du calcul de la TVA."""
    with app.app_context():
        client, user = client_user

        invoice = invoice_service.generate_invoice(
            company_id=company.id,
            client_id=client.id,
            period_year=2025,
            period_month=1
        )

        # Vérifier le calcul de TVA (7.7% de 150 CHF = 11.55 CHF)
        expected_vat = Decimal("150.00") * Decimal("0.077")
        assert invoice.vat_amount == expected_vat.quantize(Decimal("0.01"))

        # Total = Subtotal + TVA
        expected_total = Decimal("150.00") + expected_vat
        assert invoice.total == expected_total.quantize(Decimal("0.01"))


# ============================================================
# Tests de méthodes internes
# ============================================================

def test_get_billing_settings(app, invoice_service, company, billing_settings):
    """Test de récupération des paramètres de facturation."""
    with app.app_context():
        settings = invoice_service._get_billing_settings(company.id)

        assert settings is not None
        assert settings.invoice_prefix == "INV"
        assert settings.payment_terms_days == 30
        assert settings.vat_rate == 7.7


def test_get_billing_settings_create_if_missing(app, invoice_service, company):
    """Test de création automatique des paramètres si absents."""
    with app.app_context():
        # Pas de settings créés pour cette entreprise
        settings = invoice_service._get_billing_settings(company.id)

        # Devrait créer des settings par défaut
        assert settings is not None
        assert settings.company_id == company.id
        assert settings.invoice_prefix == "INV"  # Valeur par défaut


def test_get_reservations_for_period(app, invoice_service, company, client_user, completed_bookings):
    """Test de récupération des réservations d'une période."""
    with app.app_context():
        client, user = client_user

        # Récupérer les réservations de janvier 2025
        reservations = invoice_service._get_reservations_for_period(
            company_id=company.id,
            client_id=client.id,
            year=2025,
            month=1
        )

        assert len(reservations) == 3

        # Vérifier qu'elles sont bien complétées et non facturées
        for res in reservations:
            assert res.status == BookingStatus.COMPLETED
            assert res.invoice_line_id is None


# ============================================================
# Tests d'edge cases
# ============================================================

def test_generate_invoice_already_invoiced_bookings(app, invoice_service, company, client_user, completed_bookings, billing_settings):
    """Test que les réservations déjà facturées ne sont pas re-facturées."""
    with app.app_context():
        client, user = client_user

        # Première facture
        invoice1 = invoice_service.generate_invoice(
            company_id=company.id,
            client_id=client.id,
            period_year=2025,
            period_month=1
        )

        assert len(invoice1.lines) == 3

        # Essayer de re-générer une facture pour la même période
        with pytest.raises(ValueError, match="Aucune réservation trouvée"):
            invoice_service.generate_invoice(
                company_id=company.id,
                client_id=client.id,
                period_year=2025,
                period_month=1
            )


def test_generate_invoice_mixed_status_bookings(app, invoice_service, company, client_user, billing_settings):
    """Test avec des réservations de statuts mixtes."""
    with app.app_context():
        client, user = client_user

        # Créer des réservations avec différents statuts
        booking_completed = Booking(
            company_id=company.id,
            client_id=client.id,
            pickup_address="Test 1",
            dropoff_address="Test 1",
            scheduled_time=datetime(2025, 3, 1, 10, 0),
            status=BookingStatus.COMPLETED,
            amount=Decimal("50.00")
        )

        booking_pending = Booking(
            company_id=company.id,
            client_id=client.id,
            pickup_address="Test 2",
            dropoff_address="Test 2",
            scheduled_time=datetime(2025, 3, 2, 10, 0),
            status=BookingStatus.PENDING,  # Pas complété
            amount=Decimal("50.00")
        )

        booking_cancelled = Booking(
            company_id=company.id,
            client_id=client.id,
            pickup_address="Test 3",
            dropoff_address="Test 3",
            scheduled_time=datetime(2025, 3, 3, 10, 0),
            status=BookingStatus.CANCELLED,  # Annulé
            amount=Decimal("50.00")
        )

        db.session.add_all([booking_completed, booking_pending, booking_cancelled])
        db.session.commit()

        # Générer facture
        invoice = invoice_service.generate_invoice(
            company_id=company.id,
            client_id=client.id,
            period_year=2025,
            period_month=3
        )

        # Seule la réservation COMPLETED devrait être facturée
        assert len(invoice.lines) == 1
        assert invoice.subtotal == Decimal("50.00")


# ============================================================
# Tests de performance / edge cases
# ============================================================

def test_generate_invoice_large_number_of_bookings(app, invoice_service, company, client_user, billing_settings):
    """Test avec un grand nombre de réservations (performance)."""
    with app.app_context():
        client, user = client_user

        # Créer 100 réservations
        bookings = []
        for i in range(100):
            booking = Booking(
                company_id=company.id,
                client_id=client.id,
                pickup_address=f"Pickup {i}",
                dropoff_address=f"Dropoff {i}",
                scheduled_time=datetime(2025, 4, 1, 10, 0) + timedelta(hours=i),
                status=BookingStatus.COMPLETED,
                amount=Decimal("25.50")
            )
            bookings.append(booking)

        db.session.add_all(bookings)
        db.session.commit()

        # Générer la facture
        import time
        start = time.time()
        invoice = invoice_service.generate_invoice(
            company_id=company.id,
            client_id=client.id,
            period_year=2025,
            period_month=4
        )
        duration = time.time() - start

        # Vérifications
        assert len(invoice.lines) == 100
        assert invoice.subtotal == Decimal("2550.00")

        # Performance : devrait prendre moins de 5 secondes
        assert duration < 5.0, f"Génération trop lente : {duration:.2f}s"


# ============================================================
# Tests avec db_context.py (migration future)
# ============================================================

@pytest.mark.skip(reason="À activer après migration vers db_context.py")
def test_generate_invoice_with_db_context(app, invoice_service, company, client_user, completed_bookings, billing_settings):
    """Test après migration vers db_context.py."""
    from services.db_context import db_transaction

    with app.app_context():
        client, user = client_user

        # La génération de facture devrait utiliser db_transaction
        with db_transaction():
            invoice = invoice_service.generate_invoice(
                company_id=company.id,
                client_id=client.id,
                period_year=2025,
                period_month=1
            )

        # Vérifier que le commit a bien eu lieu
        db.session.expire_all()
        assert Invoice.query.filter_by(id=invoice.id).first() is not None

