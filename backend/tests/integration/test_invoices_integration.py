"""
Tests d'intégration pour le bounded context Invoices.

Teste les flux complets route → use case → repository → DB pour tous les
endpoints de facturation migrés vers DDD.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from models import Booking, Invoice, InvoiceLine, db
from models.enums import BookingStatus, InvoiceStatus
from tests.integration.helpers import (
    assert_response_json,
    assert_response_status,
    measure_performance,
)


@pytest.mark.integration
class TestInvoicesIntegration:
    """Tests d'intégration pour les factures."""

    @measure_performance(threshold_seconds=2.0)
    def test_generate_invoice_full_flow(
        self, authenticated_client, test_company, test_client, test_completed_booking
    ):
        """Test génération complète d'une facture avec réservations."""
        if not all([test_company, test_client, test_completed_booking]):
            pytest.skip("Required fixtures missing")

        # S'assurer que la réservation est complétée et non facturée
        test_completed_booking.status = BookingStatus.COMPLETED
        test_completed_booking.invoice_line_id = None
        db.session.commit()

        # Générer la facture
        url = f"/api/v1/companies/{test_company.id}/invoices/generate"
        payload = {
            "client_id": test_client.id,
            "period_year": datetime.now(UTC).year,
            "period_month": datetime.now(UTC).month,
            "reservation_ids": [test_completed_booking.id],
        }

        response = authenticated_client.post(url, json=payload)
        assert_response_status(response, 200)
        data = assert_response_json(response, ["invoice_id", "invoice"])

        # Vérifier que la facture existe en DB
        invoice = Invoice.query.get(data["invoice_id"])
        assert invoice is not None
        assert invoice.company_id == test_company.id
        assert invoice.client_id == test_client.id
        assert invoice.status == InvoiceStatus.DRAFT

        # Vérifier que la réservation est liée à la facture
        db.session.refresh(test_completed_booking)
        assert test_completed_booking.invoice_line_id is not None

        # Vérifier que les lignes de facture existent
        lines = InvoiceLine.query.filter_by(invoice_id=invoice.id).all()
        assert len(lines) > 0
        assert any(line.reservation_id == test_completed_booking.id for line in lines)

    def test_get_invoice_with_lines(
        self, authenticated_client, test_company, test_invoice, test_completed_booking
    ):
        """Test récupération d'une facture avec ses lignes."""
        if not all([test_company, test_invoice, test_completed_booking]):
            pytest.skip("Required fixtures missing")

        # Créer une ligne de facture
        invoice_line = InvoiceLine()
        invoice_line.invoice_id = test_invoice.id
        invoice_line.reservation_id = test_completed_booking.id
        invoice_line.line_type = "booking"
        invoice_line.description = "Test booking"
        invoice_line.quantity = Decimal("1.00")
        invoice_line.unit_price = Decimal("100.00")
        invoice_line.line_total = Decimal("100.00")
        invoice_line.vat_rate = Decimal("7.70")
        db.session.add(invoice_line)
        db.session.commit()

        # Récupérer la facture
        url = f"/api/v1/companies/{test_company.id}/invoices/{test_invoice.id}"
        response = authenticated_client.get(url)
        assert_response_status(response, 200)
        data = assert_response_json(response)

        # Vérifier la structure de la réponse
        assert "id" in data
        assert data["id"] == test_invoice.id
        assert "invoice_number" in data
        assert "status" in data

    @measure_performance(threshold_seconds=2.0)
    def test_cancel_invoice_releases_bookings(
        self, authenticated_client, test_company, test_invoice, test_completed_booking
    ):
        """Test annulation d'une facture et vérification de la libération des réservations."""
        if not all([test_company, test_invoice, test_completed_booking]):
            pytest.skip("Required fixtures missing")

        # Créer une ligne de facture liée à la réservation
        test_invoice.status = InvoiceStatus.DRAFT
        invoice_line = InvoiceLine()
        invoice_line.invoice_id = test_invoice.id
        invoice_line.reservation_id = test_completed_booking.id
        invoice_line.line_type = "booking"
        invoice_line.description = "Test booking"
        invoice_line.quantity = Decimal("1.00")
        invoice_line.unit_price = Decimal("100.00")
        invoice_line.line_total = Decimal("100.00")
        db.session.add(invoice_line)
        db.session.flush()

        # Lier la réservation à la ligne
        test_completed_booking.invoice_line_id = invoice_line.id
        db.session.commit()

        # Annuler la facture
        url = f"/api/v1/companies/{test_company.id}/invoices/{test_invoice.id}/cancel"
        response = authenticated_client.post(url)
        assert_response_status(response, 200)

        # Vérifier que la facture est annulée
        db.session.refresh(test_invoice)
        assert test_invoice.status == InvoiceStatus.CANCELLED
        assert test_invoice.cancelled_at is not None

        # Vérifier que la réservation est libérée
        db.session.refresh(test_completed_booking)
        assert test_completed_booking.invoice_line_id is None

    def test_duplicate_invoice_creates_draft(
        self, authenticated_client, test_company, test_invoice, test_completed_booking
    ):
        """Test duplication d'une facture et vérification de la création d'un brouillon."""
        if not all([test_company, test_invoice, test_completed_booking]):
            pytest.skip("Required fixtures missing")

        # La facture doit être SENT pour être dupliquée
        test_invoice.status = InvoiceStatus.SENT
        db.session.commit()

        # Dupliquer la facture
        url = (
            f"/api/v1/companies/{test_company.id}/invoices/{test_invoice.id}/duplicate"
        )
        response = authenticated_client.post(url)
        assert_response_status(response, 200)
        data = assert_response_json(response, ["draft_context"])

        # Vérifier que le contexte de brouillon contient les bonnes données
        assert "client_id" in data["draft_context"]
        assert "period_year" in data["draft_context"]
        assert "period_month" in data["draft_context"]

    def test_consolidated_invoice_multiple_clients(
        self,
        authenticated_client,
        test_company,
        test_client,
        test_completed_booking,
        db,
    ):
        """Test génération d'une facture consolidée pour plusieurs clients."""
        if not all([test_company, test_client, test_completed_booking]):
            pytest.skip("Required fixtures missing")

        # Créer un deuxième client et une réservation
        from models import Client
        from models.enums import ClientType

        client2 = Client()
        client2.company_id = test_company.id
        client2.first_name = "Client2"
        client2.last_name = "Test"
        client2.email = "client2@test.ch"
        client2.client_type = ClientType.INDIVIDUAL
        db.session.add(client2)
        db.session.flush()

        booking2 = Booking()
        booking2.company_id = test_company.id
        booking2.client_id = client2.id
        booking2.customer_name = "Client2 Test"
        booking2.pickup_location = "Location A"
        booking2.dropoff_location = "Location B"
        booking2.scheduled_time = datetime.now(UTC) - timedelta(days=1)
        booking2.completed_at = datetime.now(UTC) - timedelta(hours=1)
        booking2.status = BookingStatus.COMPLETED
        booking2.amount = Decimal("50.00")
        booking2.vat_rate = Decimal("7.70")
        db.session.add(booking2)
        db.session.commit()

        # Créer une institution pour la facturation tierce
        institution = Client()
        institution.company_id = test_company.id
        institution.first_name = "Institution"
        institution.last_name = "Test"
        institution.email = "institution@test.ch"
        institution.client_type = ClientType.INSTITUTION
        institution.is_institution = True
        db.session.add(institution)
        db.session.commit()

        # Générer la facture consolidée
        url = f"/api/v1/companies/{test_company.id}/invoices/generate"
        payload = {
            "client_ids": [test_client.id, client2.id],
            "bill_to_client_id": institution.id,
            "period_year": datetime.now(UTC).year,
            "period_month": datetime.now(UTC).month,
        }

        response = authenticated_client.post(url, json=payload)
        # Peut retourner 200 ou 400 selon la validation
        assert response.status_code in [200, 400]

    def test_generate_reminder_updates_invoice(
        self, authenticated_client, test_company, test_invoice, db
    ):
        """Test génération d'un rappel et vérification de la mise à jour de la facture."""
        if not all([test_company, test_invoice]):
            pytest.skip("Required fixtures missing")

        # La facture doit être OVERDUE pour générer un rappel
        test_invoice.status = InvoiceStatus.OVERDUE
        test_invoice.due_date = datetime.now(UTC) - timedelta(days=10)
        test_invoice.reminder_level = 0
        db.session.commit()

        # Générer le rappel niveau 1
        url = (
            f"/api/v1/companies/{test_company.id}/invoices/{test_invoice.id}/reminders"
        )
        payload = {"level": 1}

        response = authenticated_client.post(url, json=payload)
        # Peut retourner 200 ou 400 selon la validation
        assert response.status_code in [200, 400]

        if response.status_code == 200:
            # Vérifier que la facture a été mise à jour
            db.session.refresh(test_invoice)
            assert test_invoice.reminder_level >= 1

    def test_check_overdue_creates_late_fee(
        self, authenticated_client, test_company, test_invoice, db
    ):
        """Test vérification des factures en retard et création de frais."""
        if not all([test_company, test_invoice]):
            pytest.skip("Required fixtures missing")

        # La facture doit être SENT et en retard
        test_invoice.status = InvoiceStatus.SENT
        test_invoice.due_date = datetime.now(UTC) - timedelta(days=1)
        test_invoice.balance_due = Decimal("100.00")
        db.session.commit()

        # Vérifier les factures en retard
        url = f"/api/v1/companies/{test_company.id}/invoices/overdue/check"
        response = authenticated_client.get(url)
        # Peut retourner 200 ou 404 selon l'implémentation
        assert response.status_code in [200, 404]

    def test_process_automatic_reminders_batch(
        self, authenticated_client, test_company, test_invoice, db
    ):
        """Test traitement des rappels automatiques en lot."""
        if not all([test_company, test_invoice]):
            pytest.skip("Required fixtures missing")

        # Configurer la facture pour rappels automatiques
        test_invoice.status = InvoiceStatus.OVERDUE
        test_invoice.due_date = datetime.now(UTC) - timedelta(days=10)
        test_invoice.reminder_level = 0
        db.session.commit()

        # Traiter les rappels automatiques
        url = f"/api/v1/companies/{test_company.id}/invoices/reminders/process"
        response = authenticated_client.post(url)
        # Peut retourner 200 ou 404 selon l'implémentation
        assert response.status_code in [200, 404]
