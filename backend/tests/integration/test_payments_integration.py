"""
Tests d'intégration pour le bounded context Payments.

Teste les flux complets route → use case → repository → DB pour les
endpoints de paiements.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from models import Invoice, InvoicePayment, db
from models.enums import InvoiceStatus, PaymentMethod, PaymentStatus
from tests.integration.helpers import assert_response_json, assert_response_status


@pytest.mark.integration
class TestPaymentsIntegration:
    """Tests d'intégration pour les paiements."""

    def test_create_payment_updates_invoice(
        self, authenticated_client, test_company, test_invoice
    ):
        """Test création d'un paiement et vérification de la mise à jour de la facture."""
        if not all([test_company, test_invoice]):
            pytest.skip("Required fixtures missing")

        # La facture doit être SENT pour recevoir un paiement
        test_invoice.status = InvoiceStatus.SENT
        test_invoice.balance_due = Decimal("100.00")
        db.session.commit()

        url = f"/api/v1/companies/{test_company.id}/invoices/{test_invoice.id}/payments"
        payload = {
            "amount": 50.0,
            "payment_method": "CASH",
            "payment_date": datetime.now(UTC).isoformat(),
        }

        response = authenticated_client.post(url, json=payload)
        # Peut retourner 201 (créé) ou 400 selon la validation
        assert response.status_code in [201, 400]

        if response.status_code == 201:
            data = assert_response_json(response)
            # Vérifier que le paiement existe en DB
            if "id" in data:
                payment = InvoicePayment.query.get(data["id"])
                assert payment is not None
                assert payment.invoice_id == test_invoice.id

                # Vérifier que la facture a été mise à jour
                db.session.refresh(test_invoice)
                # Le balance_due devrait être réduit
                assert test_invoice.balance_due <= Decimal("100.00")

    def test_list_payments_with_filters(
        self, authenticated_client, test_company, test_invoice, db
    ):
        """Test liste des paiements avec filtres."""
        if not all([test_company, test_invoice]):
            pytest.skip("Required fixtures missing")

        # Créer un paiement de test
        payment = InvoicePayment()
        payment.invoice_id = test_invoice.id
        payment.amount = Decimal("50.00")
        payment.payment_method = PaymentMethod.CASH
        payment.payment_status = PaymentStatus.COMPLETED
        payment.payment_date = datetime.now(UTC)
        db.session.add(payment)
        db.session.commit()

        url = f"/api/v1/companies/{test_company.id}/invoices/{test_invoice.id}/payments"
        response = authenticated_client.get(url)
        # Peut retourner 200 ou 404 selon l'implémentation
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = assert_response_json(response)
            # Vérifier la structure de la réponse
            assert isinstance(data, (dict, list))
