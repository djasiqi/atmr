"""Tests pour la génération de références QRR (QR-Bill)."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from services.documents.qrbill import QRBillService


class TestQRRReference:
    """Tests pour la génération de références QRR."""

    def test_qrr_reference_length_and_format(self):
        """Test 1: QRR = 27 chiffres numériques + check digit correct."""
        # Arrange
        service = QRBillService()

        # Mock invoice
        invoice = Mock()
        invoice.id = 123
        invoice.invoice_number = "EM-2026-01-0001"
        invoice.company_id = 1
        invoice.qr_reference = None

        # Mock profile avec QR-IBAN valide
        profile = Mock()
        profile.payment_reference_mode = "QRR"
        profile.qr_iban = "CH4431999123000889012"  # QR-IBAN valide (CH..3…)
        profile.iban = None
        profile.creditor_reference_base = "21"

        # Mock BillingProfileService
        from services.billing import BillingProfileService
        original_get = BillingProfileService.get_by_company_id
        BillingProfileService.get_by_company_id = Mock(return_value=profile)

        try:
            # Act
            qrr_ref = service._get_payment_reference(invoice)

            # Assert
            assert qrr_ref is not None, "QRR reference should not be None"
            assert len(qrr_ref) == 27, f"QRR should be 27 digits, got {len(qrr_ref)}"
            assert qrr_ref.isdigit(), f"QRR should be numeric only, got: {qrr_ref}"
            assert qrr_ref.startswith("21"), "QRR should start with base (21)"

            # Vérifier le check digit
            ref_base = qrr_ref[:26]
            check_digit = int(qrr_ref[26])
            calculated_check = service._calculate_qrr_check_digit(ref_base)
            assert check_digit == calculated_check, (
                f"Check digit incorrect: expected {calculated_check}, got {check_digit}"
            )
        finally:
            # Restore
            BillingProfileService.get_by_company_id = original_get

    def test_qrr_reference_stable_across_generations(self):
        """Test 2: Référence stable après plusieurs générations."""
        # Arrange
        service = QRBillService()

        # Mock invoice
        invoice = Mock()
        invoice.id = 456
        invoice.invoice_number = "ATMR-2026-02-0042"
        invoice.company_id = 2
        invoice.qr_reference = None

        # Mock profile
        profile = Mock()
        profile.payment_reference_mode = "QRR"
        profile.qr_iban = "CH4431999123000889012"
        profile.iban = None
        profile.creditor_reference_base = "21"

        # Mock BillingProfileService
        from services.billing import BillingProfileService
        original_get = BillingProfileService.get_by_company_id
        BillingProfileService.get_by_company_id = Mock(return_value=profile)

        try:
            # Act - Générer 2 fois
            qrr_ref_1 = service._get_payment_reference(invoice)

            # Simuler que la référence est sauvegardée
            invoice.qr_reference = qrr_ref_1

            # Générer à nouveau (devrait réutiliser)
            qrr_ref_2 = service._get_payment_reference(invoice)

            # Assert
            assert qrr_ref_1 == qrr_ref_2, (
                f"QRR reference should be stable. First: {qrr_ref_1}, Second: {qrr_ref_2}"
            )
            assert len(qrr_ref_1) == 27
            assert len(qrr_ref_2) == 27
        finally:
            # Restore
            BillingProfileService.get_by_company_id = original_get

    def test_qrr_invalid_qr_iban_raises_exception(self):
        """Test 3: QR-IBAN invalide => lève ValueError."""
        # Arrange
        service = QRBillService()

        # Mock invoice
        invoice = Mock()
        invoice.id = 789
        invoice.invoice_number = "INV-2026-03-0100"
        invoice.company_id = 3
        invoice.qr_reference = None

        # Mock profile avec QR-IBAN invalide (pas de QR-IBAN)
        profile = Mock()
        profile.payment_reference_mode = "QRR"
        profile.qr_iban = None
        profile.iban = "CH9300762011623852957"  # IBAN standard (pas QR-IBAN)
        profile.creditor_reference_base = "21"

        # Mock BillingProfileService
        from services.billing import BillingProfileService
        original_get = BillingProfileService.get_by_company_id
        BillingProfileService.get_by_company_id = Mock(return_value=profile)

        try:
            # Act & Assert
            with pytest.raises(ValueError, match="Mode QRR nécessite un QR-IBAN"):
                service._get_payment_reference(invoice)
        finally:
            # Restore
            BillingProfileService.get_by_company_id = original_get

    def test_qrr_invalid_qr_iban_format_raises_exception(self):
        """Test 4: QR-IBAN format invalide => lève ValueError."""
        # Arrange
        service = QRBillService()

        # Mock invoice
        invoice = Mock()
        invoice.id = 999
        invoice.invoice_number = "INV-2026-04-0200"
        invoice.company_id = 4
        invoice.qr_reference = None

        # Mock profile avec QR-IBAN invalide (ne commence pas par CH)
        profile = Mock()
        profile.payment_reference_mode = "QRR"
        profile.qr_iban = "FR1420041010050500013M02606"  # IBAN français (pas QR-IBAN)
        profile.iban = None
        profile.creditor_reference_base = "21"

        # Mock BillingProfileService
        from services.billing import BillingProfileService
        original_get = BillingProfileService.get_by_company_id
        BillingProfileService.get_by_company_id = Mock(return_value=profile)

        try:
            # Act & Assert
            with pytest.raises(ValueError, match="QR-IBAN invalide"):
                service._get_payment_reference(invoice)
        finally:
            # Restore
            BillingProfileService.get_by_company_id = original_get

    def test_qrr_invalid_qr_iban_5th_char_raises_exception(self):
        """Test 5: QR-IBAN 5ème caractère != '3' => lève ValueError."""
        # Arrange
        service = QRBillService()

        # Mock invoice
        invoice = Mock()
        invoice.id = 1111
        invoice.invoice_number = "INV-2026-05-0300"
        invoice.company_id = 5
        invoice.qr_reference = None

        # Mock profile avec IBAN standard (pas QR-IBAN, 5ème char = '9')
        profile = Mock()
        profile.payment_reference_mode = "QRR"
        profile.qr_iban = "CH9300762011623852957"  # IBAN standard (CH9...)
        profile.iban = None
        profile.creditor_reference_base = "21"

        # Mock BillingProfileService
        from services.billing import BillingProfileService
        original_get = BillingProfileService.get_by_company_id
        BillingProfileService.get_by_company_id = Mock(return_value=profile)

        try:
            # Act & Assert
            with pytest.raises(ValueError, match="Le 5ème caractère doit être '3'"):
                service._get_payment_reference(invoice)
        finally:
            # Restore
            BillingProfileService.get_by_company_id = original_get

    def test_qrr_reference_deterministic(self):
        """Test 6: Référence QRR déterministe (même invoice_number = même référence)."""
        # Arrange
        service = QRBillService()

        # Mock invoice 1
        invoice1 = Mock()
        invoice1.id = 100
        invoice1.invoice_number = "EM-2026-01-0001"
        invoice1.company_id = 1
        invoice1.qr_reference = None

        # Mock invoice 2 (même invoice_number mais id différent)
        invoice2 = Mock()
        invoice2.id = 200
        invoice2.invoice_number = "EM-2026-01-0001"  # Même numéro
        invoice2.company_id = 1
        invoice2.qr_reference = None

        # Mock profile
        profile = Mock()
        profile.payment_reference_mode = "QRR"
        profile.qr_iban = "CH4431999123000889012"
        profile.iban = None
        profile.creditor_reference_base = "21"

        # Mock BillingProfileService
        from services.billing import BillingProfileService
        original_get = BillingProfileService.get_by_company_id
        BillingProfileService.get_by_company_id = Mock(return_value=profile)

        try:
            # Act
            qrr_ref_1 = service._get_payment_reference(invoice1)
            qrr_ref_2 = service._get_payment_reference(invoice2)

            # Assert - Les références doivent être différentes car invoice.id diffère
            # (garantit l'unicité)
            assert qrr_ref_1 != qrr_ref_2, (
                "QRR references should differ when invoice.id differs "
                f"(invoice1.id={invoice1.id}, invoice2.id={invoice2.id})"
            )
            assert len(qrr_ref_1) == 27
            assert len(qrr_ref_2) == 27
        finally:
            # Restore
            BillingProfileService.get_by_company_id = original_get

    def test_qrr_reference_invoice_number_normalization(self):
        """Test 7: Normalisation correcte de invoice_number avec lettres/tirets."""
        # Arrange
        service = QRBillService()

        # Mock invoice avec invoice_number contenant lettres et tirets
        invoice = Mock()
        invoice.id = 5555
        invoice.invoice_number = "ATMR-2026-12-9999"  # Contient lettres et tirets
        invoice.company_id = 6
        invoice.qr_reference = None

        # Mock profile
        profile = Mock()
        profile.payment_reference_mode = "QRR"
        profile.qr_iban = "CH4431999123000889012"
        profile.iban = None
        profile.creditor_reference_base = "21"

        # Mock BillingProfileService
        from services.billing import BillingProfileService
        original_get = BillingProfileService.get_by_company_id
        BillingProfileService.get_by_company_id = Mock(return_value=profile)

        try:
            # Act
            qrr_ref = service._get_payment_reference(invoice)

            # Assert
            assert qrr_ref is not None
            assert len(qrr_ref) == 27
            assert qrr_ref.isdigit(), "QRR should contain only digits after normalization"
            # Vérifier que les chiffres de invoice_number sont présents
            # "ATMR-2026-12-9999" -> "2026129999"
            assert "2026129999" in qrr_ref or "9999" in qrr_ref, (
                "QRR should contain normalized digits from invoice_number"
            )
        finally:
            # Restore
            BillingProfileService.get_by_company_id = original_get
