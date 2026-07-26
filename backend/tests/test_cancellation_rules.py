"""Tests unitaires pour les règles d'annulation standardisées."""

from datetime import UTC, datetime

import pytest

from application.bookings.cancellation_rules import (
    BILLABLE_REASONS,
    CANCELLATION_REASON_LABELS,
    compute_cancellation_fields,
    get_all_reason_codes,
    get_cancellation_display_label,
    is_cancellation_billable,
)


class TestIsCancellationBillable:
    """Tests pour is_cancellation_billable()."""

    @pytest.mark.parametrize("code", ["LAST_MINUTE", "NO_SHOW"])
    def test_billable_reasons(self, code: str) -> None:
        """Motifs facturables (legacy) retournent True."""
        assert is_cancellation_billable(code) is True
        assert is_cancellation_billable(code.lower()) is True

    def test_client_request_not_billable_without_context(self) -> None:
        """CLIENT_REQUEST seul ne suffit pas (dépend statut/paliers)."""
        assert is_cancellation_billable("CLIENT_REQUEST") is False

    @pytest.mark.parametrize(
        "code",
        ["COMPANY_ISSUE", "MAJOR_DELAY", "VEHICLE_ISSUE", "OTHER", "CLIENT_REQUEST"],
    )
    def test_non_billable_reasons(self, code: str) -> None:
        """Motifs non facturables retournent False."""
        assert is_cancellation_billable(code) is False

    def test_legacy_operator_cancelled_maps_to_non_billable(self) -> None:
        """OPERATOR_CANCELLED (mobile) → COMPANY_ISSUE → non facturé."""
        assert is_cancellation_billable("OPERATOR_CANCELLED") is False

    def test_none_or_empty_returns_false(self) -> None:
        """None ou vide → OTHER → non facturé."""
        assert is_cancellation_billable(None) is False
        assert is_cancellation_billable("") is False
        assert is_cancellation_billable("   ") is False


class TestGetCancellationDisplayLabel:
    """Tests pour get_cancellation_display_label()."""

    def test_known_codes_return_label(self) -> None:
        """Codes connus retournent le libellé attendu."""
        assert (
            get_cancellation_display_label("LAST_MINUTE")
            == "Annulation dernière minute"
        )
        assert (
            get_cancellation_display_label("NO_SHOW") == "Client ne s'est pas présenté"
        )
        assert get_cancellation_display_label("COMPANY_ISSUE") == "Problème entreprise"

    def test_other_with_text_returns_truncated(self) -> None:
        """OTHER + reason_text → libellé personnalisé tronqué à 80 caractères."""
        short = "Raison personnalisée"
        assert get_cancellation_display_label("OTHER", short) == f"Annulation – {short}"
        long_text = "A" * 100
        label = get_cancellation_display_label("OTHER", long_text)
        assert label.startswith("Annulation – ")
        assert len(label) <= 93  # "Annulation – " (13) + text[:80]

    def test_other_without_text_returns_default(self) -> None:
        """OTHER sans reason_text → libellé par défaut."""
        assert get_cancellation_display_label("OTHER") == "Autre raison"

    def test_unknown_code_returns_historique(self) -> None:
        """Code inconnu → Annulation (historique)."""
        assert get_cancellation_display_label("UNKNOWN") == "Annulation (historique)"
        assert get_cancellation_display_label(None) == "Annulation (historique)"


class TestComputeCancellationFields:
    """Tests pour compute_cancellation_fields()."""

    def test_returns_all_fields(self) -> None:
        """Retourne tous les champs attendus."""
        result = compute_cancellation_fields(
            reason_code="NO_SHOW",
            reason_text=None,
            cancelled_by_role="company",
        )
        assert "cancelled_at" in result
        assert "cancelled_by_role" in result
        assert "cancellation_reason_code" in result
        assert "cancellation_reason_text" in result
        assert "is_cancellation_billable" in result
        assert "cancellation_display_label" in result

    def test_billable_reason_sets_true(self) -> None:
        """Motif facturable → is_cancellation_billable=True."""
        result = compute_cancellation_fields(
            reason_code="LAST_MINUTE",
            reason_text=None,
            cancelled_by_role="company",
        )
        assert result["is_cancellation_billable"] is True
        assert result["cancellation_display_label"] == "Annulation dernière minute"

    def test_non_billable_reason_sets_false(self) -> None:
        """Motif non facturable → is_cancellation_billable=False."""
        result = compute_cancellation_fields(
            reason_code="COMPANY_ISSUE",
            reason_text=None,
            cancelled_by_role="driver",
        )
        assert result["is_cancellation_billable"] is False
        assert result["cancelled_by_role"] == "driver"

    def test_none_reason_code_returns_historique_label(self) -> None:
        """reason_code=None → Annulation (historique), code=OTHER."""
        result = compute_cancellation_fields(
            reason_code=None,
            reason_text=None,
            cancelled_by_role="company",
        )
        assert result["cancellation_reason_code"] == "OTHER"
        assert result["cancellation_display_label"] == "Annulation (historique)"
        assert result["is_cancellation_billable"] is False

    def test_legacy_operator_cancelled_mapped(self) -> None:
        """OPERATOR_CANCELLED → COMPANY_ISSUE."""
        result = compute_cancellation_fields(
            reason_code="OPERATOR_CANCELLED",
            reason_text=None,
            cancelled_by_role="company",
        )
        assert result["cancellation_reason_code"] == "COMPANY_ISSUE"
        assert result["is_cancellation_billable"] is False


class TestGetAllReasonCodes:
    """Tests pour get_all_reason_codes()."""

    def test_returns_seven_codes(self) -> None:
        """Retourne les 7 motifs."""
        codes = get_all_reason_codes()
        assert len(codes) == 7
        assert set(codes) == set(CANCELLATION_REASON_LABELS.keys())


class TestInvoiceCanceledEligible:
    """Étape 5A : facturer uniquement les annulations billables.

    canceled_eligible = (status == CANCELED and is_cancellation_billable is True).
    """

    def test_cancelled_company_issue_not_billable(self) -> None:
        """CANCELLED + COMPANY_ISSUE → non facturé (pas dans facture)."""
        assert is_cancellation_billable("COMPANY_ISSUE") is False

    def test_cancelled_no_show_billable(self) -> None:
        """CANCELLED + NO_SHOW → facturé."""
        assert is_cancellation_billable("NO_SHOW") is True

    def test_cancelled_legacy_none_not_billable(self) -> None:
        """CANCELLED + legacy (reason_code None) → non facturé."""
        assert is_cancellation_billable(None) is False
        assert is_cancellation_billable("") is False
