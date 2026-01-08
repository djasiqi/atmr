# backend/tests/services/unified_dispatch/orchestration/
# test_assignment_applier_wrapper.py
"""Tests unitaires pour AssignmentApplierWrapper.

Tests pour :
- apply : Application des assignations avec/sans perf_collector
- Gestion de company None
- Vérification des appels à AssignmentApplier
"""

from __future__ import annotations  # noqa: I001

import pytest
from flask import Flask
from unittest.mock import MagicMock, patch

from tests.factories import CompanyFactory
from services.unified_dispatch.orchestration.assignment_applier_wrapper import (
    AssignmentApplierWrapper,
)


@pytest.fixture(autouse=True)
def _app_context(app: Flask):
    """Assure que tous les tests s'exécutent dans un app context."""
    with app.app_context():
        yield


class TestApply:
    """Tests pour la méthode apply."""

    @patch(
        "services.unified_dispatch.orchestration.assignment_applier_wrapper.AssignmentApplier"
    )
    def test_apply_with_perf_collector(self, mock_applier_class):
        """Test : Application avec perf_collector utilise time_step."""
        wrapper = AssignmentApplierWrapper()

        mock_applier_instance = MagicMock()
        mock_applier_class.return_value = mock_applier_instance

        company = CompanyFactory()
        assignments = [MagicMock()]
        perf_collector = MagicMock()
        perf_collector.time_step.return_value.__enter__ = MagicMock()
        perf_collector.time_step.return_value.__exit__ = MagicMock()

        wrapper.apply(
            company=company,
            final_assignments=assignments,
            dispatch_run_id=42,
            perf_collector=perf_collector,
        )

        # Vérifier que time_step("persistence") est appelé
        perf_collector.time_step.assert_called_once_with("persistence")
        # Vérifier que apply_and_emit est appelé
        mock_applier_instance.apply_and_emit.assert_called_once_with(
            company, assignments, 42
        )

    @patch(
        "services.unified_dispatch.orchestration.assignment_applier_wrapper.AssignmentApplier"
    )
    def test_apply_without_perf_collector(self, mock_applier_class):
        """Test : Application sans perf_collector appelle directement apply_and_emit."""
        wrapper = AssignmentApplierWrapper()

        mock_applier_instance = MagicMock()
        mock_applier_class.return_value = mock_applier_instance

        company = CompanyFactory()
        assignments = [MagicMock()]

        wrapper.apply(
            company=company,
            final_assignments=assignments,
            dispatch_run_id=42,
            perf_collector=None,
        )

        # Vérifier que apply_and_emit est appelé directement
        mock_applier_instance.apply_and_emit.assert_called_once_with(
            company, assignments, 42
        )

    @patch(
        "services.unified_dispatch.orchestration.assignment_applier_wrapper.AssignmentApplier"
    )
    def test_apply_with_company_none(self, mock_applier_class):
        """Test : Retourne immédiatement si company est None."""
        wrapper = AssignmentApplierWrapper()

        mock_applier_instance = MagicMock()
        mock_applier_class.return_value = mock_applier_instance

        assignments = [MagicMock()]

        wrapper.apply(
            company=None,
            final_assignments=assignments,
            dispatch_run_id=42,
            perf_collector=None,
        )

        # Vérifier que apply_and_emit n'est PAS appelé
        mock_applier_instance.apply_and_emit.assert_not_called()

    @patch(
        "services.unified_dispatch.orchestration.assignment_applier_wrapper.AssignmentApplier"
    )
    def test_apply_with_dispatch_run_id_none(self, mock_applier_class):
        """Test : Gère dispatch_run_id None."""
        wrapper = AssignmentApplierWrapper()

        mock_applier_instance = MagicMock()
        mock_applier_class.return_value = mock_applier_instance

        company = CompanyFactory()
        assignments = [MagicMock()]

        wrapper.apply(
            company=company,
            final_assignments=assignments,
            dispatch_run_id=None,
            perf_collector=None,
        )

        # Vérifier que apply_and_emit est appelé avec dispatch_run_id=None
        mock_applier_instance.apply_and_emit.assert_called_once_with(
            company, assignments, None
        )

    @patch(
        "services.unified_dispatch.orchestration.assignment_applier_wrapper.AssignmentApplier"
    )
    def test_apply_creates_new_applier_instance(self, mock_applier_class):
        """Test : Crée une nouvelle instance AssignmentApplier à chaque appel."""
        wrapper = AssignmentApplierWrapper()

        mock_applier_instance = MagicMock()
        mock_applier_class.return_value = mock_applier_instance

        company = CompanyFactory()
        assignments = [MagicMock()]

        # Premier appel
        wrapper.apply(
            company=company,
            final_assignments=assignments,
            dispatch_run_id=1,
            perf_collector=None,
        )

        # Deuxième appel
        wrapper.apply(
            company=company,
            final_assignments=assignments,
            dispatch_run_id=2,
            perf_collector=None,
        )

        # Vérifier que AssignmentApplier est instancié deux fois
        assert mock_applier_class.call_count == 2
        # Vérifier que apply_and_emit est appelé deux fois
        assert mock_applier_instance.apply_and_emit.call_count == 2

    @patch(
        "services.unified_dispatch.orchestration.assignment_applier_wrapper.AssignmentApplier"
    )
    def test_apply_perf_collector_context_manager(self, mock_applier_class):
        """Test : Vérifie que le context manager perf_collector est utilisé
        correctement."""
        wrapper = AssignmentApplierWrapper()

        mock_applier_instance = MagicMock()
        mock_applier_class.return_value = mock_applier_instance

        company = CompanyFactory()
        assignments = [MagicMock()]
        perf_collector = MagicMock()
        context_manager = MagicMock()
        context_manager.__enter__ = MagicMock(return_value=None)
        context_manager.__exit__ = MagicMock(return_value=None)
        perf_collector.time_step.return_value = context_manager

        wrapper.apply(
            company=company,
            final_assignments=assignments,
            dispatch_run_id=42,
            perf_collector=perf_collector,
        )

        # Vérifier que le context manager est entré et sorti
        context_manager.__enter__.assert_called_once()
        context_manager.__exit__.assert_called_once()
