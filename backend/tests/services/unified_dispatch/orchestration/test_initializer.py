# backend/tests/services/unified_dispatch/orchestration/test_initializer.py
"""Tests unitaires pour DispatchInitializer.

Tests pour :
- find_and_validate_company : Recherche et validation de Company
- configure_settings : Configuration des settings avec overrides
"""

from __future__ import annotations  # noqa: I001

import pytest
from unittest.mock import MagicMock, patch

from tests.factories import CompanyFactory
from services.unified_dispatch.core.exceptions import CompanyNotFoundError
from services.unified_dispatch.orchestration.initializer import DispatchInitializer


class TestFindAndValidateCompany:
    """Tests pour la méthode find_and_validate_company."""

    def test_company_found_successfully(self, db):
        """Test : Company trouvée avec succès."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        initializer = DispatchInitializer()
        result_company, error_result = initializer.find_and_validate_company(
            company_id=company.id,
            for_date="2025-01-14",
            mode="auto",
            raise_on_company_not_found=False,
        )

        assert result_company is not None
        assert result_company.id == company.id
        assert error_result is None

    def test_company_not_found_without_raise(self, db):
        """Test : Company introuvable sans lever d'exception."""
        initializer = DispatchInitializer()
        result_company, error_result = initializer.find_and_validate_company(
            company_id=999_999,
            for_date="2025-01-14",
            mode="auto",
            raise_on_company_not_found=False,
        )

        assert result_company is None
        assert error_result is not None
        assert error_result["meta"]["reason"] == "company_not_found"
        assert "introuvable" in error_result["meta"]["error"].lower()

    def test_company_not_found_with_raise(self, db):
        """Test : Company introuvable avec levée d'exception."""
        initializer = DispatchInitializer()
        with pytest.raises(CompanyNotFoundError) as exc_info:
            initializer.find_and_validate_company(
                company_id=999_999,
                for_date="2025-01-14",
                mode="auto",
                raise_on_company_not_found=True,
            )

        assert exc_info.value.company_id == 999_999

    def test_company_found_after_flush(self, db):
        """Test : Company trouvée après flush."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.flush()  # Flush sans commit

        initializer = DispatchInitializer()
        result_company, error_result = initializer.find_and_validate_company(
            company_id=company.id,
            for_date="2025-01-14",
            mode="auto",
            raise_on_company_not_found=False,
        )

        assert result_company is not None
        assert result_company.id == company.id
        assert error_result is None

    @patch(
        "services.unified_dispatch.orchestration.initializer.track_company_not_found"
    )
    def test_tracks_metric_when_company_not_found(self, mock_track, db):
        """Test : Vérifie que la métrique est trackée quand Company introuvable."""
        initializer = DispatchInitializer()
        initializer.find_and_validate_company(
            company_id=999_999,
            for_date="2025-01-14",
            mode="auto",
            raise_on_company_not_found=False,
        )

        mock_track.assert_called_once_with(999_999, dispatch_run_id=None)


class TestConfigureSettings:
    """Tests pour la méthode configure_settings."""

    def test_basic_configuration_with_default_settings(self, db):
        """Test : Configuration basique avec settings par défaut."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        initializer = DispatchInitializer()
        settings, mode, allow_emg, is_fast_mode = initializer.configure_settings(
            company=company,
            custom_settings=None,
            overrides=None,
            allow_emergency=None,
            mode="auto",
        )

        assert settings is not None
        assert mode == "auto"
        assert isinstance(allow_emg, bool)
        assert is_fast_mode is False  # noqa: F841

    def test_configuration_with_custom_settings(self, db):
        """Test : Configuration avec custom_settings."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        custom_settings = MagicMock()
        custom_settings.features = MagicMock()
        custom_settings.emergency = MagicMock()
        custom_settings.emergency.allow_emergency_drivers = True

        initializer = DispatchInitializer()
        settings, mode, allow_emg, is_fast_mode = initializer.configure_settings(
            company=company,
            custom_settings=custom_settings,
            overrides=None,
            allow_emergency=None,
            mode="auto",
        )

        assert settings == custom_settings
        assert mode == "auto"
        assert allow_emg is True  # noqa: F841
        assert is_fast_mode is False  # noqa: F841

    def test_configuration_with_overrides(self, db):
        """Test : Configuration avec overrides."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        overrides = {
            "heuristic": {
                "driver_load_balance": 0.8,
                "proximity": 0.5,
            },
            "fairness": {
                "fairness_weight": 0.3,
            },
        }

        initializer = DispatchInitializer()
        settings, mode, _allow_emg, is_fast_mode = initializer.configure_settings(
            company=company,
            custom_settings=None,
            overrides=overrides,
            allow_emergency=None,
            mode="auto",
        )

        assert settings is not None
        assert mode == "auto"
        assert is_fast_mode is False

    def test_fast_mode_detection_and_activation(self, db):
        """Test : Détection et activation du mode rapide."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        overrides = {"fast_mode": True}

        initializer = DispatchInitializer()
        settings, mode, _allow_emg, is_fast_mode = initializer.configure_settings(
            company=company,
            custom_settings=None,
            overrides=overrides,
            allow_emergency=None,
            mode="auto",
        )

        assert mode == "heuristic_only"
        assert is_fast_mode is True
        assert settings.features.enable_solver is False
        assert settings.features.enable_rl is False
        assert settings.features.enable_parallel_heuristics is True
        assert settings.solver.time_limit_sec == 10

    def test_allow_emergency_override(self, db):
        """Test : Gestion de allow_emergency."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        initializer = DispatchInitializer()
        _settings, _mode, allow_emg, _is_fast_mode = initializer.configure_settings(
            company=company,
            custom_settings=None,
            overrides=None,
            allow_emergency=True,
            mode="auto",
        )

        assert allow_emg is True

    def test_allow_emergency_false(self, db):
        """Test : allow_emergency=False."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        initializer = DispatchInitializer()
        _settings, _mode, allow_emg, _is_fast_mode = initializer.configure_settings(
            company=company,
            custom_settings=None,
            overrides=None,
            allow_emergency=False,
            mode="auto",
        )

        assert allow_emg is False

    def test_invalid_overrides_handled_gracefully(self, db):
        """Test : Gestion gracieuse des overrides invalides."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        # Overrides avec structure invalide
        overrides = {"invalid_key": "invalid_value", "heuristic": "not_a_dict"}

        initializer = DispatchInitializer()
        # Ne doit pas lever d'exception, mais logger un warning
        settings, mode, _allow_emg, _is_fast_mode = initializer.configure_settings(
            company=company,
            custom_settings=None,
            overrides=overrides,
            allow_emergency=None,
            mode="auto",
        )

        assert settings is not None
        assert mode == "auto"

    def test_fast_mode_with_other_overrides(self, db):
        """Test : Mode rapide combiné avec d'autres overrides."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        overrides = {
            "fast_mode": True,
            "heuristic": {
                "driver_load_balance": 0.7,
            },
        }

        initializer = DispatchInitializer()
        settings, mode, _allow_emg, is_fast_mode = initializer.configure_settings(
            company=company,
            custom_settings=None,
            overrides=overrides,
            allow_emergency=None,
            mode="auto",
        )

        assert mode == "heuristic_only"
        assert is_fast_mode is True
        assert settings.features.enable_solver is False
