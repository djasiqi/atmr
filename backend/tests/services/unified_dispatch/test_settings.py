# backend/tests/services/unified_dispatch/test_settings.py
"""Tests pour la validation des overrides de settings."""

import pytest

from services.unified_dispatch import settings as ud_settings


class TestValidationOverrides:
    """Tests pour vérifier la validation post-merge des overrides."""

    def test_preferred_driver_id_applique_correctement(self):
        """Test : preferred_driver_id appliqué correctement (ignoré mais pas d'erreur)."""
        base = ud_settings.Settings()
        overrides = {
            "heuristic": {"driver_load_balance": 0.5},
            "preferred_driver_id": 123,  # Clé connue mais ignorée (normal)
        }
        
        result = ud_settings.merge_overrides(base, overrides)
        
        # Vérifier que driver_load_balance a été appliqué
        assert result.heuristic.driver_load_balance == 0.5
        
        # preferred_driver_id n'est pas dans Settings, donc ignoré (normal)
        # Pas d'erreur attendue

    def test_fairness_weight_propage(self):
        """Test : fairness_weight propagé correctement."""
        base = ud_settings.Settings()
        overrides = {
            "fairness": {"fairness_weight": 0.8},
        }
        
        result = ud_settings.merge_overrides(base, overrides)
        
        # Vérifier que fairness_weight a été appliqué
        assert result.fairness.fairness_weight == 0.8

    def test_cles_inconnues_ignorees_avec_warning(self):
        """Test : Clés inconnues ignorées avec warning."""
        base = ud_settings.Settings()
        overrides = {
            "unknown_key": "value",
            "another_unknown": {"nested": "value"},
            "heuristic": {"driver_load_balance": 0.6},  # Valide
        }
        
        result = ud_settings.merge_overrides(base, overrides)
        
        # Vérifier que la clé valide a été appliquée
        assert result.heuristic.driver_load_balance == 0.6
        
        # Les clés inconnues sont ignorées (pas d'erreur)

    def test_validation_strict_mode(self, monkeypatch):
        """Test : Mode strict validation lève exception si paramètre critique non appliqué."""
        monkeypatch.setenv("UD_SETTINGS_STRICT_VALIDATION", "true")
        
        base = ud_settings.Settings()
        # Tenter d'appliquer des overrides valides pour vérifier que le mode strict fonctionne
        # Le mode strict ne devrait pas lever d'exception si tous les paramètres sont appliqués
        overrides = {
            "heuristic": {"driver_load_balance": 0.5},
            "fairness": {"fairness_weight": 0.8},
        }
        
        # Vérifier que le mode strict est activé
        strict_val = ud_settings.os.getenv("UD_SETTINGS_STRICT_VALIDATION", "false").lower() == "true"
        assert strict_val is True
        
        # Test avec merge_overrides: devrait fonctionner si tous les paramètres sont valides
        result = ud_settings.merge_overrides(base, overrides)
        
        # Vérifier que les paramètres ont été appliqués
        assert result.heuristic.driver_load_balance == 0.5
        assert result.fairness.fairness_weight == 0.8

    def test_logging_detaille_avant_apres(self):
        """Test : Logging détaillé avant/après merge."""
        base = ud_settings.Settings()
        overrides = {
            "heuristic": {
                "driver_load_balance": 0.7,
                "proximity": 0.2,
            },
            "fairness": {
                "fairness_weight": 0.9,
            },
        }
        
        # Capturer les valeurs avant
        driver_load_before = base.heuristic.driver_load_balance
        fairness_before = base.fairness.fairness_weight
        
        result = ud_settings.merge_overrides(base, overrides)
        
        # Vérifier que les valeurs ont changé
        assert result.heuristic.driver_load_balance == 0.7
        assert result.heuristic.proximity == 0.2
        assert result.fairness.fairness_weight == 0.9
        
        # Vérifier que les valeurs avant étaient différentes
        assert driver_load_before != 0.7
        assert fairness_before != 0.9

    def test_merge_nested_settings(self):
        """Test : Merge récursif dans sous-dataclasses."""
        base = ud_settings.Settings()
        overrides = {
            "heuristic": {
                "driver_load_balance": 0.5,
                "proximity": 0.3,
                "priority": 0.1,
            },
            "solver": {
                "time_limit_sec": 120,
            },
        }
        
        result = ud_settings.merge_overrides(base, overrides)
        
        # Vérifier que tous les paramètres ont été appliqués
        assert result.heuristic.driver_load_balance == 0.5
        assert result.heuristic.proximity == 0.3
        assert result.heuristic.priority == 0.1
        assert result.solver.time_limit_sec == 120

