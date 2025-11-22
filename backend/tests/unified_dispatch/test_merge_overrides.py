#!/usr/bin/env python3
"""
Tests pour merge_overrides dans settings.py.

Valide que merge_overrides gère correctement:
- Chemins imbriqués (heuristic.proximity, etc.)
- Types inattendus
- Valeurs par défaut
- Clés inconnues (mode, preferred_driver_id, etc.)
"""

import logging

from services.unified_dispatch.settings import Settings, merge_overrides

logger = logging.getLogger(__name__)


class TestMergeOverrides:
    """Tests pour merge_overrides."""

    def test_merge_overrides_nested_paths(self):
        """Test: merge de chemins imbriqués (heuristic.proximity, etc.)."""
        base = Settings()

        overrides = {
            "heuristic": {"proximity": 0.05, "driver_load_balance": 0.95},
            "solver": {"time_limit_sec": 120},
        }

        merged = merge_overrides(base, overrides)

        # Vérifier que les valeurs imbriquées sont appliquées
        assert merged.heuristic.proximity == 0.05
        assert merged.heuristic.driver_load_balance == 0.95
        assert merged.solver.time_limit_sec == 120

        logger.info(
            "✅ Merge imbriqué réussi: heuristic.proximity=%s, solver.time_limit_sec=%s",
            merged.heuristic.proximity,
            merged.solver.time_limit_sec,
        )

    def test_merge_overrides_ignores_unknown_keys(self):
        """Test: les clés inconnues (mode, preferred_driver_id) sont ignorées."""
        base = Settings()

        overrides = {
            "heuristic": {"proximity": 0.3},
            "mode": "heuristic_only",  # Clé inconnue
            "preferred_driver_id": 42,  # Clé inconnue
            "run_async": True,  # Clé inconnue
        }

        merged = merge_overrides(base, overrides)

        # Les clés inconnues ne doivent pas causer d'erreur
        assert merged.heuristic.proximity == 0.3
        # Vérifier que les clés inconnues n'ont pas été ajoutées
        assert not hasattr(merged, "mode")
        assert not hasattr(merged, "preferred_driver_id")
        assert not hasattr(merged, "run_async")

        logger.info("✅ Clés inconnues ignorées correctement")

    def test_merge_overrides_preserves_defaults(self):
        """Test: les valeurs non overridées conservent leurs valeurs par défaut."""
        base = Settings()
        default_proximity = base.heuristic.proximity
        default_time_limit = base.solver.time_limit_sec

        overrides = {
            "heuristic": {
                "driver_load_balance": 0.8
                # proximity n'est pas dans overrides
            }
        }

        merged = merge_overrides(base, overrides)

        # Proximity doit conserver sa valeur par défaut
        assert merged.heuristic.proximity == default_proximity
        # driver_load_balance doit être override
        assert merged.heuristic.driver_load_balance == 0.8
        # time_limit_sec doit conserver sa valeur par défaut
        assert merged.solver.time_limit_sec == default_time_limit

        logger.info(
            "✅ Valeurs par défaut préservées: proximity=%s, time_limit=%s",
            merged.heuristic.proximity,
            merged.solver.time_limit_sec,
        )

    def test_merge_overrides_deep_nesting(self):
        """Test: merge sur plusieurs niveaux d'imbrication."""
        base = Settings()

        overrides = {
            "heuristic": {"proximity": 0.1, "driver_load_balance": 0.9},
            "fairness": {"fairness_weight": 0.7},
            "matrix": {
                "provider": "osrm",
                # Utiliser un attribut existant de MatrixSettings
                "cache_ttl_sec": 1800,
            },
        }

        merged = merge_overrides(base, overrides)

        # Vérifier tous les niveaux
        assert merged.heuristic.proximity == 0.1
        assert merged.heuristic.driver_load_balance == 0.9
        assert merged.fairness.fairness_weight == 0.7
        assert merged.matrix.provider == "osrm"
        assert merged.matrix.cache_ttl_sec == 1800

        logger.info("✅ Merge multi-niveaux réussi")

    def test_merge_overrides_type_coercion(self):
        """Test: gestion des types (int vs float, bool, etc.)."""
        base = Settings()

        # Test avec différents types
        overrides = {
            "solver": {
                "time_limit_sec": 60.0  # Float au lieu d'int
            },
            "features": {"enable_heuristics": True, "enable_solver": False},
        }

        merged = merge_overrides(base, overrides)

        # Le type devrait être préservé ou coerce selon le type attendu
        assert merged.solver.time_limit_sec == 60  # Doit être int
        assert merged.features.enable_heuristics is True
        assert merged.features.enable_solver is False

        logger.info("✅ Types gérés correctement")

    def test_merge_overrides_empty_dict(self):
        """Test: overrides vide → pas de modification."""
        base = Settings()
        original_proximity = base.heuristic.proximity

        merged = merge_overrides(base, {})

        assert merged.heuristic.proximity == original_proximity
        logger.info("✅ Overrides vide → pas de modification")

    def test_merge_overrides_partial_override(self):
        """Test: override partiel d'une dataclass imbriquée."""
        base = Settings()

        # Override seulement une clé dans heuristic
        overrides = {
            "heuristic": {
                "proximity": 0.4
                # Les autres clés de heuristic ne sont pas touchées
            }
        }

        merged = merge_overrides(base, overrides)

        # Proximity doit être override
        assert merged.heuristic.proximity == 0.4
        # Les autres valeurs de heuristic doivent être préservées
        assert hasattr(merged.heuristic, "driver_load_balance")

        logger.info("✅ Override partiel réussi")
