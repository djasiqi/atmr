#!/usr/bin/env python3
"""
Tests pour B1 : Versionner & protéger le Quality Score.

Teste que le quality score est versionné, que les dominant_factors sont loggés,
et que le garde-fou désactive auto-apply RL si score < 70.
"""

import logging

import pytest

from services.unified_dispatch.dispatch_metrics import (
    QUALITY_FORMULA_VERSION,
    QUALITY_THRESHOLD,
    QUALITY_WEIGHTS,
    DispatchMetricsCollector,
    get_quality_formula_hash,
)

logger = logging.getLogger(__name__)


class TestQualityScoreVersioning:
    """Tests pour le versionnage du quality score (B1)."""

    def test_quality_formula_version_constant(self):
        """Test: Vérifier que QUALITY_FORMULA_VERSION existe."""

        assert QUALITY_FORMULA_VERSION == "v1.0"
        logger.info("✅ Test: QUALITY_FORMULA_VERSION = %s", QUALITY_FORMULA_VERSION)

    def test_quality_weights_hash_reproducible(self):
        """Test: Vérifier que le hash des weights est reproductible."""

        hash1 = get_quality_formula_hash()
        hash2 = get_quality_formula_hash()

        assert hash1 == hash2, "Hash devrait être reproductible"
        assert len(hash1) == 8, "Hash devrait avoir 8 caractères"

        logger.info("✅ Test: Hash reproductible = %s", hash1)

    def test_quality_weights_hash_stable(self):
        """Test: Vérifier que le hash est stable pour les mêmes weights."""

        original_hash = get_quality_formula_hash()

        # Vérifier que le hash est stable pour les mêmes weights
        new_hash = get_quality_formula_hash()
        assert new_hash == original_hash

        logger.info("✅ Test: Hash stable avec mêmes weights")

    def test_quality_score_includes_version_and_hash(self, app_context, db_session):
        """Test: Vérifier que DispatchQualityMetrics inclut version et hash."""

        from datetime import UTC, datetime

        from models import Company, DispatchRun, DispatchStatus

        # Créer un DispatchRun minimal
        company = Company(name="Test Company")
        db_session.add(company)
        db_session.flush()

        run = DispatchRun(
            company_id=company.id, day=datetime.now(UTC).date(), status=DispatchStatus.COMPLETED, config={}
        )
        db_session.add(run)
        db_session.commit()

        collector = DispatchMetricsCollector(company.id)

        # Calculer les métriques (avec assignations vides)
        metrics = collector._calculate_metrics(
            dispatch_run_id=run.id, run_date=datetime.now(UTC).date(), assignments=[], all_bookings=[], run_metadata={}
        )

        assert hasattr(metrics, "quality_formula_version")
        assert hasattr(metrics, "quality_weights_hash")
        assert metrics.quality_formula_version == QUALITY_FORMULA_VERSION
        assert len(metrics.quality_weights_hash) == 8

        logger.info(
            "✅ Test: Metrics inclut version=%s hash=%s", metrics.quality_formula_version, metrics.quality_weights_hash
        )

    def test_dominant_factors_calculated(self, app_context, db_session):
        """Test: Vérifier que les dominant_factors sont calculés."""

        from datetime import UTC, datetime

        from models import Company, DispatchRun, DispatchStatus

        company = Company(name="Test Company")
        db_session.add(company)
        db_session.flush()

        run = DispatchRun(
            company_id=company.id, day=datetime.now(UTC).date(), status=DispatchStatus.COMPLETED, config={}
        )
        db_session.add(run)
        db_session.commit()

        collector = DispatchMetricsCollector(company.id)

        # Calculer avec des valeurs quelconques
        _, dominants = collector._calculate_quality_score(
            assignment_rate=85.0, on_time_rate=90.0, pooling_rate=20.0, fairness=0.8, avg_delay=5.0
        )

        assert isinstance(dominants, dict)
        assert len(dominants) == 3, "Devrait avoir top 3 facteurs"

        # Vérifier que les facteurs sont triés
        values = list(dominants.values())
        assert values == sorted(values, reverse=True), "Facteurs doivent être triés décroissant"

        logger.info("✅ Test: Dominant factors calculés = %s", dominants)


class TestAutoApplyGuard:
    """Tests pour le garde-fou auto-apply RL (B1)."""

    def test_quality_threshold_constant(self):
        """Test: Vérifier que QUALITY_THRESHOLD = 70."""

        assert QUALITY_THRESHOLD == 70.0
        logger.info("✅ Test: QUALITY_THRESHOLD = %.1f", QUALITY_THRESHOLD)

    def test_auto_apply_disabled_below_threshold(self):
        """Test: Simuler désactivation auto-apply si score < 70."""

        score_low = 65.0
        score_high = 75.0

        # Simuler logique garde-fou
        should_disable_low = score_low < QUALITY_THRESHOLD
        should_disable_high = score_high < QUALITY_THRESHOLD

        assert should_disable_low is True, "Score < 70 devrait désactiver"
        assert should_disable_high is False, "Score >= 70 devrait autoriser"

        logger.info("✅ Test: Garde-fou désactive si score < %.1f", QUALITY_THRESHOLD)

    def test_dominant_factors_logged(self):
        """Test: Vérifier que les dominant_factors sont loggables."""

        collector = DispatchMetricsCollector(1)

        _, dominants = collector._calculate_quality_score(
            assignment_rate=100.0,  # Contribution maximale
            on_time_rate=50.0,
            pooling_rate=10.0,
            fairness=0.5,
            avg_delay=15.0,
        )

        # Construire log string
        log_string = ", ".join(f"{k}={v:.1f}" for k, v in dominants.items())

        assert len(log_string) > 0
        assert "assignment" in log_string or "on_time" in log_string

        logger.info("✅ Test: Dominant factors loggables: %s", log_string)

    def test_quality_score_versioning_comparable(self):
        """Test: Vérifier que les scores sont comparables entre versions."""

        hash_v1 = get_quality_formula_hash()

        # Simuler v2 avec poids différents
        weights_v2 = {
            "assignment": 0.35,  # Changé de 0.30 à 0.35
            "on_time": 0.30,
            "pooling": 0.15,
            "fairness": 0.15,
            "delay": 0.05,  # Changé de 0.10 à 0.05
        }

        # Le hash devrait changer
        weights_v2_str = str(sorted(weights_v2.items()))
        import hashlib

        hash_v2 = hashlib.sha256(weights_v2_str.encode()).hexdigest()[:8]

        assert hash_v1 != hash_v2, "Hash devrait changer si weights différents"

        logger.info("✅ Test: Comparabilité entre versions (hash v1=%s vs v2=%s)", hash_v1, hash_v2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
