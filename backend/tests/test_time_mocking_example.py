"""
Exemples d'utilisation des fixtures de mock du temps.

Ce fichier sert de documentation et d'exemples pour utiliser
les fixtures frozen_time et mock_now_local dans les tests.
"""

from datetime import timedelta

import pytest

from shared import time_utils


def test_frozen_time_example(frozen_time):
    """Exemple d'utilisation de frozen_time fixture.

    frozen_time utilise freezegun pour figer le temps à une date fixe.
    """
    # Le temps est maintenant figé à 2025-01-15 10:00:00
    current_time = time_utils.now_local()
    assert current_time.year == 2025
    assert current_time.month == 1
    assert current_time.day == 15
    assert current_time.hour == 10
    assert current_time.minute == 0

    # Avancer le temps de 1 heure
    frozen_time.tick(timedelta(hours=1))
    new_time = time_utils.now_local()
    assert new_time.hour == 11
    assert new_time.minute == 0

    # Avancer le temps de 30 minutes
    frozen_time.tick(timedelta(minutes=30))
    final_time = time_utils.now_local()
    assert final_time.hour == 11
    assert final_time.minute == 30


def test_mock_now_local_example(mock_now_local):
    """Exemple d'utilisation de mock_now_local fixture.

    mock_now_local mock uniquement now_local() sans affecter datetime.now().
    """
    # now_local() retourne maintenant une date fixe
    current_time = time_utils.now_local()
    assert current_time.year == 2025
    assert current_time.month == 1
    assert current_time.day == 15

    # datetime.now() n'est PAS affecté par mock_now_local
    from datetime import UTC, datetime

    real_now = datetime.now(UTC)
    # La date réelle peut être différente (selon quand le test s'exécute)
    # mais now_local() retourne toujours la date fixe
    assert current_time != real_now


def test_frozen_time_with_relative_dates(frozen_time):
    """Exemple d'utilisation de frozen_time avec dates relatives."""
    base_time = time_utils.now_local()

    # Créer des dates relatives au temps figé
    future_time = base_time + timedelta(days=1)
    past_time = base_time - timedelta(days=1)

    assert future_time > base_time
    assert past_time < base_time

    # Avancer le temps de 1 jour
    frozen_time.tick(timedelta(days=1))
    new_base_time = time_utils.now_local()

    # Les dates relatives sont toujours valides
    assert future_time == new_base_time
    assert past_time < new_base_time
