"""
Tests des utilitaires (timezone, helpers).
"""
from datetime import UTC, datetime

import pytest


def test_time_utils_import():
    """Les helpers time_utils s'importent correctement."""
    from shared.time_utils import now_local, to_utc_from_db, iso_utc_z
    
    assert callable(now_local)
    assert callable(to_utc_from_db)
    assert callable(iso_utc_z)


def test_now_local():
    """now_local() retourne un datetime naïf."""
    from shared.time_utils import now_local
    
    dt = now_local()
    assert isinstance(dt, datetime)
    # Naïf (pas de tzinfo)
    assert dt.tzinfo is None


def test_iso_utc_z():
    """iso_utc_z convertit datetime en ISO string avec Z."""
    from shared.time_utils import iso_utc_z
    
    dt = datetime(2025, 10, 15, 12, 30, 0, tzinfo=UTC)
    iso_str = iso_utc_z(dt)
    
    assert isinstance(iso_str, str)
    assert iso_str.endswith('Z')
    assert '2025-10-15' in iso_str


def test_logging_utils_import():
    """Logging utils module s'importe correctement."""
    import shared.logging_utils as logging_utils
    
    # Vérifier que le module existe
    assert logging_utils is not None

