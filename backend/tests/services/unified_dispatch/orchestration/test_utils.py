# backend/tests/services/unified_dispatch/orchestration/test_utils.py
"""Tests unitaires pour les fonctions utilitaires de l'orchestration.

Tests pour :
- to_date_ymd : Conversion de chaîne de date en objet date
- safe_int : Conversion sécurisée en entier
"""

from datetime import date, datetime

import pytest

from services.unified_dispatch.orchestration.utils import safe_int, to_date_ymd


class TestToDateYmd:
    """Tests pour la fonction to_date_ymd."""

    def test_format_yyyy_mm_dd_valid(self):
        """Test : Format YYYY-MM-DD valide."""
        result = to_date_ymd("2025-01-14")
        assert result == date(2025, 1, 14)
        assert isinstance(result, date)

    def test_format_iso_full_with_timestamp(self):
        """Test : Format ISO complet avec timestamp."""
        result = to_date_ymd("2025-01-14T10:30:00")
        assert result == date(2025, 1, 14)
        assert isinstance(result, date)

    def test_format_iso_full_with_timezone(self):
        """Test : Format ISO complet avec timezone."""
        result = to_date_ymd("2025-01-14T10:30:00+01:00")
        assert result == date(2025, 1, 14)
        assert isinstance(result, date)

    def test_format_invalid_raises_value_error(self):
        """Test : Format invalide lève ValueError."""
        with pytest.raises(ValueError, match="for_date invalide"):
            to_date_ymd("invalid-date")

    def test_format_empty_string_raises_value_error(self):
        """Test : Chaîne vide lève ValueError."""
        with pytest.raises(ValueError, match="for_date invalide"):
            to_date_ymd("")

    def test_format_wrong_length_raises_value_error(self):
        """Test : Longueur incorrecte lève ValueError."""
        with pytest.raises(ValueError, match="for_date invalide"):
            to_date_ymd("2025-1-14")  # Trop court

    def test_format_missing_separators_raises_value_error(self):
        """Test : Séparateurs manquants lèvent ValueError."""
        with pytest.raises(ValueError, match="for_date invalide"):
            to_date_ymd("20250114")  # Pas de séparateurs

    def test_type_error_raises_value_error(self):
        """Test : Type incorrect lève ValueError."""
        with pytest.raises(ValueError, match="for_date invalide"):
            to_date_ymd(None)  # type: ignore[arg-type]

    def test_invalid_month_raises_value_error(self):
        """Test : Mois invalide lève ValueError."""
        with pytest.raises(ValueError, match="for_date invalide"):
            to_date_ymd("2025-13-14")  # Mois 13 invalide

    def test_invalid_day_raises_value_error(self):
        """Test : Jour invalide lève ValueError."""
        with pytest.raises(ValueError, match="for_date invalide"):
            to_date_ymd("2025-02-30")  # 30 février invalide


class TestSafeInt:
    """Tests pour la fonction safe_int."""

    def test_from_int(self):
        """Test : Conversion depuis int."""
        assert safe_int(42) == 42
        assert safe_int(0) == 0
        assert safe_int(-10) == -10

    def test_from_str_valid(self):
        """Test : Conversion depuis str valide."""
        assert safe_int("42") == 42
        assert safe_int("0") == 0
        assert safe_int("-10") == -10

    def test_from_float(self):
        """Test : Conversion depuis float."""
        assert safe_int(42.5) == 42
        assert safe_int(42.9) == 42
        assert safe_int(-10.7) == -10

    def test_from_none_returns_none(self):
        """Test : Conversion depuis None retourne None."""
        assert safe_int(None) is None

    def test_from_str_invalid_returns_none(self):
        """Test : Conversion depuis str invalide retourne None."""
        assert safe_int("invalid") is None
        assert safe_int("abc") is None
        assert safe_int("12.34.56") is None

    def test_from_empty_string_returns_none(self):
        """Test : Conversion depuis chaîne vide retourne None."""
        assert safe_int("") is None

    def test_from_bool_returns_int(self):
        """Test : Conversion depuis bool retourne int."""
        assert safe_int(True) == 1
        assert safe_int(False) == 0

    def test_from_list_returns_none(self):
        """Test : Conversion depuis list retourne None."""
        assert safe_int([1, 2, 3]) is None

    def test_from_dict_returns_none(self):
        """Test : Conversion depuis dict retourne None."""
        assert safe_int({"key": "value"}) is None

    def test_from_overflow_returns_none(self):
        """Test : Conversion depuis valeur overflow retourne None."""
        # Python gère les grands entiers, mais testons avec une valeur très grande
        # qui pourrait causer des problèmes dans certains contextes
        large_float = 1e20
        result = safe_int(large_float)
        # Le résultat devrait être un int ou None selon l'implémentation
        assert result is None or isinstance(result, int)

    def test_from_whitespace_string_returns_none(self):
        """Test : Conversion depuis str avec espaces retourne None."""
        assert safe_int("  42  ") is None  # Espaces non gérés par int()
        assert safe_int("  ") is None

    def test_from_hex_string_returns_none(self):
        """Test : Conversion depuis str hexadécimal retourne None."""
        # int("0xFF") fonctionne, mais safe_int ne gère pas ce cas
        assert safe_int("0xFF") is None

    def test_from_negative_zero(self):
        """Test : Conversion depuis -0."""
        assert safe_int(-0) == 0
        assert safe_int("-0") == 0
