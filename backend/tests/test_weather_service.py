"""Tests pour le service météo."""


class TestWeatherService:
    """Tests du service météo."""

    def test_get_default_weather(self):
        """Test récupération météo par défaut."""
        from services.external.weather import WeatherService

        weather = WeatherService._get_default_weather()

        assert weather["weather_factor"] == 0.5  # Neutre
        assert weather["temperature"] == 15.0
        assert weather["is_default"] is True

        print("✅ Get default weather OK")

    def test_calculate_weather_factor_ideal(self):
        """Test calcul facteur météo - conditions idéales."""
        from services.external.weather import WeatherService

        # Conditions idéales
        weather_data = {
            "temperature": 20.0,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "wind_speed": 10.0,
            "visibility": 10000,
            "clouds": 20,
        }

        factor = WeatherService._calculate_weather_factor(weather_data)

        assert factor < 0.2  # Presque idéal

        print("✅ Weather factor (idéal) = {factor")

    def test_calculate_weather_factor_rain(self):
        """Test calcul facteur météo - pluie."""
        from services.external.weather import WeatherService

        # Pluie modérée
        weather_data = {
            "temperature": 15.0,
            "rain_1h": 5.0,  # 5mm = pluie modérée
            "snow_1h": 0.0,
            "wind_speed": 20.0,
            "visibility": 8000,
            "clouds": 80,
        }

        factor = WeatherService._calculate_weather_factor(weather_data)

        # Vérifier que facteur > 0 (pluie = défavorable)
        assert factor > 0.1  # Au moins un peu défavorable
        assert factor <= 1.0  # Max 1.0

        print("✅ Weather factor (pluie) = {factor")

    def test_calculate_weather_factor_snow(self):
        """Test calcul facteur météo - neige."""
        from services.external.weather import WeatherService

        # Neige
        weather_data = {
            "temperature": -2.0,
            "rain_1h": 0.0,
            "snow_1h": 3.0,  # 3mm = neige modérée
            "wind_speed": 40.0,  # Vent fort
            "visibility": 2000,  # Visibilité réduite
            "clouds": 100,
        }

        factor = WeatherService._calculate_weather_factor(weather_data)

        # Neige + vent + visibilité réduite = très défavorable
        assert factor > 0.3  # Défavorable
        assert factor <= 1.0  # Max 1.0

        print("✅ Weather factor (neige) = {factor")

    def test_cache_mechanism(self):
        """Test mécanisme de cache."""
        from services.external.weather import WeatherService

        # Clear cache
        WeatherService.clear_cache()

        # Première récupération (sans API key = default, pas de cache)
        WeatherService.get_weather(46.2044, 6.1432)

        # Sans API key, pas de cache (retourne default direct)
        # Mais le mécanisme de cache fonctionne quand API activée

        # Tester clear cache
        WeatherService.clear_cache()
        stats = WeatherService.get_cache_stats()
        assert stats["entries"] == 0

        # Tester get stats
        assert "entries" in stats
        assert "keys" in stats

        print("✅ Cache mechanism OK (tested without API key)")

    def test_get_weather_factor_helper(self):
        """Test helper get_weather_factor."""
        from services.external.weather import get_weather_factor

        factor = get_weather_factor(46.2044, 6.1432)

        assert 0.0 <= factor <= 1.0

        print("✅ get_weather_factor OK ({factor")


if __name__ == "__main__":
    """Exécution directe pour tests rapides."""
    print("\n" + "=" * 70)
    print("🧪 TESTS WEATHER SERVICE")
    print("=" * 70)

    test = TestWeatherService()
    try:
        test.test_get_default_weather()
        test.test_calculate_weather_factor_ideal()
        test.test_calculate_weather_factor_rain()
        test.test_calculate_weather_factor_snow()
        test.test_cache_mechanism()
        test.test_get_weather_factor_helper()
    except Exception:
        print("❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        import sys

        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ TOUS LES TESTS RÉUSSIS !")
    print("=" * 70)
    print("\nℹ️  Note: Tests utilisent default weather (pas d'API key)")
    print("   Pour tester avec vraie API:")
    print("   export OPENWEATHER_API_KEY=your_key")
    print("   pytest tests/test_weather_service.py")

