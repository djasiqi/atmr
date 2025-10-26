"""Script pour configurer l'API OpenWeatherMap.

Usage:
    python scripts/setup_weather_api.py
"""
import sys
from pathlib import Path


def setup_weather_api():
    """Configure l'API OpenWeatherMap."""
    print("\n" + "="*70)
    print("🌦️ CONFIGURATION API OPENWEATHERMAP")
    print("="*70)

    print("\n📝 Étapes:")
    print("   1. Créer un compte sur https://openweathermap.org/")
    print("   2. Copier votre API key depuis https://home.openweathermap.org/api_keys")
    print("   3. Entrer la clé ci-dessous")
    print()

    api_key = input("🔑 Entrez votre API key OpenWeatherMap: ").strip()

    if not api_key or api_key == "YOUR_KEY_HERE":
        print("\n❌ Clé API invalide ou vide")
        print("   Veuillez obtenir une vraie clé sur openweathermap.org")
        sys.exit(1)

    # Vérifier longueur (généralement 32 caractères)
    if len(api_key) < 20:
        print("\n⚠️  Clé suspicieusement courte ({len(api_key)} caractères)")
        confirm = input("   Continuer quand même? (o/N): ").strip().lower()
        if confirm != "o":
            sys.exit(1)

    # Créer/mettre à jour backend/.env
    env_path = Path(__file__).parent.parent / ".env"

    env_content = f"""# Configuration OpenWeatherMap API
OPENWEATHER_API_KEY={api_key}

# Configuration ML
ML_ENABLED=true
ML_TRAFFIC_PERCENTAGE=10
FALLBACK_ON_ERROR=true
"""

    # Sauvegarder
    with Path(env_path, "w").open() as f:
        f.write(env_content)

    print("\n✅ Fichier .env créé avec succès!")
    print("   Path: {env_path}")
    print()
    print("📋 Prochaines étapes:")
    print("   1. Redémarrer le container:")
    print("      docker-compose restart api")
    print()
    print("   2. Vérifier la variable:")
    print("      docker exec atmr-api-1 python -c \"import os; print('API Key:', 'OK' if os.getenv('OPENWEATHER_API_KEY') else 'MANQUANTE')\"")
    print()
    print("   3. Tester l'API météo:")
    print("      docker exec atmr-api-1 python tests/test_weather_service.py")
    print()
    print("="*70)


if __name__ == "__main__":
    try:
        setup_weather_api()
    except KeyboardInterrupt:
        print("\n\n⚠️  Configuration annulée")
        sys.exit(1)
    except Exception:
        print("\n❌ Erreur: {e}")
        sys.exit(1)

