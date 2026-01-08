"""Test rapide pour valider l'endpoint login-test."""

import requests


def test_login_test_endpoint():
    """Tester l'endpoint /api/auth/login-test."""
    url = "http://localhost:5000/api/auth/login-test"

    # Test 1: Requête sans credentials (doit échouer)
    print("Test 1: Sans credentials...")
    response = requests.post(url, json={})
    print(f"  Status: {response.status_code}")
    print(f"  Body: {response.json()}")
    print()

    # Test 2: Avec credentials admin@test.com
    print("Test 2: Avec credentials admin@test.com...")
    response = requests.post(
        url, json={"email": "admin@test.com", "password": "test123"}
    )
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("  ✅ Login réussi!")
        print(f"  User: {data.get('user', {}).get('email')}")
        print(f"  Token: {data.get('token')[:50]}...")
    else:
        print(f"  ❌ Login échoué: {response.json()}")
    print()

    # Test 3: Avec mauvais password (doit échouer)
    print("Test 3: Avec mauvais password...")
    response = requests.post(
        url, json={"email": "admin@test.com", "password": "wrongpassword"}
    )
    print(f"  Status: {response.status_code}")
    print(f"  Body: {response.json()}")
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("Test Endpoint /api/auth/login-test")
    print("=" * 80)
    print()
    try:
        test_login_test_endpoint()
        print("=" * 80)
        print("✅ Tests terminés")
        print("=" * 80)
    except Exception as e:
        print(f"❌ Erreur: {e}")
