"""
Test rapide pour verifier que les tokens JWT generes par /login-test
ont l'audience correcte et fonctionnent avec les endpoints dispatch.
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_jwt_audience():
    print("="*80)
    print("Test JWT Audience - /login-test -> /company_dispatch/run")
    print("="*80)
    print()
    
    # Etape 1: Login
    print("[1/3] Login via /api/auth/login-test...")
    response = requests.post(
        f"{BASE_URL}/api/auth/login-test",
        json={"email": "admin@test.com", "password": "test123"},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code != 200:
        print(f"[FAIL] Login echoue: {response.status_code}")
        print(f"   Response: {response.text}")
        return False
    
    data = response.json()
    access_token = data.get("access_token")
    
    if not access_token:
        print("[FAIL] Pas de token dans la reponse")
        print(f"   Response: {json.dumps(data, indent=2)}")
        return False
    
    print(f"[OK] Login reussi!")
    print(f"   Token (preview): {access_token[:50]}...")
    print(f"   User: {data['user']['email']}")
    print()
    
    # Etape 2: Test dispatch (devrait echouer avec 404 ou autre erreur metier, mais PAS 401)
    print("[2/3] Test requete dispatch avec le nouveau token...")
    response = requests.post(
        f"{BASE_URL}/api/v1/company_dispatch/run",
        json={
            "company_id": 1,
            "date": "2026-01-09",
            "mode": "optimization"
        },
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    )
    
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.text[:200]}...")
    print()
    
    if response.status_code == 401:
        print("[FAIL] Token rejete (401 UNAUTHORIZED)")
        print("   Possible raison: audience manquante ou invalide")
        return False
    else:
        print(f"[OK] Token accepte (status {response.status_code})")
        print("   Audience JWT validee avec succes!")
        return True
    
    print()
    print("="*80)

if __name__ == "__main__":
    success = test_jwt_audience()
    print()
    print("="*80)
    if success:
        print("[SUCCESS] TEST REUSSI: Les tokens JWT ont l'audience correcte")
    else:
        print("[FAIL] TEST ECHOUE: Probleme avec l'audience JWT")
    print("="*80)
