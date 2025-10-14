#!/usr/bin/env python3
"""
Script de test pour le monitoring automatique
"""
import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5000"
TOKEN = None  # Sera rempli après le login

def login():
    """Se connecter en tant qu'entreprise"""
    global TOKEN
    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={
            "email": "entreprise@test.com",  # Adaptez avec vos credentials
            "password": "votre_mot_de_passe"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        TOKEN = data.get("access_token")
        print("✅ Connecté avec succès")
        return True
    else:
        print(f"❌ Erreur de connexion: {response.status_code}")
        print(response.text)
        return False

def get_headers():
    """Retourne les headers avec le token"""
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

def start_monitoring():
    """Démarre le monitoring automatique"""
    print("\n🚀 Démarrage du monitoring automatique...")
    response = requests.post(
        f"{BASE_URL}/api/company_dispatch/optimizer/start",
        headers=get_headers(),
        json={"check_interval_seconds": 60}  # Vérifier toutes les 60 secondes
    )
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Monitoring démarré !")
        print(f"   Statut: {json.dumps(data, indent=2)}")
        return True
    else:
        print(f"❌ Erreur lors du démarrage: {response.status_code}")
        print(response.text)
        return False

def check_status():
    """Vérifie le statut du monitoring"""
    print("\n📊 Vérification du statut...")
    response = requests.get(
        f"{BASE_URL}/api/company_dispatch/optimizer/status",
        headers=get_headers()
    )
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Statut récupéré :")
        print(f"   Running: {data.get('running')}")
        print(f"   Last check: {data.get('last_check')}")
        print(f"   Opportunities: {data.get('opportunities_count', 0)}")
        return data
    else:
        print(f"❌ Erreur: {response.status_code}")
        print(response.text)
        return None

def get_delays():
    """Récupère les retards détectés"""
    print("\n⏱️  Récupération des retards...")
    today = datetime.now().strftime("%Y-%m-%d")
    response = requests.get(
        f"{BASE_URL}/api/company_dispatch/delays/live",
        headers=get_headers(),
        params={"date": today}
    )
    
    if response.status_code == 200:
        data = response.json()
        delays = data.get("delays", [])
        summary = data.get("summary", {})
        
        print(f"✅ {len(delays)} retard(s) détecté(s)")
        print(f"   Total: {summary.get('total_delays', 0)}")
        print(f"   Critiques: {summary.get('critical', 0)}")
        print(f"   Élevés: {summary.get('high', 0)}")
        print(f"   Moyens: {summary.get('medium', 0)}")
        print(f"   Faibles: {summary.get('low', 0)}")
        
        for i, delay in enumerate(delays[:3], 1):  # Afficher les 3 premiers
            print(f"\n   Retard #{i}:")
            print(f"     Booking: #{delay.get('booking_id')}")
            print(f"     Chauffeur: #{delay.get('driver_id')}")
            print(f"     Retard: {delay.get('current_delay')} min")
            print(f"     Sévérité: {delay.get('severity')}")
            suggestions = delay.get('suggestions', [])
            if suggestions:
                print(f"     Suggestions: {len(suggestions)}")
        
        return delays
    else:
        print(f"❌ Erreur: {response.status_code}")
        print(response.text)
        return []

def get_opportunities():
    """Récupère les opportunités d'optimisation"""
    print("\n💡 Récupération des opportunités...")
    response = requests.get(
        f"{BASE_URL}/api/company_dispatch/optimizer/opportunities",
        headers=get_headers()
    )
    
    if response.status_code == 200:
        data = response.json()
        opps = data.get("opportunities", [])
        print(f"✅ {len(opps)} opportunité(s) détectée(s)")
        print(f"   Critiques: {data.get('critical_count', 0)}")
        print(f"   Élevées: {data.get('high_count', 0)}")
        
        for i, opp in enumerate(opps[:2], 1):
            print(f"\n   Opportunité #{i}:")
            print(f"     Assignment: #{opp.get('assignment_id')}")
            print(f"     Retard: {opp.get('current_delay_minutes')} min")
            print(f"     Sévérité: {opp.get('severity')}")
        
        return opps
    else:
        print(f"❌ Erreur: {response.status_code}")
        print(response.text)
        return []

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🔍 TEST DU MONITORING AUTOMATIQUE")
    print("=" * 60)
    
    # 1. Se connecter
    if not login():
        print("\n⚠️  Veuillez mettre à jour les credentials dans le script")
        return
    
    # 2. Démarrer le monitoring
    if not start_monitoring():
        return
    
    # 3. Vérifier le statut
    time.sleep(2)
    check_status()
    
    # 4. Récupérer les retards
    time.sleep(2)
    get_delays()
    
    # 5. Récupérer les opportunités
    time.sleep(2)
    get_opportunities()
    
    # 6. Attendre un peu et revérifier
    print("\n⏳ Attente de 65 secondes pour le prochain check automatique...")
    time.sleep(65)
    
    print("\n🔄 Revérification après un cycle...")
    check_status()
    get_opportunities()
    
    print("\n" + "=" * 60)
    print("✅ Test terminé !")
    print("=" * 60)
    print("\nLe monitoring continue en arrière-plan.")
    print("Pour l'arrêter, utilisez l'endpoint /optimizer/stop")

if __name__ == "__main__":
    main()

