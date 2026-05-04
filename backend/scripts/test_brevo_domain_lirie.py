#!/usr/bin/env python3
"""
Script de test pour vérifier le statut du domaine lirie.ch dans Brevo.

Usage:
    python test_brevo_domain_lirie.py

La clé API peut être fournie via:
- Variable d'environnement: BREVO_API_KEY=xxx python test_brevo_domain_lirie.py
- Ou directement dans le code (ligne 25)
"""

import os
import sys

try:
    import requests
except ImportError:
    print("❌ Module 'requests' non installé")
    print("   Installez avec: pip install requests")
    sys.exit(1)

# ⚠️ Remplacez par votre vraie clé API ou définissez BREVO_API_KEY env var
BREVO_API_KEY = os.getenv("BREVO_API_KEY") or "YOUR_BREVO_API_KEY_HERE"
DOMAIN = "lirie.ch"

if not BREVO_API_KEY or BREVO_API_KEY == "YOUR_BREVO_API_KEY_HERE":
    print("=" * 80)
    print("❌ BREVO_API_KEY non définie")
    print("=" * 80)
    print()
    print("Option 1 - Variable d'environnement:")
    print(
        "   PowerShell: $env:BREVO_API_KEY='votre_cle'; python scripts\\test_brevo_domain_lirie.py"
    )
    print()
    print("Option 2 - Modifier le script:")
    print("   Ouvrez backend/scripts/test_brevo_domain_lirie.py")
    print("   Ligne 25: BREVO_API_KEY = 'votre_cle_api'")
    print()
    sys.exit(1)

print("=" * 80)
print(f"🔍 TEST API BREVO - Vérification du domaine {DOMAIN}")
print("=" * 80)
print()

headers = {
    "accept": "application/json",
    "api-key": BREVO_API_KEY,
}

# 1. Test de connexion à l'API
print("📡 Test 1 : Connexion à l'API Brevo...")
try:
    response = requests.get(
        "https://api.brevo.com/v3/account",
        headers=headers,
        timeout=10,
    )

    if response.status_code == 200:
        account_data = response.json()
        print(f"   ✅ Connecté ! Email: {account_data.get('email', 'N/A')}")
        print(f"   📦 Plan: {account_data.get('plan', [{}])[0].get('type', 'N/A')}")
    else:
        print(f"   ❌ Erreur {response.status_code}: {response.text}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Erreur de connexion: {e}")
    sys.exit(1)

print()

# 2. Vérifier le domaine
print(f"🔍 Test 2 : Vérification du domaine {DOMAIN}...")
try:
    response = requests.get(
        f"https://api.brevo.com/v3/senders/domains/{DOMAIN}",
        headers=headers,
        timeout=10,
    )

    print(f"   📡 Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()

        # Extraire les informations
        verified = data.get("verified", False)
        authenticated = data.get("authenticated", False)
        dns_records = data.get("dns_records", {})

        print()
        print("   " + "=" * 70)
        print(f"   📧 Domaine: {DOMAIN}")
        print(f"   ✅ Verified: {verified}")
        print(f"   🔐 Authenticated: {authenticated}")
        print("   " + "=" * 70)
        print()

        if verified or authenticated:
            print("   🎉 DOMAINE VALIDÉ ! Vous pouvez envoyer des emails.")
        else:
            print("   ⏳ DOMAINE EN ATTENTE DE VALIDATION")
            print()
            print("   📋 Enregistrements DNS détectés par Brevo:")

            # SPF/Brevo Code
            brevo_code = dns_records.get("brevo_code", {})
            if brevo_code:
                print("      🔹 Brevo Code (TXT):")
                print(f"         Hôte: {brevo_code.get('host_name', '@')}")
                print(f"         Valeur: {brevo_code.get('value', 'N/A')}")
                print(f"         ✅ Valide: {brevo_code.get('is_valid', False)}")

            # DKIM 1
            dkim1 = dns_records.get("dkim1Record", {})
            if dkim1:
                print("      🔹 DKIM 1 (CNAME):")
                print(f"         Hôte: {dkim1.get('host_name', 'N/A')}")
                print(f"         Valeur: {dkim1.get('value', 'N/A')}")
                print(f"         ✅ Valide: {dkim1.get('is_valid', False)}")

            # DKIM 2
            dkim2 = dns_records.get("dkim2Record", {})
            if dkim2:
                print("      🔹 DKIM 2 (CNAME):")
                print(f"         Hôte: {dkim2.get('host_name', 'N/A')}")
                print(f"         Valeur: {dkim2.get('value', 'N/A')}")
                print(f"         ✅ Valide: {dkim2.get('is_valid', False)}")

            print()
            print("   💡 Actions suggérées:")

            # Vérifier quels enregistrements sont invalides
            brevo_valid = brevo_code.get("is_valid", False) if brevo_code else False
            dkim1_valid = dkim1.get("is_valid", False) if dkim1 else False
            dkim2_valid = dkim2.get("is_valid", False) if dkim2 else False

            if not brevo_valid:
                print(
                    "      ❌ Brevo Code (SPF) non valide → Vérifiez l'enregistrement TXT"
                )
            if not dkim1_valid:
                print("      ❌ DKIM 1 non valide → Vérifiez l'enregistrement CNAME")
            if not dkim2_valid:
                print("      ❌ DKIM 2 non valide → Vérifiez l'enregistrement CNAME")

            if brevo_valid and dkim1_valid and dkim2_valid:
                print("      ✅ Tous les enregistrements sont valides !")
                print("      ⏳ Brevo devrait valider le domaine sous peu.")
                print("      📞 Si le problème persiste, contactez le support Brevo.")

        print()
        print("   📄 Réponse complète de l'API:")
        import json

        print(json.dumps(data, indent=2))

    elif response.status_code == 404:
        print(f"   ❌ Domaine {DOMAIN} NON TROUVÉ dans Brevo")
        print()
        print("   💡 Solution:")
        print("      1. Connectez-vous à app.brevo.com")
        print("      2. Allez dans Senders & Domains → Domains")
        print("      3. Ajoutez manuellement le domaine lirie.ch")
        print("      4. Configurez les enregistrements DNS fournis")
    else:
        print(f"   ❌ Erreur API: {response.status_code}")
        print(f"   Réponse: {response.text}")

except Exception as e:
    print(f"   ❌ Erreur lors de la vérification: {e}")
    import traceback

    traceback.print_exc()

print()
print("=" * 80)
print("✅ TEST TERMINÉ")
print("=" * 80)
