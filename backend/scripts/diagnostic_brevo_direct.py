#!/usr/bin/env python3
"""
Diagnostic rapide du domaine lirie.ch dans Brevo.
Sans dépendances, juste 'requests'.
"""

import json
import sys

try:
    import requests
except ImportError:
    print("❌ Module 'requests' requis")
    print("   Installez avec: pip install requests")
    sys.exit(1)

# ⚠️ IMPORTANT : Remplacez par votre vraie clé API Brevo
# Ou définissez la variable d'environnement BREVO_API_KEY
import os

BREVO_API_KEY = os.getenv("BREVO_API_KEY", "YOUR_API_KEY_HERE")

if BREVO_API_KEY == "YOUR_API_KEY_HERE":
    print("=" * 80)
    print("❌ BREVO_API_KEY non définie")
    print("=" * 80)
    print()
    print("Collez votre clé API Brevo ci-dessous (ligne 18):")
    print("BREVO_API_KEY = 'xkeysib-VOTRE_CLE_API'")
    print()
    print("Ou exécutez:")
    print(
        '$env:BREVO_API_KEY="votre_cle"; python backend\\scripts\\diagnostic_brevo_direct.py'
    )
    sys.exit(1)

DOMAIN = "lirie.ch"

print("=" * 80)
print(f"🔍 DIAGNOSTIC BREVO - Domaine {DOMAIN}")
print("=" * 80)
print()

headers = {
    "accept": "application/json",
    "api-key": BREVO_API_KEY,
}

# 1. Test connexion
print("📡 Test 1/2 : Connexion à l'API Brevo...")
try:
    response = requests.get(
        "https://api.brevo.com/v3/account",
        headers=headers,
        timeout=10,
    )

    if response.status_code == 200:
        account = response.json()
        print(f"   ✅ Connecté ! Email: {account.get('email', 'N/A')}")
    else:
        print(f"   ❌ Erreur {response.status_code}: {response.text}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

print()

# 2. Diagnostic domaine
print(f"🔍 Test 2/2 : Diagnostic du domaine {DOMAIN}...")
try:
    response = requests.get(
        f"https://api.brevo.com/v3/senders/domains/{DOMAIN}",
        headers=headers,
        timeout=10,
    )

    print(f"   📡 Status HTTP: {response.status_code}")
    print()

    if response.status_code == 200:
        data = response.json()

        # Extraire les infos
        verified = data.get("verified", False)
        authenticated = data.get("authenticated", False)
        dns_records = data.get("dns_records", {})

        brevo_code = dns_records.get("brevo_code", {})
        dkim1 = dns_records.get("dkim1Record", {})
        dkim2 = dns_records.get("dkim2Record", {})

        brevo_valid = brevo_code.get("is_valid", False)
        dkim1_valid = dkim1.get("is_valid", False)
        dkim2_valid = dkim2.get("is_valid", False)

        # Affichage
        print("   " + "=" * 70)
        print(f"   📧 Domaine: {DOMAIN}")
        print("   " + "=" * 70)
        print()

        print("   📊 STATUT BREVO:")
        print(f"      Vérifié : {'✅ OUI' if verified else '❌ NON'}")
        print(f"      Authentifié : {'✅ OUI' if authenticated else '❌ NON'}")
        print()

        print("   🔐 VALIDATION DNS:")
        print(
            f"      Brevo Code (SPF) : {'✅ VALIDE' if brevo_valid else '❌ NON VALIDE'}"
        )
        print(f"      DKIM 1 : {'✅ VALIDE' if dkim1_valid else '❌ NON VALIDE'}")
        print(f"      DKIM 2 : {'✅ VALIDE' if dkim2_valid else '❌ NON VALIDE'}")
        print()

        # Détails des enregistrements
        print("   📋 DÉTAILS DES ENREGISTREMENTS DNS:")
        print()

        if brevo_code:
            print("      🔹 Brevo Code (TXT):")
            print(f"         Hôte : {brevo_code.get('host_name', '@')}")
            print(f"         Valeur : {brevo_code.get('value', 'N/A')}")
            print(
                f"         Statut : {'✅ Détecté par Brevo' if brevo_valid else '❌ Non détecté'}"
            )
            print()

        if dkim1:
            print("      🔹 DKIM 1 (CNAME):")
            print(f"         Hôte : {dkim1.get('host_name', 'N/A')}")
            print(f"         Valeur : {dkim1.get('value', 'N/A')}")
            print(
                f"         Statut : {'✅ Détecté par Brevo' if dkim1_valid else '❌ Non détecté'}"
            )
            print()

        if dkim2:
            print("      🔹 DKIM 2 (CNAME):")
            print(f"         Hôte : {dkim2.get('host_name', 'N/A')}")
            print(f"         Valeur : {dkim2.get('value', 'N/A')}")
            print(
                f"         Statut : {'✅ Détecté par Brevo' if dkim2_valid else '❌ Non détecté'}"
            )
            print()

        # Diagnostic et recommandations
        print("   " + "=" * 70)
        print("   💡 DIAGNOSTIC:")
        print("   " + "=" * 70)

        all_valid = brevo_valid and dkim1_valid and dkim2_valid

        if verified:
            print()
            print("   🎉 DOMAINE ENTIÈREMENT VÉRIFIÉ !")
            print("   Vous pouvez envoyer des emails depuis lirie.ch")
            print()
        elif all_valid:
            print()
            print("   ✅ Tous les enregistrements DNS sont VALIDES !")
            print("   ⏳ Brevo n'a pas encore marqué le domaine comme vérifié.")
            print()
            print("   📅 Actions:")
            print("      1. Attendre jusqu'à 72h au total")
            print("      2. Si toujours pas vérifié après 72h:")
            print("         → Contacter support@brevo.com")
            print("         → Mentionner: DNS valides mais domaine pas vérifié")
            print()
        else:
            print()
            print("   ⚠️ Certains enregistrements DNS ne sont PAS détectés par Brevo")
            print()
            print("   📅 Actions suggérées:")
            if not brevo_valid:
                print("      ❌ Brevo Code (SPF) non valide")
                print("         → Vérifier l'enregistrement TXT sur GoDaddy")
                print("         → Supprimer et recréer si nécessaire")
            if not dkim1_valid:
                print("      ❌ DKIM 1 non valide")
                print("         → Vérifier l'enregistrement CNAME brevo1._domainkey")
            if not dkim2_valid:
                print("      ❌ DKIM 2 non valide")
                print("         → Vérifier l'enregistrement CNAME brevo2._domainkey")
            print()
            print("   💡 Conseils:")
            print("      - Réduire le TTL à 300 secondes (5 minutes)")
            print("      - Attendre 15-30 minutes après modification")
            print("      - Supprimer et recréer l'enregistrement si problème")
            print()

        # Réponse complète (pour debug)
        print("   " + "=" * 70)
        print("   🔧 RÉPONSE API COMPLÈTE (debug):")
        print("   " + "=" * 70)
        print()
        print(json.dumps(data, indent=2))
        print()

    elif response.status_code == 404:
        print("   ❌ DOMAINE NON TROUVÉ DANS BREVO")
        print()
        print("   💡 Solution:")
        print("      1. Connectez-vous à app.brevo.com")
        print("      2. Senders & Domains → Domains → Add a Domain")
        print("      3. Entrez: lirie.ch")
        print("      4. Configurez les DNS fournis (déjà fait ✅)")
        print("      5. Cliquez sur 'Verify Domain'")
        print()
    else:
        print(f"   ❌ Erreur API: {response.status_code}")
        print(f"   Détails: {response.text}")
        print()

except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback

    traceback.print_exc()

print("=" * 80)
print("✅ DIAGNOSTIC TERMINÉ")
print("=" * 80)
