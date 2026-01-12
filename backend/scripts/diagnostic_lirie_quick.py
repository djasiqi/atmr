#!/usr/bin/env python3
"""
Diagnostic ULTRA RAPIDE du domaine lirie.ch dans Brevo.
Charge automatiquement depuis backend/.env
"""

import json
import os
import sys
from pathlib import Path

# Charger .env
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    print(f"✅ .env chargé: {env_path}")
except:
    print("⚠️  dotenv non disponible, utilise variables système")

print()

try:
    import requests
except ImportError:
    print("❌ 'requests' requis: pip install requests")
    sys.exit(1)

BREVO_API_KEY = os.getenv("BREVO_API_KEY")
if not BREVO_API_KEY:
    print("❌ BREVO_API_KEY non trouvée dans .env")
    print("   Ajoutez dans backend/.env: BREVO_API_KEY=xkeysib-...")
    sys.exit(1)

DOMAIN = "lirie.ch"

print("=" * 80)
print(f"🔍 DIAGNOSTIC BREVO - {DOMAIN}")
print("=" * 80)
print()

headers = {"accept": "application/json", "api-key": BREVO_API_KEY}

# Test connexion
print("📡 Connexion à Brevo...")
try:
    r = requests.get("https://api.brevo.com/v3/account", headers=headers, timeout=10)
    if r.status_code == 200:
        print(f"   ✅ Connecté ! Email: {r.json().get('email')}")
    else:
        print(f"   ❌ Erreur {r.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ {e}")
    sys.exit(1)

print()

# Diagnostic domaine
print(f"🔍 Diagnostic {DOMAIN}...")
try:
    r = requests.get(
        f"https://api.brevo.com/v3/senders/domains/{DOMAIN}",
        headers=headers,
        timeout=10,
    )

    if r.status_code == 200:
        data = r.json()

        verified = data.get("verified", False)
        authenticated = data.get("authenticated", False)
        dns = data.get("dns_records", {})

        brevo_code = dns.get("brevo_code", {})
        dkim1 = dns.get("dkim1Record", {})
        dkim2 = dns.get("dkim2Record", {})

        brevo_ok = brevo_code.get("is_valid", False)
        dkim1_ok = dkim1.get("is_valid", False)
        dkim2_ok = dkim2.get("is_valid", False)

        print()
        print("   " + "=" * 70)
        print(f"   📧 Domaine: {DOMAIN}")
        print("   " + "=" * 70)
        print()
        print("   📊 STATUT BREVO:")
        print(f"      Vérifié      : {'✅ OUI' if verified else '❌ NON'}")
        print(f"      Authentifié  : {'✅ OUI' if authenticated else '❌ NON'}")
        print()
        print("   🔐 VALIDATION DNS:")
        print(
            f"      Brevo Code (SPF) : {'✅ VALIDE' if brevo_ok else '❌ NON VALIDE'}"
        )
        print(
            f"      DKIM 1           : {'✅ VALIDE' if dkim1_ok else '❌ NON VALIDE'}"
        )
        print(
            f"      DKIM 2           : {'✅ VALIDE' if dkim2_ok else '❌ NON VALIDE'}"
        )
        print()

        all_valid = brevo_ok and dkim1_ok and dkim2_ok

        print("   " + "=" * 70)
        print("   💡 DIAGNOSTIC:")
        print("   " + "=" * 70)
        print()

        if verified:
            print("   🎉 DOMAINE COMPLÈTEMENT VÉRIFIÉ !")
            print("   ✅ Vous pouvez envoyer des emails depuis lirie.ch")
        elif all_valid:
            print("   ✅ Tous les DNS sont VALIDES !")
            print("   ⏳ Brevo n'a pas encore marqué le domaine comme vérifié")
            print()
            print("   📅 Actions:")
            print("      - Attendre jusqu'à 72h au total")
            print("      - Si pas vérifié après 72h → support@brevo.com")
        else:
            print("   ⚠️  Certains DNS ne sont PAS détectés par Brevo")
            print()
            if not brevo_ok:
                print("      ❌ Brevo Code (TXT) - Vérifier sur GoDaddy")
            if not dkim1_ok:
                print("      ❌ DKIM 1 (CNAME) - Vérifier brevo1._domainkey")
            if not dkim2_ok:
                print("      ❌ DKIM 2 (CNAME) - Vérifier brevo2._domainkey")
            print()
            print("   💡 Solutions:")
            print("      - Vérifier valeurs exactes sur GoDaddy")
            print("      - Réduire TTL à 300 secondes")
            print("      - Supprimer/recréer enregistrements")
            print("      - Attendre 15-30 min après modification")

        print()
        print("   " + "=" * 70)
        print("   📋 DÉTAILS ENREGISTREMENTS:")
        print("   " + "=" * 70)

        if brevo_code:
            print()
            print(f"   Brevo Code: {brevo_code.get('value', 'N/A')}")
            print(f"   Status: {'✅' if brevo_ok else '❌'}")

        if dkim1:
            print()
            print(
                f"   DKIM 1: {dkim1.get('host_name', 'N/A')} → {dkim1.get('value', 'N/A')}"
            )
            print(f"   Status: {'✅' if dkim1_ok else '❌'}")

        if dkim2:
            print()
            print(
                f"   DKIM 2: {dkim2.get('host_name', 'N/A')} → {dkim2.get('value', 'N/A')}"
            )
            print(f"   Status: {'✅' if dkim2_ok else '❌'}")

        print()
        print("   " + "=" * 70)
        print("   🔧 RÉPONSE API BREVO (debug):")
        print("   " + "=" * 70)
        print()
        print(json.dumps(data, indent=2))

    elif r.status_code == 404:
        print()
        print("   ❌ DOMAINE NON TROUVÉ DANS BREVO")
        print()
        print("   💡 Solution:")
        print("      1. app.brevo.com → Senders & Domains → Add Domain")
        print("      2. Entrer: lirie.ch")
        print("      3. Verify Domain")
    else:
        print(f"   ❌ Erreur {r.status_code}: {r.text}")

except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback

    traceback.print_exc()

print()
print("=" * 80)
print("✅ DIAGNOSTIC TERMINÉ")
print("=" * 80)
