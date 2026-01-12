"""
Script de test manuel pour le provider Brevo.

Usage :
    1. Définir BREVO_API_KEY dans .env
    2. python backend/services/email/test_brevo_manual.py
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire backend au path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

# ruff: noqa: E402
from services.email.brevo_provider import BrevoEmailProvider


def test_connection():
    """Test 1 : Connexion à l'API Brevo."""
    print("\n" + "=" * 60)
    print("TEST 1 : Connexion à l'API Brevo")
    print("=" * 60)

    try:
        provider = BrevoEmailProvider()
        print(f"✅ Provider initialisé avec clé : {provider.api_key[:10]}...")

        if provider.test_connection():
            print("✅ Connexion à l'API Brevo réussie !")
            return True

        print("❌ Échec de connexion à l'API Brevo")
        print("   Vérifiez que BREVO_API_KEY est valide")
        return False

    except ValueError as e:
        print(f"❌ Erreur : {e}")
        print("\n💡 Solution :")
        print("   1. Créer un compte sur app.brevo.com")
        print("   2. Obtenir une clé API dans SMTP & API → API Keys")
        print("   3. Ajouter dans .env : BREVO_API_KEY=votre_cle")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")
        return False


def test_send_simple_email(provider, to_email):
    """Test 2 : Envoi d'un email simple."""
    print("\n" + "=" * 60)
    print("TEST 2 : Envoi d'un email simple")
    print("=" * 60)

    result = provider.send_invoice_email(
        from_email="test@atmr.dev",  # À remplacer par votre domaine vérifié
        from_name="ATMR Test",
        to_email=to_email,
        to_name="Test Recipient",
        subject="Test Brevo ATMR - Email Simple",
        html_content="""
        <html>
        <body>
            <h1>✅ Test Brevo ATMR</h1>
            <p>Si vous recevez cet email, le provider Brevo fonctionne correctement !</p>
            <hr>
            <p style="color: #666; font-size: 12px;">
                Email de test envoyé depuis ATMR<br>
                Provider : Brevo (ex-Sendinblue)
            </p>
        </body>
        </html>
        """,
    )

    if result.success:
        print("✅ Email envoyé avec succès !")
        print(f"   Message ID : {result.message_id}")
        print(f"   Destinataire : {to_email}")
        return True

    print("❌ Échec de l'envoi")
    print(f"   Erreur : {result.error}")
    if "403" in str(result.error):
        print("\n💡 Cause probable : Domaine non vérifié")
        print("   Solution : Ajouter et vérifier votre domaine dans Brevo")
    return False


def test_send_email_with_pdf(provider, to_email):
    """Test 3 : Envoi d'un email avec PDF."""
    print("\n" + "=" * 60)
    print("TEST 3 : Envoi d'un email avec pièce jointe PDF")
    print("=" * 60)

    # Créer un faux PDF (pour le test)
    fake_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF"

    result = provider.send_invoice_email(
        from_email="test@atmr.dev",  # À remplacer
        from_name="ATMR Test",
        to_email=to_email,
        to_name="Test Recipient",
        subject="Test Brevo ATMR - Email avec PDF",
        html_content="""
        <html>
        <body>
            <h1>📄 Test avec Pièce Jointe</h1>
            <p>Cet email contient une pièce jointe PDF de test.</p>
            <p>Vérifiez que le fichier "facture_test.pdf" est bien attaché.</p>
        </body>
        </html>
        """,
        attachments=[{"filename": "facture_test.pdf", "content": fake_pdf}],
    )

    if result.success:
        print("✅ Email avec PDF envoyé !")
        print(f"   Message ID : {result.message_id}")
        return True

    print("❌ Échec de l'envoi")
    print(f"   Erreur : {result.error}")
    return False


def test_verify_domain(provider, domain):
    """Test 4 : Vérification de domaine."""
    print("\n" + "=" * 60)
    print("TEST 4 : Vérification de domaine")
    print("=" * 60)

    result = provider.verify_domain(domain)

    print(f"Domaine : {result.domain}")
    print(f"Vérifié : {'✅ Oui' if result.verified else '❌ Non'}")

    if result.spf_record:
        print(f"\nSPF : {result.spf_record}")
    if result.dkim_record:
        print(f"DKIM : {result.dkim_record[:50]}...")

    if result.error:
        print(f"\nErreur : {result.error}")
        print("\n💡 Pour configurer un domaine :")
        print("   1. Aller dans Brevo → Senders & Domains → Domains")
        print("   2. Ajouter votre domaine")
        print("   3. Copier les enregistrements SPF et DKIM")
        print("   4. Les ajouter chez votre hébergeur DNS")
        print("   5. Attendre 15 min puis cliquer 'Verify' dans Brevo")

    return result.verified


def main():
    """Fonction principale."""
    print("\n" + "🚀" * 30)
    print("TEST MANUEL DU PROVIDER BREVO")
    print("🚀" * 30)

    # Vérifier que la clé API existe
    api_key = os.getenv("BREVO_API_KEY")
    if not api_key:
        print("\n❌ Variable BREVO_API_KEY non définie")
        print("\n💡 Solution :")
        print("   export BREVO_API_KEY='votre_cle_api'")
        sys.exit(1)

    # Test 1 : Connexion
    if not test_connection():
        sys.exit(1)

    # Initialiser le provider
    provider = BrevoEmailProvider()

    # Demander l'email de test
    print("\n" + "-" * 60)
    to_email = input("Email de destination pour les tests : ").strip()
    if not to_email or "@" not in to_email:
        print("❌ Email invalide")
        sys.exit(1)

    # Test 2 : Email simple
    test_send_simple_email(provider, to_email)

    # Test 3 : Email avec PDF
    test_send_email_with_pdf(provider, to_email)

    # Test 4 : Vérification domaine
    domain = to_email.split("@")[1]
    test_verify_domain(provider, domain)

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)
    print("✅ Tests terminés !")
    print(f"📧 Vérifiez votre boîte mail : {to_email}")
    print("\n💡 Prochaines étapes :")
    print("   1. Configurer un domaine dans Brevo")
    print("   2. Implémenter les routes API")
    print("   3. Créer l'UI de configuration")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
