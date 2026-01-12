"""Script de test pour InvoiceTemplateBuilder.

Ce script valide l'extraction de données depuis une facture réelle
et teste la génération des templates HTML.
"""

import logging
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from models import Invoice
from services.documents.invoice_template_builder import InvoiceTemplateBuilder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def test_invoice_data_extraction():
    """Test l'extraction de données depuis une facture réelle."""
    app = create_app()

    with app.app_context():
        logger.info("=" * 80)
        logger.info("🧪 TEST EXTRACTION INVOICEDATA")
        logger.info("=" * 80)

        # Récupérer la première facture disponible
        invoice = Invoice.query.first()

        if not invoice:
            logger.error("❌ Aucune facture trouvée en base de données")
            return False

        logger.info("\n📋 Facture test:")
        logger.info("   ID: %s", invoice.id)
        logger.info("   Numéro: %s", invoice.invoice_number)
        logger.info("   Company ID: %s", invoice.company_id)
        logger.info("   Client ID: %s", invoice.client_id)
        logger.info("   Total: %s CHF", invoice.total_amount)

        # Initialiser le builder
        builder = InvoiceTemplateBuilder()
        logger.info("\n✅ InvoiceTemplateBuilder initialisé")

        # Extraire les données
        logger.info("\n🔄 Extraction des données...")
        invoice_data = builder.extract_invoice_data(invoice)

        if not invoice_data:
            logger.error("❌ Échec extraction: invoice_data est None")
            logger.error(
                "   Vérifier que CompanyBillingProfile existe pour company_id=%s",
                invoice.company_id,
            )
            return False

        logger.info("\n✅ Données extraites avec succès!")
        logger.info("")
        logger.info("%s", "=" * 80)
        logger.info("📊 DONNÉES EXTRAITES")
        logger.info("%s", "=" * 80)

        # Facture
        logger.info("\n📄 FACTURE:")
        logger.info("   Numéro: %s", invoice_data.invoice_number)
        logger.info(
            "   Date émission: %s", invoice_data.issue_date.strftime("%d.%m.%Y")
        )
        logger.info("   Date échéance: %s", invoice_data.due_date.strftime("%d.%m.%Y"))
        logger.info("   Période: %s", invoice_data.period)
        logger.info("   Total: %.2f CHF", invoice_data.total_amount)
        logger.info("   Solde dû: %.2f CHF", invoice_data.balance_due)
        logger.info("   Est rappel: %s", "Oui" if invoice_data.is_reminder else "Non")
        if invoice_data.is_reminder:
            logger.info("   Niveau rappel: %s", invoice_data.reminder_level)

        # Émetteur
        logger.info("\n🏢 ÉMETTEUR:")
        logger.info("   Nom: %s", invoice_data.emitter_name)
        logger.info("   Adresse: %s", invoice_data.emitter_street)
        logger.info("   Code postal: %s", invoice_data.emitter_postal_code)
        logger.info("   Ville: %s", invoice_data.emitter_city)
        logger.info("   Pays: %s", invoice_data.emitter_country)
        logger.info("   UID: %s", invoice_data.emitter_uid)
        logger.info("   Email: %s", invoice_data.emitter_email)
        logger.info("   Téléphone: %s", invoice_data.emitter_phone)
        logger.info(
            "   IBAN: %s...",
            invoice_data.emitter_iban[:20]
            if len(invoice_data.emitter_iban) > 20
            else invoice_data.emitter_iban,
        )

        # TVA
        logger.info("\n💰 TVA:")
        logger.info(
            "   Applicable: %s", "Oui" if invoice_data.vat_applicable else "Non"
        )
        if invoice_data.vat_applicable:
            logger.info("   Numéro TVA: %s", invoice_data.vat_number or "N/A")
            logger.info("   Taux: %s%%", invoice_data.vat_rate or "N/A")

        # Client
        logger.info("\n👤 CLIENT:")
        logger.info("   Nom: %s", invoice_data.client_name)
        logger.info(
            "   Adresse: %s", invoice_data.client_address.replace("<br/>", " / ")
        )

        # Lignes
        logger.info("\n📋 LIGNES DE FACTURE:")
        logger.info("   Nombre de lignes: %s", len(invoice_data.lines))
        if invoice_data.lines:
            logger.info("   Détail des 3 premières lignes:")
            for idx, line in enumerate(invoice_data.lines[:3], 1):
                logger.info(
                    "      %s. %s | Départ: %s | Montant: %.2f CHF",
                    idx,
                    line["date"],
                    line["departure"][:30] + "..."
                    if len(line["departure"]) > 30
                    else line["departure"],
                    line["amount"],
                )

        # Référence de paiement
        logger.info("\n💳 RÉFÉRENCE DE PAIEMENT:")
        if invoice_data.payment_reference:
            logger.info("   %s", invoice_data.payment_reference)
        else:
            logger.info("   Aucune référence générée")

        logger.info("")
        logger.info("%s", "=" * 80)
        logger.info("✅ TEST EXTRACTION: RÉUSSI")
        logger.info("=" * 80)

        return True


def test_html_generation():
    """Test la génération des composants HTML."""
    app = create_app()

    with app.app_context():
        logger.info("")
        logger.info("%s", "=" * 80)
        logger.info("🧪 TEST GÉNÉRATION HTML")
        logger.info("=" * 80)

        # Récupérer une facture
        invoice = Invoice.query.first()
        if not invoice:
            logger.error("❌ Aucune facture trouvée")
            return False

        # Extraire les données
        builder = InvoiceTemplateBuilder()
        invoice_data = builder.extract_invoice_data(invoice)

        if not invoice_data:
            logger.error("❌ Échec extraction données")
            return False

        # Tester header
        logger.info("\n📝 Test génération header HTML...")
        header_html = builder.build_header_html(invoice_data)
        logger.info("   ✅ Header généré (%s caractères)", len(header_html))

        # Tester footer
        logger.info("\n📝 Test génération footer HTML...")
        footer_html = builder.build_footer_html(invoice_data)
        logger.info("   ✅ Footer généré (%s caractères)", len(footer_html))

        # Tester template Standard
        logger.info("\n📝 Test génération table Standard...")
        standard_html = builder.build_lines_table_standard(invoice_data)
        logger.info(
            "   ✅ Template Standard généré (%s caractères)", len(standard_html)
        )

        # Tester template Minimal
        logger.info("\n📝 Test génération table Minimal...")
        minimal_html = builder.build_lines_table_minimal(invoice_data)
        logger.info("   ✅ Template Minimal généré (%s caractères)", len(minimal_html))

        # Tester template Detailed
        logger.info("\n📝 Test génération table Detailed...")
        detailed_html = builder.build_lines_table_detailed(invoice_data)
        logger.info(
            "   ✅ Template Detailed généré (%s caractères)", len(detailed_html)
        )

        logger.info("")
        logger.info("%s", "=" * 80)
        logger.info("✅ TEST GÉNÉRATION HTML: RÉUSSI")
        logger.info("=" * 80)

        return True


def main():
    """Point d'entrée principal."""
    logger.info("")
    logger.info("%s", "=" * 80)
    logger.info("🚀 DÉMARRAGE TESTS INVOICETEMPLATBUILDER")
    logger.info("=" * 80)

    success = True

    # Test 1: Extraction de données
    if not test_invoice_data_extraction():
        logger.error("\n❌ ÉCHEC TEST EXTRACTION")
        success = False

    # Test 2: Génération HTML
    if not test_html_generation():
        logger.error("\n❌ ÉCHEC TEST GÉNÉRATION HTML")
        success = False

    # Résumé
    logger.info("")
    logger.info("%s", "=" * 80)
    if success:
        logger.info("✅ TOUS LES TESTS SONT PASSÉS")
    else:
        logger.error("❌ CERTAINS TESTS ONT ÉCHOUÉ")
    logger.info("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
