#!/usr/bin/env python3
"""Script pour supprimer toutes les réservations (bookings) et factures de la base de données."""

import sys
from pathlib import Path

# Ajouter le répertoire racine du backend au PYTHONPATH si nécessaire
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def delete_all_bookings_and_invoices():
    """Supprime toutes les réservations et factures."""
    from app import create_app
    from ext import db
    from models.ab_test_result import ABTestResult
    from models.autonomous_action import AutonomousAction
    from models.booking import Booking
    from models.invoice import Invoice, InvoiceLine, InvoicePayment, InvoiceReminder
    from models.ml_prediction import MLPrediction
    from models.payment import Payment
    from models.rl_suggestion import RLSuggestion

    app = create_app()

    with app.app_context():
        print("🗑️  Suppression de toutes les réservations et factures...")
        
        # Compteurs
        deleted_invoice_payments = 0
        deleted_invoice_reminders = 0
        deleted_invoice_lines = 0
        deleted_invoices = 0
        deleted_payments = 0
        deleted_ab_test_results = 0
        deleted_rl_suggestions = 0
        deleted_autonomous_actions = 0
        deleted_ml_predictions = 0
        deleted_bookings = 0
        
        try:
            # 1. Supprimer les paiements de factures
            print("1️⃣  Suppression des paiements de factures...")
            deleted_invoice_payments = db.session.query(InvoicePayment).delete()
            print(f"   ✅ {deleted_invoice_payments} paiements de factures supprimés")
            
            # 2. Supprimer les rappels de factures
            print("2️⃣  Suppression des rappels de factures...")
            deleted_invoice_reminders = db.session.query(InvoiceReminder).delete()
            print(f"   ✅ {deleted_invoice_reminders} rappels de factures supprimés")
            
            # 3. Supprimer les lignes de factures
            print("3️⃣  Suppression des lignes de factures...")
            deleted_invoice_lines = db.session.query(InvoiceLine).delete()
            print(f"   ✅ {deleted_invoice_lines} lignes de factures supprimées")
            
            # 4. Supprimer les factures
            print("4️⃣  Suppression des factures...")
            deleted_invoices = db.session.query(Invoice).delete()
            print(f"   ✅ {deleted_invoices} factures supprimées")
            
            # 5. Supprimer les paiements (liés aux bookings)
            print("5️⃣  Suppression des paiements...")
            deleted_payments = db.session.query(Payment).delete()
            print(f"   ✅ {deleted_payments} paiements supprimés")
            
            # 6. Supprimer les résultats de tests A/B (références booking)
            print("6️⃣  Suppression des résultats de tests A/B...")
            deleted_ab_test_results = db.session.query(ABTestResult).delete()
            print(f"   ✅ {deleted_ab_test_results} résultats de tests A/B supprimés")
            
            # 7. Supprimer les suggestions RL (références booking)
            print("7️⃣  Suppression des suggestions RL...")
            deleted_rl_suggestions = db.session.query(RLSuggestion).delete()
            print(f"   ✅ {deleted_rl_suggestions} suggestions RL supprimées")
            
            # 8. Supprimer les actions autonomes (références booking)
            print("8️⃣  Suppression des actions autonomes...")
            deleted_autonomous_actions = db.session.query(AutonomousAction).delete()
            print(f"   ✅ {deleted_autonomous_actions} actions autonomes supprimées")
            
            # 9. Supprimer les prédictions ML (références booking)
            print("9️⃣  Suppression des prédictions ML...")
            deleted_ml_predictions = db.session.query(MLPrediction).delete()
            print(f"   ✅ {deleted_ml_predictions} prédictions ML supprimées")
            
            # 10. Supprimer les réservations
            print("🔟 Suppression des réservations...")
            deleted_bookings = db.session.query(Booking).delete()
            print(f"   ✅ {deleted_bookings} réservations supprimées")
            
            # Commit toutes les suppressions
            db.session.commit()
            
            print("\n✅ Suppression terminée avec succès !")
            print(f"\n📊 Résumé :")
            print(f"   - {deleted_invoice_payments} paiements de factures")
            print(f"   - {deleted_invoice_reminders} rappels de factures")
            print(f"   - {deleted_invoice_lines} lignes de factures")
            print(f"   - {deleted_invoices} factures")
            print(f"   - {deleted_payments} paiements")
            print(f"   - {deleted_ab_test_results} résultats de tests A/B")
            print(f"   - {deleted_rl_suggestions} suggestions RL")
            print(f"   - {deleted_autonomous_actions} actions autonomes")
            print(f"   - {deleted_ml_predictions} prédictions ML")
            print(f"   - {deleted_bookings} réservations")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Erreur lors de la suppression : {e}")
            raise


if __name__ == "__main__":
    delete_all_bookings_and_invoices()

