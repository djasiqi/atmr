#!/usr/bin/env python3
"""Script pour vérifier les factures dans la base de données"""

from wsgi import app
from models import Invoice, Company, Client, User

with app.app_context():
    print("=== VÉRIFICATION DES FACTURES ===")
    print(f"Total factures: {Invoice.query.count()}")
    print(f"Total entreprises: {Company.query.count()}")
    print(f"Total clients: {Client.query.count()}")
    print(f"Total utilisateurs: {User.query.count()}")
    
    print("\n=== FACTURES PAR ENTREPRISE ===")
    for company in Company.query.all():
        count = Invoice.query.filter_by(company_id=company.id).count()
        print(f"Entreprise {company.id} ({company.name}): {count} factures")
    
    print("\n=== DÉTAILS DES FACTURES ===")
    invoices = Invoice.query.all()
    for invoice in invoices:
        print(f"Facture {invoice.id}: {invoice.invoice_number} - Client {invoice.client_id} - Entreprise {invoice.company_id} - Montant {invoice.total_amount}")
        print(f"  PDF URL: {invoice.pdf_url}")
        print(f"  Statut: {invoice.status}")
    
    print("\n=== CLIENTS PAR ENTREPRISE ===")
    for company in Company.query.all():
        clients = Client.query.filter_by(company_id=company.id).all()
        print(f"Entreprise {company.id}: {len(clients)} clients")
        for client in clients:
            print(f"  - Client {client.id}: {client.user.first_name if client.user else 'N/A'} {client.user.last_name if client.user else 'N/A'}")
