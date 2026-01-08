"""Script pour générer des données de test pour le dispatch.

Crée :
- 1 compagnie (company_id=1)
- 50 chauffeurs disponibles
- 100 réservations pour le 2025-01-08

Usage :
    docker-compose exec api python tests/load_testing/seed_dispatch_data.py
"""

import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from decimal import Decimal

# Ajouter le répertoire backend au PYTHONPATH
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))


def seed_dispatch_data():
    """Créer les données de test pour le dispatch."""
    from app import create_app
    from ext import db
    from models import (
        Booking,
        BookingStatus,
        Client,
        Company,
        Driver,
        User,
        UserRole,
    )

    app = create_app()

    with app.app_context():
        print("=" * 80)
        print("Seed Dispatch Data (100 bookings × 50 drivers)")
        print("=" * 80)
        print()

        # ==============================
        # 1. Créer/Récupérer la compagnie
        # ==============================
        company = Company.query.filter_by(id=1).first()
        if not company:
            print("[1/5] Création de la compagnie...")
            # Créer utilisateur pour la compagnie
            company_user = User(
                username="test_company",
                email="company@test.com",
                role=UserRole.COMPANY,
            )
            company_user.set_password("test123")
            db.session.add(company_user)
            db.session.flush()

            # Créer la compagnie
            company = Company(
                user_id=company_user.id,
                name="Test Transport Company",
                siret="12345678901234",
                address="123 Rue de Test",
                phone="+33123456789",
                email="company@test.com",
            )
            db.session.add(company)
            db.session.flush()
            print(f"   ✅ Compagnie créée (ID: {company.id})")
        else:
            print(f"[1/5] Compagnie existante (ID: {company.id})")

        # ==============================
        # 2. Créer 50 chauffeurs
        # ==============================
        print("[2/5] Création de 50 chauffeurs...")
        existing_drivers = Driver.query.filter_by(company_id=company.id).count()
        drivers_to_create = max(0, 50 - existing_drivers)

        if drivers_to_create > 0:
            for i in range(drivers_to_create):
                # Créer utilisateur
                driver_user = User(
                    username=f"driver_test_{i + 1}",
                    email=f"driver{i + 1}@test.com",
                    role=UserRole.DRIVER,
                )
                driver_user.set_password("test123")
                db.session.add(driver_user)
                db.session.flush()

                # Créer chauffeur
                driver = Driver(
                    user_id=driver_user.id,
                    company_id=company.id,
                    license_plate=f"ABC-{i + 1:03d}-XY",
                    vehicle_assigned=f"Sedan {(i % 3) + 1}",  # Sedan 1, 2, 3
                    brand=["Peugeot", "Renault", "Citroën"][i % 3],
                    latitude=48.8566 + (i * 0.01),  # Autour de Paris
                    longitude=2.3522 + (i * 0.01),
                )
                db.session.add(driver)

            db.session.flush()
            print(f"   ✅ {drivers_to_create} chauffeurs créés")
        else:
            print(f"   ✅ {existing_drivers} chauffeurs déjà existants")

        drivers = Driver.query.filter_by(company_id=company.id).all()
        print(f"   Total : {len(drivers)} chauffeurs disponibles")

        # ==============================
        # 3. Créer 100 clients
        # ==============================
        print("[3/5] Création de 100 clients...")
        existing_clients = Client.query.count()
        clients_to_create = max(0, 100 - existing_clients)

        if clients_to_create > 0:
            for i in range(clients_to_create):
                # Créer utilisateur
                client_user = User(
                    username=f"client_test_{i + 1}",
                    email=f"client{i + 1}@test.com",
                    role=UserRole.CLIENT,
                )
                client_user.set_password("test123")
                db.session.add(client_user)
                db.session.flush()

                # Créer client
                client = Client(
                    user_id=client_user.id,
                    loyalty_points=0,
                )
                db.session.add(client)

            db.session.flush()
            print(f"   ✅ {clients_to_create} clients créés")
        else:
            print(f"   ✅ {existing_clients} clients déjà existants")

        clients = Client.query.all()
        print(f"   Total : {len(clients)} clients disponibles")

        # ==============================
        # 4. Créer 100 réservations pour le 2025-01-08
        # ==============================
        print("[4/5] Création de 100 réservations pour le 2025-01-08...")
        test_date = date(2025, 1, 8)

        # Supprimer les anciennes réservations de test (par date)
        start_of_day = datetime(2025, 1, 8, 0, 0, 0)
        end_of_day = datetime(2025, 1, 8, 23, 59, 59)
        Booking.query.filter(
            Booking.company_id == company.id,
            Booking.scheduled_time >= start_of_day,
            Booking.scheduled_time <= end_of_day,
        ).delete()

        # Créer nouvelles réservations
        for i in range(100):
            # Horaires répartis sur la journée (8h-20h)
            hour = 8 + (i * 12 // 100)  # Répartition sur 12h
            minute = (i * 60) % 60
            scheduled_datetime = datetime(2025, 1, 8, hour, minute)

            # Créer Booking avec champs minimaux
            client = clients[i % len(clients)]
            booking = Booking(
                user_id=client.user_id,
                client_id=client.id,
                company_id=company.id,
                customer_name=f"Client Test {i + 1}",
                scheduled_time=scheduled_datetime,
                pickup_location=f"Paris - Adresse {i + 1}",
                dropoff_location=f"Paris - Destination {i + 1}",
                amount=15.00 + (i % 20),
                status=BookingStatus.PENDING,
            )
            db.session.add(booking)

        db.session.flush()
        print(f"   ✅ 100 réservations créées pour le {test_date}")

        # ==============================
        # 5. Commit final
        # ==============================
        print("[5/5] Sauvegarde des données...")
        db.session.commit()
        print("   ✅ Toutes les données sauvegardées en BD")

        # ==============================
        # Résumé final
        # ==============================
        print()
        print("=" * 80)
        print("✅ SEED COMPLÉTÉ")
        print("=" * 80)
        print()
        print(f"📊 Résumé des données créées :")
        print(f"   • Compagnie       : {company.name} (ID: {company.id})")
        print(f"   • Chauffeurs      : {len(drivers)} disponibles")
        print(f"   • Clients         : {len(clients)} enregistrés")
        print(f"   • Réservations    : 100 pour le {test_date}")
        print()
        print(f"🚀 Prêt pour les tests de charge Locust !")
        print(f"   • URL : http://localhost:8089")
        print(f"   • Login : admin@test.com / test123")
        print(f"   • Company ID : {company.id}")
        print(f"   • Date : {test_date}")
        print()
        print("=" * 80)


if __name__ == "__main__":
    try:
        seed_dispatch_data()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
