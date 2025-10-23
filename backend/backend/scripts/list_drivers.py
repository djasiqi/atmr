#!/usr/bin/env python3
# ruff: noqa: T201
"""Liste tous les chauffeurs de la DB avec leurs initiales possibles"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from models import Driver

app = create_app()

with app.app_context():
    drivers = Driver.query.filter_by(company_id=1).all()
    
    print("=" * 80)
    print("👥 CHAUFFEURS DANS LA BASE DE DONNÉES")
    print("=" * 80)
    print()
    
    for d in drivers:
        if hasattr(d, "user") and d.user:
            first = d.user.first_name or ""
            last = d.user.last_name or ""
            full_name = f"{first} {last}".strip()
            
            # Générer les initiales possibles
            initials_1 = f"{first[0]}.{last[0]}" if first and last else ""
            initials_2 = f"{first[0]}{last[0]}" if first and last else ""
            
            print(f"ID: {d.id}")
            print(f"  Nom complet : {full_name}")
            print(f"  Initiales   : {initials_1} ou {initials_2}")
            print(f"  Type        : {getattr(d, 'driver_type', 'N/A')}")
            print(f"  Actif       : {getattr(d, 'is_active', False)}")
            print()

