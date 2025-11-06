#!/usr/bin/env python3
"""Script pour vérifier la valeur actuelle de emergency_penalty utilisée."""

import sys
import os
import json

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Company
from services.unified_dispatch.settings import for_company

def check_emergency_penalty(company_id: int = 1):
    """Vérifie la valeur de emergency_penalty pour une company."""
    print(f"🔍 Vérification de emergency_penalty pour company_id={company_id}\n")
    
    # Récupérer la company
    company = Company.query.get(company_id)
    if not company:
        print(f"❌ Company {company_id} non trouvée")
        return
    
    # 1. Vérifier dans autonomous_config (dispatch_overrides)
    print("=" * 60)
    print("1️⃣ Vérification dans autonomous_config (dispatch_overrides)")
    print("=" * 60)
    
    autonomous_config = company.get_autonomous_config()
    dispatch_overrides = autonomous_config.get("dispatch_overrides", {})
    
    if dispatch_overrides:
        emergency_config = dispatch_overrides.get("emergency", {})
        emergency_penalty_in_db = emergency_config.get("emergency_penalty") or emergency_config.get("emergency_per_stop_penalty")
        
        if emergency_penalty_in_db is not None:
            print(f"✅ Valeur trouvée dans DB (dispatch_overrides): {emergency_penalty_in_db}")
        else:
            print("⚠️  Pas de emergency_penalty dans dispatch_overrides")
            print(f"   Contenu emergency: {emergency_config}")
    else:
        print("⚠️  Pas de dispatch_overrides dans autonomous_config")
    
    # 2. Vérifier dans settings calculés (via for_company)
    print("\n" + "=" * 60)
    print("2️⃣ Vérification dans settings calculés (for_company)")
    print("=" * 60)
    
    settings = for_company(company)
    emergency_penalty_in_settings = getattr(settings.emergency, "emergency_penalty", None)
    
    print(f"✅ Valeur dans settings.emergency.emergency_penalty: {emergency_penalty_in_settings}")
    
    # 3. Calculer le malus appliqué
    print("\n" + "=" * 60)
    print("3️⃣ Calcul du malus appliqué")
    print("=" * 60)
    
    if emergency_penalty_in_settings:
        malus = -(emergency_penalty_in_settings / 180.0)
        print(f"📊 Pénalité: {emergency_penalty_in_settings}")
        print(f"📉 Malus appliqué au score: {malus:.3f}")
        print("   (Formule: -(penalty / 180.0))")
    
    # 4. Afficher le contenu complet de autonomous_config pour debug
    print("\n" + "=" * 60)
    print("4️⃣ Contenu complet de autonomous_config (debug)")
    print("=" * 60)
    print(json.dumps(autonomous_config, indent=2, default=str))
    
    print("\n" + "=" * 60)
    print("✅ Vérification terminée")
    print("=" * 60)

if __name__ == "__main__":
    # Utiliser Flask app context
    from app import create_app
    app = create_app()
    
    with app.app_context():
        company_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
        check_emergency_penalty(company_id)

