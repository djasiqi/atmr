#!/usr/bin/env python3
"""Script pour tester que les modifications de paramètres sont bien récupérées."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app  # noqa: E402
from ext import db  # noqa: E402
from models import Company  # noqa: E402
from services.unified_dispatch.settings import for_company  # noqa: E402


def test_settings_update():
    """Teste que les modifications sont bien récupérées."""
    app = create_app()

    with app.app_context():
        company = Company.query.get(1)
        if not company:
            print("❌ Company 1 non trouvée")
            return

        print("=" * 80)
        print("🧪 TEST DE RÉCUPÉRATION DES PARAMÈTRES")
        print("=" * 80)

        # 1. Lire les paramètres actuels
        print("\n1️⃣ Lecture des paramètres actuels")
        print("-" * 80)
        settings1 = for_company(company)
        emergency_penalty1 = settings1.emergency.emergency_penalty
        proximity1 = settings1.heuristic.proximity
        print(f"   emergency_penalty: {emergency_penalty1}")
        print(f"   proximity: {proximity1}")

        # 2. Modifier les paramètres dans la DB (simulation)
        print("\n2️⃣ Modification des paramètres dans la DB")
        print("-" * 80)
        config = company.get_autonomous_config()
        overrides = config.get("dispatch_overrides", {})

        # Sauvegarder les valeurs originales
        original_emergency = overrides.get("emergency", {}).get("emergency_penalty")
        original_proximity = overrides.get("heuristic", {}).get("proximity")

        # Modifier les valeurs
        if "emergency" not in overrides:
            overrides["emergency"] = {}
        if "heuristic" not in overrides:
            overrides["heuristic"] = {}

        # Changer emergency_penalty à 700 (si c'était 600)
        new_emergency = 700 if (original_emergency or 600) == 600 else 600
        overrides["emergency"]["emergency_penalty"] = new_emergency

        # Changer proximity à 0.1 (si c'était 0.05)
        new_proximity = 0.1 if (original_proximity or 0.05) == 0.05 else 0.05
        overrides["heuristic"]["proximity"] = new_proximity

        config["dispatch_overrides"] = overrides
        company.set_autonomous_config(config)
        db.session.add(company)
        db.session.commit()

        print(
            f"   ✅ Modifié emergency_penalty: {original_emergency} → {new_emergency}"
        )
        print(f"   ✅ Modifié proximity: {original_proximity} → {new_proximity}")

        # 3. Recharger depuis la DB (simuler un nouveau dispatch)
        print("\n3️⃣ Rechargement depuis la DB (simulation nouveau dispatch)")
        print("-" * 80)

        # Expirer la session SQLAlchemy pour forcer un rechargement depuis la DB
        db.session.expire(company)
        db.session.refresh(company)

        settings2 = for_company(company)
        emergency_penalty2 = settings2.emergency.emergency_penalty
        proximity2 = settings2.heuristic.proximity

        print(f"   emergency_penalty: {emergency_penalty2}")
        print(f"   proximity: {proximity2}")

        # 4. Vérifier que les valeurs sont bien mises à jour
        print("\n4️⃣ Vérification")
        print("-" * 80)
        if emergency_penalty2 == new_emergency:
            print(
                f"   ✅ emergency_penalty correctement récupéré: {emergency_penalty2}"
            )
        else:
            print(
                f"   ❌ emergency_penalty NON récupéré: "
                f"attendu {new_emergency}, obtenu {emergency_penalty2}"
            )

        if proximity2 == new_proximity:
            print(f"   ✅ proximity correctement récupéré: {proximity2}")
        else:
            print(
                f"   ❌ proximity NON récupéré: "
                f"attendu {new_proximity}, obtenu {proximity2}"
            )

        # 5. Restaurer les valeurs originales
        print("\n5️⃣ Restauration des valeurs originales")
        print("-" * 80)
        if original_emergency is not None:
            overrides["emergency"]["emergency_penalty"] = original_emergency
        if original_proximity is not None:
            overrides["heuristic"]["proximity"] = original_proximity

        config["dispatch_overrides"] = overrides
        company.set_autonomous_config(config)
        db.session.add(company)
        db.session.commit()
        print("   ✅ Valeurs originales restaurées")

        print("\n" + "=" * 80)
        print("✅ Test terminé")
        print("=" * 80)


if __name__ == "__main__":
    test_settings_update()
