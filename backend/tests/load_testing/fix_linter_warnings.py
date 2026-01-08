"""Script pour corriger automatiquement les warnings linter dans les tests de charge."""
import re
from pathlib import Path

def fix_dispatch_load_test():
    """Corrige les warnings dans dispatch_load_test.py"""
    file_path = Path(__file__).parent / "dispatch_load_test.py"
    content = file_path.read_text(encoding="utf-8")
    
    # 1. Remplacer × par x dans les docstrings
    content = content.replace("100 bookings × 50 drivers", "100 bookings x 50 drivers")
    content = content.replace("100×50", "100x50")
    content = content.replace("(100×50)", "(100x50)")
    
    # 2. Supprimer import random (non utilisé)
    content = re.sub(r"^import random\n", "", content, flags=re.MULTILINE)
    
    # 3. Remplacer f-strings par % formatting dans les logs
    # Ligne 57
    content = content.replace(
        'f"[SETUP] ✅ Prêt pour dispatch : date={self.test_date}, company={self.company_id}"',
        '"[SETUP] ✅ Prêt pour dispatch : date=%s, company=%s", self.test_date, self.company_id'
    )
    
    # Ligne 76
    content = content.replace(
        'f"[AUTH] ❌ Login échoué: {response.status_code}"',
        '"[AUTH] ❌ Login échoué: %s", response.status_code'
    )
    
    # Ligne 136
    content = content.replace(
        'f"Getting dispatch with heuristics for company {self.company_id}..."',
        '"Getting dispatch with heuristics for company %s...", self.company_id'
    )
    
    # Ligne 150
    content = content.replace(
        'f"[STATUS] Dispatch status: {status}"',
        '"[STATUS] Dispatch status: %s", status'
    )
    
    # Ligne 163
    content = content.replace(
        'f"[METRICS] Last dispatch: {data.get(\'last_run_duration\')}s"',
        '"[METRICS] Last dispatch: %ss", data.get(\'last_run_duration\')'
    )
    
    # Lignes 179-184 (multiline f-string)
    content = content.replace(
        '''logger.info(
                f"[DISPATCH] ✅ SUCCESS | "
                f"Duration: {duration:.2f}s | "
                f"Assignments: {num_assignments} | "
                f"Dispatch time: {dispatch_duration:.2f}s | "
                f"Assigned: {num_assigned}/{total_bookings}"
            )''',
        '''logger.info(
                "[DISPATCH] ✅ SUCCESS | Duration: %.2fs | Assignments: %s | Dispatch time: %.2fs | Assigned: %s/%s",
                duration, num_assignments, dispatch_duration, num_assigned, total_bookings
            )'''
    )
    
    # Ligne 190
    content = content.replace(
        'f"[SLO] ⚠️ Dispatch trop long: {dispatch_duration:.2f}s > 60s"',
        '"[SLO] ⚠️ Dispatch trop long: %.2fs > 60s", dispatch_duration'
    )
    
    # Lignes 195-196
    content = content.replace(
        '''logger.warning(
                        f"[SLO] ⚠️ Taux d\'assignation faible: {assigned_rate:.1f}% < 80%"
                    )''',
        '''logger.warning(
                        "[SLO] ⚠️ Taux d'assignation faible: %.1f%% < 80%%", assigned_rate
                    )'''
    )
    
    # Ligne 209
    content = content.replace(
        'f"[DISPATCH] ⏳ Async job: {job_id}"',
        '"[DISPATCH] ⏳ Async job: %s", job_id'
    )
    
    # Lignes 214-217
    content = content.replace(
        '''logger.error(
                f"[DISPATCH] ❌ FAILED | "
                f"Status: {response.status_code} | "
                f"Duration: {duration:.2f}s | "
                f"Response: {response.text[:200]}..."
            )''',
        '''logger.error(
                "[DISPATCH] ❌ FAILED | Status: %s | Duration: %.2fs | Response: %s...",
                response.status_code, duration, response.text[:200]
            )'''
    )
    
    # Ligne 235
    content = content.replace(
        'f"❌ TEST STARTUP FAILED: {e}" if e else "❌ TEST STARTUP FAILED"',
        '"❌ TEST STARTUP FAILED: %s" % e if e else "❌ TEST STARTUP FAILED"'
    )
    
    # Lignes 248-251
    content = content.replace(
        '''logger.info(f"✅ Tests complétés")
            logger.info(f"📊 Total requêtes: {stats.num_requests}")
            logger.info(f"   Succès: {stats.num_requests - stats.num_failures}")
            logger.info(f"   Échecs: {stats.num_failures}")''',
        '''logger.info("✅ Tests complétés")
            logger.info("📊 Total requêtes: %s", stats.num_requests)
            logger.info("   Succès: %s", stats.num_requests - stats.num_failures)
            logger.info("   Échecs: %s", stats.num_failures)'''
    )
    
    # Ligne 265
    content = content.replace(
        'logger.debug(f"[REQUEST] ✅ {method} {name}")',
        'logger.debug("[REQUEST] ✅ %s %s", method, name)'
    )
    
    # Ligne 267
    content = content.replace(
        'logger.error(f"[REQUEST] ❌ {name} | Exception: {exception}")',
        'logger.error("[REQUEST] ❌ %s | Exception: %s", name, exception)'
    )
    
    file_path.write_text(content, encoding="utf-8")
    print(f"[OK] Corrige: {file_path.name}")

def fix_fix_admin_permissions():
    """Corrige les warnings dans fix_admin_permissions.py"""
    file_path = Path(__file__).parent / "fix_admin_permissions.py"
    content = file_path.read_text(encoding="utf-8")
    
    # 1. Trier les imports (I001)
    content = content.replace(
        '''from app import create_app
from ext import db
from models.user import User
from models.enums import UserRole''',
        '''from app import create_app
from ext import db
from models.enums import UserRole
from models.user import User'''
    )
    
    # 2. F541 : Supprimer f-string sans placeholder
    content = content.replace(
        'print(f"[2/2] Role modifie avec succes!")',
        'print("[2/2] Role modifie avec succes!")'
    )
    
    # 3. RET503 : Ajouter return explicite à la fin
    content = content.replace(
        '''    print()
    print("="*80)

if __name__ == "__main__":''',
        '''    print()
    print("="*80)
    return False  # Fallback si aucun return n'est atteint

if __name__ == "__main__":'''
    )
    
    file_path.write_text(content, encoding="utf-8")
    print(f"[OK] Corrige: {file_path.name}")

def fix_seed_dispatch_data():
    """Corrige les warnings dans seed_dispatch_data.py"""
    file_path = Path(__file__).parent / "seed_dispatch_data.py"
    content = file_path.read_text(encoding="utf-8")
    
    # 1. Supprimer imports inutilisés
    content = content.replace(
        "from datetime import datetime, date, timedelta, UTC",
        "from datetime import UTC, date, datetime"
    )
    content = content.replace(
        "from decimal import Decimal",
        ""
    )
    
    # 2. F541 : Supprimer f-strings sans placeholders (lignes 203, 209-211)
    content = content.replace(
        'print(f"   ✅ {100 - len(clients)} clients créés")',
        'print(f"   ✅ {100 - len(clients)} clients créés")'  # Garde celui-ci car il a un placeholder
    )
    content = content.replace(
        'print(f"   Total : {len(clients)} clients disponibles")',
        'print(f"   Total : {len(clients)} clients disponibles")'  # Garde celui-ci
    )
    
    # Les f-strings à corriger sont celles SANS placeholders
    content = content.replace(
        '''print("\\n" + "="*80)
        print(f"📊 Résumé des données créées :")''',
        '''print("\\n" + "="*80)
        print("📊 Résumé des données créées :")'''
    )
    content = content.replace(
        '''print(f"\\n🚀 Prêt pour les tests de charge Locust !")
        print(f"   • URL : http://localhost:8089")
        print(f"   • Login : admin@test.com / test123")
        print(f"   • Company ID : 1")''',
        '''print("\\n🚀 Prêt pour les tests de charge Locust !")
        print("   • URL : http://localhost:8089")
        print("   • Login : admin@test.com / test123")
        print("   • Company ID : 1")'''
    )
    
    file_path.write_text(content, encoding="utf-8")
    print(f"[OK] Corrige: {file_path.name}")

if __name__ == "__main__":
    print("Correction des warnings linter...")
    print()
    try:
        fix_dispatch_load_test()
        fix_fix_admin_permissions()
        fix_seed_dispatch_data()
        print()
        print("[OK] Tous les warnings corriges !")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
