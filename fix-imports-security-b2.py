#!/usr/bin/env python3
"""
Script pour corriger automatiquement les imports après migration B2 - Module security

Ce script remplace les anciens imports par les nouveaux chemins :
- services.access_token_service → services.security.authentication
- services.refresh_token_service → services.security.authentication
- services.csrf_protection → services.security.csrf
- services.spam_protection → services.security.spam
- services.idempotency_service → services.security.idempotency
- services.safety_guards → services.security.safety
- services.secret_rotation_monitor → services.security.secret_rotation
- services.pii_masking → services.security.pii

Usage:
    python fix-imports-security-b2.py
"""

import re
from pathlib import Path

# Mapping des anciens imports vers les nouveaux
IMPORT_MAPPING = {
    r"from services\.access_token_service import": r"from services.security.authentication import",
    r"from services\.refresh_token_service import": r"from services.security.authentication import",
    r"from services\.csrf_protection import": r"from services.security.csrf import",
    r"from services\.spam_protection import": r"from services.security.spam import",
    r"from services\.idempotency_service import": r"from services.security.idempotency import",
    r"from services\.safety_guards import": r"from services.security.safety import",
    r"from services\.secret_rotation_monitor import": r"from services.security.secret_rotation import",
    r"from services\.pii_masking import": r"from services.security.pii import",
    # Imports directs des modules
    r"import services\.access_token_service": r"import services.security.authentication",
    r"import services\.refresh_token_service": r"import services.security.authentication",
    r"import services\.csrf_protection": r"import services.security.csrf",
    r"import services\.spam_protection": r"import services.security.spam",
    r"import services\.idempotency_service": r"import services.security.idempotency",
    r"import services\.safety_guards": r"import services.security.safety",
    r"import services\.secret_rotation_monitor": r"import services.security.secret_rotation",
    r"import services\.pii_masking": r"import services.security.pii",
}

# Fichiers identifiés par grep
FILES_TO_FIX = [
    "backend/app.py",
    "backend/routes/invoices.py",
    "backend/routes/companies.py",
    "backend/routes/clients.py",
    "backend/routes/bookings.py",
    "backend/routes/auth.py",
    "backend/tests/test_idempotency_service.py",
    "backend/tests/security/test_token_rotation.py",
    "backend/shared/decorators.py",
    "backend/tasks/vault_rotation_tasks.py",
    "backend/sockets/chat.py",
    "backend/services/unified_dispatch/ml/rl_optimizer.py",
    "backend/routes/secret_rotation_monitoring.py",
    "backend/routes/prometheus_metrics.py",
    "backend/tests/conftest.py",
    "backend/tests/routes/test_secret_rotation_monitoring.py",
    "backend/tests/test_safety_guards.py",
    "backend/tests/test_dispatch_integration.py",
    "backend/tests/services/test_secret_rotation_monitor.py",
    "backend/shared/logging_utils.py",
    # Fichiers du module security lui-même (à traiter séparément)
    # "backend/services/security/__init__.py",
    # "backend/services/security/authentication.py",
    # "backend/services/security/pii/__init__.py",
]

BACKEND_DIR = Path(__file__).resolve().parent / "backend"


def fix_imports_in_file(file_path: Path) -> bool:
    """Corrige les imports dans un fichier.
    
    Args:
        file_path: Chemin du fichier à corriger
    
    Returns:
        True si des changements ont été effectués, False sinon
    """
    if not file_path.exists():
        print(f"SKIP: {file_path.relative_to(BACKEND_DIR.parent)} (fichier non trouvé)")
        return False

    content = file_path.read_text(encoding="utf-8")
    original_content = content
    changes_made = False

    for old_pattern, new_replacement in IMPORT_MAPPING.items():
        new_content, count = re.subn(old_pattern, new_replacement, content)
        if count > 0:
            content = new_content
            changes_made = True

    if changes_made:
        file_path.write_text(content, encoding="utf-8")
        return True
    return False


def main():
    print("Demarrage de la correction des imports security (B2)...")
    print(f"Fichiers a traiter: {len(FILES_TO_FIX)}")
    print("=" * 60)

    fixed_count = 0
    skipped_count = 0

    for rel_path in FILES_TO_FIX:
        file_path = BACKEND_DIR.parent / rel_path
        try:
            if fix_imports_in_file(file_path):
                print(f"OK   {rel_path}")
                fixed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"ERROR: {rel_path} - {e}")

    print("=" * 60)
    print(f"Fichiers mis a jour: {fixed_count}")
    print(f"Fichiers sans changement: {skipped_count}")
    print("Correction terminee!")


if __name__ == "__main__":
    main()

