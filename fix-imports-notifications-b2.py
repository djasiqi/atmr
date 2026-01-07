#!/usr/bin/env python3
"""
Script pour corriger automatiquement les imports après migration B2 - Module notifications

Ce script remplace les anciens imports par les nouveaux chemins :
- services.notification_service → services.notifications.core
- services.push_service → services.notifications.push
- services.alerting_service → services.notifications.system
- services.proactive_alerts → services.notifications.proactive
- services.interfaces.notification_interface → services.notifications.interfaces

Usage:
    python fix-imports-notifications-b2.py
"""

import re
from pathlib import Path

# Mapping des anciens imports vers les nouveaux
IMPORT_MAPPING = {
    r"from services\.notification_service import": r"from services.notifications.core import",
    r"from services\.push_service import": r"from services.notifications.push import",
    r"from services\.alerting_service import": r"from services.notifications.system import",
    r"from services\.proactive_alerts import": r"from services.notifications.proactive import",
    r"from services\.interfaces\.notification_interface import": r"from services.notifications.interfaces import",
    # Imports directs des modules
    r"import services\.notification_service": r"import services.notifications.core",
    r"import services\.push_service": r"import services.notifications.push",
    r"import services\.alerting_service": r"import services.notifications.system",
    r"import services\.proactive_alerts": r"import services.notifications.proactive",
    r"import services\.interfaces\.notification_interface": r"import services.notifications.interfaces",
}

BACKEND_DIR = Path(__file__).resolve().parent / "backend"


def find_files_to_fix() -> list[Path]:
    """Trouve tous les fichiers Python qui peuvent contenir des imports à corriger.
    
    Returns:
        Liste des fichiers Python à analyser
    """
    files = []
    # Rechercher dans backend
    for pattern in [
        "from services.notification_service",
        "from services.push_service",
        "from services.alerting_service",
        "from services.proactive_alerts",
        "from services.interfaces.notification_interface",
    ]:
        # Utiliser grep pour trouver les fichiers
        import subprocess
        try:
            result = subprocess.run(
                ["grep", "-r", "-l", "--include=*.py", pattern, str(BACKEND_DIR)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        file_path = Path(line)
                        if file_path.exists() and file_path not in files:
                            files.append(file_path)
        except FileNotFoundError:
            # grep n'est pas disponible, utiliser une méthode Python pure
            pass
    
    # Si grep n'est pas disponible, chercher manuellement
    if not files:
        for file_path in BACKEND_DIR.rglob("*.py"):
            if "notifications" not in str(file_path):  # Éviter le module lui-même
                try:
                    content = file_path.read_text(encoding="utf-8")
                    if any(pattern in content for pattern in [
                        "services.notification_service",
                        "services.push_service",
                        "services.alerting_service",
                        "services.proactive_alerts",
                        "services.interfaces.notification_interface",
                    ]):
                        files.append(file_path)
                except Exception:
                    pass
    
    return files


def fix_imports_in_file(file_path: Path) -> bool:
    """Corrige les imports dans un fichier.
    
    Args:
        file_path: Chemin du fichier à corriger
    
    Returns:
        True si des changements ont été effectués, False sinon
    """
    if not file_path.exists():
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
    print("Demarrage de la correction des imports notifications (B2)...")
    print("=" * 60)

    files_to_fix = find_files_to_fix()
    print(f"Fichiers a traiter: {len(files_to_fix)}")

    fixed_count = 0
    skipped_count = 0

    for file_path in files_to_fix:
        try:
            rel_path = file_path.relative_to(BACKEND_DIR.parent)
            if fix_imports_in_file(file_path):
                print(f"OK   {rel_path}")
                fixed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"ERROR: {file_path} - {e}")

    print("=" * 60)
    print(f"Fichiers mis a jour: {fixed_count}")
    print(f"Fichiers sans changement: {skipped_count}")
    print("Correction terminee!")


if __name__ == "__main__":
    main()

