"""Script de validation des tests Locust.

Vérifie que les fichiers de test peuvent être importés et n'ont pas d'erreurs de syntaxe.
"""

import sys
from pathlib import Path

# Ajouter le répertoire backend au PYTHONPATH
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))


def validate_test_file(filepath: str) -> tuple[bool, str]:
    """Valider qu'un fichier de test peut être importé."""
    try:
        # Importer le module
        import importlib.util

        spec = importlib.util.spec_from_file_location("test_module", filepath)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True, f"[OK] {filepath} - Syntaxe OK"
        return False, f"[ERROR] {filepath} - Impossible de charger le module"
    except Exception as e:
        return False, f"[ERROR] {filepath} - Erreur: {e}"


def main() -> None:
    """Valider tous les fichiers de test."""
    test_dir = Path(__file__).parent
    test_files = [
        test_dir / "dispatch_load_test.py",
        test_dir / "multi_company_test.py",
        test_dir / "slow_osrm_test.py",
    ]

    print("=" * 80)
    print("Validation Tests Locust")
    print("=" * 80)
    print()

    all_valid = True
    for test_file in test_files:
        valid, message = validate_test_file(str(test_file))
        print(message)
        if not valid:
            all_valid = False

    print()
    print("=" * 80)
    if all_valid:
        print("[OK] SUCCES : Tous les tests sont syntaxiquement corrects !")
        print()
        print("[INFO] Note : Pour executer les tests, CSRF doit etre configure.")
        print("       Voir README.md pour les instructions completes.")
    else:
        print("[ERROR] ECHEC : Certains tests contiennent des erreurs")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()
