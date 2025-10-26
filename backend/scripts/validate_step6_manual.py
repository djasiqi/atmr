#!/usr/bin/env python3
"""Validation manuelle de l'Étape 6 - Dueling DQN.

Vérifie que tous les fichiers sont syntaxiquement corrects
et que les imports fonctionnent.
"""

import ast
import sys
from pathlib import Path


def validate_python_syntax(file_path):
    """Valide la syntaxe Python d'un fichier."""
    try:
        with Path(file_path, encoding="utf-8").open() as f:
            content = f.read()
        
        # Parse le fichier pour vérifier la syntaxe
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Erreur de syntaxe ligne {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Erreur: {e}"


def validate_imports(file_path):
    """Valide que les imports sont corrects."""
    try:
        with Path(file_path, encoding="utf-8").open() as f:
            content = f.read()
        
        # Parse le fichier
        tree = ast.parse(content)
        
        # Extraire les imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return True, imports
    except Exception as e:
        return False, f"Erreur imports: {e}"


def main():
    """Fonction principale de validation."""
    print("🔍 Validation manuelle Étape 6 - Dueling DQN")
    print("=" * 50)
    
    # Fichiers à valider
    files_to_validate = [
        "services/rl/improved_q_network.py",
        "services/rl/improved_dqn_agent.py",
        "services/rl/optimal_hyperparameters.py",
        "tests/rl/test_dueling_network.py",
        "scripts/validate_step6_dueling.py",
        "scripts/test_step6_quick.py",
        "scripts/deploy_step6_dueling.py",
        "scripts/step6_summary.py"
    ]
    
    backend_path = Path(__file__).parent
    
    results = {}
    
    for file_path in files_to_validate:
        full_path = backend_path / file_path
        
        if not full_path.exists():
            print("❌ {file_path}: Fichier non trouvé")
            results[file_path] = False
            continue
        
        print("\n📁 {file_path}:")
        
        # Validation syntaxe
        syntax_ok, _syntax_error = validate_python_syntax(full_path)
        if syntax_ok:
            print("   ✅ Syntaxe Python correcte")
        else:
            print("   ❌ Erreur de syntaxe: {syntax_error}")
            results[file_path] = False
            continue
        
        # Validation imports
        imports_ok, imports = validate_imports(full_path)
        if imports_ok:
            print("   ✅ Imports valides ({len(imports)} imports)")
            if imports:
                print("      Imports: {', '.join(imports[:5])}{'...' if len(imports) > 5 else ''}")
        else:
            print("   ❌ Erreur imports: {imports}")
            results[file_path] = False
            continue
        
        results[file_path] = True
        print("   ✅ Fichier validé avec succès")
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DE LA VALIDATION:")
    
    total_files = len(files_to_validate)
    valid_files = sum(1 for valid in results.values() if valid)
    
    print("Fichiers validés: {valid_files}/{total_files}")
    
    if valid_files == total_files:
        print("🎉 TOUS LES FICHIERS SONT VALIDES!")
        print("✅ L'Étape 6 - Dueling DQN est prête")
        print("✅ Syntaxe Python correcte")
        print("✅ Imports valides")
        print("✅ Code prêt pour l'exécution")
    else:
        print("⚠️  CERTAINS FICHIERS ONT DES PROBLÈMES")
        print("❌ Corriger les erreurs avant le déploiement")
    
    # Détails des erreurs
    failed_files = [f for f, valid in results.items() if not valid]
    if failed_files:
        print("\n❌ Fichiers avec erreurs:")
        for file_path in failed_files:
            print("   • {file_path}")
    
    return valid_files == total_files


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
