#!/usr/bin/env python3
"""
Script pour corriger automatiquement les erreurs d'indentation dans les fichiers Python.
"""

import subprocess

def fix_indentation_errors_in_file(file_path):
    """Corrige les erreurs d'indentation dans un fichier."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        original_lines = lines.copy()
        modified = False
        
        for i, line in enumerate(lines):
            # Corriger les lignes qui ne sont pas indentées correctement après except
            if i > 0 and lines[i-1].strip().startswith('except') and not line.strip().startswith('#'):
                # Si la ligne n'est pas indentée correctement
                if line.strip() and not line.startswith('        '):
                    # Indenter correctement (8 espaces)
                    lines[i] = '        ' + line.lstrip()
                    modified = True
        
        # Si le contenu a changé, sauvegarder
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        
        return False
        
    except Exception as e:
        print(f"Erreur lors du traitement de {file_path}: {e}")
        return False

def main():
    """Fonction principale."""
    print("🔧 Correction des erreurs d'indentation...")
    
    # Obtenir la liste des fichiers avec des erreurs de syntaxe
    try:
        result = subprocess.run([
            'python', '-m', 'ruff', 'check', '.', '--select', 'SIM117', '--output-format=json'
        ], capture_output=True, text=True, cwd='/app')
        
        if result.returncode != 0 and not result.stdout:
            print(f"Erreur lors de l'exécution de ruff: {result.stderr}")
            return
        
        errors = []
        if result.stdout:
            import json
            errors = json.loads(result.stdout)
        
        # Grouper par fichier
        files_to_fix = set()
        for error in errors:
            files_to_fix.add(error['filename'])
        
        print(f"📁 {len(files_to_fix)} fichiers à corriger")
        
        fixed_count = 0
        for file_path in files_to_fix:
            if fix_indentation_errors_in_file(file_path):
                fixed_count += 1
                print(f"✅ Corrigé: {file_path}")
        
        print(f"🎉 {fixed_count} fichiers corrigés avec succès!")
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
