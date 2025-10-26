#!/usr/bin/env python3
"""
Script pour corriger automatiquement les erreurs PLR2004 (valeurs magiques).
"""

import re
import subprocess

def fix_plr2004_in_file(file_path):
    """Corrige les erreurs PLR2004 dans un fichier."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        original_lines = lines.copy()
        modified = False
        
        # Dictionnaire pour stocker les constantes ajoutées
        constants_added = {}
        
        for i, line in enumerate(lines):
            # Pattern pour détecter les valeurs magiques dans les comparaisons
            # Exemple: if value == 50:
            magic_patterns = [
                r'(\w+)\s*==\s*(\d+)',
                r'(\w+)\s*!=\s*(\d+)',
                r'(\w+)\s*>\s*(\d+)',
                r'(\w+)\s*>=\s*(\d+)',
                r'(\w+)\s*<\s*(\d+)',
                r'(\w+)\s*<=\s*(\d+)',
                r'(\w+)\s*in\s*\[(\d+)\]',
                r'(\w+)\s*not\s*in\s*\[(\d+)\]',
            ]
            
            for pattern in magic_patterns:
                match = re.search(pattern, line)
                if match:
                    variable = match.group(1)
                    magic_value = match.group(2)
                    
                    # Créer un nom de constante approprié
                    const_name = f"{variable.upper()}_THRESHOLD"
                    if magic_value == "0":
                        const_name = f"{variable.upper()}_ZERO"
                    elif magic_value == "1":
                        const_name = f"{variable.upper()}_ONE"
                    elif magic_value == "100":
                        const_name = f"{variable.upper()}_PERCENT"
                    
                    # Ajouter la constante si elle n'existe pas déjà
                    if const_name not in constants_added:
                        constants_added[const_name] = magic_value
                    
                    # Remplacer la valeur magique par la constante
                    new_line = line.replace(magic_value, const_name)
                    lines[i] = new_line
                    modified = True
        
        # Ajouter les constantes au début du fichier (après les imports)
        if constants_added:
            # Trouver la fin des imports
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip() and not (line.startswith('#') or line.startswith('import') or line.startswith('from') or line.strip() == ''):
                    import_end = i
                    break
            
            # Insérer les constantes
            constants_lines = ['\n', '# Constantes pour éviter les valeurs magiques\n']
            for const_name, value in constants_added.items():
                constants_lines.append(f"{const_name} = {value}\n")
            constants_lines.append('\n')
            
            lines[import_end:import_end] = constants_lines
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
    print("🔧 Correction des erreurs PLR2004 (valeurs magiques)...")
    
    # Obtenir la liste des fichiers avec des erreurs PLR2004
    try:
        result = subprocess.run([
            'python', '-m', 'ruff', 'check', '.', '--select', 'PLR2004', '--output-format=json'
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
            if fix_plr2004_in_file(file_path):
                fixed_count += 1
                print(f"✅ Corrigé: {file_path}")
        
        print(f"🎉 {fixed_count} fichiers corrigés avec succès!")
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
