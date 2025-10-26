#!/usr/bin/env python3
"""
Script pour corriger automatiquement les erreurs G004 (logging format) dans tous les fichiers.
"""

import re
import subprocess

def fix_g004_in_file(file_path):
    """Corrige les erreurs G004 dans un fichier."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern pour détecter les f-strings dans les logs
        # Exemple: logger.info(f"Message {variable}")
        pattern = r'(\w+\.(?:logger|log|info|debug|warning|error|critical|exception))\(f"([^"]*)"\)'
        
        def replace_log_call(match):
            logger_name = match.group(1)
            f_string = match.group(2)
            
            # Extraire les variables de la f-string
            variables = re.findall(r'\{([^}]+)\}', f_string)
            
            # Remplacer les {variable} par %s
            format_string = f_string
            for var in variables:
                format_string = format_string.replace(f'{{{var}}}', '%s')
            
            # Créer la nouvelle ligne
            if variables:
                args = ', '.join(variables)
                return f'{logger_name}("{format_string}", {args})'
            else:
                return f'{logger_name}("{format_string}")'
        
        # Appliquer les corrections
        content = re.sub(pattern, replace_log_call, content)
        
        # Pattern pour les logs multi-lignes avec f-strings
        # Exemple: logger.info(f"Line 1 {var1} " f"Line 2 {var2}")
        multiline_pattern = r'(\w+\.(?:logger|log|info|debug|warning|error|critical|exception))\(\s*f"([^"]*)"\s*\+\s*f"([^"]*)"\s*\)'
        
        def replace_multiline_log(match):
            logger_name = match.group(1)
            line1 = match.group(2)
            line2 = match.group(3)
            
            # Extraire les variables
            all_vars = re.findall(r'\{([^}]+)\}', line1 + line2)
            
            # Créer la chaîne de format
            format_string = line1 + line2
            for var in all_vars:
                format_string = format_string.replace(f'{{{var}}}', '%s')
            
            if all_vars:
                args = ', '.join(all_vars)
                return f'{logger_name}("{format_string}", {args})'
            else:
                return f'{logger_name}("{format_string}")'
        
        content = re.sub(multiline_pattern, replace_multiline_log, content)
        
        # Si le contenu a changé, sauvegarder
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Erreur lors du traitement de {file_path}: {e}")
        return False

def main():
    """Fonction principale."""
    print("🔧 Correction des erreurs G004 (logging format)...")
    
    # Obtenir la liste des fichiers avec des erreurs G004
    try:
        result = subprocess.run([
            'python', '-m', 'ruff', 'check', '.', '--select', 'G004', '--output-format=json'
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
            if fix_g004_in_file(file_path):
                fixed_count += 1
                print(f"✅ Corrigé: {file_path}")
        
        print(f"🎉 {fixed_count} fichiers corrigés avec succès!")
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
