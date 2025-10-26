#!/usr/bin/env python3
"""
Script amélioré pour corriger automatiquement les erreurs G004 (logging format).
"""

import re
import subprocess

def fix_g004_in_file(file_path):
    """Corrige les erreurs G004 dans un fichier."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        original_lines = lines.copy()
        modified = False
        
        for i, line in enumerate(lines):
            # Pattern pour détecter les f-strings dans les logs
            # Exemple: logger.info(f"Message {variable}")
            if re.search(r'\w+\.(?:logger|log|info|debug|warning|error|critical|exception)\s*\(\s*f"', line):
                # Trouver la ligne complète (peut être sur plusieurs lignes)
                full_line = line
                j = i
                paren_count = full_line.count('(') - full_line.count(')')
                
                # Continuer jusqu'à ce que toutes les parenthèses soient fermées
                while paren_count > 0 and j + 1 < len(lines):
                    j += 1
                    full_line += lines[j]
                    paren_count += lines[j].count('(') - lines[j].count(')')
                
                # Extraire le contenu entre les parenthèses
                match = re.search(r'(\w+\.(?:logger|log|info|debug|warning|error|critical|exception))\s*\(\s*(.*)\)', full_line, re.DOTALL)
                if match:
                    logger_name = match.group(1)
                    content = match.group(2).strip()
                    
                    # Si c'est une f-string simple
                    if content.startswith('f"') and content.endswith('"'):
                        f_string = content[2:-1]  # Enlever f" et "
                        
                        # Extraire les variables
                        variables = re.findall(r'\{([^}]+)\}', f_string)
                        
                        # Remplacer les {variable} par %s
                        format_string = f_string
                        for var in variables:
                            format_string = format_string.replace(f'{{{var}}}', '%s')
                        
                        # Créer la nouvelle ligne
                        if variables:
                            args = ', '.join(variables)
                            new_line = f'{logger_name}("{format_string}", {args})\n'
                        else:
                            new_line = f'{logger_name}("{format_string}")\n'
                        
                        # Remplacer les lignes
                        for k in range(i, j + 1):
                            lines[k] = ''
                        lines[i] = new_line
                        modified = True
                    
                    # Si c'est une f-string multi-lignes
                    elif 'f"' in content and '"' in content:
                        # Extraire toutes les parties f-string
                        f_parts = re.findall(r'f"([^"]*)"', content)
                        all_text = ''.join(f_parts)
                        
                        # Extraire les variables
                        variables = re.findall(r'\{([^}]+)\}', all_text)
                        
                        # Créer la chaîne de format
                        format_string = all_text
                        for var in variables:
                            format_string = format_string.replace(f'{{{var}}}', '%s')
                        
                        # Créer la nouvelle ligne
                        if variables:
                            args = ', '.join(variables)
                            new_line = f'{logger_name}("{format_string}", {args})\n'
                        else:
                            new_line = f'{logger_name}("{format_string}")\n'
                        
                        # Remplacer les lignes
                        for k in range(i, j + 1):
                            lines[k] = ''
                        lines[i] = new_line
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
    print("🔧 Correction avancée des erreurs G004 (logging format)...")
    
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
