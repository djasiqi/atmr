#!/usr/bin/env python3
"""
Script pour corriger automatiquement les erreurs de formatage de chaînes de caractères.
"""

import re
import subprocess

def fix_formatting_errors_in_file(file_path):
    """Corrige les erreurs de formatage dans un fichier."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        modified = False
        
        # Corriger les erreurs de formatage communes
        
        # 1. Corriger les f-strings mal formées avec :.1f -> %.1f
        content = re.sub(r':(\d+\.?\d*f)', r'%\1', content)
        
        # 2. Corriger les erreurs comme "value:.1f" -> "%.1f" dans les logs
        content = re.sub(r'(\w+):(\d+\.?\d*f)', r'%\2', content)
        
        # 3. Corriger les erreurs comme "001" -> "0.001"
        content = re.sub(r'(\w+)=(\d{3,})', r'\1=0.\2', content)
        
        # 4. Corriger les erreurs d'assignation invalides comme "10 = 9"
        content = re.sub(r'^(\d+)\s*=\s*(\d+)$', r'# \1 = \2  # Constante corrigée', content, flags=re.MULTILINE)
        
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
    print("🔧 Correction automatique des erreurs de formatage...")
    
    # Obtenir la liste des fichiers Python
    try:
        result = subprocess.run([
            'find', '/app', '-name', '*.py', '-type', 'f'
        ], capture_output=True, text=True, cwd='/app')
        
        if result.returncode != 0:
            print(f"Erreur lors de la recherche de fichiers: {result.stderr}")
            return
        
        files = result.stdout.strip().split('\n')
        files = [f for f in files if f and f.endswith('.py')]
        
        print(f"📁 {len(files)} fichiers Python trouvés")
        
        fixed_count = 0
        for file_path in files:
            if fix_formatting_errors_in_file(file_path):
                fixed_count += 1
                print(f"✅ Corrigé: {file_path}")
        
        print(f"🎉 {fixed_count} fichiers corrigés avec succès!")
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
