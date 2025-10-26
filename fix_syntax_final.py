#!/usr/bin/env python3
"""
Script pour corriger les erreurs de syntaxe introduites par les corrections précédentes.
"""

import os
import re

def fix_syntax_errors():
    """Corrige les erreurs de syntaxe introduites."""
    print("🔧 Correction des erreurs de syntaxe...")
    
    # Patterns de correction
    patterns = [
        # Corriger les f-strings malformées
        (r'f"([^"]*):\.([^"]*)"', r'f"\1"'),
        (r'f"([^"]*):\.([^"]*)"', r'f"\1"'),
        
        # Corriger les erreurs d'indentation
        (r'except ImportError:\n        DispatchEnv = None\n    ImprovedDQNAgent = None', 
         'except ImportError:\n    DispatchEnv = None\n    ImprovedDQNAgent = None'),
        
        # Corriger les erreurs de formatage dans les logs
        (r'logger\.info\("([^"]*)", ([^)]*):\.([^)]*)\)', r'logger.info("\1", \2)'),
        
        # Corriger les erreurs de formatage dans les f-strings
        (r'f"([^"]*):\.([^"]*)"', r'f"\1"'),
    ]
    
    # Trouver tous les fichiers Python
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Ignorer les dossiers venv et __pycache__
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git']]
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"📁 {len(python_files)} fichiers Python trouvés")
    
    fixed_count = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Appliquer les corrections
            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            # Si le contenu a changé, sauvegarder
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Corrigé: {file_path}")
                fixed_count += 1
                
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {file_path}: {e}")
    
    print(f"🎉 {fixed_count} fichiers corrigés avec succès!")

if __name__ == "__main__":
    fix_syntax_errors()
