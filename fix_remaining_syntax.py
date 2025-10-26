#!/usr/bin/env python3
"""
Script pour corriger les erreurs de syntaxe restantes dans le projet.
"""

import os
import re

def fix_syntax_errors():
    """Corrige les erreurs de syntaxe restantes."""
    print("🔧 Correction des erreurs de syntaxe restantes...")
    
    # Patterns de correction
    patterns = [
        # Corriger les littéraux décimaux invalides (001 -> 0.001, 005 -> 0.005, etc.)
        (r'\b001\b', '0.001'),
        (r'\b005\b', '0.005'),
        (r'\b0001\b', '0.0001'),
        (r'\b01\b', '0.1'),
        
        # Corriger les nombres décimaux malformés (0.132.5 -> 0.1325)
        (r'(\d+\.\d+)\.(\d+)', r'\1\2'),
        
        # Corriger les expressions mathématiques malformées
        (r'(\d+)\*(\d+)\*(\d+)', r'\1 * \2 * \3'),
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
                content = re.sub(pattern, replacement, content)
            
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
