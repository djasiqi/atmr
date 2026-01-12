#!/usr/bin/env python3
"""
Script de migration automatique des styles shadow* vers shadowPresets
Usage: python3 migrate_shadows_auto.py
"""

import re
import os
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MOBILE_APP_ROOT = PROJECT_ROOT / "mobile" / "operations-app"

# Pattern pour détecter les imports existants
IMPORT_PATTERN = r"^import\s+.*from\s+['\"]react-native['\"];"
SHADOW_STYLES_IMPORT = 'import { shadowPresets } from "@/styles/shadowStyles";'

# Patterns de remplacement pour les styles shadow
SHADOW_PATTERNS = {
    # Large shadows (modals, dropdowns)
    r'shadowColor:\s*"rgba\(15,54,43,0\.15\)",\s*shadowOffset:\s*\{\s*width:\s*0,\s*height:\s*12\s*\},\s*shadowOpacity:\s*1,\s*shadowRadius:\s*24,\s*elevation:\s*8,': 
        '...shadowPresets.large, // ✅ Compatible web/native',
    
    # Medium shadows (cards importantes)
    r'shadowColor:\s*"rgba\(15,54,43,0\.08\)",\s*shadowOffset:\s*\{\s*width:\s*0,\s*height:\s*4\s*\},\s*shadowOpacity:\s*1,\s*shadowRadius:\s*12,\s*elevation:\s*4,': 
        '...shadowPresets.medium, // ✅ Compatible web/native',
    
    # Small shadows (boutons, cards simples)
    r'shadowColor:\s*"rgba\(15,54,43,0\.08\)",\s*shadowOffset:\s*\{\s*width:\s*0,\s*height:\s*2\s*\},\s*shadowOpacity:\s*1,\s*shadowRadius:\s*6,\s*elevation:\s*2,': 
        '...shadowPresets.small, // ✅ Compatible web/native',
    
    r'shadowColor:\s*"#000",\s*shadowOffset:\s*\{\s*width:\s*0,\s*height:\s*1\s*\},\s*shadowOpacity:\s*0\.05,\s*shadowRadius:\s*2,\s*elevation:\s*1,': 
        '...shadowPresets.small, // ✅ Compatible web/native',
    
    # Accent shadows (éléments accentués)
    r'shadowColor:\s*"rgba\(10,127,89,0\.3\)",\s*shadowOffset:\s*\{\s*width:\s*0,\s*height:\s*4\s*\},\s*shadowOpacity:\s*1,\s*shadowRadius:\s*12,\s*elevation:\s*4,': 
        '...shadowPresets.accent, // ✅ Compatible web/native',
}


def add_import_if_missing(content: str) -> str:
    """Ajoute l'import shadowPresets si absent"""
    if "shadowPresets" in content:
        return content
    
    # Trouver la position après les imports React Native
    match = re.search(IMPORT_PATTERN, content, re.MULTILINE)
    if match:
        insert_pos = match.end()
        return content[:insert_pos] + "\n" + SHADOW_STYLES_IMPORT + content[insert_pos:]
    
    return content


def migrate_file(file_path: Path) -> tuple[bool, str]:
    """
    Migre un fichier vers shadowPresets
    Returns: (changed, message)
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content
        
        # Verifier si le fichier contient des shadow styles
        if not any(keyword in content for keyword in ["shadowColor", "shadowOffset", "shadowOpacity", "shadowRadius"]):
            return False, "Pas de styles shadow trouves"
        
        # Appliquer les remplacements
        changes_made = 0
        for pattern, replacement in SHADOW_PATTERNS.items():
            new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
            if count > 0:
                content = new_content
                changes_made += count
        
        if changes_made == 0:
            return False, "Patterns shadow non standard (migration manuelle requise)"
        
        # Ajouter l'import si necessaire
        content = add_import_if_missing(content)
        
        if content != original_content:
            file_path.write_text(content, encoding="utf-8")
            return True, f"OK - {changes_made} style(s) migre(s)"
        
        return False, "Aucun changement"
        
    except Exception as e:
        return False, f"Erreur: {e}"


def find_files_to_migrate():
    """Trouve tous les fichiers .tsx avec shadow styles"""
    files_to_migrate = []
    
    for pattern in ["**/*.tsx", "**/*.ts"]:
        for file_path in MOBILE_APP_ROOT.glob(pattern):
            # Ignorer node_modules, shadowStyles.ts, etc.
            if "node_modules" in str(file_path) or "shadowStyles" in file_path.name:
                continue
            
            try:
                content = file_path.read_text(encoding="utf-8")
                if any(keyword in content for keyword in ["shadowColor", "shadowOffset"]):
                    files_to_migrate.append(file_path)
            except Exception:
                pass
    
    return files_to_migrate


def main():
    print("Recherche des fichiers a migrer...")
    files = find_files_to_migrate()
    
    if not files:
        print("OK - Aucun fichier a migrer !")
        return
    
    print(f"{len(files)} fichiers trouves avec styles shadow*\n")
    
    migrated = 0
    skipped = 0
    errors = 0
    
    for file_path in files:
        rel_path = file_path.relative_to(MOBILE_APP_ROOT)
        changed, message = migrate_file(file_path)
        
        if changed:
            print(f"[OK] {rel_path}: {message}")
            migrated += 1
        elif "Erreur" in message:
            print(f"[ERROR] {rel_path}: {message}")
            errors += 1
        else:
            print(f"[SKIP] {rel_path}: {message}")
            skipped += 1
    
    print(f"\nRESUME:")
    print(f"   Migres: {migrated}")
    print(f"   Ignores: {skipped}")
    print(f"   Erreurs: {errors}")
    print(f"   Total: {len(files)}")
    
    if skipped > 0:
        print(f"\nWARNING: {skipped} fichiers necessitent une migration manuelle")
        print("   (patterns shadow non standard)")


if __name__ == "__main__":
    main()
