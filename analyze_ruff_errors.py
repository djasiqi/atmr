#!/usr/bin/env python3
"""Script pour analyser les erreurs Ruff et identifier les fichiers problématiques."""

import json
import subprocess
from collections import defaultdict

def analyze_ruff_errors():
    """Analyse les erreurs Ruff et retourne les statistiques."""
    try:
        # Exécuter ruff check et capturer la sortie JSON
        result = subprocess.run([
            'python', '-m', 'ruff', 'check', '.', '--output-format=json'
        ], capture_output=True, text=True, cwd='/app')
        
        if result.returncode != 0 and not result.stdout:
            print(f"Erreur lors de l'exécution de ruff: {result.stderr}")
            return
        
        # Parser le JSON
        errors = json.loads(result.stdout)
        
        # Compter les erreurs par fichier
        files_count = defaultdict(int)
        error_types = defaultdict(int)
        
        for error in errors:
            filename = error.get('filename', 'unknown')
            code = error.get('code', 'unknown')
            files_count[filename] += 1
            error_types[code] += 1
        
        # Afficher les fichiers avec le plus d'erreurs
        print("=== TOP 20 FICHIERS AVEC LE PLUS D'ERREURS ===")
        sorted_files = sorted(files_count.items(), key=lambda x: x[1], reverse=True)
        for i, (filename, count) in enumerate(sorted_files[:20], 1):
            print(f"{i:2d}. {count:4d} erreurs - {filename}")
        
        print(f"\nTotal fichiers avec erreurs: {len(files_count)}")
        print(f"Total erreurs: {sum(files_count.values())}")
        
        # Afficher les types d'erreurs les plus fréquents
        print("\n=== TOP 15 TYPES D'ERREURS LES PLUS FRÉQUENTS ===")
        sorted_types = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        for i, (code, count) in enumerate(sorted_types[:15], 1):
            print(f"{i:2d}. {code:8s} - {count:4d} occurrences")
        
        return files_count, error_types
        
    except Exception as e:
        print(f"Erreur lors de l'analyse: {e}")
        return None, None

if __name__ == "__main__":
    analyze_ruff_errors()
