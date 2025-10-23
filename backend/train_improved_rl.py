#!/usr/bin/env python3
"""
Script d'entraînement RL amélioré avec toutes les données disponibles.
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.rl_train_offline import train_offline

def train_improved_rl():
    """Lance l'entraînement RL avec toutes les données disponibles."""
    
    print("=================================================================================")
    print("🚀 ENTRAÎNEMENT RL AMÉLIORÉ - TOUTES LES DONNÉES")
    print("=================================================================================")
    
    # Configuration améliorée
    config = {
        "historical_data_file": "data/rl/historical_dispatches_corrected.json",
        "num_episodes": 25000,  # Plus d'épisodes pour convergence
        "save_path": "data/rl/models/dispatch_optimized_v3_improved.pth",
        "learning_rate": 0.00005,  # Learning rate réduit pour stabilité
        "batch_size": 128,  # Batch size augmenté
        "target_update_freq": 50,  # Mise à jour plus fréquente
    }
    
    print("📊 Configuration d'entraînement:")
    for key, value in config.items():
        print(f"   - {key}: {value}")
    print()
    
    # Lancer l'entraînement
    try:
        train_offline(**config)
        print("\n🎉 Entraînement amélioré terminé avec succès !")
        return True
    except Exception as e:
        print(f"\n❌ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_improved_rl()
    sys.exit(0 if success else 1)
