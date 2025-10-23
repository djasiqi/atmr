#!/usr/bin/env python3
"""
Script de monitoring pour l'entraînement RL amélioré.
"""

import time
from pathlib import Path

# pyright: reportMissingImports=false
try:
    import torch  # noqa: F401
except ImportError:
    torch = None


def monitor_training():
    """Surveille l'entraînement RL amélioré."""

    print("=================================================================================")  # noqa: T201
    print("📊 MONITORING ENTRAÎNEMENT RL AMÉLIORÉ")  # noqa: T201
    print("=================================================================================")  # noqa: T201

    model_path = Path("data/rl/models/dispatch_optimized_v3_improved.pth")

    print(f"📂 Modèle cible : {model_path}")  # noqa: T201
    print("📊 Données : 215 dispatches, 2220 bookings")  # noqa: T201
    print("🎯 Objectif : Écart ≤ 1 course")  # noqa: T201
    print()  # noqa: T201

    print("⏳ Surveillance en cours...")  # noqa: T201
    print("   (Ctrl+C pour arrêter)")  # noqa: T201
    print()  # noqa: T201

    try:
        while True:
            # Vérifier si le modèle existe
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                mod_time = time.ctime(model_path.stat().st_mtime)
                print(f"✅ Modèle trouvé : {size_mb:.1f} MB (modifié: {mod_time})")  # noqa: T201

                # Essayer de charger le modèle pour voir les métriques
                try:
                    if torch is not None:
                        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)

                        if 'episode_count' in checkpoint:
                            episodes = checkpoint['episode_count']
                            epsilon = checkpoint.get('epsilon', 'N/A')
                            training_steps = checkpoint.get('training_step', 'N/A')

                            print(f"📈 Progrès : {episodes} épisodes")  # noqa: T201
                            print(f"🎲 Epsilon : {epsilon}")  # noqa: T201
                            print(f"🔢 Training steps : {training_steps}")  # noqa: T201

                            if 'losses' in checkpoint and checkpoint['losses']:
                                recent_losses = checkpoint['losses'][-10:]
                                avg_loss = sum(recent_losses) / len(recent_losses)
                                print(f"📉 Loss récente : {avg_loss:.4f}")  # noqa: T201

                            print()  # noqa: T201
                    else:
                        print("⚠️  PyTorch non disponible")  # noqa: T201
                        print()  # noqa: T201

                except Exception as e:
                    print(f"⚠️  Erreur lecture modèle : {e}")  # noqa: T201
                    print()  # noqa: T201
            else:
                print("⏳ Modèle pas encore créé...")  # noqa: T201
                print()  # noqa: T201

            time.sleep(30)  # Vérifier toutes les 30 secondes

    except KeyboardInterrupt:
        print("\n🛑 Monitoring arrêté")  # noqa: T201


if __name__ == "__main__":
    monitor_training()
