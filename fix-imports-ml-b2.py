#!/usr/bin/env python3
"""
Script pour corriger automatiquement les imports après migration B2 - Module ML/RL

Ce script remplace les anciens imports par les nouveaux chemins :
- services.ml_features → services.ml.features
- services.ml_monitoring_service → services.ml.monitoring
- services.ml.demand_prediction → services.ml.models.demand_prediction
- services.ml.eta_delay_model → services.ml.models.eta_delay
- services.ml.model_registry → services.ml.models.registry
- services.ml.training_metadata_schema → services.ml.models.training_metadata
- services.rl.* → services.ml.rl.*

Usage:
    python fix-imports-ml-b2.py
"""

import re
from pathlib import Path

# Mapping des anciens imports vers les nouveaux
IMPORT_MAPPING = {
    # Fichiers racine ML
    r"from services\.ml_features import": r"from services.ml.features import",
    r"from services\.ml_monitoring_service import": r"from services.ml.monitoring import",
    r"import services\.ml_features": r"import services.ml.features",
    r"import services\.ml_monitoring_service": r"import services.ml.monitoring",
    
    # Modèles ML (ancien ml/ vers ml/models/)
    r"from services\.ml\.demand_prediction import": r"from services.ml.models.demand_prediction import",
    r"from services\.ml\.eta_delay_model import": r"from services.ml.models.eta_delay import",
    r"from services\.ml\.model_registry import": r"from services.ml.models.registry import",
    r"from services\.ml\.training_metadata_schema import": r"from services.ml.models.training_metadata import",
    r"import services\.ml\.demand_prediction": r"import services.ml.models.demand_prediction",
    r"import services\.ml\.eta_delay_model": r"import services.ml.models.eta_delay",
    r"import services\.ml\.model_registry": r"import services.ml.models.registry",
    r"import services\.ml\.training_metadata_schema": r"import services.ml.models.training_metadata",
    
    # Module RL (services.rl.* vers services.ml.rl.*)
    r"from services\.rl\.dispatch_env import": r"from services.ml.rl.dispatch_env import",
    r"from services\.rl\.distributional_dqn import": r"from services.ml.rl.distributional_dqn import",
    r"from services\.rl\.hyperparameter_tuner import": r"from services.ml.rl.hyperparameter_tuner import",
    r"from services\.rl\.improved_dqn_agent import": r"from services.ml.rl.improved_dqn_agent import",
    r"from services\.rl\.improved_q_network import": r"from services.ml.rl.improved_q_network import",
    r"from services\.rl\.n_step_buffer import": r"from services.ml.rl.n_step_buffer import",
    r"from services\.rl\.noisy_networks import": r"from services.ml.rl.noisy_networks import",
    r"from services\.rl\.optimal_hyperparameters import": r"from services.ml.rl.optimal_hyperparameters import",
    r"from services\.rl\.replay_buffer import": r"from services.ml.rl.replay_buffer import",
    r"from services\.rl\.reward_shaping import": r"from services.ml.rl.reward_shaping import",
    r"from services\.rl\.rl_logger import": r"from services.ml.rl.rl_logger import",
    r"from services\.rl\.shadow_mode_manager import": r"from services.ml.rl.shadow_mode_manager import",
    r"from services\.rl\.suggestion_generator import": r"from services.ml.rl.suggestion_generator import",
    
    # Imports directs de modules
    r"import services\.rl\.dispatch_env": r"import services.ml.rl.dispatch_env",
    r"import services\.rl\.distributional_dqn": r"import services.ml.rl.distributional_dqn",
    r"import services\.rl\.hyperparameter_tuner": r"import services.ml.rl.hyperparameter_tuner",
    r"import services\.rl\.improved_dqn_agent": r"import services.ml.rl.improved_dqn_agent",
    r"import services\.rl\.improved_q_network": r"import services.ml.rl.improved_q_network",
    r"import services\.rl\.n_step_buffer": r"import services.ml.rl.n_step_buffer",
    r"import services\.rl\.noisy_networks": r"import services.ml.rl.noisy_networks",
    r"import services\.rl\.optimal_hyperparameters": r"import services.ml.rl.optimal_hyperparameters",
    r"import services\.rl\.replay_buffer": r"import services.ml.rl.replay_buffer",
    r"import services\.rl\.reward_shaping": r"import services.ml.rl.reward_shaping",
    r"import services\.rl\.rl_logger": r"import services.ml.rl.rl_logger",
    r"import services\.rl\.shadow_mode_manager": r"import services.ml.rl.shadow_mode_manager",
    r"import services\.rl\.suggestion_generator": r"import services.ml.rl.suggestion_generator",
}

BACKEND_DIR = Path(__file__).resolve().parent / "backend"


def find_files_to_fix() -> list[Path]:
    """Trouve tous les fichiers Python qui contiennent des imports à corriger."""
    files = []
    # Rechercher tous les fichiers Python dans backend
    for file_path in BACKEND_DIR.rglob("*.py"):
        # Éviter le module ml lui-même (on le traitera séparément)
        if "services/ml" in str(file_path).replace("\\", "/"):
            # Mais inclure les fichiers ml qui ont des imports internes
            try:
                content = file_path.read_text(encoding="utf-8")
                # Vérifier si le fichier contient des imports à corriger
                if any(pattern in content for pattern in [
                    "services.ml_features",
                    "services.ml_monitoring_service",
                    "services.ml.demand_prediction",
                    "services.ml.eta_delay_model",
                    "services.ml.model_registry",
                    "services.ml.training_metadata_schema",
                    "services.rl.",
                ]):
                    files.append(file_path)
            except Exception:
                pass
        else:
            # Pour les autres fichiers, vérifier s'ils importent ML/RL
            try:
                content = file_path.read_text(encoding="utf-8")
                if any(pattern in content for pattern in [
                    "services.ml_features",
                    "services.ml_monitoring_service",
                    "services.ml.demand_prediction",
                    "services.ml.eta_delay_model",
                    "services.ml.model_registry",
                    "services.ml.training_metadata_schema",
                    "services.rl.",
                ]):
                    files.append(file_path)
            except Exception:
                pass
    
    return files


def fix_imports_in_file(file_path: Path) -> bool:
    """Corrige les imports dans un fichier."""
    if not file_path.exists():
        return False

    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content
        changes_made = False

        for old_pattern, new_replacement in IMPORT_MAPPING.items():
            new_content, count = re.subn(old_pattern, new_replacement, content)
            if count > 0:
                content = new_content
                changes_made = True

        if changes_made:
            file_path.write_text(content, encoding="utf-8")
            return True
    except Exception as e:
        print(f"ERROR: {file_path} - {e}")
    
    return False


def main():
    print("Demarrage de la correction des imports ML/RL (B2)...")
    print("=" * 60)

    files_to_fix = find_files_to_fix()
    print(f"Fichiers a traiter: {len(files_to_fix)}")
    print("=" * 60)

    fixed_count = 0
    skipped_count = 0

    for file_path in files_to_fix:
        try:
            rel_path = file_path.relative_to(BACKEND_DIR.parent)
            if fix_imports_in_file(file_path):
                print(f"OK   {rel_path}")
                fixed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"ERROR: {file_path} - {e}")

    print("=" * 60)
    print(f"Fichiers mis a jour: {fixed_count}")
    print(f"Fichiers sans changement: {skipped_count}")
    print("Correction terminee!")


if __name__ == "__main__":
    main()

