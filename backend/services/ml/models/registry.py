#!/usr/bin/env python3
# pyright: reportMissingImports=false

# Constantes pour éviter les valeurs magiques
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import torch

MODEL_SIZE_MB_THRESHOLD = 1000
MODEL_AGE_DAYS_THRESHOLD = 30
MIN_VERSIONS_FOR_ROLLBACK = 2

"""Système de registre de modèles MLOps pour l'Étape 13.

Ce module implémente un système complet de gestion des modèles avec :
- Versioning strict des modèles
- Promotion contrôlée (canary)
- Traçabilité complète training → déploiement
- Rollback simple et sécurisé
"""


# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


class ModelMetadata:
    """Métadonnées complètes d'un modèle."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        model_name: str,
        model_arch: str,
        version: str,
        created_at: datetime,
        training_config: Dict[str, Any],
        performance_metrics: Dict[str, float],
        features_config: Dict[str, Any],
        scalers_config: Dict[str, Any],
        optuna_study_id: str | None = None,
        hyperparameters: Dict[str, Any] | None = None,
        dataset_info: Dict[str, Any] | None = None,
        model_size_mb: float | None = None,
        checksum: str | None = None,
    ):
        """Initialise les métadonnées d'un modèle.

        Args:
            model_name: Nom du modèle (ex: "dqn_dispatch")
            model_arch: Architecture du modèle (ex: "dueling_dqn", "c51", "qr_dqn")
            version: Version du modèle (ex: "v1.23")
            created_at: Date de création
            training_config: Configuration d'entraînement
            performance_metrics: Métriques de performance
            features_config: Configuration des features
            scalers_config: Configuration des scalers
            optuna_study_id: ID de l'étude Optuna
            hyperparameters: Hyperparamètres utilisés
            dataset_info: Informations sur le dataset
            model_size_mb: Taille du modèle en MB
            checksum: Checksum du modèle

        """
        self.model_name = model_name
        self.model_arch = model_arch
        self.version = version
        self.created_at = created_at
        self.training_config = training_config
        self.performance_metrics = performance_metrics
        self.features_config = features_config
        self.scalers_config = scalers_config
        self.optuna_study_id = optuna_study_id
        self.hyperparameters = hyperparameters or {}
        self.dataset_info = dataset_info or {}
        self.model_size_mb = model_size_mb
        self.checksum = checksum

    def to_dict(self) -> Dict[str, Any]:
        """Convertit les métadonnées en dictionnaire."""
        return {
            "model_name": self.model_name,
            "model_arch": self.model_arch,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "training_config": self.training_config,
            "performance_metrics": self.performance_metrics,
            "features_config": self.features_config,
            "scalers_config": self.scalers_config,
            "optuna_study_id": self.optuna_study_id,
            "hyperparameters": self.hyperparameters,
            "dataset_info": self.dataset_info,
            "model_size_mb": self.model_size_mb,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Crée des métadonnées à partir d'un dictionnaire."""
        return cls(
            model_name=data["model_name"],
            model_arch=data["model_arch"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            training_config=data["training_config"],
            performance_metrics=data["performance_metrics"],
            features_config=data["features_config"],
            scalers_config=data["scalers_config"],
            optuna_study_id=data.get("optuna_study_id"),
            hyperparameters=data.get("hyperparameters", {}),
            dataset_info=data.get("dataset_info", {}),
            model_size_mb=data.get("model_size_mb"),
            checksum=data.get("checksum"),
        )


class ModelRegistry:
    """Registre de modèles avec gestion des versions et promotion."""

    def __init__(self, registry_path: Path):  # pyright: ignore[reportMissingSuperCall]
        """Initialise le registre de modèles.

        Args:
            registry_path: Chemin vers le répertoire du registre

        """
        self.registry_path = registry_path
        self.models_path = registry_path / "models"
        self.metadata_path = registry_path / "metadata"
        self.current_path = registry_path / "current"

        # Créer les répertoires si nécessaire
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self.current_path.mkdir(parents=True, exist_ok=True)

        # Charger le registre existant
        self.registry_file = registry_path / "registry.json"
        self.registry = self._load_registry()

        # Sauvegarder le registre initial s'il n'existe pas
        if not self.registry_file.exists():
            self._save_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Charge le registre depuis le fichier."""
        if self.registry_file.exists():
            with self.registry_file.open("r", encoding="utf-8") as f:
                return cast(Dict[str, Any], json.load(f))
        return {
            "models": {},
            "current_models": {},
            "promotion_history": [],
            "last_updated": datetime.now(UTC).isoformat(),
        }

    def _save_registry(self):
        """Sauvegarde le registre dans le fichier."""
        self.registry["last_updated"] = datetime.now(UTC).isoformat()
        with self.registry_file.open("w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def register_model(
        self,
        model: torch.nn.Module,
        metadata: ModelMetadata,
        _model_file_path: Path | None = None,
    ) -> Path:
        """Enregistre un nouveau modèle dans le registre.

        Args:
            model: Modèle PyTorch à enregistrer
            metadata: Métadonnées du modèle
            model_file_path: Chemin optionnel pour le fichier du modèle

        Returns:
            Chemin vers le fichier du modèle enregistré

        """
        # Générer le nom de fichier
        model_filename = (
            f"{metadata.model_name}_{metadata.model_arch}_{metadata.version}.pth"
        )
        model_path = self.models_path / model_filename

        # Sauvegarder le modèle
        torch.save(model.state_dict(), model_path)

        # Calculer la taille et le checksum
        metadata.model_size_mb = model_path.stat().st_size / (1024 * 1024)
        metadata.checksum = self._calculate_checksum(model_path)

        # Sauvegarder les métadonnées
        metadata_filename = (
            f"{metadata.model_name}_{metadata.model_arch}_{metadata.version}.json"
        )
        metadata_path = self.metadata_path / metadata_filename

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

        # Mettre à jour le registre
        model_key = f"{metadata.model_name}_{metadata.model_arch}"
        if model_key not in self.registry["models"]:
            self.registry["models"][model_key] = []

        self.registry["models"][model_key].append(
            {
                "version": metadata.version,
                "created_at": metadata.created_at.isoformat(),
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "performance_metrics": metadata.performance_metrics,
                "model_size_mb": metadata.model_size_mb,
                "checksum": metadata.checksum,
            }
        )

        # Trier par date de création (plus récent en premier)
        self.registry["models"][model_key].sort(
            key=lambda x: x["created_at"], reverse=True
        )

        self._save_registry()

        return model_path

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calcule le checksum d'un fichier
        (SHA-256 au lieu de MD5 pour meilleures pratiques)."""
        import hashlib

        hash_sha256 = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def get_model_versions(
        self, model_name: str, model_arch: str
    ) -> List[Dict[str, Any]]:
        """Obtient toutes les versions d'un modèle.

        Args:
            model_name: Nom du modèle
            model_arch: Architecture du modèle

        Returns:
            Liste des versions du modèle

        """
        model_key = f"{model_name}_{model_arch}"
        return cast(List[Dict[str, Any]], self.registry["models"].get(model_key, []))

    def get_latest_model(
        self, model_name: str, model_arch: str
    ) -> Dict[str, Any] | None:
        """Obtient la dernière version d'un modèle.

        Args:
            model_name: Nom du modèle
            model_arch: Architecture du modèle

        Returns:
            Dernière version du modèle ou None

        """
        versions = self.get_model_versions(model_name, model_arch)
        return versions[0] if versions else None

    def promote_model(
        self,
        model_name: str,
        model_arch: str,
        version: str,
        kpi_thresholds: Dict[str, float],
        force: bool = False,
    ) -> bool:
        """Promouvoit un modèle vers la production (canary promotion).

        Args:
            model_name: Nom du modèle
            model_arch: Architecture du modèle
            version: Version à promouvoir
            kpi_thresholds: Seuils KPI pour la promotion
            force: Forcer la promotion sans validation KPI

        Returns:
            True si la promotion a réussi

        """
        # Trouver le modèle
        model_key = f"{model_name}_{model_arch}"
        versions = self.get_model_versions(model_name, model_arch)

        target_model = None
        for model in versions:
            if model["version"] == version:
                target_model = model
                break

        if not target_model:
            msg = f"Modèle {model_name}_{model_arch} version {version} non trouvé"
            raise ValueError(msg)

        # Charger les métadonnées complètes
        metadata_path = Path(target_model["metadata_path"])
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata_data = json.load(f)

        metadata = ModelMetadata.from_dict(metadata_data)

        # Valider les KPIs si pas forcé
        if not force and not self._validate_kpis(
            metadata.performance_metrics, kpi_thresholds
        ):
            return False

        # Créer le lien symbolique vers le modèle actuel
        current_model_path = self.current_path / f"{model_name}_{model_arch}.pth"
        if current_model_path.exists():
            current_model_path.unlink()

        # Copier le modèle vers current
        shutil.copy2(Path(target_model["model_path"]), current_model_path)

        # Mettre à jour le registre
        self.registry["current_models"][model_key] = {
            "version": version,
            "promoted_at": datetime.now(UTC).isoformat(),
            "model_path": str(current_model_path),
            "metadata_path": str(metadata_path),
            "performance_metrics": metadata.performance_metrics,
            "kpi_thresholds": kpi_thresholds,
        }

        # Ajouter à l'historique de promotion
        self.registry["promotion_history"].append(
            {
                "model_name": model_name,
                "model_arch": model_arch,
                "version": version,
                "promoted_at": datetime.now(UTC).isoformat(),
                "performance_metrics": metadata.performance_metrics,
                "kpi_thresholds": kpi_thresholds,
                "forced": force,
            }
        )

        self._save_registry()

        return True

    def _validate_kpis(
        self, performance_metrics: Dict[str, float], thresholds: Dict[str, float]
    ) -> bool:
        """Valide que les métriques de performance respectent les seuils.

        Args:
            performance_metrics: Métriques de performance du modèle
            thresholds: Seuils KPI requis

        Returns:
            True si tous les seuils sont respectés

        """
        for metric_name, threshold in thresholds.items():
            if metric_name not in performance_metrics:
                print(f"⚠️ Métrique {metric_name} manquante dans les performances")
                return False

            metric_value = performance_metrics[metric_name]

            # Pour les métriques de qualité (ponctualité, etc.), on veut >=
            # seuil
            if metric_name in ["punctuality_rate", "accuracy", "f1_score"]:
                if metric_value < threshold:
                    print(f"❌ {metric_name}: {metric_value:.2f} < {threshold:.2f}")
                    return False

            # Pour les métriques de coût (distance, retards), on veut <= seuil
            elif metric_name in ["avg_distance", "avg_delay", "cost"]:
                if metric_value > threshold:
                    print(f"❌ {metric_name}: {metric_value:.2f} > {threshold:.2f}")
                    return False

            else:
                print("⚠️ Type de métrique {metric_name} non reconnu")
                return False

        return True

    def rollback_model(
        self, model_name: str, model_arch: str, target_version: str | None = None
    ) -> bool:
        """Effectue un rollback vers une version précédente.

        Args:
            model_name: Nom du modèle
            model_arch: Architecture du modèle
            target_version: Version cible (si None, utilise la version précédente)

        Returns:
            True si le rollback a réussi

        """
        model_key = f"{model_name}_{model_arch}"

        if model_key not in self.registry["current_models"]:
            print("❌ Aucun modèle actuel trouvé pour {model_key}")
            return False

        current_version = self.registry["current_models"][model_key]["version"]

        # Si pas de version cible, utiliser la version précédente
        if target_version is None:
            versions = self.get_model_versions(model_name, model_arch)
            if len(versions) < MIN_VERSIONS_FOR_ROLLBACK:
                print(f"❌ Pas de version précédente disponible pour {model_key}")
                return False

            # Trouver la version précédente (pas la version actuelle)
            for version in versions[1:]:  # Skip la première (actuelle)
                if version["version"] != current_version:
                    target_version = version["version"]
                    break

        if target_version is None:
            print("❌ Version cible non trouvée pour {model_key}")
            return False

        # Promouvoir la version cible
        return self.promote_model(
            model_name, model_arch, target_version, kpi_thresholds={}, force=True
        )

    def get_current_model(
        self, model_name: str, model_arch: str
    ) -> Dict[str, Any] | None:
        """Obtient le modèle actuellement en production.

        Args:
            model_name: Nom du modèle
            model_arch: Architecture du modèle

        Returns:
            Informations du modèle actuel ou None

        """
        model_key = f"{model_name}_{model_arch}"
        return cast(
            Dict[str, Any] | None, self.registry["current_models"].get(model_key)
        )

    def list_models(self) -> Dict[str, List[str]]:
        """Liste tous les modèles disponibles.

        Returns:
            Dictionnaire des modèles par architecture

        """
        result: dict[str, list[str]] = {}
        for model_key, _versions in self.registry["models"].items():
            model_name, model_arch = model_key.rsplit("_", 1)
            if model_name not in result:
                result[model_name] = []
            result[model_name].append(model_arch)
        return result

    def get_promotion_history(self) -> List[Dict[str, Any]]:
        """Obtient l'historique des promotions.

        Returns:
            Historique des promotions

        """
        return cast(List[Dict[str, Any]], self.registry["promotion_history"])

    def cleanup_old_versions(
        self, model_name: str, model_arch: str, keep_versions: int = 5
    ):
        """Nettoie les anciennes versions d'un modèle.

        Args:
            model_name: Nom du modèle
            model_arch: Architecture du modèle
            keep_versions: Nombre de versions à conserver

        """
        model_key = f"{model_name}_{model_arch}"
        versions = self.get_model_versions(model_name, model_arch)

        if len(versions) <= keep_versions:
            return

        # Supprimer les versions anciennes
        versions_to_remove = versions[keep_versions:]

        for version in versions_to_remove:
            # Supprimer le fichier du modèle
            model_path = Path(version["model_path"])
            if model_path.exists():
                model_path.unlink()

            # Supprimer le fichier de métadonnées
            metadata_path = Path(version["metadata_path"])
            if metadata_path.exists():
                metadata_path.unlink()

        # Mettre à jour le registre
        self.registry["models"][model_key] = versions[:keep_versions]
        self._save_registry()

        print(f"🧹 Nettoyage terminé: {len(versions_to_remove)} versions supprimées")


class ModelPromotionValidator:
    """Validateur pour la promotion de modèles."""

    def __init__(self, registry: ModelRegistry):
        """Initialise le validateur.

        Args:
            registry: Registre de modèles

        """
        super().__init__()  # Appel explicite pour object
        self.registry = registry

    def validate_model_for_promotion(
        self,
        model_name: str,
        model_arch: str,
        version: str,
        kpi_thresholds: Dict[str, float],
    ) -> Tuple[bool, List[str]]:
        """Valide qu'un modèle peut être promu.

        Args:
            model_name: Nom du modèle
            model_arch: Architecture du modèle
            version: Version à valider
            kpi_thresholds: Seuils KPI

        Returns:
            Tuple (is_valid, list_of_issues)

        """
        issues = []

        # Vérifier que le modèle existe
        versions = self.registry.get_model_versions(model_name, model_arch)
        target_model = None
        for model in versions:
            if model["version"] == version:
                target_model = model
                break

        if not target_model:
            issues.append(
                f"Modèle {model_name}_{model_arch} version {version} non trouvé"
            )
            return False, issues

        # Charger les métadonnées
        metadata_path = Path(target_model["metadata_path"])
        if not metadata_path.exists():
            issues.append(f"Fichier de métadonnées manquant: {metadata_path}")
            return False, issues

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata_data = json.load(f)

        metadata = ModelMetadata.from_dict(metadata_data)

        # Valider les KPIs
        if not self.registry._validate_kpis(
            metadata.performance_metrics, kpi_thresholds
        ):
            issues.append(
                "Les métriques de performance ne respectent pas les seuils KPI"
            )

        # Vérifier la taille du modèle
        if (
            metadata.model_size_mb and metadata.model_size_mb > MODEL_SIZE_MB_THRESHOLD
        ):  # 1GB
            issues.append(f"Modèle trop volumineux: {metadata.model_size_mb:.1f} MB")

        # Vérifier l'âge du modèle
        model_age_days = (datetime.now(UTC) - metadata.created_at).days
        if model_age_days > MODEL_AGE_DAYS_THRESHOLD:
            issues.append(f"Modèle trop ancien: {model_age_days} jours")

        return len(issues) == 0, issues


def create_model_registry(registry_path: Path) -> ModelRegistry:
    """Factory function pour créer un registre de modèles.

    Args:
        registry_path: Chemin vers le répertoire du registre

    Returns:
        Instance du registre de modèles

    """
    return ModelRegistry(registry_path)
