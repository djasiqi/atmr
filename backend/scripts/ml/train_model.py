"""
Script d'entraînement du modèle ML de prédiction de retards.

Entraîne un RandomForestRegressor et évalue ses performances.

Usage:
    python scripts/ml/train_model.py [--train data/ml/train_data.csv] [--test data/ml/test_data.csv]
"""
# ruff: noqa: T201, N803
# pyright: reportArgumentType=false, reportReturnType=false, reportOperatorIssue=false
# print() est intentionnel dans les scripts ML
# X_train, X_test = convention ML (ignorer N803)

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def load_datasets(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les datasets train et test."""
    print("\n" + "="*70)
    print("📂 CHARGEMENT DES DATASETS")
    print("="*70)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"✅ Train: {train_df.shape}")
    print(f"✅ Test:  {test_df.shape}")

    return train_df, test_df


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = 'actual_delay_minutes',
    exclude_cols: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les features et le target.

    Args:
        df: DataFrame complet
        target_col: Nom de la colonne target
        exclude_cols: Colonnes à exclure des features (IDs, etc.)

    Returns:
        Tuple (X features, y target)
    """
    if exclude_cols is None:
        exclude_cols = [
            'booking_id', 'driver_id', 'assignment_id', 'company_id',
            target_col
        ]

    # Features = toutes colonnes sauf target et IDs
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]

    print("\n📊 Features préparées:")
    print(f"   Features: {X.shape[1]}")
    print(f"   Échantillons: {X.shape[0]}")
    print(f"   Target: {target_col}")

    return X, y  # type: ignore[return-value]


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42
) -> RandomForestRegressor:
    """
    Entraîne un Random Forest Regressor.

    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        n_estimators: Nombre d'arbres (défaut: 100)
        max_depth: Profondeur max des arbres (None = illimité)
        random_state: Seed pour reproductibilité

    Returns:
        Modèle entraîné
    """
    print("\n" + "="*70)
    print("🌳 ENTRAÎNEMENT RANDOM FOREST")
    print("="*70)
    print("Paramètres:")
    print(f"   n_estimators: {n_estimators}")
    print(f"   max_depth: {max_depth or 'Illimité'}")
    print(f"   random_state: {random_state}")
    print("="*70)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,  # Utiliser tous les CPUs
        verbose=0
    )

    print("\n⏱️ Entraînement en cours...")
    start_time = time.time()

    model.fit(X_train, y_train)

    elapsed = time.time() - start_time
    print(f"✅ Entraînement terminé en {elapsed:.2f}s")

    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Évalue les performances du modèle.

    Métriques calculées:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R² score
    - Temps de prédiction

    Returns:
        Dict avec toutes les métriques
    """
    print("\n" + "="*70)
    print("📊 ÉVALUATION DU MODÈLE")
    print("="*70)

    # Prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Métriques Train
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Métriques Test
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    # Temps de prédiction (moyenne sur 100 prédictions)
    sample = X_test.iloc[:100] if len(X_test) >= 100 else X_test
    start = time.time()
    for _ in range(100):
        model.predict(sample)
    avg_pred_time = (time.time() - start) / 100 * 1000  # en ms

    print("\n🎯 MÉTRIQUES TRAIN SET:")
    print(f"   MAE  : {train_mae:.2f} min")
    print(f"   RMSE : {train_rmse:.2f} min")
    print(f"   R²   : {train_r2:.4f}")

    print("\n🎯 MÉTRIQUES TEST SET:")
    print(f"   MAE  : {test_mae:.2f} min {'✅' if test_mae < 5.0 else '⚠️'} (cible: < 5 min)")
    print(f"   RMSE : {test_rmse:.2f} min")
    print(f"   R²   : {test_r2:.4f} {'✅' if test_r2 > 0.6 else '⚠️'} (cible: > 0.6)")

    print("\n⚡ PERFORMANCE:")
    print(f"   Temps prédiction: {avg_pred_time:.2f}ms {'✅' if avg_pred_time < 100 else '⚠️'} (cible: < 100ms)")

    # Overfitting check
    overfitting = train_r2 - test_r2
    print("\n🔍 OVERFITTING CHECK:")
    print(f"   Diff R² (train - test): {overfitting:.4f}")
    if overfitting > 0.15:
        print("   ⚠️ Overfitting détecté (diff > 0.15)")
    else:
        print("   ✅ Pas d'overfitting significatif")

    return {
        "train": {
            "mae": float(train_mae),
            "rmse": float(train_rmse),
            "r2": float(train_r2),
        },
        "test": {
            "mae": float(test_mae),
            "rmse": float(test_rmse),
            "r2": float(test_r2),
        },
        "prediction_time_ms": float(avg_pred_time),
        "overfitting": float(overfitting),
    }


def cross_validate_model(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5
) -> dict:
    """
    Validation croisée pour estimer la robustesse du modèle.

    Args:
        model: Modèle à valider
        X: Features
        y: Target
        cv: Nombre de folds (défaut: 5)

    Returns:
        Dict avec scores CV
    """
    print("\n" + "="*70)
    print(f"🔄 VALIDATION CROISÉE ({cv}-FOLD CV)")
    print("="*70)

    print("\n⏱️ Cross-validation en cours...")

    # Scorer sur MAE (négatif par convention sklearn)
    cv_mae_scores = -cross_val_score(
        model, X, y,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    # Scorer sur R²
    cv_r2_scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring='r2',
        n_jobs=-1
    )

    print(f"\n📊 Résultats {cv}-Fold CV:")
    print("\n   MAE:")
    print(f"      Moyenne : {cv_mae_scores.mean():.2f} min")
    print(f"      Std     : ±{cv_mae_scores.std():.2f} min")
    print(f"      Min/Max : {cv_mae_scores.min():.2f} / {cv_mae_scores.max():.2f} min")

    print("\n   R²:")
    print(f"      Moyenne : {cv_r2_scores.mean():.4f}")
    print(f"      Std     : ±{cv_r2_scores.std():.4f}")
    print(f"      Min/Max : {cv_r2_scores.min():.4f} / {cv_r2_scores.max():.4f}")

    # Stabilité
    cv_stability = cv_r2_scores.std()
    print("\n🔍 STABILITÉ:")
    print(f"   Std R² = {cv_stability:.4f}")
    if cv_stability < 0.05:
        print("   ✅ Modèle très stable (std < 0.05)")
    elif cv_stability < 0.10:
        print("   ✅ Modèle stable (std < 0.10)")
    else:
        print("   ⚠️ Modèle instable (std > 0.10)")

    return {
        "cv_mae_mean": float(cv_mae_scores.mean()),
        "cv_mae_std": float(cv_mae_scores.std()),
        "cv_r2_mean": float(cv_r2_scores.mean()),
        "cv_r2_std": float(cv_r2_scores.std()),
        "stability": float(cv_stability),
    }


def analyze_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str],
    top_n: int = 15
) -> pd.DataFrame:
    """
    Analyse l'importance des features.

    Args:
        model: Modèle entraîné
        feature_names: Noms des features
        top_n: Nombre de top features à afficher

    Returns:
        DataFrame avec importances triées
    """
    print("\n" + "="*70)
    print(f"🎯 IMPORTANCE DES FEATURES (TOP {top_n})")
    print("="*70)

    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print(f"\n   {'Rang':<5} {'Feature':<30} {'Importance':<12} {'Cumul %'}")
    print("   " + "-"*65)

    cumul = 0.0
    for i, row in feature_importance.head(top_n).iterrows():  # type: ignore[attr-defined]
        cumul += row['importance']
        bar = "█" * int(row['importance'] * 50)
        idx = int(i) + 1 if isinstance(i, (int, float, np.integer)) else 1  # type: ignore[arg-type]
        print(f"   {idx:<5} {row['feature']:<30} {row['importance']:.4f}  {bar:10s} {cumul*100:.1f}%")

    print(f"\n✅ Top {top_n} features expliquent {cumul*100:.1f}% de la variance")

    return feature_importance


def save_model(
    model: RandomForestRegressor,
    feature_names: list[str],
    metrics: dict,
    output_path: str
) -> None:
    """
    Sauvegarde le modèle et ses métadonnées.

    Args:
        model: Modèle entraîné
        feature_names: Liste des features utilisées
        metrics: Métriques de performance
        output_path: Chemin de sauvegarde
    """
    print("\n" + "="*70)
    print("💾 SAUVEGARDE DU MODÈLE")
    print("="*70)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Données à sauvegarder
    model_data = {
        "model": model,
        "feature_names": feature_names,
        "metrics": metrics,
        "n_features": len(feature_names),
        "trained_at": pd.Timestamp.now().isoformat(),
    }

    # Sauvegarder en pickle
    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)

    file_size = output_file.stat().st_size / 1024  # en KB

    print(f"✅ Modèle sauvegardé: {output_file}")
    print(f"   Taille: {file_size:.1f} KB")
    print(f"   Features: {len(feature_names)}")
    print(f"   MAE (test): {metrics['test']['mae']:.2f} min")
    print(f"   R² (test): {metrics['test']['r2']:.4f}")


def generate_training_report(
    metrics: dict,
    cv_results: dict,
    feature_importance: pd.DataFrame,
    output_dir: Path
) -> None:
    """Génère un rapport d'entraînement complet."""
    print("\n" + "="*70)
    print("📝 GÉNÉRATION DU RAPPORT")
    print("="*70)

    report_path = output_dir / 'TRAINING_REPORT.md'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 🤖 RAPPORT D'ENTRAÎNEMENT DU MODÈLE ML\n\n")

        # Métriques
        f.write("## 📊 MÉTRIQUES DE PERFORMANCE\n\n")
        f.write("### Test Set\n\n")
        f.write(f"- **MAE** : {metrics['test']['mae']:.2f} min")
        f.write(" ✅\n" if metrics['test']['mae'] < 5.0 else " ⚠️\n")
        f.write(f"- **RMSE** : {metrics['test']['rmse']:.2f} min\n")
        f.write(f"- **R²** : {metrics['test']['r2']:.4f}")
        f.write(" ✅\n" if metrics['test']['r2'] > 0.6 else " ⚠️\n")
        f.write(f"- **Temps prédiction** : {metrics['prediction_time_ms']:.2f}ms")
        f.write(" ✅\n" if metrics['prediction_time_ms'] < 100 else " ⚠️\n")

        # Validation croisée
        f.write("\n### Validation Croisée (5-Fold)\n\n")
        f.write(f"- **MAE (CV)** : {cv_results['cv_mae_mean']:.2f} ± {cv_results['cv_mae_std']:.2f} min\n")
        f.write(f"- **R² (CV)** : {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}\n")
        f.write(f"- **Stabilité** : {cv_results['stability']:.4f}")
        f.write(" ✅\n" if cv_results['stability'] < 0.10 else " ⚠️\n")

        # Overfitting
        f.write("\n### Overfitting Check\n\n")
        f.write(f"- **Diff R² (train - test)** : {metrics['overfitting']:.4f}\n")
        if metrics['overfitting'] > 0.15:
            f.write("- ⚠️ **Overfitting détecté**\n")
        else:
            f.write("- ✅ **Pas d'overfitting significatif**\n")

        # Top features
        f.write("\n## 🎯 TOP 10 FEATURES\n\n")
        f.write("| Rang | Feature | Importance |\n")
        f.write("|------|---------|------------|\n")

        for i, row in feature_importance.head(10).iterrows():  # type: ignore[attr-defined]
            idx = int(i) + 1 if isinstance(i, (int, float, np.integer)) else 1  # type: ignore[arg-type]
            f.write(f"| {idx} | `{row['feature']}` | {row['importance']:.4f} |\n")

        f.write("\n---\n\n")
        f.write("**Rapport généré automatiquement par `train_model.py`**\n")

    print(f"✅ Rapport sauvegardé: {report_path}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Entraînement modèle ML")
    parser.add_argument("--train", type=str, default="data/ml/train_data.csv",
                       help="Fichier CSV train")
    parser.add_argument("--test", type=str, default="data/ml/test_data.csv",
                       help="Fichier CSV test")
    parser.add_argument("--output", type=str, default="data/ml/models/delay_predictor.pkl",
                       help="Fichier de sortie du modèle")
    parser.add_argument("--n-estimators", type=int, default=100,
                       help="Nombre d'arbres (défaut: 100)")
    parser.add_argument("--max-depth", type=int, default=None,
                       help="Profondeur max (défaut: illimité)")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🤖 ENTRAÎNEMENT MODÈLE ML - PRÉDICTION DE RETARDS")
    print("="*70)
    print(f"Train  : {args.train}")
    print(f"Test   : {args.test}")
    print(f"Output : {args.output}")
    print("="*70)

    try:
        # 1. Charger datasets
        train_df, test_df = load_datasets(args.train, args.test)

        # 2. Préparer features et target
        X_train, y_train = prepare_features_and_target(train_df)
        X_test, y_test = prepare_features_and_target(test_df)

        # 3. Entraîner modèle
        model = train_random_forest(
            X_train, y_train,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth
        )

        # 4. Évaluer
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

        # 5. Validation croisée
        cv_results = cross_validate_model(model, X_train, y_train, cv=5)

        # 6. Feature importance
        feature_importance = analyze_feature_importance(model, X_train.columns.tolist(), top_n=15)

        # 7. Sauvegarder modèle
        save_model(model, X_train.columns.tolist(), metrics, args.output)

        # 8. Rapport
        output_dir = Path(args.output).parent
        generate_training_report(metrics, cv_results, feature_importance, output_dir)

        # 9. Métadonnées
        metadata = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": len(X_train.columns),
            "metrics": metrics,
            "cv_results": cv_results,
            "top_features": feature_importance.head(10).to_dict('records'),
        }

        metadata_path = output_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Métadonnées: {metadata_path}")

        # Résumé final
        print("\n" + "="*70)
        print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
        print("="*70)
        print("\n🎯 Performance Test Set:")
        print(f"   MAE  : {metrics['test']['mae']:.2f} min {'✅' if metrics['test']['mae'] < 5.0 else '❌'}")
        print(f"   R²   : {metrics['test']['r2']:.4f} {'✅' if metrics['test']['r2'] > 0.6 else '❌'}")
        print(f"   Temps: {metrics['prediction_time_ms']:.2f}ms {'✅' if metrics['prediction_time_ms'] < 100 else '❌'}")

        print("\n📊 Validation Croisée:")
        print(f"   MAE (CV): {cv_results['cv_mae_mean']:.2f} ± {cv_results['cv_mae_std']:.2f} min")
        print(f"   R² (CV) : {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")

        print("\n" + "="*70)

        # Check objectifs
        success = (
            metrics['test']['mae'] < 5.0 and
            metrics['test']['r2'] > 0.6 and
            metrics['prediction_time_ms'] < 100
        )

        if success:
            print("🎉 TOUS LES OBJECTIFS ATTEINTS !")
            print("="*70 + "\n")
            sys.exit(0)
        else:
            print("⚠️ Certains objectifs non atteints")
            print("   → Considérer fine-tuning hyperparamètres")
            print("="*70 + "\n")
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

