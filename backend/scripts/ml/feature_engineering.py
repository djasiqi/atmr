"""
Script de feature engineering avancé pour le modèle ML de prédiction de retards.

Crée des features dérivées, interactions, et normalise les données.

Usage:
    python scripts/ml/feature_engineering.py [--input data/ml/training_data.csv] [--output data/ml/]
"""
# ruff: noqa: T201
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportReturnType=false
# print() est intentionnel dans les scripts ML
# Pandas/sklearn ont des types complexes, ignorer warnings stricts

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features d'interaction entre variables importantes.

    Interactions créées :
    - distance × traffic_density : Impact combiné distance + trafic
    - distance × weather_factor : Impact combiné distance + météo
    - traffic_density × weather_factor : Conditions défavorables combinées
    - is_medical × distance : Urgence médicale longue distance
    - is_urgent × traffic_density : Urgence en heure de pointe
    """
    print("\n" + "="*70)
    print("🔗 CRÉATION DES FEATURES D'INTERACTION")
    print("="*70)

    df_new = df.copy()

    # Interaction 1: Distance × Trafic (effet combiné majeur)
    if 'distance_km' in df.columns and 'traffic_density' in df.columns:
        df_new['distance_x_traffic'] = df['distance_km'] * df['traffic_density']
        print("✅ distance_x_traffic = distance × traffic")

    # Interaction 2: Distance × Météo
    if 'distance_km' in df.columns and 'weather_factor' in df.columns:
        df_new['distance_x_weather'] = df['distance_km'] * df['weather_factor']
        print("✅ distance_x_weather = distance × weather")

    # Interaction 3: Trafic × Météo (conditions défavorables)
    if 'traffic_density' in df.columns and 'weather_factor' in df.columns:
        df_new['traffic_x_weather'] = df['traffic_density'] * df['weather_factor']
        print("✅ traffic_x_weather = traffic × weather")

    # Interaction 4: Médical × Distance (urgence longue distance)
    if 'is_medical' in df.columns and 'distance_km' in df.columns:
        df_new['medical_x_distance'] = df['is_medical'] * df['distance_km']
        print("✅ medical_x_distance = is_medical × distance")

    # Interaction 5: Urgent × Trafic (urgence en pointe)
    if 'is_urgent' in df.columns and 'traffic_density' in df.columns:
        df_new['urgent_x_traffic'] = df['is_urgent'] * df['traffic_density']
        print("✅ urgent_x_traffic = is_urgent × traffic")

    n_new_features = len(df_new.columns) - len(df.columns)
    print(f"\n✅ {n_new_features} features d'interaction créées")

    return df_new


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features temporelles avancées.

    Features créées :
    - is_rush_hour : Binaire heures de pointe (7-9h, 17-19h)
    - is_weekend : Binaire weekend (samedi-dimanche)
    - hour_sin, hour_cos : Encodage cyclique de l'heure
    - day_sin, day_cos : Encodage cyclique du jour
    - is_morning_peak : Binaire pic matin (7-9h)
    - is_evening_peak : Binaire pic soir (17-19h)
    """
    print("\n" + "="*70)
    print("⏰ CRÉATION DES FEATURES TEMPORELLES")
    print("="*70)

    df_new = df.copy()

    # Heures de pointe (7-9h et 17-19h)
    if 'time_of_day' in df.columns:
        df_new['is_rush_hour'] = df['time_of_day'].apply(
            lambda h: 1.0 if h in [7, 8, 17, 18] else 0.0
        )
        print("✅ is_rush_hour (7-9h, 17-19h)")

        df_new['is_morning_peak'] = df['time_of_day'].apply(
            lambda h: 1.0 if h in [7, 8] else 0.0
        )
        print("✅ is_morning_peak (7-9h)")

        df_new['is_evening_peak'] = df['time_of_day'].apply(
            lambda h: 1.0 if h in [17, 18] else 0.0
        )
        print("✅ is_evening_peak (17-19h)")

        # Encodage cyclique de l'heure (évite discontinuité 23h → 0h)
        df_new['hour_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 24)
        df_new['hour_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 24)
        print("✅ hour_sin, hour_cos (encodage cyclique)")

    # Weekend (samedi-dimanche)
    if 'day_of_week' in df.columns:
        df_new['is_weekend'] = df['day_of_week'].apply(
            lambda d: 1.0 if d >= 5 else 0.0
        )
        print("✅ is_weekend (samedi-dimanche)")

        # Encodage cyclique du jour (évite discontinuité dimanche → lundi)
        df_new['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df_new['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        print("✅ day_sin, day_cos (encodage cyclique)")

    # Midi (12-14h)
    if 'time_of_day' in df.columns:
        df_new['is_lunch_time'] = df['time_of_day'].apply(
            lambda h: 1.0 if h in [12, 13] else 0.0
        )
        print("✅ is_lunch_time (12-14h)")

    n_new_features = len(df_new.columns) - len(df.columns)
    print(f"\n✅ {n_new_features} features temporelles créées")

    return df_new


def create_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features agrégées basées sur l'historique et les patterns.

    Features créées :
    - delay_by_hour : Retard moyen par heure
    - delay_by_day : Retard moyen par jour
    - delay_by_driver_exp : Retard moyen par niveau d'expérience driver
    - distance_category : Catégorie de distance (courte/moyenne/longue)
    - traffic_level : Niveau de trafic (faible/moyen/élevé)
    """
    print("\n" + "="*70)
    print("📊 CRÉATION DES FEATURES AGRÉGÉES")
    print("="*70)

    df_new = df.copy()
    target = 'actual_delay_minutes'

    # Retard moyen par heure
    if 'time_of_day' in df.columns and target in df.columns:
        hour_delays = df.groupby('time_of_day')[target].mean()
        df_new['delay_by_hour'] = df['time_of_day'].map(hour_delays)  # type: ignore[arg-type]
        print("✅ delay_by_hour (retard moyen par heure)")

    # Retard moyen par jour
    if 'day_of_week' in df.columns and target in df.columns:
        day_delays = df.groupby('day_of_week')[target].mean()
        df_new['delay_by_day'] = df['day_of_week'].map(day_delays)  # type: ignore[arg-type]
        print("✅ delay_by_day (retard moyen par jour)")

    # Catégorie d'expérience driver
    if 'driver_total_bookings' in df.columns:
        df_new['driver_experience_level'] = pd.cut(
            df['driver_total_bookings'],
            bins=[0, 50, 200, float('inf')],
            labels=[0, 1, 2]  # 0=novice, 1=intermédiaire, 2=expert
        ).astype(float)  # type: ignore[attr-defined]
        print("✅ driver_experience_level (novice/inter/expert)")

        # Retard moyen par niveau d'expérience
        if target in df.columns:
            exp_delays = df.groupby(
                pd.cut(df['driver_total_bookings'], bins=[0, 50, 200, float('inf')]), observed=True
            )[target].mean()
            df_new['delay_by_driver_exp'] = pd.cut(
                df['driver_total_bookings'],
                bins=[0, 50, 200, float('inf')]
            ).map(exp_delays)  # type: ignore[attr-defined,arg-type]
            print("✅ delay_by_driver_exp (retard par niveau exp)")

    # Catégorie de distance
    if 'distance_km' in df.columns:
        df_new['distance_category'] = pd.cut(
            df['distance_km'],
            bins=[0, 5, 10, 20, float('inf')],
            labels=[0, 1, 2, 3]  # 0=courte, 1=moyenne, 2=longue, 3=très longue
        ).astype(float)  # type: ignore[attr-defined]
        print("✅ distance_category (courte/moyenne/longue)")

    # Niveau de trafic
    if 'traffic_density' in df.columns:
        df_new['traffic_level'] = pd.cut(
            df['traffic_density'],
            bins=[0, 0.4, 0.7, 1.0],
            labels=[0, 1, 2]  # 0=faible, 1=moyen, 2=élevé
        ).astype(float)  # type: ignore[attr-defined]
        print("✅ traffic_level (faible/moyen/élevé)")

    n_new_features = len(df_new.columns) - len(df.columns)
    print(f"\n✅ {n_new_features} features agrégées créées")

    return df_new


def create_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features polynomiales pour capturer relations non-linéaires.

    Features créées :
    - distance_squared : Distance au carré (relation quadratique)
    - traffic_squared : Trafic au carré
    - driver_exp_log : Log de l'expérience (rendements décroissants)
    """
    print("\n" + "="*70)
    print("📐 CRÉATION DES FEATURES POLYNOMIALES")
    print("="*70)

    df_new = df.copy()

    # Distance au carré (relation quadratique possible)
    if 'distance_km' in df.columns:
        df_new['distance_squared'] = df['distance_km'] ** 2
        print("✅ distance_squared = distance²")

    # Trafic au carré
    if 'traffic_density' in df.columns:
        df_new['traffic_squared'] = df['traffic_density'] ** 2
        print("✅ traffic_squared = traffic²")

    # Log de l'expérience driver (rendements décroissants)
    if 'driver_total_bookings' in df.columns:
        df_new['driver_exp_log'] = np.log1p(df['driver_total_bookings'])  # log(1+x)
        print("✅ driver_exp_log = log(1 + exp)")

    n_new_features = len(df_new.columns) - len(df.columns)
    print(f"\n✅ {n_new_features} features polynomiales créées")

    return df_new


def normalize_features(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Normalise les features continues avec StandardScaler et MinMaxScaler.

    Args:
        df: DataFrame à normaliser
        exclude_cols: Colonnes à exclure de la normalisation

    Returns:
        Tuple (DataFrame normalisé, dict des scalers)
    """
    print("\n" + "="*70)
    print("📏 NORMALISATION DES FEATURES")
    print("="*70)

    if exclude_cols is None:
        exclude_cols = [
            'booking_id', 'driver_id', 'assignment_id', 'company_id',
            'actual_delay_minutes'  # Target
        ]

    df_new = df.copy()
    scalers = {}

    # Séparer features numériques continues et binaires
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Features binaires (0/1) - ne pas normaliser
    binary_cols = [col for col in numeric_cols if df_new[col].isin([0.0, 1.0]).all()]  # type: ignore[arg-type]

    # Features continues - normaliser avec StandardScaler
    continuous_cols = [col for col in numeric_cols if col not in binary_cols]

    if len(continuous_cols) > 0:
        print(f"\n🔧 StandardScaler sur {len(continuous_cols)} features continues :")
        for col in continuous_cols[:5]:  # Afficher les 5 premières
            print(f"   - {col}")
        if len(continuous_cols) > 5:
            print(f"   ... et {len(continuous_cols) - 5} autres")

        scaler = StandardScaler()
        df_new[continuous_cols] = scaler.fit_transform(df_new[continuous_cols])
        scalers['standard'] = {
            'scaler': scaler,
            'columns': continuous_cols
        }

    if len(binary_cols) > 0:
        print(f"\n✅ {len(binary_cols)} features binaires conservées sans normalisation")

    print(f"\n✅ Normalisation terminée : {len(continuous_cols)} features normalisées")

    return df_new, scalers


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split le dataset en train/test avec stratification.

    Args:
        df: DataFrame complet
        test_size: Proportion du test set (défaut: 0.2 = 20%)
        random_state: Seed pour reproductibilité

    Returns:
        Tuple (train_df, test_df)
    """
    print("\n" + "="*70)
    print("✂️ SPLIT TRAIN/TEST")
    print("="*70)

    # Stratifier sur les bins de retard pour avoir distribution similaire
    target = 'actual_delay_minutes'
    if target in df.columns:
        # Créer bins pour stratification (3 bins pour éviter classes trop petites)
        try:
            bins = pd.cut(df[target], bins=3, labels=False, duplicates='drop')

            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=bins
            )
        except ValueError:
            # Si stratification échoue, split sans stratification
            print("⚠️ Stratification impossible, split simple")
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state
            )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

    print(f"✅ Train set : {len(train_df)} échantillons ({(1-test_size)*100:.0f}%)")
    print(f"✅ Test set  : {len(test_df)} échantillons ({test_size*100:.0f}%)")

    # Vérifier distribution du target
    if target in df.columns:
        print("\n📊 Distribution du target :")
        print(f"   Train - Moyenne : {train_df[target].mean():.2f} min")
        print(f"   Test  - Moyenne : {test_df[target].mean():.2f} min")
        print(f"   Différence      : {abs(train_df[target].mean() - test_df[target].mean()):.2f} min")

    return train_df, test_df


def generate_feature_report(
    original_df: pd.DataFrame,
    engineered_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Génère un rapport détaillé du feature engineering."""
    print("\n" + "="*70)
    print("📝 GÉNÉRATION DU RAPPORT")
    print("="*70)

    report_path = output_dir / 'FEATURE_ENGINEERING_REPORT.md'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 🔧 RAPPORT DE FEATURE ENGINEERING\n\n")

        # Résumé
        f.write("## 📊 RÉSUMÉ\n\n")
        f.write(f"- **Features originales** : {len(original_df.columns)}\n")
        f.write(f"- **Features après engineering** : {len(engineered_df.columns)}\n")
        f.write(f"- **Nouvelles features créées** : {len(engineered_df.columns) - len(original_df.columns)}\n\n")

        # Liste des nouvelles features
        new_features = [col for col in engineered_df.columns if col not in original_df.columns]

        f.write("## 🆕 NOUVELLES FEATURES CRÉÉES\n\n")

        # Par catégorie
        interaction_features = [f for f in new_features if '_x_' in f]
        temporal_features = [f for f in new_features if any(x in f for x in ['is_', 'hour_', 'day_'])]
        aggregated_features = [f for f in new_features if any(x in f for x in ['delay_by_', '_level', '_category'])]
        polynomial_features = [f for f in new_features if any(x in f for x in ['squared', '_log'])]

        f.write("### Interactions\n\n")
        for feat in interaction_features:
            f.write(f"- `{feat}`\n")

        f.write("\n### Temporelles\n\n")
        for feat in temporal_features:
            f.write(f"- `{feat}`\n")

        f.write("\n### Agrégées\n\n")
        for feat in aggregated_features:
            f.write(f"- `{feat}`\n")

        f.write("\n### Polynomiales\n\n")
        for feat in polynomial_features:
            f.write(f"- `{feat}`\n")

        f.write("\n---\n\n")
        f.write("**Rapport généré automatiquement par `feature_engineering.py`**\n")

    print(f"✅ Rapport sauvegardé : {report_path}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Feature engineering pour ML")
    parser.add_argument("--input", type=str, default="data/ml/training_data.csv",
                       help="Fichier CSV d'entrée")
    parser.add_argument("--output", type=str, default="data/ml/",
                       help="Dossier de sortie")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proportion du test set (défaut: 0.2)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("🔧 FEATURE ENGINEERING AVANCÉ")
    print("="*70)
    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print("="*70)

    try:
        # 1. Charger données
        print("\n📂 Chargement des données...")
        df_original = pd.read_csv(args.input)
        print(f"✅ {len(df_original)} échantillons × {len(df_original.columns)} features")

        # 2. Feature engineering
        df = df_original.copy()

        df = create_interaction_features(df)
        df = create_temporal_features(df)
        df = create_aggregated_features(df)
        df = create_polynomial_features(df)

        print(f"\n✅ Total features : {len(df_original.columns)} → {len(df.columns)} (+{len(df.columns) - len(df_original.columns)})")

        # 3. Split train/test AVANT normalisation (pour éviter data leakage)
        train_df, test_df = split_train_test(df, test_size=args.test_size)

        # 4. Normalisation (fit sur train, transform sur train et test)
        print("\n🔧 Normalisation du train set...")
        train_normalized, scalers = normalize_features(train_df)

        print("\n🔧 Normalisation du test set (avec scalers du train)...")
        test_normalized = test_df.copy()
        if 'standard' in scalers:
            scaler = scalers['standard']['scaler']
            cols = scalers['standard']['columns']
            test_normalized[cols] = scaler.transform(test_df[cols])

        # 5. Sauvegarder
        print("\n💾 Sauvegarde des fichiers...")

        # Dataset complet (avant split)
        full_path = output_dir / 'training_data_engineered.csv'
        df.to_csv(full_path, index=False)
        print(f"✅ Dataset complet : {full_path}")

        # Train/test normalisés
        train_path = output_dir / 'train_data.csv'
        test_path = output_dir / 'test_data.csv'
        train_normalized.to_csv(train_path, index=False)
        test_normalized.to_csv(test_path, index=False)
        print(f"✅ Train set : {train_path}")
        print(f"✅ Test set  : {test_path}")

        # Scalers
        scalers_path = output_dir / 'scalers.json'
        scalers_data = {
            'standard_scaler': {
                'columns': scalers['standard']['columns'],
                'mean': scalers['standard']['scaler'].mean_.tolist(),
                'scale': scalers['standard']['scaler'].scale_.tolist(),
            }
        } if 'standard' in scalers else {}

        with open(scalers_path, 'w') as f:
            json.dump(scalers_data, f, indent=2)
        print(f"✅ Scalers : {scalers_path}")

        # Rapport
        generate_feature_report(df_original, df, output_dir)

        # Métadonnées
        metadata = {
            "original_features": len(df_original.columns),
            "engineered_features": len(df.columns),
            "new_features": len(df.columns) - len(df_original.columns),
            "train_samples": len(train_normalized),
            "test_samples": len(test_normalized),
            "test_size": args.test_size,
            "normalized": True,
        }

        metadata_path = output_dir / 'feature_engineering_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Métadonnées : {metadata_path}")

        print("\n" + "="*70)
        print("✅ FEATURE ENGINEERING TERMINÉ AVEC SUCCÈS !")
        print("="*70)
        print("\n📊 Résumé :")
        print(f"   Features  : {len(df_original.columns)} → {len(df.columns)} (+{len(df.columns) - len(df_original.columns)})")
        print(f"   Train set : {len(train_normalized)} ({(1-args.test_size)*100:.0f}%)")
        print(f"   Test set  : {len(test_normalized)} ({args.test_size*100:.0f}%)")
        print("="*70)

    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

