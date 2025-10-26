"""Script d'analyse exploratoire des données (EDA) pour le dataset ML.

Génère des visualisations, statistiques et un rapport complet.

Usage:
    python scripts/ml/analyze_data.py [--input data/ml/training_data.csv] [--output reports/eda/]
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# print() est intentionnel dans les scripts d'analyse
# matplotlib, seaborn = bibliothèques externes, ignorer warnings d'import

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from scipy import stats


def load_data(input_path: str) -> pd.DataFrame:
    """Charge le dataset depuis CSV."""
    print("\n📂 Chargement des données depuis {input_path}...")

    df = pd.read_csv(input_path)
    print("✅ Dataset chargé : {len(df)} lignes × {len(df.columns)} colonnes")

    return df


def analyze_basic_statistics(df: pd.DataFrame) -> dict:
    """Analyse statistique de base."""
    print("\n" + "="*70)
    print("📊 STATISTIQUES DESCRIPTIVES")
    print("="*70)

    stats_dict = {
        "total_samples": len(df),
        "features": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
    }

    # Statistiques sur la target
    target = "actual_delay_minutes"
    if target in df.columns:
        print("\n🎯 Target: {target}")
        print("   Moyenne    : {df[target].mean()")
        print("   Médiane    : {df[target].median()")
        print("   Écart-type : {df[target].std()")
        print("   Min / Max  : {df[target].min()")
        print("   Q1 / Q3    : {df[target].quantile(0.25)")

        stats_dict["target_stats"] = {
            "mean": float(df[target].mean()),
            "median": float(df[target].median()),
            "std": float(df[target].std()),
            "min": float(df[target].min()),
            "max": float(df[target].max()),
            "q1": float(df[target].quantile(0.25)),
            "q3": float(df[target].quantile(0.75)),
        }

    # Valeurs manquantes
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️ Valeurs manquantes détectées :")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            print("   - {col:25s} : {count:5d} ({pct%5.2f}%)")
    else:
        print("\n✅ Aucune valeur manquante")

    return stats_dict


def analyze_correlations(df: pd.DataFrame, output_dir: Path) -> dict:
    """Analyse des corrélations et génération de heatmap."""
    print("\n" + "="*70)
    print("🔗 ANALYSE DES CORRÉLATIONS")
    print("="*70)

    # Sélectionner colonnes numériques seulement
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclure les IDs
    id_cols = ["booking_id", "driver_id", "assignment_id", "company_id"]
    numeric_cols = [col for col in numeric_cols if col not in id_cols]

    if len(numeric_cols) < 2:
        print("⚠️ Pas assez de colonnes numériques pour calculer les corrélations")
        return {}

    # Calculer corrélations
    corr_matrix = df[numeric_cols].corr()

    # Afficher top corrélations avec target
    target = "actual_delay_minutes"
    if target in corr_matrix.columns:
        print("\n🎯 Corrélations avec {target} :")
        target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)

        for feature, corr in target_corr.items():
            if abs(corr) > 0.1:
                symbol = "⭐" if abs(corr) > 0.5 else "📊" if abs(corr) > 0.3 else "📉"
                print("   {symbol} {feature:25s} : {corr:+.3f}")

    # Générer heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8})
    plt.title("Matrice de Corrélation des Features", fontsize=16, fontweight="bold")
    plt.tight_layout()

    heatmap_path = output_dir / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=0.300, bbox_inches="tight")
    plt.close()

    print("\n✅ Heatmap sauvegardée : {heatmap_path}")

    return {
        "correlation_matrix": corr_matrix.to_dict(),
        "target_correlations": target_corr.to_dict() if target in corr_matrix.columns else {},
    }


def analyze_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyse des distributions des variables."""
    print("\n" + "="*70)
    print("📈 ANALYSE DES DISTRIBUTIONS")
    print("="*70)

    # Distribution de la target
    target = "actual_delay_minutes"
    if target in df.columns:
        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Histogramme
        axes[0, 0].hist(df[target], bins=50, edgecolor="black", alpha=0.7)
        axes[0, 0].axvline(df[target].mean(), color="red", linestyle="--",
                          linewidth=2, label=f"Moyenne: {df[target].mean()")
        axes[0, 0].axvline(df[target].median(), color="green", linestyle="--",
                          linewidth=2, label=f"Médiane: {df[target].median()")
        axes[0, 0].set_xlabel("Retard (minutes)")
        axes[0, 0].set_ylabel("Fréquence")
        axes[0, 0].set_title("Distribution des Retards")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Histogramme + KDE
        axes[0, 1].hist(df[target], bins=50, density=True, alpha=0.6, edgecolor="black")
        df[target].plot(kind="kde", ax=axes[0, 1], color="red", linewidth=2)
        axes[0, 1].set_xlabel("Retard (minutes)")
        axes[0, 1].set_ylabel("Densité")
        axes[0, 1].set_title("Distribution + Kernel Density Estimation")
        axes[0, 1].grid(True, alpha=0.3)

        # Box plot
        axes[1, 0].boxplot(df[target], vert=True)
        axes[1, 0].set_ylabel("Retard (minutes)")
        axes[1, 0].set_title("Box Plot - Détection Outliers")
        axes[1, 0].grid(True, alpha=0.3)

        # Q-Q plot
        stats.probplot(df[target], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot - Test Normalité")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f"Analyse de Distribution: {target}", fontsize=16, fontweight="bold", y=1.00)
        plt.tight_layout()

        dist_path = output_dir / "target_distribution.png"
        plt.savefig(dist_path, dpi=0.300, bbox_inches="tight")
        plt.close()

        print("✅ Distribution target sauvegardée : {dist_path}")

    # Distribution des features numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    id_cols = ["booking_id", "driver_id", "assignment_id", "company_id"]
    feature_cols = [col for col in numeric_cols if col not in id_cols and col != target]

    if len(feature_cols) > 0:
        n_features = len(feature_cols)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        _fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for idx, col in enumerate(feature_cols):
            if idx < len(axes):
                axes[idx].hist(df[col], bins=30, edgecolor="black", alpha=0.7)
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel("Fréquence")
                axes[idx].set_title(f"Distribution: {col}")
                axes[idx].grid(True, alpha=0.3)

        # Masquer axes inutilisés
        for idx in range(len(feature_cols), len(axes)):
            axes[idx].axis("off")

        plt.suptitle("Distribution des Features", fontsize=16, fontweight="bold")
        plt.tight_layout()

        features_path = output_dir / "features_distributions.png"
        plt.savefig(features_path, dpi=0.300, bbox_inches="tight")
        plt.close()

        print("✅ Distributions features sauvegardées : {features_path}")


def analyze_outliers(df: pd.DataFrame) -> dict:
    """Détection et analyse des outliers."""
    print("\n" + "="*70)
    print("🔍 DÉTECTION DES OUTLIERS")
    print("="*70)

    target = "actual_delay_minutes"
    outliers_info = {}

    if target in df.columns:
        # Méthode IQR
        Q1 = df[target].quantile(0.25)
        Q3 = df[target].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[target] < lower_bound) | (df[target] > upper_bound)]
        n_outliers = len(outliers)
        pct_outliers = (n_outliers / len(df)) * 100

        print("\n📊 Méthode IQR (Interquartile Range) :")
        print("   Q1           : {Q1")
        print("   Q3           : {Q3")
        print("   IQR          : {IQR")
        print("   Borne inf    : {lower_bound")
        print("   Borne sup    : {upper_bound")
        print("   Outliers     : {n_outliers} ({pct_outliers")

        outliers_info["iqr"] = {
            "Q1": float(Q1),
            "Q3": float(Q3),
            "IQR": float(IQR),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "n_outliers": int(n_outliers),
            "pct_outliers": float(pct_outliers),
        }

        # Méthode Z-score
        z_scores = np.abs(stats.zscore(df[target]))  # type: ignore[arg-type]
        outliers_z = df[z_scores > 3]
        n_outliers_z = len(outliers_z)
        pct_outliers_z = (n_outliers_z / len(df)) * 100

        print("\n📊 Méthode Z-score (|z| > 3) :")
        print("   Outliers     : {n_outliers_z} ({pct_outliers_z")

        outliers_info["zscore"] = {
            "n_outliers": int(n_outliers_z),
            "pct_outliers": float(pct_outliers_z),
        }

    return outliers_info


def analyze_temporal_patterns(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyse des patterns temporels."""
    print("\n" + "="*70)
    print("⏰ ANALYSE DES PATTERNS TEMPORELS")
    print("="*70)

    target = "actual_delay_minutes"

    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Retard par heure de la journée
    if "time_of_day" in df.columns and target in df.columns:
        hourly_stats = df.groupby("time_of_day")[target].agg(["mean", "median", "std"])
        axes[0, 0].plot(hourly_stats.index, hourly_stats["mean"], marker="o",
                       linewidth=2, label="Moyenne")
        axes[0, 0].fill_between(hourly_stats.index,
                                hourly_stats["mean"] - hourly_stats["std"],
                                hourly_stats["mean"] + hourly_stats["std"],
                                alpha=0.2)
        axes[0, 0].set_xlabel("Heure de la journée")
        axes[0, 0].set_ylabel("Retard (minutes)")
        axes[0, 0].set_title("Retard Moyen par Heure")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        print("\n🕐 Heures de pointe (retard > moyenne) :")
        peak_hours = hourly_stats[hourly_stats["mean"] > hourly_stats["mean"].mean()]
        for hour, row in peak_hours.iterrows():  # type: ignore[attr-defined]
            hour_int = int(hour) if isinstance(hour, (int, float, np.integer)) else 0  # type: ignore[arg-type]
            print("   - {hour_int:02d}h : {row['mean']")

    # Retard par jour de la semaine
    if "day_of_week" in df.columns and target in df.columns:
        days = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        daily_stats = df.groupby("day_of_week")[target].agg(["mean", "median", "count"])
        axes[0, 1].bar(daily_stats.index, daily_stats["mean"], alpha=0.7, edgecolor="black")
        axes[0, 1].set_xlabel("Jour de la semaine")
        axes[0, 1].set_ylabel("Retard moyen (minutes)")
        axes[0, 1].set_title("Retard Moyen par Jour")
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(days)
        axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Retard par mois
    if "month" in df.columns and target in df.columns:
        months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                 "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
        monthly_stats = df.groupby("month")[target].agg(["mean", "median", "count"])
        axes[1, 0].bar(monthly_stats.index, monthly_stats["mean"], alpha=0.7, edgecolor="black")
        axes[1, 0].set_xlabel("Mois")
        axes[1, 0].set_ylabel("Retard moyen (minutes)")
        axes[1, 0].set_title("Retard Moyen par Mois")
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(months, rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Heatmap heure × jour
    if all(col in df.columns for col in ["time_of_day", "day_of_week", target]):
        pivot = df.pivot_table(values=target, index="time_of_day",
                               columns="day_of_week", aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                   ax=axes[1, 1], cbar_kws={"label": "Retard (min)"})
        axes[1, 1].set_xlabel("Jour de la semaine")
        axes[1, 1].set_ylabel("Heure")
        axes[1, 1].set_title("Heatmap Retard: Heure × Jour")
        axes[1, 1].set_xticklabels(days)

    plt.suptitle("Analyse Temporelle des Retards", fontsize=16, fontweight="bold")
    plt.tight_layout()

    temporal_path = output_dir / "temporal_patterns.png"
    plt.savefig(temporal_path, dpi=0.300, bbox_inches="tight")
    plt.close()

    print("\n✅ Patterns temporels sauvegardés : {temporal_path}")


def analyze_feature_relationships(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyse des relations entre features et target."""
    print("\n" + "="*70)
    print("🔗 ANALYSE DES RELATIONS FEATURES-TARGET")
    print("="*70)

    target = "actual_delay_minutes"
    if target not in df.columns:
        return

    # Relations clés
    key_features = ["distance_km", "traffic_density", "weather_factor", "driver_total_bookings"]
    available_features = [f for f in key_features if f in df.columns]

    if len(available_features) > 0:
        n_features = len(available_features)
        _fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4))
        if n_features == 1:
            axes = [axes]

        for idx, feature in enumerate(available_features):
            axes[idx].scatter(df[feature], df[target], alpha=0.3, s=10)

            # Régression linéaire
            z = np.polyfit(df[feature], df[target], 1)
            p = np.poly1d(z)
            axes[idx].plot(df[feature], p(df[feature]), "r--", linewidth=2, alpha=0.8)

            # Corrélation
            corr = df[feature].corr(df[target])  # type: ignore[arg-type]
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel(target)
            axes[idx].set_title(f"{feature}\n(corr: {corr")
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle("Relations Features-Target", fontsize=16, fontweight="bold")
        plt.tight_layout()

        relations_path = output_dir / "feature_relationships.png"
        plt.savefig(relations_path, dpi=0.300, bbox_inches="tight")
        plt.close()

        print("✅ Relations features sauvegardées : {relations_path}")


def generate_summary_report(df: pd.DataFrame, stats: dict, corr_info: dict,
                            outliers: dict, output_dir: Path) -> None:
    """Génère un rapport de synthèse en texte."""
    print("\n" + "="*70)
    print("📝 GÉNÉRATION DU RAPPORT DE SYNTHÈSE")
    print("="*70)

    report_path = output_dir / "EDA_SUMMARY_REPORT.md"

    with Path(report_path, "w", encoding="utf-8").open() as f:
        f.write("# 📊 RAPPORT D'ANALYSE EXPLORATOIRE (EDA)\n\n")
        f.write(f"**Dataset** : {len(df)} échantillons × {len(df.columns)} features\n\n")
        f.write("---\n\n")

        # Statistiques de base
        f.write("## 📈 STATISTIQUES DESCRIPTIVES\n\n")
        if "target_stats" in stats:
            target_stats = stats["target_stats"]
            f.write("### Target: `actual_delay_minutes`\n\n")
            f.write(f"- **Moyenne** : {target_stats['mean']")
            f.write(f"- **Médiane** : {target_stats['median']")
            f.write(f"- **Écart-type** : {target_stats['std']")
            f.write(f"- **Min / Max** : {target_stats['min']")
            f.write(f"- **Q1 / Q3** : {target_stats['q1']")

        # Corrélations
        f.write("## 🔗 CORRÉLATIONS PRINCIPALES\n\n")
        if "target_correlations" in corr_info:
            target_corr = corr_info["target_correlations"]
            sorted_corr = sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True)

            f.write("| Feature | Corrélation | Force |\n")
            f.write("|---------|-------------|-------|\n")

            for feature, corr in sorted_corr[:10]:
                if abs(corr) > 0.1:
                    force = "Forte" if abs(corr) > 0.5 else "Moyenne" if abs(corr) > 0.3 else "Faible"
                    f.write(f"| `{feature}` | {corr:+.3f} | {force} |\n")

            f.write("\n")

        # Outliers
        f.write("## 🔍 OUTLIERS DÉTECTÉS\n\n")
        if "iqr" in outliers:
            iqr_info = outliers["iqr"]
            f.write(f"**Méthode IQR** : {iqr_info['n_outliers']} outliers ({iqr_info['pct_outliers']")
            f.write(f"- Borne inférieure : {iqr_info['lower_bound']")
            f.write(f"- Borne supérieure : {iqr_info['upper_bound']")

        if "zscore" in outliers:
            z_info = outliers["zscore"]
            f.write(f"**Méthode Z-score** : {z_info['n_outliers']} outliers ({z_info['pct_outliers']")

        # Recommandations
        f.write("## 💡 INSIGHTS & RECOMMANDATIONS\n\n")
        f.write("### Points Clés\n\n")

        if "target_correlations" in corr_info:
            top_feature = max(corr_info["target_correlations"].items(), key=lambda x: abs(x[1]))
            f.write(f"1. **Feature la plus prédictive** : `{top_feature[0]}` (corr: {top_feature[1]:+.3f})\n")

        if "iqr" in outliers and outliers["iqr"]["pct_outliers"] > 5:
            f.write(f"2. ⚠️ **Outliers significatifs** : {outliers['iqr']['pct_outliers']")
            f.write("   - Recommandation : Investiguer ces cas extrêmes\n")

        f.write("\n### Prochaines Étapes\n\n")
        f.write("1. **Feature Engineering** : Créer interactions entre top features\n")
        f.write("2. **Traitement Outliers** : Décider de conserver ou transformer\n")
        f.write("3. **Normalisation** : Préparer features pour ML\n")
        f.write("4. **Split Train/Test** : 80/20 avec stratification\n\n")

        f.write("---\n\n")
        f.write("**Rapport généré automatiquement par `analyze_data.py`**\n")

    print("✅ Rapport de synthèse sauvegardé : {report_path}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Analyse exploratoire des données ML")
    parser.add_argument("--input", type=str, default="data/ml/training_data.csv",
                       help="Fichier CSV d'entrée")
    parser.add_argument("--output", type=str, default="data/ml/reports/eda/",
                       help="Dossier de sortie pour les rapports")

    args = parser.parse_args()

    # Créer dossier de sortie
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("📊 ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    print("="*70)
    print("Input  : {args.input}")
    print("Output : {args.output}")
    print("="*70)

    try:
        # Charger données
        df = load_data(args.input)

        # Analyses
        stats = analyze_basic_statistics(df)
        corr_info = analyze_correlations(df, output_dir)
        analyze_distributions(df, output_dir)
        outliers = analyze_outliers(df)
        analyze_temporal_patterns(df, output_dir)
        analyze_feature_relationships(df, output_dir)

        # Rapport final
        generate_summary_report(df, stats, corr_info, outliers, output_dir)

        # Sauvegarder métadonnées
        metadata = {
            "input_file": args.input,
            "n_samples": len(df),
            "n_features": len(df.columns),
            "statistics": stats,
            "correlations": corr_info.get("target_correlations", {}),
            "outliers": outliers,
        }

        metadata_path = output_dir / "eda_metadata.json"
        with Path(metadata_path, "w").open() as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "="*70)
        print("✅ ANALYSE TERMINÉE AVEC SUCCÈS !")
        print("="*70)
        print("\nFichiers générés :")
        print("  - {output_dir / 'correlation_heatmap.png'}")
        print("  - {output_dir / 'target_distribution.png'}")
        print("  - {output_dir / 'features_distributions.png'}")
        print("  - {output_dir / 'temporal_patterns.png'}")
        print("  - {output_dir / 'feature_relationships.png'}")
        print("  - {output_dir / 'EDA_SUMMARY_REPORT.md'}")
        print("  - {output_dir / 'eda_metadata.json'}")
        print("\n" + "="*70)

    except Exception as e:
        print("\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

