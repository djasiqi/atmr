#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Script d'analyse des données du Shadow Mode.

Génère des rapports détaillés et des visualisations pour comparer
les performances du DQN avec le système actuel.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_shadow_data(log_dir: str, start_date: str, end_date: str) -> tuple:
    """Charge les données shadow mode pour une période donnée.
    
    Args:
        log_dir: Répertoire des logs
        start_date: Date de début (YYYYMMDD)
        end_date: Date de fin (YYYYMMDD)
        
    Returns:
        (predictions DataFrame, comparisons DataFrame)

    """
    log_path = Path(log_dir)

    # Générer la liste des dates
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    # Charger toutes les prédictions
    all_predictions = []
    for date in dates:
        pred_file = log_path / f"predictions_{date}.jsonl"
        if pred_file.exists():
            with Path(pred_file, encoding="utf-8").open() as f:
                for line in f:
                    pred = json.loads(line)
                    pred["date"] = date
                    all_predictions.append(pred)

    # Charger toutes les comparaisons
    all_comparisons = []
    for date in dates:
        comp_file = log_path / f"comparisons_{date}.jsonl"
        if comp_file.exists():
            with Path(comp_file, encoding="utf-8").open() as f:
                for line in f:
                    comp = json.loads(line)
                    comp["date"] = date
                    all_comparisons.append(comp)

    predictions_df = pd.DataFrame(all_predictions) if all_predictions else pd.DataFrame()
    comparisons_df = pd.DataFrame(all_comparisons) if all_comparisons else pd.DataFrame()

    return predictions_df, comparisons_df


def analyze_agreement_rates(comparisons_df: pd.DataFrame) -> dict:
    """Analyse les taux d'accord entre DQN et système actuel."""
    if comparisons_df.empty:
        return {}

    total = len(comparisons_df)
    agreements = comparisons_df["agreement"].sum()
    agreement_rate = agreements / total if total > 0 else 0

    # Par date
    daily_agreement = comparisons_df.groupby("date")["agreement"].mean()

    # Par confiance
    if "confidence" in comparisons_df.columns:
        high_conf = comparisons_df[comparisons_df["confidence"] > 0.8]
        low_conf = comparisons_df[comparisons_df["confidence"] < 0.3]

        high_conf_agreement = high_conf["agreement"].mean() if len(high_conf) > 0 else 0
        low_conf_agreement = low_conf["agreement"].mean() if len(low_conf) > 0 else 0
    else:
        high_conf_agreement = 0
        low_conf_agreement = 0

    return {
        "overall_agreement_rate": agreement_rate,
        "total_comparisons": total,
        "agreements": int(agreements),
        "daily_agreement": daily_agreement.to_dict(),
        "high_confidence_agreement": high_conf_agreement,
        "low_confidence_agreement": low_conf_agreement
    }


def analyze_action_distribution(predictions_df: pd.DataFrame, comparisons_df: pd.DataFrame) -> dict:
    """Analyse la distribution des actions (assign vs wait)."""
    if predictions_df.empty or comparisons_df.empty:
        return {}

    # DQN actions
    dqn_assigns = (predictions_df["action_type"] == "assign").sum()
    dqn_waits = (predictions_df["action_type"] == "wait").sum()
    dqn_total = len(predictions_df)

    # Actual actions
    actual_assigns = comparisons_df["actual_driver_id"].notna().sum()
    actual_waits = comparisons_df["actual_driver_id"].isna().sum()
    actual_total = len(comparisons_df)

    return {
        "dqn": {
            "assigns": int(dqn_assigns),
            "waits": int(dqn_waits),
            "total": int(dqn_total),
            "assign_rate": float(dqn_assigns / dqn_total) if dqn_total > 0 else 0
        },
        "actual": {
            "assigns": int(actual_assigns),
            "waits": int(actual_waits),
            "total": int(actual_total),
            "assign_rate": float(actual_assigns / actual_total) if actual_total > 0 else 0
        }
    }


def generate_visualizations(
    predictions_df: pd.DataFrame,
    comparisons_df: pd.DataFrame,
    output_dir: str
) -> list:
    """Génère des visualisations des données shadow mode."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_files = []

    if comparisons_df.empty:
        print("⚠️  Pas de données de comparaison pour les visualisations")
        return generated_files

    # 1. Taux d'accord par jour
    if "date" in comparisons_df.columns:
        daily_agreement = comparisons_df.groupby("date")["agreement"].mean()

        plt.figure(figsize=(12, 6))
        plt.plot(daily_agreement.index, daily_agreement.values, marker="o", linewidth=2)
        plt.axhline(y=daily_agreement.mean(), color="r", linestyle="--", label="Moyenne")
        plt.title("Taux d'accord DQN vs Système Actuel (par jour)")
        plt.xlabel("Date")
        plt.ylabel("Taux d'accord")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        agreement_file = output_path / "agreement_rate_daily.png"
        plt.savefig(agreement_file, dpi=0.300)
        plt.close()
        generated_files.append(str(agreement_file))
        print("✅ Graphique sauvegardé: {agreement_file}")

    # 2. Distribution des actions
    action_dist = analyze_action_distribution(predictions_df, comparisons_df)

    if action_dist:
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # DQN
        dqn_data = [action_dist["dqn"]["assigns"], action_dist["dqn"]["waits"]]
        ax1.pie(dqn_data, labels=["Assign", "Wait"], autopct="%1.1f%%", startangle=90)
        ax1.set_title(f'DQN Actions\n(Total: {action_dist["dqn"]["total"]})')

        # Actual
        actual_data = [action_dist["actual"]["assigns"], action_dist["actual"]["waits"]]
        ax2.pie(actual_data, labels=["Assign", "Wait"], autopct="%1.1f%%", startangle=90)
        ax2.set_title(f'Système Actuel\n(Total: {action_dist["actual"]["total"]})')

        plt.tight_layout()

        actions_file = output_path / "action_distribution.png"
        plt.savefig(actions_file, dpi=0.300)
        plt.close()
        generated_files.append(str(actions_file))
        print("✅ Graphique sauvegardé: {actions_file}")

    # 3. Confiance vs Accord
    if "confidence" in comparisons_df.columns:
        bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
        labels = ["Très faible", "Faible", "Moyen", "Élevé", "Très élevé"]
        comparisons_df["confidence_bin"] = pd.cut(
            comparisons_df["confidence"],
            bins=bins,
            labels=labels
        )

        conf_agreement = comparisons_df.groupby("confidence_bin", observed=True)["agreement"].agg(["mean", "count"])

        plt.figure(figsize=(10, 6))
        x = range(len(conf_agreement))
        plt.bar(x, conf_agreement["mean"], alpha=0.7)
        plt.plot(x, conf_agreement["mean"], "ro-", linewidth=2)

        # Ajouter les counts au-dessus des barres
        for i, (mean_val, count_val) in enumerate(zip(conf_agreement["mean"], conf_agreement["count"], strict=False)):
            plt.text(i, mean_val + 0.02, f"n={int(count_val)}", ha="center")

        plt.xticks(x, conf_agreement.index)
        plt.title("Taux d'accord par niveau de confiance")
        plt.xlabel("Niveau de confiance DQN")
        plt.ylabel("Taux d'accord")
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        conf_file = output_path / "confidence_vs_agreement.png"
        plt.savefig(conf_file, dpi=0.300)
        plt.close()
        generated_files.append(str(conf_file))
        print("✅ Graphique sauvegardé: {conf_file}")

    return generated_files


def generate_report(
    predictions_df: pd.DataFrame,
    comparisons_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_file: str
) -> None:
    """Génère un rapport JSON complet."""
    agreement_analysis = analyze_agreement_rates(comparisons_df)
    action_analysis = analyze_action_distribution(predictions_df, comparisons_df)

    report = {
        "metadata": {
            "start_date": start_date,
            "end_date": end_date,
            "generated_at": datetime.utcnow().isoformat(),
            "total_predictions": len(predictions_df),
            "total_comparisons": len(comparisons_df)
        },
        "agreement_analysis": agreement_analysis,
        "action_distribution": action_analysis,
        "summary": {
            "shadow_mode_status": "active",
            "model_path": "data/rl/models/dqn_best.pth",
            "recommendation": _generate_recommendation(agreement_analysis, action_analysis)
        }
    }

    with Path(output_file, "w", encoding="utf-8").open() as f:
        json.dump(report, f, indent=2)

    print("\n📄 Rapport sauvegardé: {output_file}")


def _generate_recommendation(agreement_analysis: dict, action_analysis: dict) -> str:
    """Génère une recommandation basée sur l'analyse."""
    if not agreement_analysis or not action_analysis:
        return "Pas assez de données pour générer une recommandation"

    agreement_rate = agreement_analysis.get("overall_agreement_rate", 0)

    if agreement_rate > 0.75:
        return "✅ Taux d'accord élevé (>75%). Prêt pour Phase 2 (A/B Testing 50/50)"
    if agreement_rate > 0.60:
        return "⚠️  Taux d'accord moyen (60-75%). Analyser les désaccords avant Phase 2"
    return "❌ Taux d'accord faible (<60%). Investiguer les différences avant déploiement"


def main():
    parser = argparse.ArgumentParser(
        description="Analyser les données du Shadow Mode DQN"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="data/rl/shadow_mode",
        help="Répertoire des logs shadow mode"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Date de début (YYYYMMDD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.utcnow().strftime("%Y%m%d"),
        help="Date de fin (YYYYMMDD, défaut: aujourd'hui)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/rl/shadow_mode/analysis",
        help="Répertoire de sortie pour les rapports"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("📊 ANALYSE SHADOW MODE DQN")
    print("=" * 70)
    print("Période : {args.start_date} → {args.end_date}")
    print("Log dir : {args.log_dir}")
    print()

    # Charger les données
    print("📥 Chargement des données...")
    predictions_df, comparisons_df = load_shadow_data(
        args.log_dir,
        args.start_date,
        args.end_date
    )

    print("   Prédictions : {len(predictions_df)}")
    print("   Comparaisons: {len(comparisons_df)}")
    print()

    if predictions_df.empty and comparisons_df.empty:
        print("❌ Aucune donnée trouvée pour la période spécifiée")
        return 1

    # Analyse
    print("🔍 Analyse des données...")
    agreement_analysis = analyze_agreement_rates(comparisons_df)

    if agreement_analysis:
        print("\n📈 TAUX D'ACCORD:")
        print("   Global    : {agreement_analysis['overall_agreement_rate']")
        print("   Total     : {agreement_analysis['total_comparisons']} comparaisons")
        print("   Accords   : {agreement_analysis['agreements']}")

        if "high_confidence_agreement" in agreement_analysis:
            print("   High conf : {agreement_analysis['high_confidence_agreement']")
            print("   Low conf  : {agreement_analysis['low_confidence_agreement']")

    action_analysis = analyze_action_distribution(predictions_df, comparisons_df)

    if action_analysis:
        print("\n📊 DISTRIBUTION DES ACTIONS:")
        print("   DQN     : {action_analysis['dqn']['assign_rate']")
        print("   Actuel  : {action_analysis['actual']['assign_rate']")

    # Visualisations
    print("\n📊 Génération des visualisations...")
    generate_visualizations(predictions_df, comparisons_df, args.output_dir)
    print("   {len(viz_files)} graphiques générés")

    # Rapport
    report_file = Path(args.output_dir) / f"report_{args.start_date}_{args.end_date}.json"
    generate_report(predictions_df, comparisons_df, args.start_date, args.end_date, str(report_file))

    print("\n" + "=" * 70)
    print("✅ ANALYSE TERMINÉE")
    print("=" * 70)

    # Recommandation
    _generate_recommendation(agreement_analysis, action_analysis)
    print("\n💡 RECOMMANDATION:")
    print("   {recommendation}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

