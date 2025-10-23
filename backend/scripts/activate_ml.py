# ruff: noqa: T201
"""
Script pour activer/désactiver le ML en production.

Usage:
    # Activer ML à 10%
    python scripts/activate_ml.py --enable --percentage 10
    # Augmenter à 25%
    python scripts/activate_ml.py --percentage 25
    # Désactiver ML
    python scripts/activate_ml.py --disable
    # Voir le statut
    python scripts/activate_ml.py --status
"""
import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def activate_ml(percentage: int, dry_run: bool = False) -> None:
    """
    Active le ML avec un pourcentage de trafic donné.
    Args:
        percentage: Pourcentage de trafic (0-100)
        dry_run: Si True, simule sans appliquer
    """
    from feature_flags import FeatureFlags

    if not 0 <= percentage <= 100:
        print(f"❌ Erreur: Le pourcentage doit être entre 0 et 100 (fourni: {percentage})")
        sys.exit(1)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}🚀 Activation ML à {percentage}%")
    print("=" * 60)

    if not dry_run:
        FeatureFlags.set_ml_enabled(True)
        FeatureFlags.set_ml_traffic_percentage(percentage)

    print(f"✅ ML activé à {percentage}% du trafic")
    print(f"✅ Fallback automatique: {'Activé' if FeatureFlags.should_fallback_on_error() else 'Désactivé'}")

    # Recommandations
    print("\n📋 Recommandations:")
    if percentage < 25:
        print("   - Monitorer pendant 24h avant d'augmenter")
        print("   - Vérifier dashboard toutes les heures")
        print("   - Alertes configurées pour taux erreur > 5%")
    elif percentage < 50:
        print("   - Phase de test élargi")
        print("   - Comparer métriques ML vs heuristique")
        print("   - Collecter feedback utilisateurs")
    elif percentage < 100:
        print("   - Avant-dernière étape")
        print("   - Valider stabilité sur 48h")
        print("   - Préparer rollout 100%")
    else:
        print("   - ML activé à 100% ! 🎉")
        print("   - Monitoring continu essentiel")
        print("   - Plan de rollback prêt")

    print("\n💡 Prochaines étapes:")
    print("   1. Vérifier logs: docker logs -f atmr-api-1 | grep 'FeatureFlag'")
    print("   2. Tester: curl http://localhost:5001/api/feature-flags/status")
    print("   3. Dashboard: http://localhost:3000/ml-monitoring")

    if percentage < 100:
        next_percentage = min(percentage * 2, 100) if percentage < 50 else 100
        print(f"   4. Augmenter: python scripts/activate_ml.py --percentage {next_percentage}")

    print("=" * 60)


def deactivate_ml(dry_run: bool = False) -> None:
    """
    Désactive complètement le ML.
    Args:
        dry_run: Si True, simule sans appliquer
    """
    from feature_flags import FeatureFlags

    print(f"\n{'[DRY RUN] ' if dry_run else ''}🛑 Désactivation ML")
    print("=" * 60)

    if not dry_run:
        FeatureFlags.set_ml_enabled(False)
        FeatureFlags.set_ml_traffic_percentage(0)

    print("✅ ML désactivé")
    print("✅ Toutes les prédictions utilisent maintenant l'heuristique")

    print("\n⚠️ Impact:")
    print("   - Prédictions moins précises (heuristique simple)")
    print("   - Pas d'anticipation des retards complexes")
    print("   - Buffer ETA non optimisé")

    print("\n💡 Pour réactiver:")
    print("   python scripts/activate_ml.py --enable --percentage 10")

    print("=" * 60)


def show_status() -> None:
    """Affiche le statut actuel du système ML."""
    from feature_flags import get_feature_flags_status

    status = get_feature_flags_status()

    print("\n📊 STATUT FEATURE FLAGS ML")
    print("=" * 60)

    # Configuration
    print("\n⚙️ Configuration:")
    config = status["config"]
    print(f"   ML Activé : {'✅ Oui' if config['ML_ENABLED'] else '❌ Non'}")
    print(f"   Trafic ML : {config['ML_TRAFFIC_PERCENTAGE']}%")
    print(f"   Fallback  : {'✅ Activé' if config['FALLBACK_ON_ERROR'] else '❌ Désactivé'}")

    # Statistiques
    print("\n📈 Statistiques:")
    stats = status["stats"]
    print(f"   Total requêtes    : {stats['total_requests']}")
    print(f"   Requêtes ML       : {stats['ml_requests']} ({stats['ml_usage_rate']:.1%})")
    print(f"   Succès ML         : {stats['ml_successes']}")
    print(f"   Erreurs ML        : {stats['ml_failures']}")
    print(f"   Taux succès       : {stats['ml_success_rate']:.1%}")
    print(f"   Requêtes fallback : {stats['fallback_requests']}")

    # Santé
    print("\n🏥 Santé:")
    health = status["health"]
    health_icon = "✅" if health["status"] == "healthy" else "⚠️"
    print(f"   Statut       : {health_icon} {health['status'].upper()}")
    print(f"   Taux succès  : {health['success_rate']}")
    print(f"   Taux erreur  : {health['error_rate']}")

    # Alertes
    if stats['ml_success_rate'] < 0.95 and stats['ml_requests'] > 10:
        print("\n⚠️ ALERTES:")
        print(f"   Taux de succès bas ({stats['ml_success_rate']:.1%})")
        print("   Action recommandée: Vérifier logs et considérer rollback")

    print("=" * 60)


def main() -> None:
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(
        description="Activer/désactiver le ML en production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Activer ML à 10%
  python scripts/activate_ml.py --enable --percentage 10
  # Augmenter progressivement
  python scripts/activate_ml.py --percentage 25
  python scripts/activate_ml.py --percentage 50
  python scripts/activate_ml.py --percentage 100
  # Désactiver ML
  python scripts/activate_ml.py --disable
  # Voir le statut
  python scripts/activate_ml.py --status
  # Test (dry run)
  python scripts/activate_ml.py --enable --percentage 50 --dry-run
        """,
    )

    parser.add_argument(
        "--enable",
        action="store_true",
        help="Activer le ML",
    )
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Désactiver le ML",
    )
    parser.add_argument(
        "--percentage",
        type=int,
        default=10,
        help="Pourcentage de trafic ML (0-100, défaut: 10)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Afficher le statut actuel",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulation sans appliquer les changements",
    )

    args = parser.parse_args()

    # Si aucun argument, afficher le statut
    if not (args.enable or args.disable or args.status):
        show_status()
        return

    # Statut
    if args.status:
        show_status()
        return

    # Désactivation
    if args.disable:
        deactivate_ml(dry_run=args.dry_run)
        return

    # Activation
    if args.enable or args.percentage is not None:
        activate_ml(args.percentage, dry_run=args.dry_run)
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Opération annulée par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

