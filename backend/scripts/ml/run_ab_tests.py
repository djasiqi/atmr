# ruff: noqa: T201, W293
# pyright: reportAttributeAccessIssue=false
"""
Script pour exécuter tests A/B : ML vs Heuristique.
"""
import sys
from pathlib import Path

# Ajouter backend au path
backend_path = str(Path(__file__).parent.parent.parent)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)


def run_ab_tests_from_db(limit: int = 50) -> dict:
    """
    Exécute tests A/B sur bookings existants en DB.
    
    Args:
        limit: Nombre de tests à exécuter
        
    Returns:
        dict avec résultats et métriques
    """
    from app import create_app
    from db import db
    from models.ab_test_result import ABTestResult
    from models.booking import Booking
    from models.driver import Driver
    from services.ab_testing_service import ABTestingService
    
    app = create_app()
    
    with app.app_context():
        print("\n" + "="*70)
        print("A/B TESTING : ML vs HEURISTIQUE")
        print("="*70)
        print(f"Limite : {limit} tests")
        print()
        
        # Récupérer des bookings et drivers
        bookings = Booking.query.filter(
            Booking.pickup_lat.isnot(None),
            Booking.pickup_lon.isnot(None),
            Booking.dropoff_lat.isnot(None),
            Booking.dropoff_lon.isnot(None),
        ).limit(limit).all()
        
        if not bookings:
            print("❌ Aucun booking trouvé avec coordonnées valides")
            return {}
        
        drivers = Driver.query.filter(
            Driver.is_active == True  # noqa: E712
        ).limit(limit).all()
        
        if not drivers:
            print("❌ Aucun driver actif trouvé")
            return {}
        
        print(f"✅ {len(bookings)} bookings trouvés")
        print(f"✅ {len(drivers)} drivers actifs trouvés")
        print()
        
        # Exécuter tests A/B
        results = []
        tests_to_run = min(len(bookings), len(drivers))
        
        print(f"🧪 Exécution de {tests_to_run} tests A/B...")
        print()
        
        for i, (booking, driver) in enumerate(zip(bookings[:tests_to_run], drivers[:tests_to_run], strict=False), 1):
            try:
                result = ABTestingService.run_ab_test(booking, driver)
                results.append(result)
                
                # Sauvegarder en DB
                ab_test = ABTestResult(
                    booking_id=result["booking_id"],
                    driver_id=result["driver_id"],
                    test_timestamp=result["test_timestamp"],
                    ml_delay_minutes=result["ml_delay_minutes"],
                    ml_confidence=result["ml_confidence"],
                    ml_risk_level=result["ml_risk_level"],
                    ml_prediction_time_ms=result["ml_prediction_time_ms"],
                    ml_weather_factor=result["ml_weather_factor"],
                    heuristic_delay_minutes=result["heuristic_delay_minutes"],
                    heuristic_prediction_time_ms=result["heuristic_prediction_time_ms"],
                    difference_minutes=result["difference_minutes"],
                    ml_faster=result["ml_faster"],
                    speed_advantage_ms=result["speed_advantage_ms"],
                )
                db.session.add(ab_test)
                
                if i % 10 == 0:
                    print(f"  {i}/{tests_to_run} tests complétés...")
                    db.session.commit()
            except Exception as e:
                print(f"  ❌ Erreur test {i}: {e}")
        
        db.session.commit()
        
        print(f"\n✅ {len(results)} tests A/B complétés")
        print()
        
        # Calculer métriques
        if results:
            metrics = ABTestingService.calculate_metrics(results)
            
            print("="*70)
            print("RÉSULTATS AGRÉGÉS")
            print("="*70)
            print()
            
            print("📊 PRÉDICTIONS")
            print(f"  ML moyen           : {metrics['ml_avg_delay']:.2f} min")
            print(f"  Heuristique moyen  : {metrics['heuristic_avg_delay']:.2f} min")
            print(f"  Différence absolue : {metrics['avg_difference_minutes']:.2f} min")
            print()
            
            print("⚡ PERFORMANCE")
            print(f"  ML temps moyen        : {metrics['ml_avg_time_ms']:.1f} ms")
            print(f"  Heuristique temps moy : {metrics['heuristic_avg_time_ms']:.1f} ms")
            print(f"  ML plus rapide        : {metrics['ml_faster_percentage']:.1f}%")
            print()
            
            print("🎯 QUALITÉ ML")
            print(f"  Confiance moyenne : {metrics['ml_avg_confidence']:.3f}")
            print()
            
            # Sauvegarder rapport
            report_path = Path("data/ml/ab_test_report.txt")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("RAPPORT A/B TESTING : ML vs HEURISTIQUE\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Total tests : {metrics['total_tests']}\n\n")
                
                f.write("PRÉDICTIONS\n")
                f.write(f"  ML moyen           : {metrics['ml_avg_delay']:.2f} min\n")
                f.write(f"  Heuristique moyen  : {metrics['heuristic_avg_delay']:.2f} min\n")
                f.write(f"  Différence absolue : {metrics['avg_difference_minutes']:.2f} min\n\n")
                
                f.write("PERFORMANCE\n")
                f.write(f"  ML temps moyen        : {metrics['ml_avg_time_ms']:.1f} ms\n")
                f.write(f"  Heuristique temps moy : {metrics['heuristic_avg_time_ms']:.1f} ms\n")
                f.write(f"  ML plus rapide        : {metrics['ml_faster_percentage']:.1f}%\n\n")
                
                f.write("QUALITÉ ML\n")
                f.write(f"  Confiance moyenne : {metrics['ml_avg_confidence']:.3f}\n\n")
                
                f.write("="*70 + "\n")
            
            print(f"📄 Rapport sauvegardé : {report_path}")
            print()
            
            return {
                "results": results,
                "metrics": metrics,
            }
        else:
            print("❌ Aucun résultat à analyser")
            return {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="A/B Testing: ML vs Heuristique")
    parser.add_argument("--limit", type=int, default=50, help="Nombre de tests (défaut: 50)")
    args = parser.parse_args()
    
    try:
        results = run_ab_tests_from_db(limit=args.limit)
        
        if results:
            print("="*70)
            print("✅ A/B TESTING TERMINÉ AVEC SUCCÈS !")
            print("="*70)
            sys.exit(0)
        else:
            print("⚠️  Aucun résultat produit")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

