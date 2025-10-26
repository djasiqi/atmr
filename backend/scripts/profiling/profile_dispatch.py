"""Script de profiling pour mesurer les performances du dispatch.

Usage:
    python scripts/profiling/profile_dispatch.py
"""
# print() et date.today() sont intentionnels dans les scripts de profiling

import sys
import time
from datetime import date

from sqlalchemy import event
from sqlalchemy.engine import Engine

# Liste des queries exécutées
queries_log = []
query_count = 0


@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Enregistre le temps avant l'exécution d'une query."""
    context._query_start_time = time.time()


@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Calcule le temps d'exécution et log les queries lentes."""
    global query_count
    query_count += 1

    total_time = time.time() - context._query_start_time

    # Logger queries > 50ms (lentes)
    if total_time > 0.050:
        queries_log.append({
            "query": statement[:300],  # Limiter taille
            "time": total_time,
            "params": str(parameters)[:150] if parameters else ""
        })


def profile_dispatch(company_id=1, for_date=None):
    """Profile un dispatch complet.

    Args:
        company_id: ID de l'entreprise
        for_date: Date du dispatch (YYYY-MM-DD)

    Returns:
        Dict avec métriques de performance

    """
    # Import après event listeners
    from app import create_app
    from services.unified_dispatch import engine as dispatch_engine

    app = create_app()

    with app.app_context():
        global queries_log, query_count
        queries_log = []
        query_count = 0

        if for_date is None:
            for_date = date.today().isoformat()

        print("\n" + "="*70)
        print("PROFILING DISPATCH - DEMARRAGE")
        print("="*70)
        print("Company ID  : {company_id}")
        print("Date        : {for_date}")
        print("Database    : {app.config.get('SQLALCHEMY_DATABASE_URI', 'N/A')[:50]}...")
        print("="*70)

        # Mesurer temps total
        start_time = time.time()

        try:
            # Exécuter dispatch (fonction directe, pas une classe)
            result = dispatch_engine.run(company_id=company_id, for_date=for_date)

            end_time = time.time()
            total_time = end_time - start_time

            # Analyser résultats
            print("\n" + "="*70)
            print("RESULTATS PROFILING")
            print("="*70)
            print("\nTemps total          : {total_time")
            print("Assignments crees    : {len(result.get('assignments', []))}")
            print("Total queries SQL    : {query_count}")
            print("Queries lentes (>50ms) : {len(queries_log)}")

            # Trier les queries lentes pour affichage et rapport
            sorted_queries = sorted(queries_log, key=lambda x: x["time"], reverse=True) if queries_log else []

            if sorted_queries:
                print("\n" + "="*70)
                print("TOP 10 QUERIES LES PLUS LENTES")
                print("="*70)

                for i, q in enumerate(sorted_queries[:10], 1):
                    print("\n{i}. Temps: {q['time']*1000")
                    print("   Query: {q['query'][:200]}")
                    if q["params"]:
                        print("   Params: {q['params'][:100]}")

            # Sauvegarder rapport détaillé
            report_path = "scripts/profiling/profiling_results.txt"
            with Path(report_path, "w", encoding="utf-8").open() as f:
                f.write("="*70 + "\n")
                f.write("PROFILING RESULTS - DISPATCH\n")
                f.write("="*70 + "\n\n")
                f.write(f"Date: {for_date}\n")
                f.write(f"Company ID: {company_id}\n")
                f.write(f"Temps total: {total_time")
                f.write(f"Total queries: {query_count}\n")
                f.write(f"Queries lentes (>50ms): {len(queries_log)}\n")
                f.write(f"Assignments créés: {len(result.get('assignments', []))}\n\n")

                if sorted_queries:
                    f.write("="*70 + "\n")
                    f.write("TOP QUERIES LENTES\n")
                    f.write("="*70 + "\n\n")

                    for i, q in enumerate(sorted_queries, 1):
                        f.write(f"{i}. Temps: {q['time']*1000")
                        f.write(f"   Query: {q['query']}\n")
                        if q["params"]:
                            f.write(f"   Params: {q['params']}\n")
                        f.write("\n")

            print("\nRapport sauvegarde dans {report_path}")
            print("="*70)

            return {
                "total_time": total_time,
                "query_count": query_count,
                "slow_queries": len(queries_log),
                "queries": sorted_queries,
                "assignments": len(result.get("assignments", []))
            }

        except Exception as e:
            print("\nERREUR: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    print("\n>> Lancement du profiling...\n")
    results = profile_dispatch(company_id=1)

    if results:
        print("\n>> Profiling termine avec succes !")
        print("\n>> Resume:")
        print("   - Temps: {results['total_time']")
        print("   - Queries: {results['query_count']}")
        print("   - Lentes: {results['slow_queries']}")
    else:
        print("\n>> Profiling echoue")
        sys.exit(1)

