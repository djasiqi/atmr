"""✅ 3.4: Tâches Celery pour profiling automatique CPU/mémoire.

Profiling hebdomadaire pour identifier les top-10 fonctions chaudes
et optimiser les performances de l'application.
"""

from __future__ import annotations

import cProfile
import logging
import os
import pstats
import time
from datetime import UTC, datetime, timedelta
from io import StringIO
from typing import Any

try:
    import psutil  # type: ignore[import-untyped]

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

from celery import Task

from celery_app import celery, get_flask_app

logger = logging.getLogger(__name__)

# ✅ 3.4: Constantes
TOP_FUNCTIONS_COUNT = 10
PROFILING_DURATION_SECONDS = int(os.getenv("PROFILING_DURATION_SECONDS", "30"))  # 30s par défaut


def _collect_system_metrics() -> dict[str, Any]:
    """✅ 3.4: Collecte les métriques système (CPU, mémoire) si psutil disponible."""
    if not PSUTIL_AVAILABLE or psutil is None:
        return {
            "cpu_percent": None,
            "memory_percent": None,
            "memory_available_mb": None,
            "memory_total_mb": None,
            "psutil_available": False,
        }
    
    try:
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_info = process.memory_info()
        
        # Mémoire système
        system_memory = psutil.virtual_memory()
        
        return {
            "cpu_percent": round(cpu_percent, 2),
            "memory_percent": round(process.memory_percent(), 2),
            "memory_available_mb": round(memory_info.rss / 1024 / 1024, 2),
            "memory_total_mb": round(system_memory.total / 1024 / 1024, 2),
            "memory_system_percent": round(system_memory.percent, 2),
            "psutil_available": True,
        }
    except Exception as e:
        logger.warning("[3.4 Profiling] Erreur collecte métriques système: %s", e)
        return {
            "cpu_percent": None,
            "memory_percent": None,
            "psutil_available": False,
            "error": str(e)[:200],
        }


def _profile_endpoints(profiler: cProfile.Profile, duration_seconds: int = 30) -> dict[str, Any]:
    """✅ 3.4: Profile quelques endpoints critiques pendant une durée définie.
    
    Args:
        profiler: Instance cProfile.Profile
        duration_seconds: Durée du profiling
        
    Returns:
        dict avec top_functions, total_time, stats
    """
    from flask import Flask
    
    app = get_flask_app()
    
    if not isinstance(app, Flask):
        logger.warning("[3.4 Profiling] App n'est pas une instance Flask valide")
        return {"error": "Invalid Flask app"}
    
    profiler.enable()
    start_time = time.time()
    
    try:
        # Simuler quelques requêtes pendant la durée définie
        with app.test_client() as client:
            request_count = 0
            while time.time() - start_time < duration_seconds:
                # Faire quelques requêtes de test
                try:
                    # GET /health (endpoint simple)
                    client.get("/health", follow_redirects=True)
                    request_count += 1
                    
                    # Attendre un peu entre les requêtes
                    time.sleep(0.1)
                except Exception as e:
                    logger.debug("[3.4 Profiling] Erreur requête test: %s", e)
                    break
        
        actual_duration = time.time() - start_time
        profiler.disable()
        
        logger.info(
            "[3.4 Profiling] Profiling terminé: %d requêtes en %.2fs",
            request_count,
            actual_duration
        )
        
        # Analyser les résultats
        stats_buffer = StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats("cumulative")
        
        # Extraire top N fonctions
        top_functions = []
        stats.print_stats(TOP_FUNCTIONS_COUNT)
        output = stats_buffer.getvalue()
        
        # Parser les résultats
        HEADER_LINES = 5
        FOOTER_LINES = 3
        lines = output.split("\n")[HEADER_LINES:-FOOTER_LINES]  # Skip header/footer
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            MIN_PARTS_COUNT = 5
            if len(parts) >= MIN_PARTS_COUNT:
                try:
                    ncalls = parts[0]
                    tottime = float(parts[1])
                    cumtime = float(parts[3])
                    filename_line = " ".join(parts[5:])
                    
                    # Extraire nom fonction (format: filename:lineno(function_name))
                    if "(" in filename_line and ")" in filename_line:
                        func_part = filename_line.split("(")[-1].rstrip(")")
                        file_part = filename_line.split("(")[0]
                        top_functions.append({
                            "function": func_part,
                            "file": file_part,
                            "ncalls": ncalls,
                            "tottime": round(tottime, 4),
                            "cumtime": round(cumtime, 4),
                            "raw": filename_line[:200],  # Limiter taille
                        })
                except (ValueError, IndexError) as e:
                    logger.debug("[3.4 Profiling] Erreur parsing ligne: %s", e)
                    continue
        
        return {
            "request_count": request_count,
            "duration_seconds": round(actual_duration, 2),
            "top_functions": top_functions[:TOP_FUNCTIONS_COUNT],
            "total_stats": {
                "total_calls": stats.total_calls,
                "primitive_calls": stats.primitive_calls,
            },
        }
        
    except Exception as e:
        profiler.disable()
        logger.exception("[3.4 Profiling] Erreur profiling endpoints: %s", e)
        return {"error": str(e)[:500]}


@celery.task(bind=True, name="tasks.profiling_tasks.run_weekly_profiling")
def run_weekly_profiling(self: Task) -> dict[str, Any]:  # noqa: ARG001
    """✅ 3.4: Profiling automatique hebdomadaire.
    
    Profile l'application pendant une durée définie et identifie les top-10 fonctions chaudes.
    Stocke les résultats en base de données pour analyse historique.
    
    Returns:
        dict avec status, top_functions, system_metrics, profiling_results
    """
    try:
        app = get_flask_app()
        with app.app_context():
            from ext import db
            
            logger.info("[3.4 Profiling] Début profiling hebdomadaire...")
            
            # Collecter métriques système avant profiling
            system_metrics_before = _collect_system_metrics()
            
            # Créer profiler
            profiler = cProfile.Profile()
            
            # Profiler pendant la durée définie
            profiling_results = _profile_endpoints(
                profiler,
                duration_seconds=PROFILING_DURATION_SECONDS
            )
            
            # Collecter métriques système après profiling
            system_metrics_after = _collect_system_metrics()
            
            if "error" in profiling_results:
                logger.error("[3.4 Profiling] Erreur profiling: %s", profiling_results["error"])
                return {
                    "status": "error",
                    "error": profiling_results["error"],
                }
            
            # Extraire top fonctions
            top_functions = profiling_results.get("top_functions", [])
            
            logger.info(
                "[3.4 Profiling] ✅ Profiling terminé: %d fonctions identifiées",
                len(top_functions)
            )
            
            # Stocker les résultats en base de données
            profiling_record = None
            try:
                from models.profiling_metrics import ProfilingMetrics
                
                profiling_record = ProfilingMetrics()
                profiling_record.profiling_date = datetime.now(UTC)
                profiling_record.duration_seconds = profiling_results.get("duration_seconds", 0)
                profiling_record.request_count = profiling_results.get("request_count", 0)
                profiling_record.top_functions = top_functions
                profiling_record.system_metrics_before = system_metrics_before
                profiling_record.system_metrics_after = system_metrics_after
                profiling_record.total_stats = profiling_results.get("total_stats", {})
                db.session.add(profiling_record)
                db.session.commit()
                
                logger.info(
                    "[3.4 Profiling] ✅ Métriques stockées (ID: %s)",
                    profiling_record.id
                )
            except ImportError:
                # Modèle ProfilingMetrics n'existe pas encore - logger mais continuer
                logger.warning(
                    "[3.4 Profiling] ⚠️ Modèle ProfilingMetrics non trouvé - résultats non stockés"
                )
            except Exception as e:
                logger.exception("[3.4 Profiling] Erreur stockage métriques: %s", e)
                db.session.rollback()
            
            # Générer rapport textuel
            report_lines = []
            separator_line = "=" * 80
            separator_dash = "-" * 80
            report_lines.append(separator_line)
            report_lines.append("✅ 3.4 PROFILING HEBDOMADAIRE")
            report_lines.append(separator_line)
            report_lines.append(f"Date: {datetime.now(UTC).isoformat()}")
            report_lines.append(f"Durée: {profiling_results.get('duration_seconds', 0)}s")
            report_lines.append(f"Requêtes: {profiling_results.get('request_count', 0)}")
            report_lines.append("")
            
            if top_functions:
                report_lines.append("TOP 10 FONCTIONS CHAUDES:")
                report_lines.append(separator_dash)
                for i, func in enumerate(top_functions, 1):
                    func_name = func.get("function", "unknown")
                    cumtime = func.get("cumtime", 0)
                    ncalls = func.get("ncalls", "0")
                    report_lines.append(
                        f"{i:2d}. {func_name:50s} {cumtime:8.4f}s ({ncalls} calls)"
                    )
            else:
                report_lines.append("⚠️ Aucune fonction identifiée")
            
            report_lines.append("")
            report_lines.append("MÉTRIQUES SYSTÈME:")
            report_lines.append(separator_dash)
            if system_metrics_before.get("psutil_available"):
                cpu_percent = system_metrics_before.get("cpu_percent", 0)
                memory_mb = system_metrics_before.get("memory_available_mb", 0)
                memory_percent = system_metrics_before.get("memory_percent", 0)
                report_lines.append(f"CPU: {cpu_percent}%")
                report_lines.append(f"Mémoire: {memory_mb} MB ({memory_percent}%)")
            else:
                report_lines.append("⚠️ psutil non disponible - métriques système limitées")
            
            report_lines.append(separator_line)
            report_text = "\n".join(report_lines)
            
            logger.info("\n%s", report_text)
            
            return {
                "status": "success",
                "profiling_date": datetime.now(UTC).isoformat(),
                "duration_seconds": profiling_results.get("duration_seconds", 0),
                "request_count": profiling_results.get("request_count", 0),
                "top_functions": top_functions,
                "system_metrics_before": system_metrics_before,
                "system_metrics_after": system_metrics_after,
                "report": report_text,
                "profiling_id": profiling_record.id if profiling_record else None,
            }
            
    except Exception as e:
        logger.exception("[3.4 Profiling] ❌ Erreur profiling hebdomadaire: %s", e)
        raise


@celery.task(bind=True, name="tasks.profiling_tasks.generate_profiling_report")
def generate_profiling_report(self: Task, days: int = 7) -> dict[str, Any]:  # noqa: ARG001
    """✅ 3.4: Génère un rapport consolidé sur les dernières X semaines de profiling.
    
    Args:
        days: Nombre de jours à analyser (défaut: 7)
        
    Returns:
        dict avec rapport consolidé, trends, recommendations
    """
    try:
        app = get_flask_app()
        with app.app_context():
            from models.profiling_metrics import ProfilingMetrics
            
            cutoff_date = datetime.now(UTC) - timedelta(days=days)
            
            logger.info(
                "[3.4 Profiling] Génération rapport profiling (derniers %d jours)",
                days
            )
            
            # Récupérer tous les profils récents
            recent_profiles = ProfilingMetrics.query.filter(
                ProfilingMetrics.profiling_date >= cutoff_date
            ).order_by(ProfilingMetrics.profiling_date.desc()).all()
            
            if not recent_profiles:
                return {
                    "status": "success",
                    "message": "Aucun profil récent trouvé",
                    "days_analyzed": days,
                    "profile_count": 0,
                }
            
            # Analyser les top fonctions les plus fréquentes
            function_counts: dict[str, int] = {}
            function_times: dict[str, list[float]] = {}
            
            for profile in recent_profiles:
                top_functions = profile.top_functions or []
                for func in top_functions:
                    func_name = func.get("function", "unknown")
                    cumtime = func.get("cumtime", 0.0)
                    
                    function_counts[func_name] = function_counts.get(func_name, 0) + 1
                    if func_name not in function_times:
                        function_times[func_name] = []
                    function_times[func_name].append(cumtime)
            
            # Identifier les fonctions les plus fréquemment chaudes
            most_frequent = sorted(
                function_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:TOP_FUNCTIONS_COUNT]
            
            # Calculer temps moyen pour chaque fonction
            avg_times = {}
            for func_name, times in function_times.items():
                if times:
                    avg_times[func_name] = sum(times) / len(times)
            
            # Générer recommandations
            recommendations = []
            HOT_FUNCTION_THRESHOLD = 0.5  # 50% des profils
            SLOW_FUNCTION_THRESHOLD_SECONDS = 1.0  # 1 seconde
            CRITICAL_THRESHOLD_SECONDS = 2.0  # 2 secondes
            
            for func_name, count in most_frequent:
                appearance_rate = count / len(recent_profiles) if recent_profiles else 0
                if appearance_rate >= HOT_FUNCTION_THRESHOLD:  # Apparaît dans >50% des profils
                    avg_time = avg_times.get(func_name, 0.0)
                    if avg_time > SLOW_FUNCTION_THRESHOLD_SECONDS:  # > 1 seconde
                        recommendations.append({
                            "function": func_name,
                            "reason": f"Fonction chaude apparaissant dans {count}/{len(recent_profiles)} profils (temps moyen: {avg_time:.2f}s)",
                            "priority": "high" if avg_time > CRITICAL_THRESHOLD_SECONDS else "medium",
                        })
            
            return {
                "status": "success",
                "days_analyzed": days,
                "profile_count": len(recent_profiles),
                "most_frequent_hot_functions": [
                    {
                        "function": func_name,
                        "appearance_count": count,
                        "appearance_rate": round((count / len(recent_profiles)) * 100, 1),
                        "avg_time_seconds": round(avg_times.get(func_name, 0.0), 4),
                    }
                    for func_name, count in most_frequent
                ],
                "recommendations": recommendations,
                "last_profiling_date": recent_profiles[0].profiling_date.isoformat() if recent_profiles else None,
            }
            
    except ImportError:
        logger.warning("[3.4 Profiling] Modèle ProfilingMetrics non trouvé")
        return {
            "status": "error",
            "error": "ProfilingMetrics model not found",
        }
    except Exception as e:
        logger.exception("[3.4 Profiling] ❌ Erreur génération rapport: %s", e)
        raise

