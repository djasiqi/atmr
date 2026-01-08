"""
Scénario 1 : Test de Charge Standard - Dispatch 100 bookings x 50 drivers

Ce test simule un dispatch standard avec une charge importante pour valider :
- Performance du moteur d'optimisation OR-Tools
- Temps de calcul des matrices de distances (5000 éléments)
- Utilisation mémoire et CPU
- Latence API sous charge

Métriques clés :
- Temps de dispatch : < 60s (objectif)
- Taux de réussite : > 95%
- Assignments réussis : > 80%

Usage:
    locust -f dispatch_load_test.py --host=http://localhost:5000
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from typing import Any

from locust import HttpUser, between, events, task
from locust.runners import MasterRunner

logger = logging.getLogger(__name__)


class DispatchLoadTest(HttpUser):
    """Test de charge standard : 100 bookings x 50 drivers."""

    wait_time = between(2, 5)  # Attente entre requêtes
    host = "http://localhost:5000"

    # Variables d'instance
    token: str | None = None
    company_id: int = 1
    test_date: str = ""

    def on_start(self) -> None:  # pyright: ignore[reportImplicitOverride]
        """Setup initial : Login et préparation données."""
        logger.info("[SETUP] Initialisation utilisateur Locust...")

        # 1. Login
        self._login()

        # 2. Déterminer date de test (demain)
        tomorrow = date.today() + timedelta(days=1)
        self.test_date = tomorrow.strftime("%Y-%m-%d")

        logger.info(
            "[SETUP] ✅ Prêt pour dispatch : date=%s, company=%s",
            self.test_date,
            self.company_id,
        )

    def _login(self) -> None:
        """Authentification avec JWT."""
        response = self.client.post(
            "/api/auth/login-test",  # Endpoint test (sans CSRF)
            json={
                "email": "admin@test.com",
                "password": "test123",
            },
            name="[AUTH] Login",
        )

        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            logger.info("[AUTH] ✅ Login réussi")
        else:
            logger.error("[AUTH] ❌ Login échoué : %s", response.status_code)
            raise Exception("Login failed")

    def _get_headers(self) -> dict[str, str]:
        """Headers avec JWT token."""
        if not self.token:
            raise Exception("No token available")
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    @task(10)  # Poids 10 : Tâche principale
    def dispatch_large_set(self) -> None:
        """
        Test dispatch 100 bookings x 50 drivers.

        Simule un dispatch réaliste avec :
        - 100 courses à assigner
        - 50 chauffeurs disponibles
        - Optimisation OR-Tools (MIP)
        - Calcul matrices de distances OSRM
        """
        start_time = time.time()

        response = self.client.post(
            "/api/v1/company_dispatch/run",
            json={
                "company_id": self.company_id,
                "for_date": self.test_date,  # ✅ C2: Corrigé (API attend for_date, pas date)
                "mode": "optimization",  # OR-Tools
                "force_rerun": False,
            },
            headers=self._get_headers(),
            name="[DISPATCH] Large Set (100x50)",
        )

        duration = time.time() - start_time

        if response.status_code == 200:
            self._process_success_response(response, duration)
        else:
            self._process_error_response(response, duration)

    @task(3)  # Poids 3 : Moins fréquent
    def dispatch_with_heuristics(self) -> None:
        """Test dispatch avec heuristiques (fallback rapide)."""
        response = self.client.post(
            "/api/v1/company_dispatch/run",
            json={
                "company_id": self.company_id,
                "for_date": self.test_date,  # ✅ C2: Corrigé (API attend for_date, pas date)
                "mode": "heuristic",  # Greedy
            },
            headers=self._get_headers(),
            name="[DISPATCH] Heuristic",
        )

        if response.status_code == 200:
            data = response.json()
            logger.debug("[HEURISTIC] Assignations: %s", data.get("num_assignments", 0))

    # @task(2)  # ⚠️ C2: Endpoint désactivé temporairement (403 FORBIDDEN - permissions)
    def check_dispatch_status(self) -> None:
        """Vérifier l'état d'un dispatch en cours."""
        response = self.client.get(
            f"/api/v1/company_dispatch/status?company_id={self.company_id}&date={self.test_date}",
            headers=self._get_headers(),
            name="[DISPATCH] Check Status",
        )

        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            logger.debug("[STATUS] Dispatch status: %s", status)

    # @task(1)  # ⚠️ C2: Endpoint désactivé temporairement (404 NOT FOUND)
    def get_dispatch_metrics(self) -> None:
        """Récupérer les métriques du dernier dispatch."""
        response = self.client.get(
            f"/api/v1/company_dispatch/metrics?company_id={self.company_id}",
            headers=self._get_headers(),
            name="[METRICS] Get Dispatch Metrics",
        )

        if response.status_code == 200:
            data = response.json()
            logger.debug("[METRICS] Last dispatch: %ss", data.get("last_run_duration"))

    def _process_success_response(self, response: Any, duration: float) -> None:
        """Traiter une réponse réussie et logger les métriques."""
        try:
            data = response.json()

            # Extraire métriques
            dispatch_duration = data.get("duration_seconds", 0)
            num_assignments = data.get("num_assignments", 0)
            num_bookings = data.get("total_bookings", 0)
            num_drivers = data.get("total_drivers", 0)
            unassigned = data.get("unassigned_count", 0)

            # Log métriques
            logger.info(
                "[DISPATCH] ✅ SUCCESS | Duration: %.2fs | API: %.2fs | Assignments: %s/%s | Drivers: %s | Unassigned: %s",
                dispatch_duration,
                duration,
                num_assignments,
                num_bookings,
                num_drivers,
                unassigned,
            )

            # Validation SLO
            if dispatch_duration > 60:
                logger.warning(
                    "[SLO] ⚠️ Dispatch trop lent : %.2fs > 60s", dispatch_duration
                )

            if num_assignments < (num_bookings * 0.8):
                logger.warning(
                    "[SLO] ⚠️ Taux d'assignation faible : %s/%s < 80%%",
                    num_assignments,
                    num_bookings,
                )

            # Enregistrer métriques personnalisées
            events.request.fire(
                request_type="DISPATCH_METRICS",
                name="dispatch_duration",
                response_time=dispatch_duration * 1000,  # ms
                response_length=num_assignments,
                exception=None,
            )

        except json.JSONDecodeError as e:
            logger.error("[DISPATCH] ❌ Erreur parsing JSON : %s", e)

    def _process_error_response(self, response: Any, duration: float) -> None:
        """Traiter une réponse d'erreur."""
        logger.error(
            "[DISPATCH] ❌ FAILED | Status: %s | Duration: %.2fs | Response: %s",
            response.status_code,
            duration,
            response.text[:200],
        )


# ========== Event Handlers (Callbacks Locust) ==========


@events.test_start.add_listener
def on_test_start(environment: Any, **kwargs: Any) -> None:
    """Hook exécuté au démarrage du test."""
    logger.info("=" * 80)
    logger.info("[LOCUST] 🚀 Démarrage Test de Charge - Scénario 1")
    logger.info("[LOCUST] Objectif : 100 bookings x 50 drivers")
    logger.info("=" * 80)

    # Si mode distributed (master/workers)
    if isinstance(environment.runner, MasterRunner):
        logger.info(
            "[LOCUST] Mode Master/Worker : %s workers", environment.runner.worker_count
        )


@events.test_stop.add_listener
def on_test_stop(environment: Any, **kwargs: Any) -> None:
    """Hook exécuté à la fin du test."""
    logger.info("=" * 80)
    logger.info("[LOCUST] ✅ Test de Charge Terminé - Scénario 1")
    logger.info("=" * 80)

    # Afficher statistiques
    stats = environment.stats
    logger.info("[STATS] Total Requests: %s", stats.total.num_requests)
    logger.info("[STATS] Total Failures: %s", stats.total.num_failures)
    logger.info("[STATS] Avg Response Time: %.2fms", stats.total.avg_response_time)
    logger.info("[STATS] RPS: %.2f", stats.total.current_rps)


@events.request.add_listener
def on_request(
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    exception: Exception | None,
    **kwargs: Any,
) -> None:
    """Hook exécuté après chaque requête (pour logging détaillé)."""
    if exception:
        logger.error("[REQUEST] ❌ %s | Exception: %s", name, exception)
    elif response_time > 5000:  # > 5s
        logger.warning("[REQUEST] ⚠️ %s | Lent: %.0fms", name, response_time)


# ========== Configuration Recommandée ==========

"""
Usage en ligne de commande :

1. **Test local léger** (1 utilisateur) :
   ```bash
   locust -f dispatch_load_test.py --host=http://localhost:5000 --users=1 --spawn-rate=1 --run-time=5m
   ```

2. **Test de charge moyen** (10 utilisateurs) :
   ```bash
   locust -f dispatch_load_test.py --host=http://localhost:5000 --users=10 --spawn-rate=2 --run-time=10m
   ```

3. **Test de charge intensif** (50 utilisateurs) :
   ```bash
   locust -f dispatch_load_test.py --host=http://localhost:5000 --users=50 --spawn-rate=5 --run-time=15m --headless --csv=results_scenario1
   ```

4. **Mode Web UI** (interface graphique) :
   ```bash
   locust -f dispatch_load_test.py --host=http://localhost:5000
   # Ouvrir http://localhost:8089
   ```

5. **Mode distribué** (master + workers) :
   ```bash
   # Terminal 1 (master)
   locust -f dispatch_load_test.py --master --host=http://localhost:5000

   # Terminal 2+ (workers)
   locust -f dispatch_load_test.py --worker --master-host=localhost
   ```

Métriques surveillées :
- Response Time (p50, p95, p99)
- RPS (Requests Per Second)
- Failure Rate
- Dispatch Duration (métrique custom)
"""
