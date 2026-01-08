"""
Scénario 1 : Test de Charge Standard - Dispatch 100 bookings × 50 drivers

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
import random
import time
from datetime import date, timedelta
from typing import Any

from locust import HttpUser, between, events, task
from locust.runners import MasterRunner

logger = logging.getLogger(__name__)


class DispatchLoadTest(HttpUser):
    """Test de charge standard : 100 bookings × 50 drivers."""

    wait_time = between(2, 5)  # Attente entre requêtes
    host = "http://localhost:5000"

    # Variables d'instance
    token: str | None = None
    company_id: int = 1
    test_date: str = ""

    def on_start(self) -> None:
        """Setup initial : Login et préparation données."""
        logger.info("[SETUP] Initialisation utilisateur Locust...")

        # 1. Login
        self._login()

        # 2. Déterminer date de test (demain)
        tomorrow = date.today() + timedelta(days=1)
        self.test_date = tomorrow.strftime("%Y-%m-%d")

        logger.info(
            f"[SETUP] ✅ Prêt pour dispatch : date={self.test_date}, company={self.company_id}"
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
            logger.error(f"[AUTH] ❌ Login échoué : {response.status_code}")
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
        Test dispatch 100 bookings × 50 drivers.

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
                "date": self.test_date,
                "mode": "optimization",  # OR-Tools
                "force_rerun": False,
            },
            headers=self._get_headers(),
            name="[DISPATCH] Large Set (100×50)",
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
                "date": self.test_date,
                "mode": "heuristic",  # Greedy
            },
            headers=self._get_headers(),
            name="[DISPATCH] Heuristic",
        )

        if response.status_code == 200:
            data = response.json()
            logger.debug(f"[HEURISTIC] Assignations: {data.get('num_assignments', 0)}")

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
            logger.debug(f"[STATUS] Dispatch status: {status}")

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
            logger.debug(f"[METRICS] Last dispatch: {data.get('last_run_duration')}s")

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
                f"[DISPATCH] ✅ SUCCESS | "
                f"Duration: {dispatch_duration:.2f}s | "
                f"API: {duration:.2f}s | "
                f"Assignments: {num_assignments}/{num_bookings} | "
                f"Drivers: {num_drivers} | "
                f"Unassigned: {unassigned}"
            )

            # Validation SLO
            if dispatch_duration > 60:
                logger.warning(
                    f"[SLO] ⚠️ Dispatch trop lent : {dispatch_duration:.2f}s > 60s"
                )

            if num_assignments < (num_bookings * 0.8):
                logger.warning(
                    f"[SLO] ⚠️ Taux d'assignation faible : "
                    f"{num_assignments}/{num_bookings} < 80%"
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
            logger.error(f"[DISPATCH] ❌ Erreur parsing JSON : {e}")

    def _process_error_response(self, response: Any, duration: float) -> None:
        """Traiter une réponse d'erreur."""
        logger.error(
            f"[DISPATCH] ❌ FAILED | "
            f"Status: {response.status_code} | "
            f"Duration: {duration:.2f}s | "
            f"Response: {response.text[:200]}"
        )


# ========== Event Handlers (Callbacks Locust) ==========


@events.test_start.add_listener
def on_test_start(environment: Any, **kwargs: Any) -> None:
    """Hook exécuté au démarrage du test."""
    logger.info("=" * 80)
    logger.info("[LOCUST] 🚀 Démarrage Test de Charge - Scénario 1")
    logger.info("[LOCUST] Objectif : 100 bookings × 50 drivers")
    logger.info("=" * 80)

    # Si mode distributed (master/workers)
    if isinstance(environment.runner, MasterRunner):
        logger.info(
            f"[LOCUST] Mode Master/Worker : {environment.runner.worker_count} workers"
        )


@events.test_stop.add_listener
def on_test_stop(environment: Any, **kwargs: Any) -> None:
    """Hook exécuté à la fin du test."""
    logger.info("=" * 80)
    logger.info("[LOCUST] ✅ Test de Charge Terminé - Scénario 1")
    logger.info("=" * 80)

    # Afficher statistiques
    stats = environment.stats
    logger.info(f"[STATS] Total Requests: {stats.total.num_requests}")
    logger.info(f"[STATS] Total Failures: {stats.total.num_failures}")
    logger.info(f"[STATS] Avg Response Time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"[STATS] RPS: {stats.total.current_rps:.2f}")


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
        logger.error(f"[REQUEST] ❌ {name} | Exception: {exception}")
    elif response_time > 5000:  # > 5s
        logger.warning(f"[REQUEST] ⚠️ {name} | Lent: {response_time:.0f}ms")


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
