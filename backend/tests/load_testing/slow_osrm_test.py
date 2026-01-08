"""
Scénario 3 : Test OSRM Lent - Résilience avec latence 500ms

Ce test simule un environnement avec OSRM lent/dégradé pour valider :
- Timeouts appropriés (pas de hang)
- Fallback vers Haversine si OSRM fail
- Circuit breaker fonctionnel
- Cache OSRM efficace

Objectifs :
- Dispatch réussit malgré OSRM lent
- Temps total dispatch < 2min (même avec 500ms latency)
- Fallback automatique si timeout
- Pas d'erreurs 500

Métriques clés :
- OSRM latency : ~500ms (simulée)
- Dispatch duration : < 120s
- Fallback rate : Mesurer usage Haversine
- Success rate : > 90%

Usage:
    # 1. Démarrer OSRM avec latence simulée (Docker)
    docker-compose up -d osrm-slow

    # 2. Lancer test
    locust -f slow_osrm_test.py --host=http://localhost:5000
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from typing import Any

from locust import HttpUser, between, events, task

logger = logging.getLogger(__name__)


class SlowOSRMDispatchTest(HttpUser):
    """Test dispatch avec OSRM lent (500ms latency)."""

    wait_time = between(3, 7)  # Attente plus longue (OSRM lent)
    host = "http://localhost:5000"

    # Variables
    token: str | None = None
    company_id: int = 1
    test_date: str = ""

    # Compteurs pour statistiques
    osrm_success_count: int = 0
    osrm_fallback_count: int = 0
    osrm_timeout_count: int = 0

    def on_start(self) -> None:
        """Setup initial."""
        logger.info("[SETUP] Initialisation test OSRM lent...")

        # Login
        self._login()

        # Date de test
        tomorrow = date.today() + timedelta(days=1)
        self.test_date = tomorrow.strftime("%Y-%m-%d")

        logger.info(f"[SETUP] ✅ Prêt pour test OSRM lent | Date: {self.test_date}")

    def _login(self) -> None:
        """Authentification."""
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
        """Headers avec JWT."""
        if not self.token:
            raise Exception("No token available")
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    @task(10)  # Tâche principale
    def dispatch_slow_osrm(self) -> None:
        """
        Test dispatch avec OSRM lent (500ms latency).

        Le système doit :
        - Tolérer la latence OSRM
        - Utiliser le cache efficacement
        - Fallback vers Haversine si nécessaire
        - Ne pas timeout sur l'API dispatch
        """
        start_time = time.time()

        response = self.client.post(
            "/api/v1/company_dispatch/run",
            json={
                "company_id": self.company_id,
                "date": self.test_date,
                "mode": "optimization",
                "osrm_timeout": 10,  # 10s timeout pour OSRM
                "enable_fallback": True,  # Fallback Haversine si OSRM fail
            },
            headers=self._get_headers(),
            name="[DISPATCH] OSRM Slow",
            timeout=150,  # Timeout API : 2.5min
        )

        duration = time.time() - start_time

        if response.status_code == 200:
            self._process_success(response, duration)
        elif response.status_code == 504:
            # Gateway Timeout (OSRM timeout)
            logger.error(
                f"[DISPATCH] ❌ Gateway Timeout (504) | Duration: {duration:.2f}s"
            )
            self.osrm_timeout_count += 1
        else:
            logger.error(
                f"[DISPATCH] ❌ Failed | "
                f"Status: {response.status_code} | "
                f"Duration: {duration:.2f}s"
            )

    @task(5)
    def dispatch_with_cache(self) -> None:
        """
        Test dispatch avec cache OSRM chaud.

        Après le 1er dispatch, le cache doit être rempli.
        Les dispatches suivants doivent être plus rapides.
        """
        start_time = time.time()

        response = self.client.post(
            "/api/v1/company_dispatch/run",
            json={
                "company_id": self.company_id,
                "date": self.test_date,
                "mode": "optimization",
                "use_cache": True,  # Forcer utilisation cache
            },
            headers=self._get_headers(),
            name="[DISPATCH] OSRM Cached",
        )

        duration = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            cache_hit_rate = data.get("osrm_cache_hit_rate", 0)
            dispatch_duration = data.get("duration_seconds", 0)

            logger.info(
                f"[CACHE] ✅ Dispatch avec cache | "
                f"Duration: {dispatch_duration:.2f}s | "
                f"Cache hit rate: {cache_hit_rate:.1%}"
            )

            # Valider que le cache accélère vraiment
            if cache_hit_rate > 0.8 and dispatch_duration > 60:
                logger.warning(
                    f"[CACHE] ⚠️ Cache hit rate élevé mais dispatch lent : "
                    f"{dispatch_duration:.2f}s"
                )

    @task(3)
    def test_haversine_fallback(self) -> None:
        """
        Test fallback vers Haversine (distances à vol d'oiseau).

        Forcer le fallback pour vérifier qu'il fonctionne.
        """
        response = self.client.post(
            "/api/v1/company_dispatch/run",
            json={
                "company_id": self.company_id,
                "date": self.test_date,
                "mode": "optimization",
                "distance_mode": "haversine",  # Forcer Haversine
            },
            headers=self._get_headers(),
            name="[DISPATCH] Haversine Fallback",
        )

        if response.status_code == 200:
            data = response.json()
            distance_mode = data.get("distance_mode_used", "unknown")

            if distance_mode == "haversine":
                logger.info("[FALLBACK] ✅ Haversine utilisé correctement")
                self.osrm_fallback_count += 1
            else:
                logger.warning(
                    f"[FALLBACK] ⚠️ Mode attendu: haversine, obtenu: {distance_mode}"
                )

    @task(2)
    def check_osrm_health(self) -> None:
        """Vérifier la santé du service OSRM."""
        response = self.client.get(
            "/api/osrm/health",
            name="[OSRM] Health Check",
        )

        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            latency = data.get("avg_latency_ms", 0)

            logger.debug(f"[OSRM] Status: {status} | Latency: {latency:.0f}ms")

            if latency > 500:
                logger.warning(f"[OSRM] ⚠️ Latence élevée : {latency:.0f}ms")

    @task(1)
    def check_circuit_breaker(self) -> None:
        """Vérifier l'état du circuit breaker OSRM."""
        response = self.client.get(
            "/api/osrm/circuit-breaker",
            headers=self._get_headers(),
            name="[OSRM] Circuit Breaker Status",
        )

        if response.status_code == 200:
            data = response.json()
            state = data.get("state", "unknown")  # closed, open, half-open
            failure_rate = data.get("failure_rate", 0)

            logger.debug(
                f"[CIRCUIT BREAKER] State: {state} | Failure rate: {failure_rate:.1%}"
            )

            if state == "open":
                logger.warning(
                    "[CIRCUIT BREAKER] ⚠️ Circuit ouvert (trop d'échecs OSRM)"
                )

    def _process_success(self, response: Any, duration: float) -> None:
        """Traiter succès et analyser métriques."""
        try:
            data = response.json()

            # Métriques
            dispatch_duration = data.get("duration_seconds", 0)
            num_assignments = data.get("num_assignments", 0)
            num_bookings = data.get("total_bookings", 0)
            distance_mode = data.get("distance_mode_used", "osrm")
            osrm_calls = data.get("osrm_api_calls", 0)
            osrm_cache_hits = data.get("osrm_cache_hits", 0)

            # Calculer cache hit rate
            cache_hit_rate = 0
            if osrm_calls + osrm_cache_hits > 0:
                cache_hit_rate = osrm_cache_hits / (osrm_calls + osrm_cache_hits)

            # Log détaillé
            logger.info(
                f"[DISPATCH] ✅ SUCCESS | "
                f"Duration: {dispatch_duration:.2f}s | "
                f"API: {duration:.2f}s | "
                f"Assignments: {num_assignments}/{num_bookings} | "
                f"Distance mode: {distance_mode} | "
                f"OSRM calls: {osrm_calls} | "
                f"Cache hits: {osrm_cache_hits} ({cache_hit_rate:.1%})"
            )

            # Comptabiliser
            if distance_mode == "osrm":
                self.osrm_success_count += 1
            elif distance_mode == "haversine":
                self.osrm_fallback_count += 1

            # Validations SLO
            if dispatch_duration > 120:
                logger.error(
                    f"[SLO] ❌ Dispatch trop lent avec OSRM lent : "
                    f"{dispatch_duration:.2f}s > 120s"
                )

            if num_assignments < (num_bookings * 0.7):
                logger.warning(
                    f"[SLO] ⚠️ Taux d'assignation faible : "
                    f"{num_assignments}/{num_bookings} < 70%"
                )

            # Métriques custom
            events.request.fire(
                request_type="OSRM_METRICS",
                name=f"osrm_{distance_mode}_duration",
                response_time=dispatch_duration * 1000,
                response_length=osrm_calls,
                exception=None,
            )

        except json.JSONDecodeError as e:
            logger.error(f"[DISPATCH] ❌ Erreur parsing JSON : {e}")


# ========== Event Handlers ==========


@events.test_start.add_listener
def on_test_start(environment: Any, **kwargs: Any) -> None:
    """Hook démarrage test."""
    logger.info("=" * 80)
    logger.info("[LOCUST] 🚀 Démarrage Test OSRM Lent - Scénario 3")
    logger.info("[LOCUST] Objectif : Valider résilience avec OSRM 500ms")
    logger.info("[LOCUST] Validation : Timeouts, Fallback, Circuit Breaker, Cache")
    logger.info("=" * 80)


@events.test_stop.add_listener
def on_test_stop(environment: Any, **kwargs: Any) -> None:
    """Hook fin test avec statistiques OSRM."""
    logger.info("=" * 80)
    logger.info("[LOCUST] ✅ Test OSRM Lent Terminé - Scénario 3")
    logger.info("=" * 80)

    # Stats globales
    stats = environment.stats
    logger.info(f"[STATS] Total Requests: {stats.total.num_requests}")
    logger.info(f"[STATS] Total Failures: {stats.total.num_failures}")
    logger.info(f"[STATS] Avg Response Time: {stats.total.avg_response_time:.2f}ms")

    # Stats OSRM (approximatif, basé sur 1 user)
    # Dans un vrai test, il faudrait agréger entre tous les users
    logger.info("=" * 80)
    logger.info("[OSRM STATS] Approximation (1 user) :")
    logger.info(f"  - OSRM Success: {SlowOSRMDispatchTest.osrm_success_count}")
    logger.info(f"  - Haversine Fallback: {SlowOSRMDispatchTest.osrm_fallback_count}")
    logger.info(f"  - Timeouts: {SlowOSRMDispatchTest.osrm_timeout_count}")


# ========== Configuration Recommandée ==========

"""
Usage en ligne de commande :

0. **Setup Docker : OSRM avec latence** :
   ```bash
   # Option A : Modifier docker-compose.yml pour ajouter latence
   # (voir documentation Docker tc netem)

   # Option B : Simuler latence via proxy (toxiproxy, etc.)

   # Option C : Tests sans latence simulée (OSRM normal)
   # Le test reste utile pour valider cache, fallback, circuit breaker
   ```

1. **Test résilience OSRM** (configuration recommandée) :
   ```bash
   locust -f slow_osrm_test.py \\
       --host=http://localhost:5000 \\
       --users=5 \\
       --spawn-rate=1 \\
       --run-time=10m \\
       --headless \\
       --csv=results_scenario3
   ```

2. **Test intensif** :
   ```bash
   locust -f slow_osrm_test.py \\
       --host=http://localhost:5000 \\
       --users=10 \\
       --spawn-rate=2 \\
       --run-time=15m \\
       --headless \\
       --csv=results_scenario3_intensive
   ```

3. **Mode Web UI** :
   ```bash
   locust -f slow_osrm_test.py --host=http://localhost:5000
   # Ouvrir http://localhost:8089
   ```

Points de validation :
- ✅ Dispatch réussit malgré OSRM lent (< 120s)
- ✅ Cache OSRM réduit appels (hit rate > 80%)
- ✅ Fallback Haversine fonctionne si OSRM fail
- ✅ Circuit breaker s'ouvre si trop d'échecs
- ✅ Pas de timeout fatal (504)
"""
