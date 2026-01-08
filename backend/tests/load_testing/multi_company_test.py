"""
Scénario 2 : Test Multi-Entreprises - 10 entreprises en parallèle

Ce test simule un environnement SaaS multi-tenant avec :
- 10 entreprises actives simultanément
- Dispatches en parallèle
- Validation isolation des données
- Mesure contention DB et locks Redis

Objectifs :
- Valider isolation données (pas de leak entre entreprises)
- Tester performance sous contention DB
- Vérifier gestion locks Redis (SETNX)
- Mesurer impact parallélisme sur latence

Métriques clés :
- Temps dispatch par entreprise : < 60s
- Pas de deadlocks DB
- Pas de collision locks Redis
- Isolation données : 100%

Usage:
    locust -f multi_company_test.py --host=http://localhost:5000 --users=10 --spawn-rate=10
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import date, timedelta
from typing import Any

from locust import HttpUser, between, events, task

logger = logging.getLogger(__name__)


class MultiCompanyDispatchTest(HttpUser):
    """Test multi-entreprises : 10 entreprises en parallèle."""

    wait_time = between(0, 2)  # Parallélisme agressif
    host = "http://localhost:5000"

    # Variables d'instance
    token: str | None = None
    company_id: int = 0  # Sera assigné dynamiquement
    test_date: str = ""
    user_index: int = 0

    def on_start(self) -> None:  # type: ignore[reportImplicitOverride]
        """Setup : Login et assignation entreprise."""
        # Assigner un ID d'entreprise unique (1-10)
        self.user_index = (id(self) % 10) + 1
        self.company_id = self.user_index

        logger.info("[SETUP] User %s → Company %s", self.user_index, self.company_id)

        # Login
        self._login()

        # Date de test
        tomorrow = date.today() + timedelta(days=1)
        self.test_date = tomorrow.strftime("%Y-%m-%d")

        logger.info(
            "[SETUP] ✅ Company %s prête | Date: %s", self.company_id, self.test_date
        )

    def _login(self) -> None:
        """Authentification spécifique à l'entreprise."""
        # Login avec credentials de l'entreprise
        email = f"company{self.company_id}@test.com"
        password = "test123"

        response = self.client.post(
            "/api/auth/login-test",  # Endpoint test (sans CSRF)
            json={
                "email": email,
                "password": password,
            },
            name=f"[AUTH] Login Company {self.company_id}",
        )

        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            logger.info("[AUTH] ✅ Company %s authentifiée", self.company_id)
        else:
            logger.error(
                "[AUTH] ❌ Company %s login échoué : %s",
                self.company_id,
                response.status_code,
            )
            # Fallback : utiliser admin token
            self._login_as_admin()

    def _login_as_admin(self) -> None:
        """Fallback : Login admin (pour tests)."""
        response = self.client.post(
            "/api/auth/login-test",  # Endpoint test (sans CSRF)
            json={
                "email": "admin@test.com",
                "password": "test123",
            },
            name="[AUTH] Login Admin (fallback)",
        )

        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            logger.info("[AUTH] ✅ Company %s utilise admin token", self.company_id)

    def _get_headers(self) -> dict[str, str]:
        """Headers avec JWT token."""
        if not self.token:
            raise Exception(f"No token for company {self.company_id}")
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    @task(10)  # Tâche principale
    def dispatch_company(self) -> None:
        """
        Dispatch pour une entreprise spécifique.

        Teste :
        - Isolation données (chaque entreprise voit seulement ses données)
        - Locks Redis (éviter conflits entre entreprises)
        - Contention DB (queries simultanées)
        """
        start_time = time.time()

        response = self.client.post(
            "/api/v1/company_dispatch/run",
            json={
                "company_id": self.company_id,
                "date": self.test_date,
                "mode": "optimization",
            },
            headers=self._get_headers(),
            name=f"[DISPATCH] Company {self.company_id}",
            catch_response=True,
        )

        duration = time.time() - start_time

        if response.status_code == 200:
            self._validate_response(response, duration)
        elif response.status_code == 409:
            # Conflit (dispatch déjà en cours)
            logger.warning(
                "[DISPATCH] ⚠️ Company %s | Dispatch déjà en cours (409)",
                self.company_id,
            )
            response.success()  # Pas une erreur réelle
        elif response.status_code == 423:
            # Locked (Redis lock actif)
            logger.warning(
                "[DISPATCH] ⚠️ Company %s | Redis lock actif (423)", self.company_id
            )
            response.success()
        else:
            logger.error(
                "[DISPATCH] ❌ Company %s | Status: %s | Response: %s",
                self.company_id,
                response.status_code,
                response.text[:200],
            )
            response.failure(f"Status {response.status_code}")

    @task(3)
    def check_company_bookings(self) -> None:
        """Vérifier les bookings de l'entreprise (isolation données)."""
        response = self.client.get(
            f"/api/bookings?company_id={self.company_id}&date={self.test_date}",
            headers=self._get_headers(),
            name=f"[BOOKINGS] Get Company {self.company_id}",
        )

        if response.status_code == 200:
            data = response.json()
            bookings = data.get("bookings", [])
            logger.debug(
                "[BOOKINGS] Company %s : %s bookings", self.company_id, len(bookings)
            )

            # Validation isolation : vérifier que tous les bookings appartiennent
            # bien à cette entreprise
            for booking in bookings:
                if booking.get("company_id") != self.company_id:
                    logger.error(
                        "[ISOLATION] ❌ LEAK DÉTECTÉ ! Booking %s de Company %s visible par Company %s",
                        booking.get("id"),
                        booking.get("company_id"),
                        self.company_id,
                    )

    @task(2)
    def check_company_drivers(self) -> None:
        """Vérifier les drivers de l'entreprise (isolation données)."""
        response = self.client.get(
            f"/api/drivers?company_id={self.company_id}&date={self.test_date}",
            headers=self._get_headers(),
            name=f"[DRIVERS] Get Company {self.company_id}",
        )

        if response.status_code == 200:
            data = response.json()
            drivers = data.get("drivers", [])
            logger.debug(
                "[DRIVERS] Company %s : %s drivers", self.company_id, len(drivers)
            )

    @task(1)
    def test_redis_lock(self) -> None:
        """Tester acquisition/release de lock Redis."""
        # Tenter d'acquérir un lock pour cette entreprise/date
        response = self.client.post(
            "/api/dispatch/acquire-lock",
            json={
                "company_id": self.company_id,
                "date": self.test_date,
            },
            headers=self._get_headers(),
            name=f"[LOCK] Acquire Company {self.company_id}",
        )

        if response.status_code == 200:
            logger.debug("[LOCK] ✅ Company %s lock acquis", self.company_id)

            # Attendre un peu
            time.sleep(random.uniform(0.1, 0.5))

            # Relâcher le lock
            release_response = self.client.post(
                "/api/dispatch/release-lock",
                json={
                    "company_id": self.company_id,
                    "date": self.test_date,
                },
                headers=self._get_headers(),
                name=f"[LOCK] Release Company {self.company_id}",
            )

            if release_response.status_code == 200:
                logger.debug("[LOCK] ✅ Company %s lock relâché", self.company_id)

    def _validate_response(self, response: Any, duration: float) -> None:
        """Valider la réponse et vérifier isolation données."""
        try:
            data = response.json()

            # Extraire métriques
            dispatch_duration = data.get("duration_seconds", 0)
            num_assignments = data.get("num_assignments", 0)
            num_bookings = data.get("total_bookings", 0)
            company_returned = data.get("company_id")

            # Validation isolation
            if company_returned and company_returned != self.company_id:
                logger.error(
                    "[ISOLATION] ❌ Company mismatch ! Expected: %s, Got: %s",
                    self.company_id,
                    company_returned,
                )
                response.failure(f"Company leak: {company_returned}")
                return

            # Log succès
            logger.info(
                "[DISPATCH] ✅ Company %s | Duration: %.2fs | API: %.2fs | Assignments: %s/%s",
                self.company_id,
                dispatch_duration,
                duration,
                num_assignments,
                num_bookings,
            )

            # Métriques custom
            events.request.fire(
                request_type="COMPANY_DISPATCH",
                name=f"company_{self.company_id}_duration",
                response_time=dispatch_duration * 1000,
                response_length=num_assignments,
                exception=None,
            )

            response.success()

        except json.JSONDecodeError as e:
            logger.error(
                "[DISPATCH] ❌ Company %s | Erreur parsing JSON : %s",
                self.company_id,
                e,
            )
            response.failure(f"JSON parse error: {e}")


# ========== Event Handlers ==========


@events.test_start.add_listener
def on_test_start(environment: Any, **kwargs: Any) -> None:
    """Hook démarrage test."""
    logger.info("=" * 80)
    logger.info("[LOCUST] 🚀 Démarrage Test Multi-Entreprises - Scénario 2")
    logger.info("[LOCUST] Objectif : 10 entreprises en parallèle")
    logger.info("[LOCUST] Validation : Isolation données, Redis locks, DB contention")
    logger.info("=" * 80)


@events.test_stop.add_listener
def on_test_stop(environment: Any, **kwargs: Any) -> None:
    """Hook fin test."""
    logger.info("=" * 80)
    logger.info("[LOCUST] ✅ Test Multi-Entreprises Terminé - Scénario 2")
    logger.info("=" * 80)

    stats = environment.stats
    logger.info("[STATS] Total Requests: %s", stats.total.num_requests)
    logger.info("[STATS] Total Failures: %s", stats.total.num_failures)
    logger.info("[STATS] Avg Response Time: %.2fms", stats.total.avg_response_time)


# ========== Configuration Recommandée ==========

"""
Usage en ligne de commande :

1. **Test 10 entreprises** (configuration recommandée) :
   ```bash
   locust -f multi_company_test.py \\
       --host=http://localhost:5000 \\
       --users=10 \\
       --spawn-rate=10 \\
       --run-time=10m \\
       --headless \\
       --csv=results_scenario2
   ```

2. **Test intensif 20 entreprises** :
   ```bash
   locust -f multi_company_test.py \\
       --host=http://localhost:5000 \\
       --users=20 \\
       --spawn-rate=5 \\
       --run-time=15m \\
       --headless \\
       --csv=results_scenario2_intensive
   ```

3. **Mode Web UI** :
   ```bash
   locust -f multi_company_test.py --host=http://localhost:5000
   # Ouvrir http://localhost:8089
   # Configurer : Users=10, Spawn rate=10
   ```

Points de validation :
- ✅ Aucun leak de données entre entreprises
- ✅ Locks Redis fonctionnent (pas de double dispatch)
- ✅ Performance stable avec 10+ entreprises
- ✅ Pas de deadlocks DB
"""
