"""
Locust load testing script for ATMR Backend API
Usage: locust -f locustfile.py --host=http://localhost:5000

Installation: pip install -r requirements-profiling.txt
"""
from locust import HttpUser, task, between, events  # type: ignore[import-untyped]
import random

# Test credentials (dev environment)
TEST_USERS = [
    {"username": "company1", "password": "test123", "role": "company"},
    {"username": "driver1", "password": "test123", "role": "driver"},
    {"username": "client1", "password": "test123", "role": "client"},
]


class CompanyUser(HttpUser):
    """Simule un utilisateur entreprise (endpoints les plus utilisés)."""
    
    wait_time = between(2, 5)  # 2-5s entre requêtes
    
    def on_start(self):
        """Login au démarrage."""
        user = random.choice([u for u in TEST_USERS if u["role"] == "company"])
        response = self.client.post("/api/auth/login", json={
            "username": user["username"],
            "password": user["password"]
        }, name="/api/auth/login [company]")
        
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
            self.company_id = response.json().get("company_id", 1)
        else:
            self.token = None
            self.headers = {}
    
    @task(5)  # Poids 5 (50% du trafic)
    def get_bookings(self):
        """GET /api/bookings (endpoint critique N+1)."""
        if not self.token:
            return
        
        self.client.get(
            f"/api/bookings?company_id={self.company_id}",
            headers=self.headers,
            name="/api/bookings [list]"
        )
    
    @task(3)  # Poids 3 (30%)
    def get_drivers(self):
        """GET /api/companies/me/drivers."""
        if not self.token:
            return
        
        self.client.get(
            "/api/companies/me/drivers",
            headers=self.headers,
            name="/api/companies/me/drivers"
        )
    
    @task(1)  # Poids 1 (10%)
    def get_dispatch_status(self):
        """GET /api/companies/me/dispatch/status."""
        if not self.token:
            return
        
        self.client.get(
            "/api/companies/me/dispatch/status",
            headers=self.headers,
            name="/api/companies/me/dispatch/status"
        )
    
    @task(1)  # Poids 1 (10%, coûteux)
    def run_dispatch(self):
        """POST /api/dispatch/run (test charge lourde)."""
        if not self.token:
            return
        
        self.client.post(
            "/api/dispatch/run",
            json={
                "company_id": self.company_id,
                "date": "2025-10-21",
                "mode": "optimize"
            },
            headers=self.headers,
            name="/api/dispatch/run [optimize]",
            timeout=60  # OSRM peut prendre du temps
        )


class DriverUser(HttpUser):
    """Simule un chauffeur mobile (moins de trafic)."""
    
    wait_time = between(10, 20)  # Moins fréquent
    
    def on_start(self):
        """Login."""
        user = random.choice([u for u in TEST_USERS if u["role"] == "driver"])
        response = self.client.post("/api/auth/login", json={
            "username": user["username"],
            "password": user["password"]
        }, name="/api/auth/login [driver]")
        
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
            self.driver_id = response.json().get("driver_id", 1)
        else:
            self.token = None
    
    @task(5)
    def get_my_bookings(self):
        """GET /api/drivers/<id>/bookings."""
        if not self.token:
            return
        
        self.client.get(
            f"/api/drivers/{self.driver_id}/bookings",
            headers=self.headers,
            name="/api/drivers/<id>/bookings"
        )
    
    @task(2)
    def update_location(self):
        """POST /api/drivers/<id>/location (simule GPS)."""
        if not self.token:
            return
        
        # Position aléatoire autour de Genève
        lat = 46.2 + random.uniform(-0.1, 0.1)
        lon = 6.1 + random.uniform(-0.1, 0.1)
        
        self.client.post(
            f"/api/drivers/{self.driver_id}/location",
            json={
                "latitude": lat,
                "longitude": lon,
                "accuracy": 10.0
            },
            headers=self.headers,
            name="/api/drivers/<id>/location [update]"
        )


# ============================================================
# Métriques custom (latence p95/p99 par endpoint)
# ============================================================
stats_by_endpoint = {}

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Tracker latence par endpoint."""
    if name not in stats_by_endpoint:
        stats_by_endpoint[name] = []
    
    stats_by_endpoint[name].append(response_time)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Afficher métriques finales."""
    print("\n" + "="*60)
    print("📊 MÉTRIQUES FINALES PAR ENDPOINT")
    print("="*60)
    
    for endpoint, times in sorted(stats_by_endpoint.items()):
        if not times:
            continue
        
        times_sorted = sorted(times)
        p50 = times_sorted[len(times_sorted) // 2]
        p95 = times_sorted[int(len(times_sorted) * 0.95)]
        p99 = times_sorted[int(len(times_sorted) * 0.99)]
        avg = sum(times) / len(times)
        
        print(f"\n{endpoint}")
        print(f"  Requests: {len(times)}")
        print(f"  Average:  {avg:.0f}ms")
        print(f"  p50:      {p50:.0f}ms")
        print(f"  p95:      {p95:.0f}ms {'❌ >800ms' if p95 > 800 else '✅'}")
        print(f"  p99:      {p99:.0f}ms {'❌ >2000ms' if p99 > 2000 else '✅'}")
    
    print("\n" + "="*60)
    print("✅ Tests terminés. Vérifier que p95 < 800ms sur endpoints critiques.")
    print("="*60 + "\n")

