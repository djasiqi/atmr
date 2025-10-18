"""
Locust load testing scenarios for ATMR API
Usage: locust -f locust_load_test.py --host=http://localhost:5000
Open: http://localhost:8089
"""
from locust import HttpUser, task, between, events
from datetime import datetime, timedelta

class ATMRUser(HttpUser):
    """Simulated ATMR user (company role)"""
    wait_time = between(1, 3)  # Entre 1 et 3 secondes entre chaque tâche
    
    def on_start(self):
        """Login au démarrage de chaque utilisateur simulé"""
        response = self.client.post("/api/auth/login", json={
            "email": "test@test.com",
            "password": "password123"
        }, catch_response=True)
        
        if response.status_code == 200:
            data = response.json()
            self.token = data.get('access_token')
            self.headers = {"Authorization": f"Bearer {self.token}"}
            response.success()
        else:
            response.failure(f"Login failed: {response.status_code}")
            self.headers = {}
    
    @task(4)  # 40% du trafic
    def get_bookings(self):
        """Récupérer les réservations du jour"""
        today = datetime.now().strftime("%Y-%m-%d")
        with self.client.get(
            f"/api/bookings?date={today}",
            headers=self.headers,
            catch_response=True,
            name="/api/bookings [date]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    response.success()
                    # Log metrics
                    if hasattr(response, 'elapsed'):
                        events.request.fire(
                            request_type="GET",
                            name="/api/bookings [items]",
                            response_time=response.elapsed.total_seconds() * 1000,
                            response_length=len(data),
                            exception=None,
                            context={}
                        )
                else:
                    response.failure("Response not a list")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(2)  # 20% du trafic
    def get_drivers(self):
        """Récupérer la liste des chauffeurs"""
        with self.client.get(
            "/api/drivers",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(2)  # 20% du trafic
    def get_company_me(self):
        """Récupérer les infos de l'entreprise"""
        with self.client.get(
            "/api/companies/me",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(1)  # 10% du trafic
    def create_booking(self):
        """Créer une nouvelle réservation"""
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
        
        payload = {
            "customer_name": "Test Client",
            "pickup_location": "Rue de Lausanne 1, Genève",
            "dropoff_location": "Hôpital Cantonal, Genève",
            "scheduled_time": tomorrow,
            "amount": 50.0,
            "client_id": 1,  # Doit exister dans DB de test
            "company_id": 1,
            "user_id": 1
        }
        
        with self.client.post(
            "/api/bookings",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            elif response.status_code == 400:
                # Acceptable si données invalides (DB contraintes)
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(1)  # 10% du trafic (lourd)
    def run_dispatch(self):
        """Lancer un dispatch automatique"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        payload = {
            "date": today,
            "mode": "auto",
            "regular_first": True
        }
        
        with self.client.post(
            "/api/company_dispatch/run",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/company_dispatch/run",
            timeout=30  # Dispatch peut prendre du temps
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Vérifier structure réponse
                if 'assignments' in data or 'meta' in data:
                    response.success()
                else:
                    response.failure("Invalid response structure")
            elif response.status_code == 202:
                # Accepted (async)
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


class ATMRDriverUser(HttpUser):
    """Simulated ATMR driver user"""
    wait_time = between(2, 5)
    
    def on_start(self):
        """Login driver"""
        response = self.client.post("/api/auth/login", json={
            "email": "driver@test.com",
            "password": "password123"
        }, catch_response=True)
        
        if response.status_code == 200:
            data = response.json()
            self.token = data.get('access_token')
            self.headers = {"Authorization": f"Bearer {self.token}"}
            response.success()
        else:
            response.failure(f"Driver login failed: {response.status_code}")
            self.headers = {}
    
    @task(5)  # 50% du trafic driver
    def get_driver_bookings(self):
        """Récupérer ses propres réservations"""
        with self.client.get(
            "/api/driver/bookings",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(3)  # 30% du trafic
    def update_booking_status(self):
        """Mettre à jour le statut d'une réservation"""
        # Simulé (booking_id devrait exister)
        booking_id = 1
        
        payload = {"status": "IN_PROGRESS"}
        
        with self.client.patch(
            f"/api/bookings/{booking_id}",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/bookings/{id} [PATCH]"
        ) as response:
            if response.status_code in [200, 404]:  # 404 acceptable si booking inexistant
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


# ============================================================
# Event listeners pour métriques custom
# ============================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Hook au démarrage du test"""
    print("🔥 Starting ATMR load test...")
    print(f"   Target: {environment.host}")
    print(f"   Users: {environment.runner.target_user_count if environment.runner else 'N/A'}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Hook à la fin du test"""
    print("\n✅ Load test completed!")
    print(f"   Total requests: {environment.stats.total.num_requests}")
    print(f"   Total failures: {environment.stats.total.num_failures}")
    
    # Vérifier critères d'acceptation
    total_requests = environment.stats.total.num_requests
    total_failures = environment.stats.total.num_failures
    
    if total_requests > 0:
        error_rate = (total_failures / total_requests) * 100
        
        print("\n📊 RÉSULTATS:")
        print(f"   Error rate: {error_rate:.2f}%")
        
        if error_rate < 5:
            print("   ✅ PASSED: Error rate < 5%")
        else:
            print(f"   ❌ FAILED: Error rate {error_rate:.2f}% >= 5%")
        
        # Latence moyenne
        avg_response_time = environment.stats.total.avg_response_time
        print(f"   Average response time: {avg_response_time:.0f}ms")
        
        # RPS (Requests per second)
        if environment.stats.total.total_rps:
            print(f"   Throughput: {environment.stats.total.total_rps:.2f} req/s")


# ============================================================
# Configuration recommandée
# ============================================================
"""
Scenarios de test:

1. Light load (warmup):
   - Users: 10
   - Spawn rate: 2/s
   - Duration: 2min

2. Normal load:
   - Users: 100
   - Spawn rate: 10/s
   - Duration: 5min
   
   Critères:
   - Error rate < 1%
   - Avg response time < 200ms

3. Stress test:
   - Users: 200
   - Spawn rate: 20/s
   - Duration: 10min
   
   Critères:
   - Error rate < 5%
   - Avg response time < 500ms

4. Spike test:
   - Users: 0 -> 300 -> 0
   - Spawn rate: 50/s
   - Duration: 3min
   
   Critères:
   - System recovery < 30s
   - No crashes
"""

