import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Métriques personnalisées
const errorRate = new Rate('errors');

// Configuration
export const options = {
  stages: [
    { duration: '2m', target: 100 }, // Ramp-up 2 min
    { duration: '5m', target: 100 }, // Hold 5 min
    { duration: '1m', target: 0 },    // Ramp-down 1 min
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // p95 < 500ms
    http_req_failed: ['rate<0.01'],   // < 1% erreurs
    errors: ['rate<0.01'],
  },
};

// Base URL (à adapter)
const BASE_URL = __ENV.BASE_URL || 'http://localhost:5000';

// Tokens JWT (à générer avant le test)
let adminToken = __ENV.ADMIN_TOKEN || '';
let driverToken = __ENV.DRIVER_TOKEN || '';
let clientToken = __ENV.CLIENT_TOKEN || '';

// Fonction login
function login(email, password, role) {
  const res = http.post(`${BASE_URL}/api/v1/auth/login`, JSON.stringify({
    email,
    password,
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  if (res.status === 200) {
    const body = JSON.parse(res.body);
    return body.access_token;
  }
  return null;
}

// Scénario Read Heavy (Admin/Company)
export function readHeavyScenario() {
  const token = adminToken || login('admin@test.com', 'password', 'admin');
  
  const headers = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  };
  
  // 1. Fetch missions/bookings
  const bookingsRes = http.get(`${BASE_URL}/api/v1/company/reservations?day=2025-01-15`, { headers });
  check(bookingsRes, {
    'bookings status 200': (r) => r.status === 200,
    'bookings response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);
  
  // 2. Fetch dispatch status
  const dispatchRes = http.get(`${BASE_URL}/api/v1/company/dispatch/status?day=2025-01-15`, { headers });
  check(dispatchRes, {
    'dispatch status 200': (r) => r.status === 200,
    'dispatch response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);
  
  sleep(5); // Simuler 5-10s entre requêtes
}

// Scénario Write Heavy (Driver)
export function writeHeavyScenario() {
  const token = driverToken || login('driver@test.com', 'password', 'driver');
  
  const headers = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  };
  
  // 1. Fetch missions
  const missionsRes = http.get(`${BASE_URL}/api/v1/driver/missions`, { headers });
  check(missionsRes, {
    'missions status 200': (r) => r.status === 200,
  }) || errorRate.add(1);
  
  // 2. Update status (si mission disponible)
  if (missionsRes.status === 200) {
    const missions = JSON.parse(missionsRes.body);
    if (missions.length > 0) {
      const missionId = missions[0].id;
      const updateRes = http.patch(
        `${BASE_URL}/api/v1/driver/missions/${missionId}/status`,
        JSON.stringify({ status: 'IN_PROGRESS' }),
        { headers }
      );
      check(updateRes, {
        'update status 200': (r) => r.status === 200,
        'update response time < 1000ms': (r) => r.timings.duration < 1000,
      }) || errorRate.add(1);
    }
  }
  
  // 3. Send position
  const positionRes = http.post(
    `${BASE_URL}/api/v1/driver/location`,
    JSON.stringify({
      latitude: 46.5197 + Math.random() * 0.01,
      longitude: 6.6323 + Math.random() * 0.01,
      timestamp: Date.now(),
    }),
    { headers }
  );
  check(positionRes, {
    'position status 200': (r) => r.status === 200,
    'position response time < 1000ms': (r) => r.timings.duration < 1000,
  }) || errorRate.add(1);
  
  sleep(5); // Simuler 5-30s entre mises à jour
}

// Scénario Client
export function clientScenario() {
  const token = clientToken || login('client@test.com', 'password', 'client');
  
  const headers = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  };
  
  // Fetch reservations
  const res = http.get(`${BASE_URL}/api/v1/client/reservations`, { headers });
  check(res, {
    'reservations status 200': (r) => r.status === 200,
    'reservations response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);
  
  sleep(10); // Simuler consultation occasionnelle
}

// Répartition: 10% admin, 70% driver, 20% client
export default function () {
  const rand = Math.random();
  if (rand < 0.1) {
    readHeavyScenario();
  } else if (rand < 0.8) {
    writeHeavyScenario();
  } else {
    clientScenario();
  }
}
