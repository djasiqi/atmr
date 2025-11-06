/**
 * Test de charge minimale (smoke test) pour valider la disponibilité de l'API.
 *
 * Usage:
 *   k6 run scripts/k6-smoke.js
 *
 * Variables d'environnement:
 *   BASE_URL: URL de base de l'API (défaut: http://localhost:5000)
 *   VUS: Nombre d'utilisateurs virtuels (défaut: 20)
 *   DURATION: Durée du test (défaut: 1m)
 */

import http from "k6/http";
import { sleep, check } from "k6";
import { Rate } from "k6/metrics";

const errorRate = new Rate("errors");
const p95Latency = new Rate("p95_ok");

export let options = {
  stages: [
    { duration: "30s", target: 10 }, // Montée progressive
    { duration: "1m", target: 20 }, // Charge nominale
    { duration: "30s", target: 0 }, // Descente
  ],
  thresholds: {
    http_req_duration: ["p(95)<500"], // p95 < 500ms
    "http_req_duration{endpoint:/health}": ["p(95)<100"], // Healthcheck < 100ms
    errors: ["rate<0.01"], // < 1% d'erreurs
    http_req_failed: ["rate<0.01"], // < 1% de requêtes échouées
  },
};

const baseUrl = __ENV.BASE_URL || "http://localhost:5000";

export default function () {
  // 1. Healthcheck simple
  let res = http.get(`${baseUrl}/health`, {
    tags: { endpoint: "/health", type: "healthcheck" },
  });
  let healthOk = check(res, {
    "health status 200": (r) => r.status === 200,
    "health response time < 100ms": (r) => r.timings.duration < 100,
  });
  if (!healthOk) errorRate.add(1);

  sleep(0.2);

  // 2. Healthcheck détaillé (si disponible)
  res = http.get(`${baseUrl}/health/detailed`, {
    tags: { endpoint: "/health/detailed", type: "healthcheck" },
  });
  let detailedOk = check(res, {
    "detailed health 200": (r) => r.status === 200,
    "database component ok": (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.components?.database === "ok";
      } catch {
        return false;
      }
    },
  });
  if (!detailedOk) errorRate.add(1);

  sleep(0.3);

  // 3. Métriques Prometheus (si disponible)
  res = http.get(`${baseUrl}/prometheus/metrics`, {
    tags: { endpoint: "/prometheus/metrics", type: "metrics" },
  });
  check(res, {
    "prometheus metrics 200": (r) => r.status === 200,
    "prometheus format valid": (r) =>
      r.body.includes("HELP") || r.body.includes("# TYPE"),
  }) || errorRate.add(1);

  // Vérifier p95 < 500ms global
  if (res.timings.duration < 500) {
    p95Latency.add(1);
  }

  sleep(0.3);
}

export function handleSummary(data) {
  return {
    stdout: textSummary(data),
    "artifacts/smoke-test-summary.json": JSON.stringify(data),
  };
}

function textSummary(data) {
  let summary = "\n";
  summary += "=".repeat(80) + "\n";
  summary += "K6 SMOKE TEST SUMMARY\n";
  summary += "=".repeat(80) + "\n\n";

  summary += `Base URL: ${baseUrl}\n`;
  summary += `Total requests: ${data.metrics.http_reqs.values.count}\n`;
  summary += `Failed requests: ${
    data.metrics.http_req_failed.values.rate * 100
  }%\n`;
  summary += `p50 latency: ${data.metrics.http_req_duration.values.med}ms\n`;
  summary += `p95 latency: ${data.metrics.http_req_duration.values["p(95)"]}ms\n`;
  summary += `p99 latency: ${data.metrics.http_req_duration.values["p(99)"]}ms\n\n`;

  if (data.metrics.http_req_duration.values["p(95)"] < 500) {
    summary += "✅ p95 latency < 500ms: OK\n";
  } else {
    summary += "❌ p95 latency >= 500ms: FAIL\n";
  }

  if (data.metrics.http_req_failed.values.rate < 0.01) {
    summary += "✅ Error rate < 1%: OK\n";
  } else {
    summary += "❌ Error rate >= 1%: FAIL\n";
  }

  summary += "=".repeat(80) + "\n";
  return summary;
}
