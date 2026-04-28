import http from "k6/http";
import ws from "k6/ws";
import { check, sleep } from "k6";
import { Counter, Rate } from "k6/metrics";

const baseUrl = __ENV.BASE_URL || "http://localhost:5000";
const wsUrl = __ENV.WS_URL || "ws://localhost:5000/socket.io/?EIO=4&transport=websocket";
const useWs = (__ENV.ENABLE_WS || "false").toLowerCase() === "true";

const appErrors = new Counter("app_errors_total");
const wsConnectFailures = new Counter("ws_connect_failures_total");
const realtimeLoss = new Rate("realtime_loss_rate");

export const options = {
  scenarios: {
    api_gate: {
      executor: "ramping-arrival-rate",
      startRate: 200,
      timeUnit: "1s",
      preAllocatedVUs: 200,
      maxVUs: 4000,
      stages: [
        { target: 1000, duration: "10m" },
        { target: 2000, duration: "1h40m" },
        { target: 2000, duration: "10m" },
      ],
      exec: "apiScenario",
    },
    ws_gate: {
      executor: "constant-vus",
      vus: Number(__ENV.WS_VUS || "100"),
      duration: __ENV.WS_DURATION || "5m",
      exec: "wsScenario",
      startTime: "0s",
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.005"],
    http_req_duration: ["p(95)<300", "p(99)<800"],
    app_errors_total: ["count<20"],
    ws_connect_failures_total: ["count<10"],
    realtime_loss_rate: ["rate<0.0001"],
  },
};

export function apiScenario() {
  const health = http.get(`${baseUrl}/health`, {
    tags: { endpoint: "/health" },
  });
  check(health, {
    "health is 200": (r) => r.status === 200,
    "health p95 budget": (r) => r.timings.duration < 100,
  }) || appErrors.add(1);

  const metrics = http.get(`${baseUrl}/prometheus/metrics`, {
    tags: { endpoint: "/prometheus/metrics" },
  });
  check(metrics, {
    "metrics is 200": (r) => r.status === 200,
    "metrics has prom format": (r) => r.body.includes("# TYPE"),
  }) || appErrors.add(1);

  sleep(0.2);
}

export function wsScenario() {
  if (!useWs) {
    sleep(1);
    return;
  }

  const response = ws.connect(wsUrl, {}, (socket) => {
    let received = 0;
    socket.on("open", () => {
      socket.send('40');
    });
    socket.on("message", () => {
      received += 1;
    });
    socket.setTimeout(() => {
      realtimeLoss.add(received === 0 ? 1 : 0);
      socket.close();
    }, 5000);
  });

  if (!response || response.status !== 101) {
    wsConnectFailures.add(1);
    realtimeLoss.add(1);
  }
}

