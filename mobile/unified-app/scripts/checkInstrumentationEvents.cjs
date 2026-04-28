const fs = require("fs");
const path = require("path");

const telemetryFile = path.resolve(
  __dirname,
  "..",
  "src",
  "core",
  "observability",
  "driverTelemetry.ts"
);

const expectedEvents = [
  "auth.refresh.failure",
  "auth.bootstrap.failure",
  "realtime.socket.disconnect",
  "realtime.socket.reconnect",
  "tracking.permission.denied",
  "tracking.send.backoff",
  "tracking.send.failure",
  "tracking.send.recovered",
  "transition.queue.retry",
  "transition.queue.flush",
  "transition.queue.failure",
  "push.token.registered",
  "push.notification.received",
  "push.notification.opened",
  "driver.runtime.heartbeat",
];

const content = fs.readFileSync(telemetryFile, "utf8");
const missing = expectedEvents.filter((name) => !content.includes(`"${name}"`));

if (missing.length > 0) {
  console.error("[check-instrumentation-events] missing events:", missing.join(", "));
  process.exit(1);
}

console.log(
  `[check-instrumentation-events] OK (${expectedEvents.length} events validated in driverTelemetry.ts)`
);
