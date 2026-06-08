import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

const batchOk = new Rate('driver_location_batch_ok');
const batchLatency = new Trend('driver_location_batch_latency_ms');
const batchesSent = new Counter('driver_location_batches_sent');

export const options = {
  scenarios: {
    fleet_drivers: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: __ENV.TARGET_DRIVERS ? Number(__ENV.TARGET_DRIVERS) : 50 },
        { duration: '10m', target: __ENV.TARGET_DRIVERS ? Number(__ENV.TARGET_DRIVERS) : 50 },
        { duration: '1m', target: 0 },
      ],
    },
  },
  thresholds: {
    driver_location_batch_ok: ['rate>0.95'],
    driver_location_batch_latency_ms: ['p(95)<2000'],
  },
};

const BASE_WS = __ENV.WS_URL || 'ws://localhost:5000/socket.io/?EIO=4&transport=websocket';
const DRIVER_TOKEN = __ENV.DRIVER_TOKEN || '';
const BATCH_INTERVAL_SEC = Number(__ENV.BATCH_INTERVAL_SEC || '5');

function makePosition(vu, seq) {
  const baseLat = 46.2044 + (vu % 20) * 0.001;
  const baseLon = 6.1432 + (vu % 20) * 0.001;
  const now = new Date().toISOString();
  return {
    latitude: baseLat + seq * 0.00001,
    longitude: baseLon + seq * 0.00001,
    accuracy: 8 + (vu % 5),
    speed: 12,
    heading: (seq * 15) % 360,
    timestamp: now,
    recorded_at: now,
    location_mode: 'mission_live',
    is_background: false,
    tracking_event_id: `k6_${vu}_${seq}_${Date.now()}`,
    sequence_id: seq,
    platform: vu % 2 === 0 ? 'android' : 'ios',
  };
}

export default function fleetDriverSimulation() {
  if (!DRIVER_TOKEN) {
    console.error('DRIVER_TOKEN required');
    return;
  }

  const url = BASE_WS;
  const res = ws.connect(url, { headers: { Authorization: `Bearer ${DRIVER_TOKEN}` } }, (socket) => {
    socket.on('open', () => {
      socket.send('40');
    });

    let seq = 0;
    socket.setInterval(() => {
      seq += 1;
      const payload = {
        tracking_session_id: `k6_session_${__VU}`,
        positions: [makePosition(__VU, seq)],
      };
      const started = Date.now();
      socket.send(`42["driver_location_batch",${JSON.stringify(payload)}]`);
      batchLatency.add(Date.now() - started);
      batchesSent.add(1);
    }, BATCH_INTERVAL_SEC * 1000);

    socket.setTimeout(() => {
      socket.close();
    }, 11 * 60 * 1000);
  });

  check(res, { 'ws connected': (r) => r && r.status === 101 });
  batchOk.add(res && res.status === 101);
  sleep(1);
}
