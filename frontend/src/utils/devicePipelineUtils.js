/** Preuve de vie device-health (heartbeat + tracking natif). */

const DEVICE_HEARTBEAT_FRESH_MS = 120_000;

function parseHeartbeatEpochMs(hb) {
  if (hb == null || hb === '') return null;
  if (typeof hb === 'number' && Number.isFinite(hb)) {
    return hb < 1e12 ? hb * 1000 : hb;
  }
  const asNum = Number(hb);
  if (Number.isFinite(asNum)) {
    return asNum < 1e12 ? asNum * 1000 : asNum;
  }
  const parsed = Date.parse(String(hb));
  return Number.isFinite(parsed) ? parsed : null;
}

function truthyDeviceFlag(value) {
  return value === true || value === 1 || value === '1';
}

function isIosDeviceHealth(dh) {
  const platform = String(dh?.platform ?? '').trim().toLowerCase();
  return platform === 'ios' || platform === 'iphone' || platform === 'apple';
}

export function isDeviceHeartbeatFresh(driver, nowMs = Date.now()) {
  const dh = driver?.device_health;
  if (!dh || typeof dh !== 'object') return false;
  const hbMs = parseHeartbeatEpochMs(dh.last_heartbeat_at);
  if (hbMs == null) return false;
  const ageMs = nowMs - hbMs;
  return Number.isFinite(ageMs) && ageMs >= 0 && ageMs <= DEVICE_HEARTBEAT_FRESH_MS;
}

/**
 * Pipeline vivant : heartbeat frais + signal natif crédible.
 * `fgs_running` est Android-only — sur iOS son absence/0 ne pénalise pas le pipeline.
 */
export function isDevicePipelineAlive(driver, nowMs = Date.now()) {
  const dh = driver?.device_health;
  if (!dh || typeof dh !== 'object') return false;
  if (!isDeviceHeartbeatFresh(driver, nowMs)) return false;

  if (truthyDeviceFlag(dh.tracking_active)) return true;
  if (truthyDeviceFlag(dh.native_task_running)) return true;

  if (!isIosDeviceHealth(dh) && truthyDeviceFlag(dh.fgs_running)) {
    return true;
  }

  const state = String(dh.tracking_state || '').toLowerCase();
  return state === 'starting' || state === 'active';
}

export const isDeviceHealthSignalActive = (driver, nowMs = Date.now()) =>
  isDevicePipelineAlive(driver, nowMs);
