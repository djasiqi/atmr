/**
 * Circuit breaker reconnexion Socket.IO (fenêtre glissante + cooldown).
 */

const DEFAULT_CONFIG = {
  failureWindowMs: 60_000,
  failureThreshold: 8,
  cooldownMs: 30_000,
};

export function createReconnectCircuitBreaker(config = {}) {
  const settings = { ...DEFAULT_CONFIG, ...config };
  const failureTimestamps = [];
  let cooldownUntil = 0;
  let cooldownTimer = null;

  const clearCooldownTimer = () => {
    if (cooldownTimer) {
      clearTimeout(cooldownTimer);
      cooldownTimer = null;
    }
  };

  const pruneFailures = (now) => {
    while (
      failureTimestamps.length > 0 &&
      now - failureTimestamps[0] > settings.failureWindowMs
    ) {
      failureTimestamps.shift();
    }
  };

  const setSocketReconnection = (socket, enabled) => {
    if (socket?.io?.opts) {
      socket.io.opts.reconnection = enabled;
    }
  };

  return {
    isCooldownActive() {
      return Date.now() < cooldownUntil;
    },

    shouldAllowReconnectAttempt() {
      return Date.now() >= cooldownUntil;
    },

    recordSuccess() {
      failureTimestamps.length = 0;
      cooldownUntil = 0;
      clearCooldownTimer();
    },

    recordFailure(socket) {
      const now = Date.now();
      failureTimestamps.push(now);
      pruneFailures(now);

      if (failureTimestamps.length < settings.failureThreshold) {
        return false;
      }

      cooldownUntil = now + settings.cooldownMs;
      setSocketReconnection(socket, false);
      clearCooldownTimer();
      cooldownTimer = setTimeout(() => {
        cooldownTimer = null;
        failureTimestamps.length = 0;
        cooldownUntil = 0;
        setSocketReconnection(socket, true);
      }, settings.cooldownMs);

      return true;
    },

    reset(socket) {
      failureTimestamps.length = 0;
      cooldownUntil = 0;
      clearCooldownTimer();
      setSocketReconnection(socket, true);
    },
  };
}

export const companyReconnectCircuitBreaker = createReconnectCircuitBreaker();
