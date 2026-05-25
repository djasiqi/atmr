import type { Manager, Socket } from "socket.io-client";

export type ReconnectCircuitBreakerConfig = {
  failureWindowMs: number;
  failureThreshold: number;
  cooldownMs: number;
};

const DEFAULT_CONFIG: ReconnectCircuitBreakerConfig = {
  failureWindowMs: 60_000,
  failureThreshold: 8,
  cooldownMs: 30_000,
};

export function createReconnectCircuitBreaker(config: Partial<ReconnectCircuitBreakerConfig> = {}) {
  const settings = { ...DEFAULT_CONFIG, ...config };
  const failureTimestamps: number[] = [];
  let cooldownUntil = 0;
  let cooldownTimer: ReturnType<typeof setTimeout> | null = null;

  const clearCooldownTimer = () => {
    if (cooldownTimer) {
      clearTimeout(cooldownTimer);
      cooldownTimer = null;
    }
  };

  const pruneFailures = (now: number) => {
    while (
      failureTimestamps.length > 0 &&
      now - failureTimestamps[0] > settings.failureWindowMs
    ) {
      failureTimestamps.shift();
    }
  };

  const setSocketReconnection = (socket: Socket | null, enabled: boolean) => {
    const manager = socket?.io as Manager | undefined;
    if (manager?.opts) {
      manager.opts.reconnection = enabled;
    }
  };

  return {
    isCooldownActive(): boolean {
      return Date.now() < cooldownUntil;
    },

    shouldAllowReconnectAttempt(): boolean {
      return !this.isCooldownActive();
    },

    recordSuccess() {
      failureTimestamps.length = 0;
      cooldownUntil = 0;
      clearCooldownTimer();
    },

    recordFailure(socket?: Socket | null): boolean {
      const now = Date.now();
      failureTimestamps.push(now);
      pruneFailures(now);

      if (failureTimestamps.length < settings.failureThreshold) {
        return false;
      }

      cooldownUntil = now + settings.cooldownMs;
      setSocketReconnection(socket ?? null, false);
      clearCooldownTimer();
      cooldownTimer = setTimeout(() => {
        cooldownTimer = null;
        failureTimestamps.length = 0;
        cooldownUntil = 0;
        setSocketReconnection(socket ?? null, true);
      }, settings.cooldownMs);

      return true;
    },

    reset(socket?: Socket | null) {
      failureTimestamps.length = 0;
      cooldownUntil = 0;
      clearCooldownTimer();
      setSocketReconnection(socket ?? null, true);
    },

    getSnapshot() {
      return {
        failureCount: failureTimestamps.length,
        cooldownUntil,
        isCooldownActive: Date.now() < cooldownUntil,
      };
    },
  };
}

/** Instance partagée pour les bridges Socket.IO mobile. */
export const mobileReconnectCircuitBreaker = createReconnectCircuitBreaker();
