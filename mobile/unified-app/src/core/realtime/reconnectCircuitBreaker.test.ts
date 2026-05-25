import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { createReconnectCircuitBreaker } from "./reconnectCircuitBreaker";

describe("reconnectCircuitBreaker", () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("enters cooldown after N failures in window", () => {
    const breaker = createReconnectCircuitBreaker({
      failureWindowMs: 60_000,
      failureThreshold: 3,
      cooldownMs: 10_000,
    });
    const mockSocket = { io: { opts: { reconnection: true } } };

    breaker.recordFailure(mockSocket as never);
    breaker.recordFailure(mockSocket as never);
    expect(breaker.isCooldownActive()).toBe(false);

    const tripped = breaker.recordFailure(mockSocket as never);
    expect(tripped).toBe(true);
    expect(breaker.isCooldownActive()).toBe(true);
    expect(breaker.shouldAllowReconnectAttempt()).toBe(false);
    expect(mockSocket.io.opts.reconnection).toBe(false);
  });

  it("clears cooldown after duration and allows reconnect", () => {
    const breaker = createReconnectCircuitBreaker({
      failureThreshold: 1,
      cooldownMs: 5_000,
    });
    const mockSocket = { io: { opts: { reconnection: true } } };

    breaker.recordFailure(mockSocket as never);
    expect(breaker.isCooldownActive()).toBe(true);

    jest.advanceTimersByTime(5_000);
    expect(breaker.isCooldownActive()).toBe(false);
    expect(breaker.shouldAllowReconnectAttempt()).toBe(true);
    expect(mockSocket.io.opts.reconnection).toBe(true);
  });

  it("recordSuccess resets failure state", () => {
    const breaker = createReconnectCircuitBreaker({ failureThreshold: 2 });
    breaker.recordFailure();
    breaker.recordSuccess();
    expect(breaker.getSnapshot().failureCount).toBe(0);
    expect(breaker.isCooldownActive()).toBe(false);
  });
});
