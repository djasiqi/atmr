/**
 * @jest-environment jsdom
 */

import { createReconnectCircuitBreaker } from './reconnectCircuitBreaker';

describe('reconnectCircuitBreaker', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('blocks reconnect attempts during cooldown', () => {
    const breaker = createReconnectCircuitBreaker({
      failureThreshold: 2,
      cooldownMs: 15_000,
    });
    const mockSocket = { io: { opts: { reconnection: true } } };

    breaker.recordFailure(mockSocket);
    breaker.recordFailure(mockSocket);

    expect(breaker.shouldAllowReconnectAttempt()).toBe(false);
    expect(mockSocket.io.opts.reconnection).toBe(false);
  });

  it('recovers after cooldown expires', () => {
    const breaker = createReconnectCircuitBreaker({
      failureThreshold: 1,
      cooldownMs: 10_000,
    });
    const mockSocket = { io: { opts: { reconnection: true } } };

    breaker.recordFailure(mockSocket);
    jest.advanceTimersByTime(10_000);

    expect(breaker.shouldAllowReconnectAttempt()).toBe(true);
    expect(mockSocket.io.opts.reconnection).toBe(true);
  });
});
