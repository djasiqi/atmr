import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import {
  OfflineMutationAction,
  OfflineMutationQueue,
} from "./offlineMutationQueue";

type TestAction = OfflineMutationAction & { missionId: number };

const storage = new Map<string, string>();
const mockGetItem = jest.fn(async (key: string) => storage.get(key) ?? null);
const mockSetItem = jest.fn(async (key: string, value: string) => {
  storage.set(key, value);
});

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: (key: string) => mockGetItem(key),
  setItem: (key: string, value: string) => mockSetItem(key, value),
}));

describe("offline mutation queue", () => {
  beforeEach(() => {
    storage.clear();
    mockGetItem.mockClear();
    mockSetItem.mockClear();
  });

  it("enqueues actions and reports pending count", async () => {
    const execute = jest.fn(async () => undefined);
    const queue = new OfflineMutationQueue<TestAction>({
      storageKey: "test_queue",
      maxRetries: 3,
      replayWindowMs: 10_000,
      backoffBaseMs: 10,
      backoffMaxMs: 100,
      execute,
    });

    await queue.enqueue({
      id: "a1",
      missionId: 7,
      queuedAt: Date.now(),
      retryCount: 0,
    });

    expect(await queue.count()).toBe(1);
    expect(execute).not.toHaveBeenCalled();
  });

  it("keeps action for replay after transient failure", async () => {
    jest.useFakeTimers();
    const execute = jest
      .fn(async () => undefined)
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(undefined);
    const queue = new OfflineMutationQueue<TestAction>({
      storageKey: "test_queue_retry",
      maxRetries: 3,
      replayWindowMs: 10_000,
      backoffBaseMs: 10,
      backoffMaxMs: 100,
      execute,
    });

    await queue.enqueue({
      id: "a2",
      missionId: 8,
      queuedAt: Date.now(),
      retryCount: 0,
    });

    const firstFlushPromise = queue.flush();
    await jest.advanceTimersByTimeAsync(20);
    const firstFlush = await firstFlushPromise;
    expect(firstFlush.failed).toBe(1);
    expect(await queue.count()).toBe(1);

    const secondFlush = await queue.flush();
    expect(secondFlush.sent).toBe(1);
    expect(await queue.count()).toBe(0);
    jest.useRealTimers();
  });
});
