import React from "react";
import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import { act, create } from "react-test-renderer";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useDriverAvailability } from "./useDriverAvailability";
import {
  getDriverAvailabilityActive,
  resetDriverAvailabilityBridgeForTests,
} from "../services/driverAvailabilityBridge";

const mockGetDriverProfile = jest.fn() as jest.MockedFunction<() => Promise<unknown>>;
const mockReadDriverProfileCache = jest.fn() as jest.MockedFunction<
  (opts?: { allowStale?: boolean }) => Promise<{ profile?: unknown }>
>;
const mockWriteDriverProfileCache = jest.fn() as jest.MockedFunction<
  (profile: unknown) => Promise<void>
>;

jest.mock("../api", () => ({
  getDriverProfile: () => mockGetDriverProfile(),
  updateDriverAvailability: jest.fn(),
}));

jest.mock("../services/driverProfileCache", () => ({
  readDriverProfileCache: (opts?: { allowStale?: boolean }) => mockReadDriverProfileCache(opts),
  writeDriverProfileCache: (profile: unknown) => mockWriteDriverProfileCache(profile),
}));

type HookValue = ReturnType<typeof useDriverAvailability>;

function HookProbe({ onValue }: { onValue: (value: HookValue) => void }) {
  const value = useDriverAvailability();
  onValue(value);
  return null;
}

async function renderHook(): Promise<{ latest: () => HookValue; unmount: () => void }> {
  let latest: HookValue | null = null;
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  let renderer: ReturnType<typeof create>;
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={client}>
        <HookProbe
          onValue={(value) => {
            latest = value;
          }}
        />
      </QueryClientProvider>
    );
  });
  return {
    latest: () => {
      if (!latest) throw new Error("hook not mounted");
      return latest;
    },
    unmount: () => {
      act(() => {
        renderer!.unmount();
      });
    },
  };
}

describe("useDriverAvailability", () => {
  beforeEach(() => {
    resetDriverAvailabilityBridgeForTests();
    mockGetDriverProfile.mockReset();
    mockReadDriverProfileCache.mockReset();
    mockWriteDriverProfileCache.mockReset();
  });

  afterEach(() => {
    resetDriverAvailabilityBridgeForTests();
  });

  it("cold start avant hydratation → UNKNOWN, pas AVAILABLE", async () => {
    mockReadDriverProfileCache.mockReturnValue(new Promise(() => undefined));
    mockGetDriverProfile.mockReturnValue(new Promise(() => undefined));
    const { latest, unmount } = await renderHook();
    expect(latest().isAvailable).toBeNull();
    expect(latest().availabilityLoading).toBe(true);
    expect(getDriverAvailabilityActive()).toBeNull();
    unmount();
  });

  it("échec profil sans cache → reste UNKNOWN, jamais true", async () => {
    mockReadDriverProfileCache.mockResolvedValue({});
    mockGetDriverProfile.mockRejectedValue(new Error("network"));
    const { latest, unmount } = await renderHook();
    expect(latest().isAvailable).toBeNull();
    expect(latest().availabilityLoading).toBe(true);
    expect(getDriverAvailabilityActive()).toBeNull();
    unmount();
  });

  it("cache is_available=false → UNAVAILABLE", async () => {
    mockReadDriverProfileCache.mockResolvedValue({ profile: { is_available: false } });
    mockGetDriverProfile.mockReturnValue(new Promise(() => undefined));
    const { latest, unmount } = await renderHook();
    expect(latest().isAvailable).toBe(false);
    expect(latest().availabilityLoading).toBe(false);
    expect(getDriverAvailabilityActive()).toBe(false);
    unmount();
  });
});
