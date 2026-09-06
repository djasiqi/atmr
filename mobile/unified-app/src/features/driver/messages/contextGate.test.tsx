import React from "react";
import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { act, create } from "react-test-renderer";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

/**
 * P1-C3 : les pollers du hub driver ne doivent JAMAIS interroger le backend
 * hors contexte chauffeur actif (ex. apres bascule vers l'espace entreprise).
 */

const mockDriverContextId = jest.fn() as jest.MockedFunction<() => string | null>;
const mockFetchHubUnreadCount = jest.fn() as jest.MockedFunction<
  (companyId: number) => Promise<number>
>;
const mockGetDriverMissionEta = jest.fn() as jest.MockedFunction<
  (bookingId: number, opts?: unknown) => Promise<unknown>
>;

jest.mock("../hooks", () => ({
  useActiveDriverContextId: () => mockDriverContextId(),
}));

jest.mock("../../../core/sessionProvider", () => ({
  useSession: () => ({ bootstrap: null, status: "ready" }),
}));

jest.mock("../../../core/realtime/realtimeManager", () => ({
  realtimeManager: { isDriverSocketReady: () => false },
}));

jest.mock("./api", () => ({
  fetchHubUnreadCount: (companyId: number) => mockFetchHubUnreadCount(companyId),
}));

jest.mock("../api", () => ({
  getDriverMissionEta: (bookingId: number, opts?: unknown) =>
    mockGetDriverMissionEta(bookingId, opts),
}));

import {
  resetDriverSessionNetworkGateForTests,
  setDriverSessionNetworkReady,
} from "../../../core/network/driverSessionNetworkGate";
import { useHubUnreadCount, useMissionEtaMinutes } from "./hooks";

function HookProbe() {
  useHubUnreadCount(1);
  useMissionEtaMinutes(42, "IN_PROGRESS");
  return null;
}

async function renderProbe(): Promise<() => void> {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  let renderer: ReturnType<typeof create>;
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={client}>
        <HookProbe />
      </QueryClientProvider>
    );
  });
  await act(async () => {
    await Promise.resolve();
  });
  return () => {
    act(() => {
      renderer!.unmount();
    });
    // Stoppe timers/refetch retenus par le cache (sinon Jest ne sort pas).
    client.clear();
    client.unmount();
  };
}

describe("P1-C3 - gate contexte des pollers hub driver", () => {
  beforeEach(() => {
    resetDriverSessionNetworkGateForTests();
    setDriverSessionNetworkReady(true);
    mockDriverContextId.mockReset();
    mockFetchHubUnreadCount.mockReset();
    mockGetDriverMissionEta.mockReset();
    mockFetchHubUnreadCount.mockResolvedValue(0);
    mockGetDriverMissionEta.mockResolvedValue({});
  });

  it("contexte company (pas de contexte driver) : AUCUNE requete reseau", async () => {
    mockDriverContextId.mockReturnValue(null);

    const unmount = await renderProbe();

    expect(mockFetchHubUnreadCount).not.toHaveBeenCalled();
    expect(mockGetDriverMissionEta).not.toHaveBeenCalled();
    unmount();
  });

  it("contexte driver actif : les requetes partent normalement", async () => {
    mockDriverContextId.mockReturnValue("driver:9");

    const unmount = await renderProbe();

    expect(mockFetchHubUnreadCount).toHaveBeenCalledWith(1);
    expect(mockGetDriverMissionEta).toHaveBeenCalledWith(42, {
      missionStatus: "IN_PROGRESS",
    });
    unmount();
  });

  it("transition driver -> company : AUCUN nouveau poll apres le switch (timers 60 s)", async () => {
    jest.useFakeTimers();
    try {
      mockDriverContextId.mockReturnValue("driver:9");
      const client = new QueryClient({
        defaultOptions: { queries: { retry: false, gcTime: 0 } },
      });
      const makeTree = () => (
        <QueryClientProvider client={client}>
          <HookProbe />
        </QueryClientProvider>
      );
      let renderer: ReturnType<typeof create>;
      await act(async () => {
        renderer = create(makeTree());
      });
      await act(async () => {
        await Promise.resolve();
      });
      expect(mockFetchHubUnreadCount).toHaveBeenCalledTimes(1);
      expect(mockGetDriverMissionEta).toHaveBeenCalledTimes(1);

      // Sanity : le polling est reellement actif en contexte driver
      // (unread 15 s, eta 20 s -> 1 intervalle chacun apres 21 s).
      await act(async () => {
        jest.advanceTimersByTime(21_000);
      });
      await act(async () => {
        await Promise.resolve();
      });
      expect(mockFetchHubUnreadCount.mock.calls.length).toBeGreaterThanOrEqual(2);
      expect(mockGetDriverMissionEta.mock.calls.length).toBeGreaterThanOrEqual(2);

      // Switch reel : provider -> company, QueryClient et composant CONSERVES.
      mockDriverContextId.mockReturnValue(null);
      await act(async () => {
        renderer!.update(makeTree());
      });
      await act(async () => {
        await Promise.resolve();
      });
      const unreadAtSwitch = mockFetchHubUnreadCount.mock.calls.length;
      const etaAtSwitch = mockGetDriverMissionEta.mock.calls.length;

      // >= 3 intervalles de poll apres la transition.
      await act(async () => {
        jest.advanceTimersByTime(60_000);
      });
      await act(async () => {
        await Promise.resolve();
      });

      expect(mockFetchHubUnreadCount.mock.calls.length).toBe(unreadAtSwitch);
      expect(mockGetDriverMissionEta.mock.calls.length).toBe(etaAtSwitch);

      act(() => {
        renderer!.unmount();
      });
      client.clear();
      client.unmount();
    } finally {
      jest.useRealTimers();
    }
  });
});
