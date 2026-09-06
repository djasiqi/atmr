import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import {
  cancelCompanyRide,
  simulateCompanyPricing,
  getDispatchStatus,
  getCompanyDispatchMessages,
  getCompanyDispatchModes,
  getCompanyDispatchDelays,
  getCompanyPartnershipsForTransfer,
  getDispatchMissions,
  getDriversLocationsSnapshot,
  getOptimizerStatus,
  markCompanyRideUrgent,
  getRealtimeDashboard,
  resetCompanyAssignments,
  scheduleCompanyRide,
  switchCompanyDispatchMode,
  transferCompanyRide,
  updateCompanyRide,
} from "./companyApi";

const mockGet = jest.fn<(...args: any[]) => any>();
const mockPost = jest.fn<(...args: any[]) => any>();
const mockPut = jest.fn<(...args: any[]) => any>();
const mockDelete = jest.fn<(...args: any[]) => any>();
const mockEmitCompanyDispatchTelemetry = jest.fn<(...args: any[]) => any>();

jest.mock("../../../core/api/client", () => ({
  apiClient: {
    get: (...args: unknown[]) => mockGet(...args),
    post: (...args: unknown[]) => mockPost(...args),
    put: (...args: unknown[]) => mockPut(...args),
    delete: (...args: unknown[]) => mockDelete(...args),
  },
}));

jest.mock("../telemetry/companyTelemetry", () => ({
  emitCompanyDispatchTelemetry: (...args: unknown[]) => mockEmitCompanyDispatchTelemetry(...args),
}));

jest.mock("../dispatch/dispatchModeLock", () => ({
  SEMI_AUTO_DISPATCH_ENABLED: true,
  OPTIMIZER_ENABLED: true,
  FULLY_AUTO_DISPATCH_ENABLED: true,
  shouldMountDispatchEngine: () => true,
  assertSemiAutoDispatchEnabled: () => undefined,
  assertOptimizerEnabled: () => undefined,
  assertFullyAutoDispatchEnabled: () => undefined,
  assertDispatchModeSwitchAllowed: () => undefined,
  DispatchFeatureDisabledError: class DispatchFeatureDisabledError extends Error {},
}));

describe("company api normalization", () => {
  beforeEach(() => {
    mockGet.mockReset();
    mockPost.mockReset();
    mockPut.mockReset();
    mockDelete.mockReset();
    mockEmitCompanyDispatchTelemetry.mockReset();
  });

  it("normalizes missions from backend raw payload and resolves id aliases", async () => {
    mockGet.mockResolvedValueOnce({
      data: {
        items: [
          {
            booking_id: "101",
            company_id: "42",
            status: "assigned",
            client: { name: "Dupont" },
            route: { pickup_address: "A", dropoff_address: "B" },
            driver: { id: "7" },
          },
        ],
      },
    });

    const result = await getDispatchMissions({
      contextId: "company:42",
      date: "2026-01-01",
    });

    expect(result.missions).toEqual([
      expect.objectContaining({
        mission_id: 101,
        company_id: 42,
        status: "assigned",
        client_name: "Dupont",
        driver_id: 7,
        partner_company_name: null,
      }),
    ]);
    expect(mockGet.mock.calls[0][0]).toEqual("/company_mobile/dispatch/v1/rides");
    expect(mockGet.mock.calls[0][1]).toEqual(
      expect.objectContaining({
        params: expect.objectContaining({ page: 1, page_size: 50, date: "2026-01-01" }),
      })
    );
    expect(result.total).toBe(1);
    expect(result.loaded).toBe(1);
    expect(result.page_size).toBe(50);
    expect(result.is_complete).toBe(true);
    expect(result.date).toBe("2026-01-01");
  });

  it("conserve total/page serveur pour une page partielle", async () => {
    mockGet.mockResolvedValueOnce({
      data: {
        page: 1,
        page_size: 50,
        total: 87,
        items: [{ booking_id: 1, status: "assigned" }],
      },
    });
    const result = await getDispatchMissions({
      contextId: "company:42",
      date: "2026-09-06",
      page: 1,
    });
    expect(result.total).toBe(87);
    expect(result.loaded).toBe(1);
    expect(result.is_complete).toBe(false);
    expect(result.next_page).toBe(2);
  });

  it("normalizes partner_company_name from transfer payload", async () => {
    mockGet.mockResolvedValueOnce({
      data: {
        items: [
          {
            booking_id: 202,
            status: "assigned",
            transfer: { partner_company_name: "Emmenez-moi" },
          },
        ],
      },
    });

    const result = await getDispatchMissions({
      contextId: "company:42",
      date: "2026-01-01",
    });

    expect(result.missions[0]?.partner_company_name).toBe("Emmenez-moi");
  });

  it("priorise booking_id sur mission_id pour l’alignement retards `/company_dispatch/delays`", async () => {
    mockGet.mockResolvedValueOnce({
      data: {
        items: [
          {
            mission_id: 99999,
            booking_id: "101",
            status: "assigned",
            client: { name: "Dupont" },
            route: { pickup_address: "A", dropoff_address: "B" },
          },
        ],
      },
    });

    const result = await getDispatchMissions({
      contextId: "company:42",
      date: "2026-01-01",
    });

    expect(result.missions[0]?.mission_id).toBe(101);
  });

  it("marks dashboard metrics absent when the API key is missing (no implicite zero)", async () => {
    mockGet.mockResolvedValueOnce({
      data: {
        timestamp: "2026-01-01T10:00:00.000Z",
      },
    });
    const dashboard = await getRealtimeDashboard({
      contextId: "company:42",
      date: "2026-01-01",
    });
    expect(dashboard.delayed_bookings_metrics_available).toBe(false);
    expect(dashboard.opportunities_metrics_available).toBe(false);
    expect(dashboard.delayed_bookings).toBe(0);
    expect(dashboard.opportunities).toBe(0);
  });

  it("normalizes dashboard and optimizer payloads", async () => {
    mockGet
      .mockResolvedValueOnce({
        data: {
          stats: { delayed_bookings: 3 },
          opportunities: [{ id: 1 }, { id: 2 }],
          quality_metrics: { avg_delay: 8 },
          timestamp: "2026-01-01T10:00:00.000Z",
        },
      })
      .mockResolvedValueOnce({
        data: {
          optimizer: {
            active: true,
            running: true,
            last_tick: "2026-01-01T10:00:00.000Z",
            next_window_start: "2026-01-01T10:05:00.000Z",
          },
        },
      });

    const dashboard = await getRealtimeDashboard({
      contextId: "company:42",
      date: "2026-01-01",
    });
    const optimizer = await getOptimizerStatus({ contextId: "company:42" });

    expect(dashboard).toEqual(
      expect.objectContaining({
        delayed_bookings_metrics_available: true,
        delayed_bookings: 3,
        opportunities_metrics_available: true,
        opportunities: 2,
        avg_delay_minutes: 8,
      })
    );
    expect(optimizer.status).toEqual(
      expect.objectContaining({
        optimizer_enabled: true,
        optimizer_state: "running",
      })
    );
  });

  it("falls back from drivers snapshot endpoint to live endpoint", async () => {
    mockGet
      .mockRejectedValueOnce(new Error("primary endpoint down"))
      .mockResolvedValueOnce({
        data: {
          drivers: [
            {
              driver_id: 7,
              lat: 46.5,
              lon: 6.6,
              timestamp: "2026-01-01T10:00:00.000Z",
              recorded_at: "2026-01-01T10:00:00.000Z",
            },
          ],
        },
      });

    const locations = await getDriversLocationsSnapshot({ contextId: "company:42" });

    expect(locations.locations[0]).toEqual(
      expect.objectContaining({
        driver_id: 7,
        latitude: 46.5,
        longitude: 6.6,
        recorded_at: "2026-01-01T10:00:00.000Z",
      })
    );
    expect(mockGet).toHaveBeenCalledTimes(2);
  });

  it("uses company_mobile dispatch chat endpoint first", async () => {
    mockGet.mockResolvedValueOnce({
      data: {
        messages: [
          { id: 1, content: "hello", created_at: "2026-01-01T10:00:00.000Z" },
        ],
      },
    });

    const payload = await getCompanyDispatchMessages({
      contextId: "company:42",
      date: "2026-01-01",
    });

    expect(payload).toEqual(
      expect.objectContaining({
        messages: expect.any(Array),
      })
    );
    expect(mockGet).toHaveBeenCalledTimes(1);
    expect(mockGet.mock.calls[0][0]).toEqual("/company_mobile/dispatch/v1/chat/messages");
  });

  it("falls back chat endpoint to legacy dispatch endpoints when company_mobile fails", async () => {
    mockGet
      .mockRejectedValueOnce({ response: { status: 404 } })
      .mockResolvedValueOnce({
        data: {
          messages: [
            { id: 2, content: "mobile chat", created_at: "2026-01-01T11:00:00.000Z" },
          ],
        },
      });

    const payload = await getCompanyDispatchMessages({
      contextId: "company:42",
      date: "2026-01-01",
    });

    expect(payload).toEqual(expect.objectContaining({ messages: expect.any(Array) }));
    expect(mockGet.mock.calls[0][0]).toEqual("/company_mobile/dispatch/v1/chat/messages");
    expect(mockGet.mock.calls[1][0]).toEqual("/dispatch/v1/messages");
  });

  it("falls back to legacy dispatch mode endpoint for read + switch", async () => {
    mockGet
      .mockRejectedValueOnce({ response: { status: 404 } })
      .mockResolvedValueOnce({ data: { dispatch_mode: "manual" } });
    mockPut
      .mockRejectedValueOnce({ response: { status: 404 } })
      .mockResolvedValueOnce({ data: { dispatch_mode: "semi_auto" } });
    mockPost.mockRejectedValueOnce({ response: { status: 404 } });

    const modes = await getCompanyDispatchModes({ contextId: "company:42" });
    const switched = await switchCompanyDispatchMode({
      contextId: "company:42",
      mode: "semi_auto",
    });

    expect(modes).toEqual(expect.objectContaining({ dispatch_mode: "manual" }));
    expect(switched).toEqual(expect.objectContaining({ dispatch_mode: "semi_auto" }));
    expect(mockGet.mock.calls[0][0]).toEqual("/company_mobile/dispatch/v1/mode");
    expect(mockGet.mock.calls[1][0]).toEqual("/dispatch/v1/modes");
    expect(mockPut.mock.calls[0][0]).toEqual("/company_mobile/dispatch/v1/mode");
    expect(mockPost.mock.calls[0][0]).toEqual("/dispatch/v1/modes/switch");
    expect(mockPut.mock.calls[1][0]).toEqual("/dispatch/v1/mode");
  });

  it("falls back reset endpoint from reset-assignments to reset", async () => {
    mockPost
      .mockRejectedValueOnce({ response: { status: 404 } })
      .mockResolvedValueOnce({ data: { ok: true } });

    const payload = await resetCompanyAssignments({
      contextId: "company:42",
      date: "2026-01-01",
    });

    expect(payload).toEqual(expect.objectContaining({ ok: true }));
    expect(mockPost.mock.calls[0][0]).toEqual("/dispatch/v1/reset-assignments");
    expect(mockPost.mock.calls[1][0]).toEqual("/dispatch/v1/reset");
  });

  it("posts schedule payload on dispatch schedule endpoint", async () => {
    mockPut.mockRejectedValueOnce({ response: { status: 404 } });
    mockPost.mockResolvedValueOnce({ data: { ok: true } });

    await scheduleCompanyRide({
      contextId: "company:42",
      missionId: 123,
      payload: {
        pickup_at: "2026-01-02T10:30:00.000Z",
        timezone: "Europe/Zurich",
        note: "pilot schedule",
      },
    });

    expect(mockPost).toHaveBeenCalledWith(
      "/company_mobile/dispatch/v1/rides/123/schedule",
      expect.objectContaining({
        pickup_at: "2026-01-02T10:30:00.000Z",
        timezone: "Europe/Zurich",
      }),
      expect.objectContaining({
        headers: expect.objectContaining({
          "X-Active-Context-Id": "company:42",
        }),
      })
    );
  });

  it("met à jour une course via l’endpoint mobile dispatch", async () => {
    mockPut.mockResolvedValueOnce({ data: { ok: true } });

    await updateCompanyRide({
      contextId: "company:42",
      missionId: 456,
      payload: {
        pickup_address: "Rue A",
        dropoff_address: "Rue B",
        scheduled_time: "2026-06-22T09:45:00",
      },
    });

    expect(mockPut).toHaveBeenCalledWith(
      "/company_mobile/dispatch/v1/rides/456",
      expect.objectContaining({
        scheduled_time: "2026-06-22T09:45:00",
      }),
      expect.objectContaining({
        headers: expect.objectContaining({
          "X-Active-Context-Id": "company:42",
        }),
      })
    );
  });

  it("uses cancel parity payload with reason_code and note", async () => {
    mockDelete.mockRejectedValueOnce({ response: { status: 404 } });
    mockPost.mockResolvedValueOnce({ data: { ok: true } });

    await cancelCompanyRide({
      contextId: "company:42",
      missionId: 321,
      reasonCode: "client_no_show",
      note: "No answer after two calls",
    });

    expect(mockPost).toHaveBeenCalledWith(
      "/company_mobile/dispatch/v1/rides/321/cancel",
      expect.objectContaining({
        reason_code: "client_no_show",
        note: "No answer after two calls",
        reason: "client_no_show",
      }),
      expect.any(Object)
    );
  });

  it("sends typed urgent payload with default source", async () => {
    mockPost
      .mockRejectedValueOnce({ response: { status: 404 } })
      .mockResolvedValueOnce({ data: { ok: true } });

    await markCompanyRideUrgent({
      contextId: "company:42",
      missionId: 77,
      payload: {
        urgent: true,
        reason_code: "medical_priority",
      },
    });

    expect(mockPost).toHaveBeenCalledWith(
      "/company_mobile/dispatch/v1/rides/77/urgent",
      expect.objectContaining({
        urgent: true,
        reason_code: "medical_priority",
        reason: "medical_priority",
        extra_delay_minutes: 15,
        source: "mobile_unified_company",
      }),
      expect.any(Object)
    );
  });

  it("normalizes explicit dispatch status endpoint payload", async () => {
    mockGet.mockResolvedValueOnce({
      data: {
        dispatch_mode: "semi_auto",
        dispatch_state: "running",
      },
    });

    const status = await getDispatchStatus({
      contextId: "company:42",
      date: "2026-01-01",
    });

    expect(status).toEqual(
      expect.objectContaining({
        context_id: "company:42",
        dispatch_mode: "semi_auto",
        dispatch_state: "running",
        source: "scheduler_runtime",
      })
    );
  });

  it("falls back dashboard/status/optimizer to company_mobile dispatch endpoints", async () => {
    mockGet
      // getRealtimeDashboard: /company_mobile/dispatch/v1/dashboard/realtime -> fallback
      .mockRejectedValueOnce({ response: { status: 404 } })
      .mockResolvedValueOnce({
        data: {
          stats: { delayed_bookings: 1 },
          opportunities: [{ id: 1 }],
          quality_metrics: { avg_delay: 3 },
          timestamp: "2026-01-01T10:00:00.000Z",
        },
      })
      // getDispatchStatus: /company_mobile/dispatch/v1/status -> fallback
      .mockRejectedValueOnce({ response: { status: 404 } })
      .mockResolvedValueOnce({
        data: {
          dispatch_mode: "manual",
          dispatch_state: "idle",
        },
      })
      // getOptimizerStatus: /company_mobile/dispatch/v1/status -> fallback
      .mockRejectedValueOnce({ response: { status: 404 } })
      .mockResolvedValueOnce({
        data: {
          optimizer: {
            active: true,
            running: false,
          },
        },
      });

    const dashboard = await getRealtimeDashboard({
      contextId: "company:42",
      date: "2026-01-01",
    });
    const dispatchStatus = await getDispatchStatus({
      contextId: "company:42",
      date: "2026-01-01",
    });
    const optimizer = await getOptimizerStatus({ contextId: "company:42" });

    expect(dashboard.delayed_bookings).toBe(1);
    expect(dispatchStatus.dispatch_mode).toBe("manual");
    expect(dispatchStatus.dispatch_state).toBe("idle");
    expect(optimizer.status.optimizer_enabled).toBe(true);
    expect(mockGet.mock.calls[0][0]).toBe("/company_mobile/dispatch/v1/dashboard/realtime");
    expect(mockGet.mock.calls[1][0]).toBe("/dispatch/v1/dashboard/realtime");
    expect(mockGet.mock.calls[2][0]).toBe("/company_mobile/dispatch/v1/status");
    expect(mockGet.mock.calls[3][0]).toBe("/dispatch/v1/status");
    expect(mockGet.mock.calls[4][0]).toBe("/company_mobile/dispatch/v1/status");
    expect(mockGet.mock.calls[5][0]).toBe("/dispatch/v1/status");
  });

  it("surfaces transfer conflict 409 with explicit message", async () => {
    mockPost.mockRejectedValueOnce({ response: { status: 409 }, isAxiosError: true });

    await expect(
      transferCompanyRide({
        contextId: "company:42",
        missionId: 55,
        partnershipId: 12,
      })
    ).rejects.toThrow(/Conflit de transfert detecte/);

    expect(mockEmitCompanyDispatchTelemetry).toHaveBeenCalledWith(
      "company.dispatch.transfer_conflict",
      expect.objectContaining({
        mission_id: 55,
        partnership_id: 12,
      }),
      { allowWhenDisabled: true }
    );
  });

  it("loads partnerships for transfer from company_mobile endpoint", async () => {
    mockGet.mockResolvedValueOnce({
      data: {
        data: [
          {
            id: 7,
            partner_company_id: 99,
            partner_company_name: "Partenaire Test",
          },
        ],
      },
    });

    const result = await getCompanyPartnershipsForTransfer({ contextId: "company:42" });
    expect(result.items).toEqual([
      { id: 7, label: "Partenaire Test", partnerCompanyId: 99 },
    ]);
    expect(mockGet.mock.calls[0][0]).toBe("/company_mobile/partnerships/for-transfer");
  });

  it("posts transfer with partnership id and booking_id", async () => {
    mockPost.mockResolvedValueOnce({ data: { data: { id: 1 } } });

    await transferCompanyRide({
      contextId: "company:42",
      missionId: 55,
      partnershipId: 12,
    });

    expect(mockPost.mock.calls[0][0]).toBe("/company_mobile/partnerships/12/transfers");
    expect(mockPost.mock.calls[0][1]).toEqual({ booking_id: 55 });
  });

  it("fusionne les retards live et snapshot comme le tableau web", async () => {
    mockGet
      .mockResolvedValueOnce({
        data: { delays: [{ booking_id: 1, delay_minutes: 4 }] },
      })
      .mockResolvedValueOnce({
        data: [{ booking_id: 1, delay_minutes: 9, pickup_eta: "2026-01-01T10:00:00.000Z" }],
      });

    const rows = await getCompanyDispatchDelays({ contextId: "company:42", date: "2026-01-01" });

    expect(mockGet.mock.calls[0]?.[0]).toBe("/company_dispatch/delays/live");
    expect(mockGet.mock.calls[1]?.[0]).toBe("/company_dispatch/delays");
    expect(rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          booking_id: 1,
          delay_minutes: 9,
          pickup_eta: "2026-01-01T10:00:00.000Z",
        }),
      ]),
    );
  });

  it("posts pricing simulation payload to pricing endpoint", async () => {
    mockPost.mockResolvedValueOnce({
      data: {
        amount: 48.2,
        pricing: { amount: 48.2 },
      },
    });

    const payload = {
      pricing_profile_version_id: 10,
      booking: {
        pickup_at: "2026-05-05T14:30:00",
        pickup_lat: 46.2,
        pickup_lng: 6.1,
        dropoff_lat: 46.21,
        dropoff_lng: 6.15,
      },
    };
    const response = await simulateCompanyPricing({
      contextId: "company:42",
      payload,
    });

    expect(response).toEqual(expect.objectContaining({ amount: 48.2 }));
    expect(mockPost).toHaveBeenCalledWith(
      "/pricing/simulate",
      payload,
      expect.objectContaining({
        headers: expect.objectContaining({
          "X-Active-Context-Id": "company:42",
        }),
      })
    );
  });
});
