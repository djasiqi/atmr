import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import {
  getDriverBookingsAll,
  getDriverCompanyBookingsToday,
  getDriverCompletedTrips,
  getDriverProfile,
  getDriverRoute,
  triggerDriverTestPush,
  updateDriverPhoto,
  updateDriverProfile,
} from "./api";

type ApiResponse = { data: unknown };
type MockedApiMethod = (...args: unknown[]) => Promise<ApiResponse>;

const mockGet = jest.fn<MockedApiMethod>();
const mockPut = jest.fn<MockedApiMethod>();
const mockPost = jest.fn<MockedApiMethod>();

jest.mock("../../core/api/client", () => ({
  apiClient: {
    get: (...args: unknown[]) => mockGet(...args),
    put: (...args: unknown[]) => mockPut(...args),
    post: (...args: unknown[]) => mockPost(...args),
  },
}));

describe("driver secondary api contracts", () => {
  beforeEach(() => {
    mockGet.mockReset();
    mockPut.mockReset();
    mockPost.mockReset();
  });

  it("loads profile and route from driver endpoints", async () => {
    mockGet
      .mockResolvedValueOnce({ data: { full_name: "Driver One", email: "d1@example.com" } })
      .mockResolvedValueOnce({ data: { points: [{}, {}, {}] } });

    const profile = await getDriverProfile();
    const route = await getDriverRoute();

    expect(profile).toEqual(expect.objectContaining({ full_name: "Driver One" }));
    expect(route).toEqual(expect.objectContaining({ points: expect.any(Array) }));
    expect(mockGet.mock.calls[0][0]).toEqual("/driver/me/profile");
    expect(mockGet.mock.calls[1][0]).toEqual("/driver/me/route");
  });

  it("updates profile and photo on dedicated endpoints", async () => {
    mockPut
      .mockResolvedValueOnce({ data: { full_name: "Driver Updated" } })
      .mockResolvedValueOnce({ data: { photo_url: "https://cdn/p.png" } });

    const updated = await updateDriverProfile({ full_name: "Driver Updated" });
    const photo = await updateDriverPhoto("https://cdn/p.png");

    expect(updated).toEqual(expect.objectContaining({ full_name: "Driver Updated" }));
    expect(photo).toEqual(expect.objectContaining({ photo_url: "https://cdn/p.png" }));
    expect(mockPut.mock.calls[0][0]).toEqual("/driver/me/profile");
    expect(mockPut.mock.calls[1][0]).toEqual("/driver/me/photo");
  });

  it("normalizes list-like payloads for all/today/completed endpoints", async () => {
    mockGet
      .mockResolvedValueOnce({ data: { items: [{ id: 1 }, { id: 2 }] } })
      .mockResolvedValueOnce({ data: { data: [{ id: 3 }] } })
      .mockResolvedValueOnce({ data: { bookings: [{ id: 9, status: "COMPLETED" }] } });

    const allBookings = await getDriverBookingsAll();
    const todayBookings = await getDriverCompanyBookingsToday();
    const completed = await getDriverCompletedTrips(88);

    expect(allBookings).toHaveLength(2);
    expect(todayBookings).toHaveLength(1);
    expect(completed).toHaveLength(1);
    expect(mockGet.mock.calls[0][0]).toEqual("/driver/me/bookings/all");
    expect(mockGet.mock.calls[1][0]).toEqual("/driver/me/company-bookings/today");
    expect(mockGet.mock.calls[2][0]).toEqual("/drivers/88/completed-trips");
  });

  it("triggers test push endpoint for support checks", async () => {
    mockPost.mockResolvedValueOnce({ data: { ok: true } });

    await triggerDriverTestPush();

    expect(mockPost).toHaveBeenCalledTimes(1);
    expect(mockPost.mock.calls[0][0]).toEqual("/driver/me/test-push");
  });
});

