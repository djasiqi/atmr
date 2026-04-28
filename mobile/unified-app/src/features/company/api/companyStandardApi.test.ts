import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import {
  createStandardCompanyClient,
  createStandardCompanyClientStay,
  fetchStandardCompanyBillingParties,
  linkStandardCompanyClientBillingParty,
} from "./companyStandardApi";

type ApiResponse = { data: unknown };
type MockedApiMethod = (...args: unknown[]) => Promise<ApiResponse>;

const mockGet = jest.fn<MockedApiMethod>();
const mockPost = jest.fn<MockedApiMethod>();

jest.mock("../../../core/api/client", () => ({
  apiClient: {
    get: (...args: unknown[]) => mockGet(...args),
    post: (...args: unknown[]) => mockPost(...args),
  },
}));

describe("company standard api minimal pilot", () => {
  beforeEach(() => {
    mockGet.mockReset();
    mockPost.mockReset();
  });

  it("creates a client with active context header", async () => {
    mockPost.mockResolvedValueOnce({ data: { id: 12 } });

    await createStandardCompanyClient({
      contextId: "company:42",
      payload: {
        first_name: "Lea",
        last_name: "Martin",
        gender: "female",
      },
    });

    expect(mockPost).toHaveBeenCalledWith(
      "/companies/me/clients",
      expect.objectContaining({ first_name: "Lea" }),
      expect.objectContaining({
        headers: expect.objectContaining({ "X-Active-Context-Id": "company:42" }),
      })
    );
  });

  it("creates a client stay using standard endpoint", async () => {
    mockPost.mockResolvedValueOnce({ data: { ok: true } });

    await createStandardCompanyClientStay({
      contextId: "company:42",
      clientId: 12,
      payload: {
        company_id: 42,
        start_date: "2026-05-01",
      },
    });

    expect(mockPost).toHaveBeenCalledWith(
      "/clients/12/stays",
      expect.objectContaining({ company_id: 42 }),
      expect.any(Object)
    );
  });

  it("normalizes billing parties payload", async () => {
    mockGet.mockResolvedValueOnce({
      data: {
        data: [{ id: "9", display_name: "Assureur A", type: "insurance" }],
      },
    });

    const parties = await fetchStandardCompanyBillingParties({ contextId: "company:42" });
    expect(parties).toEqual([
      expect.objectContaining({
        id: 9,
        display_name: "Assureur A",
        type: "insurance",
      }),
    ]);
  });

  it("links billing party to client", async () => {
    mockPost.mockResolvedValueOnce({ data: { ok: true } });

    await linkStandardCompanyClientBillingParty({
      contextId: "company:42",
      clientId: 12,
      payload: {
        billing_party_id: 9,
        is_default: true,
      },
    });

    expect(mockPost).toHaveBeenCalledWith(
      "/clients/12/billing-parties",
      expect.objectContaining({ billing_party_id: 9 }),
      expect.any(Object)
    );
  });
});
