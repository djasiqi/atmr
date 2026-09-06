import { describe, expect, it } from "@jest/globals";
import { AxiosError } from "axios";
import {
  COMPANY_OFFLINE_ACTION_MESSAGE,
  COMPANY_OFFLINE_DAY_TITLE,
  CompanyOfflineActionError,
  assertCompanyOnlineForMutation,
  isCompanyNetworkOffline,
  isCompanyNetworkRequestError,
  resolveDispatchDayEmptyKind,
  shouldRetryCompanyMutation,
  shouldRetryCompanyQuery,
} from "./companyOfflinePolicy";

describe("companyOfflinePolicy", () => {
  it("détecte l’offline uniquement si connected=false", () => {
    expect(isCompanyNetworkOffline({ connected: false, internetReachable: false, type: "none", cellularGeneration: null, updatedAt: "" })).toBe(true);
    expect(isCompanyNetworkOffline({ connected: true, internetReachable: false, type: "wifi", cellularGeneration: null, updatedAt: "" })).toBe(false);
  });

  it("refuse une mutation immédiatement hors ligne", () => {
    expect(() => {
      if (isCompanyNetworkOffline({ connected: false, internetReachable: false, type: "none", cellularGeneration: null, updatedAt: "" })) {
        throw new CompanyOfflineActionError();
      }
    }).toThrow(COMPANY_OFFLINE_ACTION_MESSAGE);
    expect(assertCompanyOnlineForMutation).toBeDefined();
  });

  it("ne retry pas une query / mutation réseau", () => {
    const network = new AxiosError("Network Error");
    network.code = "ERR_NETWORK";
    expect(shouldRetryCompanyQuery(0, network)).toBe(false);
    expect(shouldRetryCompanyMutation(0, network)).toBe(false);
    expect(isCompanyNetworkRequestError(new CompanyOfflineActionError())).toBe(true);
  });

  it("journée absente du cache offline ≠ J précédent ni spinner", () => {
    expect(
      resolveDispatchDayEmptyKind({
        missionCount: 0,
        hasCachedDay: false,
        isLoading: true,
        isError: false,
        isOffline: true,
        isNetworkError: false,
        searchActive: false,
        isDayComplete: false,
      })
    ).toBe("offline_unavailable");
    expect(
      resolveDispatchDayEmptyKind({
        missionCount: 3,
        hasCachedDay: true,
        isLoading: false,
        isError: true,
        isOffline: true,
        isNetworkError: true,
        searchActive: false,
        isDayComplete: true,
      })
    ).toBe("list");
    expect(COMPANY_OFFLINE_DAY_TITLE).toContain("hors connexion");
  });

  it("cache présent + recherche locale reste une liste / filtre", () => {
    expect(
      resolveDispatchDayEmptyKind({
        missionCount: 0,
        hasCachedDay: true,
        isLoading: false,
        isError: false,
        isOffline: true,
        isNetworkError: false,
        searchActive: true,
        isDayComplete: true,
      })
    ).toBe("search_none");
  });
});
