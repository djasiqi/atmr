import React from "react";
import { act, create } from "react-test-renderer";
import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import QuickActionScreen from "../../app/quick-action";

const mockUseLocalSearchParams = jest.fn() as jest.Mock<any>;
const mockRouterReplace = jest.fn() as jest.Mock<any>;
const mockQuickAccept = jest.fn() as jest.Mock<any>;
const mockQuickReject = jest.fn() as jest.Mock<any>;
const mockQuickStart = jest.fn() as jest.Mock<any>;
const mockQuickComplete = jest.fn() as jest.Mock<any>;
const mockRedirect = jest.fn() as jest.Mock<any>;

jest.mock("expo-router", () => ({
  useLocalSearchParams: () => mockUseLocalSearchParams(),
  useRouter: () => ({ replace: (...args: unknown[]) => mockRouterReplace(...args) }),
  Redirect: (props: { href: unknown }) => {
    mockRedirect(props);
    return null;
  },
}));

jest.mock("../../src/features/driver/api", () => ({
  quickAcceptDriverMission: (...args: unknown[]) => mockQuickAccept(...args),
  quickRejectDriverMission: (...args: unknown[]) => mockQuickReject(...args),
  quickStartDriverMission: (...args: unknown[]) => mockQuickStart(...args),
  quickCompleteDriverMission: (...args: unknown[]) => mockQuickComplete(...args),
}));

describe("quick-action route", () => {
  beforeEach(() => {
    mockUseLocalSearchParams.mockReset();
    mockRouterReplace.mockReset();
    mockQuickAccept.mockReset();
    mockQuickReject.mockReset();
    mockQuickStart.mockReset();
    mockQuickComplete.mockReset();
    mockRedirect.mockReset();
    mockQuickAccept.mockResolvedValue(undefined);
    mockQuickReject.mockResolvedValue(undefined);
    mockQuickStart.mockResolvedValue(undefined);
    mockQuickComplete.mockResolvedValue(undefined);
  });

  it("redirects to invalid-link fallback for invalid params", async () => {
    mockUseLocalSearchParams.mockReturnValue({ missionId: "", action: "" });
    await act(async () => {
      create(<QuickActionScreen />);
      await Promise.resolve();
    });

    expect(mockRedirect).toHaveBeenCalledWith(
      expect.objectContaining({
        href: "/(public)/fallback/invalid-link?reason=quick_action_invalid",
      })
    );
  });
});
