import React from "react";
import { act, create } from "react-test-renderer";
import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { SafeAreaProvider } from "react-native-safe-area-context";
import BookingStatusScreen from "../../app/(public)/booking-status";

jest.mock("../../assets/images/landing-background.png", () => 1);

/** ScrollView dans Screen ne rend pas toujours les enfants avec react-test-renderer. */
jest.mock("../../src/design/responsive", () => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports -- jest.mock factory scope
  const ReactMod = require("react") as typeof import("react");
  // eslint-disable-next-line @typescript-eslint/no-require-imports -- jest.mock factory scope
  const { View } = require("react-native") as typeof import("react-native");
  const actual =
    jest.requireActual<typeof import("../../src/design/responsive")>(
      "../../src/design/responsive"
    );
  return {
    ...actual,
    Screen: ({ children }: { children: ReactMod.ReactNode }) =>
      ReactMod.createElement(View, { style: { flex: 1 } }, children),
  };
});

const safeAreaMetrics = {
  frame: { x: 0, y: 0, width: 390, height: 844 },
  insets: { top: 0, left: 0, right: 0, bottom: 0 },
};

function renderWithSafeArea(node: React.ReactElement) {
  return create(<SafeAreaProvider initialMetrics={safeAreaMetrics}>{node}</SafeAreaProvider>);
}

const mockUseLocalSearchParams = jest.fn() as jest.Mock<any>;
const mockRedirect = jest.fn() as jest.Mock<any>;
const mockFetchPublicBookingStatus = jest.fn() as jest.Mock<any>;
const mockFetchGuestBookingStatus = jest.fn() as jest.Mock<any>;

jest.mock("expo-router", () => ({
  useLocalSearchParams: () => mockUseLocalSearchParams(),
  usePathname: () => "/(public)/booking-status",
  useRouter: () => ({ push: jest.fn(), replace: jest.fn(), canGoBack: () => false, back: jest.fn() }),
  Redirect: (props: { href: unknown }) => {
    mockRedirect(props);
    return null;
  },
}));

jest.mock("../../src/core/sessionProvider", () => ({
  useSession: () => ({ bootstrap: { is_authenticated: false } }),
}));

jest.mock("../../src/core/api/client", () => ({
  fetchPublicBookingStatus: (...args: unknown[]) => mockFetchPublicBookingStatus(...args),
  fetchGuestBookingStatus: (...args: unknown[]) => mockFetchGuestBookingStatus(...args),
  linkGuestBookingToAccount: jest.fn(),
}));

describe("booking-status route", () => {
  beforeEach(() => {
    mockUseLocalSearchParams.mockReset();
    mockRedirect.mockReset();
    mockFetchPublicBookingStatus.mockReset();
    mockFetchGuestBookingStatus.mockReset();
  });

  it("redirects to invalid-link fallback when token is invalid", async () => {
    mockUseLocalSearchParams.mockReturnValue({ token: "bad_token" });
    mockFetchPublicBookingStatus.mockRejectedValue({ status: 401 });
    mockFetchGuestBookingStatus.mockRejectedValue({ status: 401 });
    let tree: any;
    await act(async () => {
      tree = renderWithSafeArea(<BookingStatusScreen />);
      await Promise.resolve();
    });
    const pressables = tree.root.findAll(
      (node: { props?: { onPress?: unknown } }) => typeof node.props?.onPress === "function"
    );
    expect(pressables.length).toBeGreaterThanOrEqual(2);
    await act(async () => {
      pressables[1].props.onPress();
      await Promise.resolve();
    });
    expect(mockRedirect).toHaveBeenCalledWith(
      expect.objectContaining({
        href: "/(public)/fallback/invalid-link?reason=token_invalid",
      })
    );
  });

  it("falls back to guest booking status when standard status is not found", async () => {
    mockUseLocalSearchParams.mockReturnValue({ token: "guest_token_1" });
    mockFetchPublicBookingStatus.mockRejectedValue({ status: 404 });
    mockFetchGuestBookingStatus.mockResolvedValue({
      guest_booking_id: "gb_123",
      status: "pending_payment",
      departure: "Geneve",
      destination: "Lausanne",
      amount: 42,
      currency: "CHF",
    });
    let tree: any;
    await act(async () => {
      tree = renderWithSafeArea(<BookingStatusScreen />);
      await Promise.resolve();
    });
    const pressables = tree.root.findAll(
      (node: { props?: { onPress?: unknown } }) => typeof node.props?.onPress === "function"
    );
    expect(pressables.length).toBeGreaterThanOrEqual(2);
    await act(async () => {
      pressables[1].props.onPress();
      await Promise.resolve();
    });
    const textContent = tree.root.findAllByType("Text").map((node: any) => node.props.children).flat().join(" ");
    expect(textContent).toContain("gb_123");
    expect(mockFetchGuestBookingStatus).toHaveBeenCalledWith("guest_token_1");
  });
});
