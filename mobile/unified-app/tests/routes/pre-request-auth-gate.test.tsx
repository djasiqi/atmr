import React from "react";
import { act, create } from "react-test-renderer";
import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { SafeAreaProvider } from "react-native-safe-area-context";
import PreRequestAuthGateScreen from "../../app/(public)/pre-request/auth-gate";

const safeAreaMetrics = {
  frame: { x: 0, y: 0, width: 390, height: 844 },
  insets: { top: 0, left: 0, right: 0, bottom: 0 },
};

/** ScrollView dans Screen ne rend pas toujours les enfants avec react-test-renderer ; on neutralise pour ce test d’intention route. */
jest.mock("../../src/design/responsive", () => {
  /* Jest interdit les imports externes dans la factory (hoisting) ; require() est le pattern officiel. */
  // eslint-disable-next-line @typescript-eslint/no-require-imports -- see jest.mock factory scope
  const ReactMod = require("react") as typeof import("react");
  // eslint-disable-next-line @typescript-eslint/no-require-imports -- see jest.mock factory scope
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

const mockUseLocalSearchParams = jest.fn() as jest.Mock<any>;
const mockRouterReplace = jest.fn() as jest.Mock<any>;

jest.mock("expo-router", () => ({
  useLocalSearchParams: () => mockUseLocalSearchParams(),
  useRouter: () => ({
    replace: (...args: unknown[]) => mockRouterReplace(...args),
  }),
}));

describe("pre-request auth gate route", () => {
  beforeEach(() => {
    mockUseLocalSearchParams.mockReset();
    mockRouterReplace.mockReset();
  });

  it(
    "builds next route with public draft id",
    async () => {
      mockUseLocalSearchParams.mockReturnValue({ draftId: "draft_abc" });
      let tree: ReturnType<typeof create> | any;
      await act(async () => {
        tree = create(
          <SafeAreaProvider initialMetrics={safeAreaMetrics}>
            <PreRequestAuthGateScreen />
          </SafeAreaProvider>
        );
        await Promise.resolve();
      });
      const pressables = tree.root.findAll(
        (node: { props?: { onPress?: unknown } }) => typeof node.props?.onPress === "function"
      );
      expect(pressables.length).toBeGreaterThan(0);
      await act(async () => {
        pressables[0].props.onPress();
        await Promise.resolve();
      });
      expect(mockRouterReplace).toHaveBeenCalledWith(
        expect.objectContaining({
          pathname: "/(public)/login",
          params: expect.objectContaining({
            next: "/(app)/(client)/booking/new?publicDraftId=draft_abc",
          }),
        })
      );
    },
    15_000
  );
});
