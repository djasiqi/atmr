import React from "react";
import { act, create } from "react-test-renderer";
import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import PreRequestAuthGateScreen from "../../app/(public)/pre-request/auth-gate";

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
        tree = create(<PreRequestAuthGateScreen />);
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
