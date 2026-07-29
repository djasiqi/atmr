import { describe, expect, it, jest } from "@jest/globals";
import { createElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { View } from "react-native";

jest.mock("./useAccessibilityScale", () => ({
  useAccessibilityScale: jest.fn(),
}));

jest.mock("./useResponsiveTokens", () => ({
  useResponsiveTokens: () => ({ spacingSm: 8 }),
}));

const { useAccessibilityScale } = require("./useAccessibilityScale") as {
  useAccessibilityScale: jest.Mock;
};
const { ModalFooterActions } = require("../ui/ModalFooterActions");

describe("ModalFooterActions", () => {
  it("passe en column quand shouldStackRows", () => {
    useAccessibilityScale.mockReturnValue({ shouldStackRows: true });
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(
        createElement(ModalFooterActions, {
          secondary: createElement(View, { testID: "sec" }),
          primary: createElement(View, { testID: "pri" }),
        })
      );
    });
    const rows = tree!.root.findAllByType(View);
    const layout = rows.find((v) => v.props.style && Array.isArray(v.props.style) &&
      v.props.style.some((s: { flexDirection?: string }) => s && s.flexDirection === "column"));
    expect(layout).toBeTruthy();
  });

  it("reste en row quand shouldStackRows est false", () => {
    useAccessibilityScale.mockReturnValue({ shouldStackRows: false });
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(
        createElement(ModalFooterActions, {
          primary: createElement(View, { testID: "pri" }),
        })
      );
    });
    const rows = tree!.root.findAllByType(View);
    const layout = rows.find((v) => v.props.style && Array.isArray(v.props.style) &&
      v.props.style.some((s: { flexDirection?: string }) => s && s.flexDirection === "row"));
    expect(layout).toBeTruthy();
  });
});
