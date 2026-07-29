import { describe, expect, it, jest } from "@jest/globals";
import { createElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { Text } from "react-native";

jest.mock("./useResponsiveTokens", () => ({
  useResponsiveTokens: () => ({
    formButtonMinHeight: 48,
    buttonFontSize: 13,
    bodyLineHeightRatio: 1.3,
    spacingMd: 16,
    spacingSm: 8,
    radiusMd: 12,
  }),
}));

const { AppButton } = require("../ui/AppButton");

describe("AppButton wrapping", () => {
  it("autorise le wrap du label (pas de numberOfLines forcé)", () => {
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(
        createElement(AppButton, {
          title: "Confirmer la libération de la mission maintenant",
        })
      );
    });
    const text = tree!.root.findByType(Text);
    expect(text.props.numberOfLines).toBeUndefined();
    expect(text.props.style.flexShrink).toBe(1);
  });
});
