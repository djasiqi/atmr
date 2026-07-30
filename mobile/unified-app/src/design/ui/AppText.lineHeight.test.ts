import { afterEach, describe, expect, it } from "@jest/globals";
import { Platform } from "react-native";
import { CONTENT_FONT_CAP } from "../responsive/fontScaleCaps";
import { scaleLineHeightForFontScale } from "./AppText";

const originalOs = Platform.OS;

function setPlatformOs(os: typeof Platform.OS): void {
  Object.defineProperty(Platform, "OS", {
    configurable: true,
    get: () => os,
  });
}

describe("scaleLineHeightForFontScale", () => {
  afterEach(() => {
    setPlatformOs(originalOs);
  });

  it("ne modifie rien hors Android", () => {
    setPlatformOs("ios");
    expect(
      scaleLineHeightForFontScale({ fontSize: 16, lineHeight: 20 }, 2, CONTENT_FONT_CAP)
    ).toBeNull();
  });

  it("sur Android, multiplie lineHeight par le fontScale plafonné", () => {
    setPlatformOs("android");
    expect(
      scaleLineHeightForFontScale({ fontSize: 16, lineHeight: 20 }, 2, CONTENT_FONT_CAP)
    ).toEqual({ lineHeight: 40 });
    expect(
      scaleLineHeightForFontScale({ fontSize: 16, lineHeight: 20 }, 2.5, 1.3)
    ).toEqual({ lineHeight: 26 });
    expect(
      scaleLineHeightForFontScale({ fontSize: 16, lineHeight: 20 }, 1, CONTENT_FONT_CAP)
    ).toBeNull();
  });
});
