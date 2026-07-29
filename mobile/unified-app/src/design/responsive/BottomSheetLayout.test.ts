import { describe, expect, it } from "@jest/globals";
import { computeBottomSheetLayout } from "./BottomSheetLayout";

const tokens = {
  modalSheetMaxHeightRatio: 0.9,
  modalSheetMaxHeightCap: 720,
};

describe("computeBottomSheetLayout", () => {
  it("avec bottomInset > 0 : paddingBottom inclut l'inset, sans double-soustraction dans reservedChromeHeight", () => {
    const usableHeight = 800;
    const bottomInset = 34;
    const reservedChromeHeight = 120;
    const bottomPaddingExtra = 8;

    const layout = computeBottomSheetLayout(usableHeight, bottomInset, tokens, {
      reservedChromeHeight,
      bottomPaddingExtra,
    });

    const expectedPaddingBottom = Math.max(16, bottomInset + bottomPaddingExtra);
    expect(layout.paddingBottom).toBe(expectedPaddingBottom);
    expect(layout.paddingBottom).toBe(42);

    const expectedCardMaxHeight = Math.min(
      Math.round(usableHeight * tokens.modalSheetMaxHeightRatio),
      tokens.modalSheetMaxHeightCap
    );
    expect(layout.cardMaxHeight).toBe(expectedCardMaxHeight);

    // scrollMaxHeight = cardMaxHeight - reservedChromeHeight - paddingBottom
    // (safe area déjà dans paddingBottom, pas dans reservedChromeHeight)
    expect(layout.scrollMaxHeight).toBe(
      expectedCardMaxHeight - reservedChromeHeight - layout.paddingBottom
    );
    expect(layout.scrollMaxHeight).not.toBe(
      expectedCardMaxHeight - reservedChromeHeight - bottomInset - layout.paddingBottom
    );
  });
});
