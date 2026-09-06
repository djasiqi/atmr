import { describe, expect, it } from "@jest/globals";
import {
  FLOATING_TAB_BAR_OVERLAY_STYLE,
  FLOATING_TAB_SAFE_AREA_NONE,
} from "./floatingTabScreenOptions";

describe("floatingTabScreenOptions overlay", () => {
  it("place la barre en overlay sans fond opaque", () => {
    expect(FLOATING_TAB_BAR_OVERLAY_STYLE.position).toBe("absolute");
    expect(FLOATING_TAB_BAR_OVERLAY_STYLE.backgroundColor).toBe("transparent");
    expect(FLOATING_TAB_BAR_OVERLAY_STYLE.borderTopWidth).toBe(0);
  });

  it("n’absorbe pas le safe-area inférieur au niveau de la scène", () => {
    expect(FLOATING_TAB_SAFE_AREA_NONE.bottom).toBe(0);
    expect(FLOATING_TAB_SAFE_AREA_NONE.top).toBe(0);
  });
});
