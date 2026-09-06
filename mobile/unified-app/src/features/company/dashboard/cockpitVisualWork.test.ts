import { describe, expect, it } from "@jest/globals";
import {
  resolveCockpitVisualWork,
  shouldFreezeCockpitMapData,
} from "./cockpitVisualWork";

describe("cockpitVisualWork", () => {
  it("active le travail visuel uniquement au focus", () => {
    expect(resolveCockpitVisualWork(true)).toEqual({
      visualWorkEnabled: true,
      shouldRecordScreenRender: true,
    });
    expect(resolveCockpitVisualWork(false)).toEqual({
      visualWorkEnabled: false,
      shouldRecordScreenRender: false,
    });
  });

  it("gèle les données carte seulement tant que les deux frames sont hors focus", () => {
    expect(shouldFreezeCockpitMapData(false, false)).toBe(true);
    expect(shouldFreezeCockpitMapData(false, true)).toBe(false);
    expect(shouldFreezeCockpitMapData(true, false)).toBe(false);
    expect(shouldFreezeCockpitMapData(true, true)).toBe(false);
  });
});
