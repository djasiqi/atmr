import { describe, expect, it } from "@jest/globals";
import { createCaptureId } from "./captureId";

describe("createCaptureId", () => {
  it("génère un identifiant non vide distinct du suivant", () => {
    const a = createCaptureId();
    const b = createCaptureId();
    expect(a).toBeTruthy();
    expect(b).toBeTruthy();
    expect(a).not.toBe(b);
  });
});
