import {
  FORM_CHECKBOX_ROW_MIN_HEIGHT,
  FORM_CHECKBOX_SIZE,
  FORM_CONTROL_HEIGHT,
  FORM_CONTROL_RADIUS,
  FORM_SEGMENT_HEIGHT,
  FORM_TEXTAREA_MIN_HEIGHT,
} from "./formControls";

describe("formControls", () => {
  it("aligne input, select et date sur le gabarit compact de la réservation", () => {
    expect(FORM_CONTROL_HEIGHT).toBe(36);
    expect(FORM_CONTROL_RADIUS).toBe(12);
  });

  it("garde une case petite et une ligne tactile", () => {
    expect(FORM_CHECKBOX_SIZE).toBeGreaterThanOrEqual(20);
    expect(FORM_CHECKBOX_SIZE).toBeLessThanOrEqual(22);
    expect(FORM_CHECKBOX_ROW_MIN_HEIGHT).toBeGreaterThanOrEqual(44);
    expect(FORM_CHECKBOX_ROW_MIN_HEIGHT).toBeGreaterThan(FORM_CHECKBOX_SIZE);
  });

  it("borne le segment et la zone de texte", () => {
    expect(FORM_SEGMENT_HEIGHT).toBe(32);
    expect(FORM_TEXTAREA_MIN_HEIGHT).toBeGreaterThanOrEqual(76);
    expect(FORM_TEXTAREA_MIN_HEIGHT).toBeLessThanOrEqual(88);
  });
});
