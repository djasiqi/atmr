import { contrastRatio, lirieColor, lirieStatusLabel } from "./tokens";

describe("jetons Signature LIRIE", () => {
  it("sépare la marque des statuts", () => {
    expect(lirieColor.green).toBe("#00796B");
    expect(lirieColor.ink).toBe("#263532");
    expect(lirieColor.canvas).toBe("#F5F7F6");
    expect(lirieColor.white).toBe("#FFFFFF");
    expect(lirieStatusLabel.assigned).toBe("Assignée");
    expect(lirieStatusLabel.cancelled).toBe("Annulée");
  });

  it("garde un contraste AA pour le texte et le bouton", () => {
    expect(contrastRatio(lirieColor.ink, lirieColor.white)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lirieColor.ink, lirieColor.canvas)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lirieColor.inkMuted, lirieColor.white)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lirieColor.onBrand, lirieColor.green)).toBeGreaterThanOrEqual(3);
    expect(contrastRatio(lirieColor.ink, lirieColor.statusFill)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lirieColor.disabledInk, lirieColor.disabledFill)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(lirieColor.danger, lirieColor.dangerFill)).toBeGreaterThanOrEqual(4.5);
  });
});
