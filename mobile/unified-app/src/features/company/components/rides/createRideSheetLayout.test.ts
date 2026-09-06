import { describe, expect, it } from "@jest/globals";
import {
  computeCreateRideAddressListMaxHeight,
  computeCreateRideResultsMaxHeight,
  computeCreateRideSheetLayout,
} from "./createRideSheetLayout";

describe("createRideSheetLayout", () => {
  it("plafonne sans forcer une grande hauteur (clavier fermé)", () => {
    const layout = computeCreateRideSheetLayout({
      windowHeight: 800,
      keyboardHeight: 0,
      resizedBySystem: false,
    });
    expect(layout.keyboardOpen).toBe(false);
    expect(layout.liftBottom).toBe(0);
    expect(layout.maxSheetHeight).toBe(Math.round(800 * 0.62));
    expect(layout.maxSheetHeight).toBeLessThan(800 * 0.7);
  });

  it("soulève le sheet sans l’étendre au clavier", () => {
    const layout = computeCreateRideSheetLayout({
      windowHeight: 800,
      keyboardHeight: 320,
      resizedBySystem: false,
    });
    expect(layout.keyboardOpen).toBe(true);
    expect(layout.liftBottom).toBe(320);
    expect(layout.availableHeight).toBe(480);
    expect(layout.maxSheetHeight).toBeLessThanOrEqual(472);
    expect(layout.maxSheetHeight).toBeGreaterThan(300);
    expect(layout.maxSheetHeight).toBeLessThan(layout.availableHeight);
  });

  it("n’ajoute pas de lift si adjustResize a déjà réduit la fenêtre", () => {
    const layout = computeCreateRideSheetLayout({
      windowHeight: 480,
      keyboardHeight: 320,
      resizedBySystem: true,
    });
    expect(layout.liftBottom).toBe(0);
    expect(layout.maxSheetHeight).toBeLessThanOrEqual(472);
    expect(layout.maxSheetHeight).toBeLessThan(layout.availableHeight);
  });

  it("borne la liste de résultats après le chrome, sans flex vide", () => {
    expect(computeCreateRideResultsMaxHeight(400, 180, 52)).toBe(168);
    expect(computeCreateRideResultsMaxHeight(280, 200, 52)).toBe(72);
  });

  it("borne la liste d’adresses au-dessus du clavier", () => {
    expect(
      computeCreateRideAddressListMaxHeight({
        windowHeight: 800,
        keyboardHeight: 0,
        resizedBySystem: false,
      })
    ).toBe(230);
    const capped = computeCreateRideAddressListMaxHeight({
      windowHeight: 800,
      keyboardHeight: 360,
      resizedBySystem: false,
    });
    expect(capped).toBeLessThanOrEqual(230);
    expect(capped).toBeGreaterThanOrEqual(120);
  });
});
