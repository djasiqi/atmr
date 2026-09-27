import {
  driverAvatarTone,
  driverInitials,
  driverPickerSummary,
  filterDriverOptions,
  uniqueDriverInitials,
} from "./assignDriverPicker";

describe("assignDriverPicker", () => {
  it("construit les initiales d'un nom composé", () => {
    expect(driverInitials("Emmenez Moi")).toBe("EM");
    expect(driverInitials("  ")).toBe("?");
    expect(driverInitials("Ada")).toBe("AD");
  });

  it("distingue deux chauffeurs qui partageraient les mêmes initiales", () => {
    const initials = uniqueDriverInitials([
      { id: 1, label: "Léa Martin" },
      { id: 2, label: "Lucas Morel" },
      { id: 3, label: "Chloé Roux" },
    ]);
    expect(initials.get(1)).toBe("LMA");
    expect(initials.get(2)).toBe("LMO");
    expect(initials.get(3)).toBe("CR");
    expect(new Set(initials.values()).size).toBe(3);
  });

  it("attribue une teinte stable à chaque nom", () => {
    const tone = driverAvatarTone("Léa Martin");
    expect(driverAvatarTone("Léa Martin")).toEqual(tone);
    expect(tone.background.length).toBeGreaterThan(0);
    expect(tone.foreground.length).toBeGreaterThan(0);
  });

  it("filtre les chauffeurs sans tenir compte de la casse", () => {
    const drivers = [
      { id: 1, label: "Emmenez Moi" },
      { id: 2, label: "Ada Lovelace" },
    ];
    expect(filterDriverOptions(drivers, "  emmenez ")).toEqual([drivers[0]]);
    expect(filterDriverOptions(drivers, "")).toEqual(drivers);
  });

  it("résume la liste ou la recherche", () => {
    expect(driverPickerSummary(1, 1, "")).toBe("1 chauffeur disponible");
    expect(driverPickerSummary(3, 3, "")).toBe("3 chauffeurs disponibles");
    expect(driverPickerSummary(3, 1, "ada")).toBe("1 résultat");
    expect(driverPickerSummary(3, 0, "zzz")).toBe("0 résultats");
  });
});
