import type { CompanyDriverLiveLocation } from "../api/contracts";
import {
  driverFleetMarkerInitials,
  resolveDriverDisplayName,
  resolveFleetMarkerInitialsFromDisplayName,
} from "./companyDriverMapStatus";

const baseDriver = (overrides?: Partial<CompanyDriverLiveLocation>): CompanyDriverLiveLocation => ({
  driver_id: 7,
  latitude: 46.2,
  longitude: 6.14,
  timestamp: new Date().toISOString(),
  ...overrides,
});

describe("resolveDriverDisplayName", () => {
  it("utilise le nom d'entreprise si aucune identité chauffeur", () => {
    expect(
      resolveDriverDisplayName(baseDriver(), { organizationName: "Emmenez-moi" })
    ).toBe("Emmenez-moi");
  });

  it("préfère le nom personnel au nom d'entreprise", () => {
    expect(
      resolveDriverDisplayName(baseDriver({ driver_name: "Giuseppe Rossi" }), {
        organizationName: "Emmenez-moi",
      })
    ).toBe("Giuseppe Rossi");
  });

  it("utilise le nom mission avant l'entreprise", () => {
    expect(
      resolveDriverDisplayName(baseDriver(), {
        missionDriverName: "Dris K.",
        organizationName: "Emmenez-moi",
      })
    ).toBe("Dris K.");
  });

  it("fusionne prénom et nom du snapshot", () => {
    expect(
      resolveDriverDisplayName(
        baseDriver({ first_name: "Marie", last_name: "Dupont" })
      )
    ).toBe("Marie Dupont");
  });
});

describe("driverFleetMarkerInitials", () => {
  it("utilise prénom + nom quand disponibles (parité web)", () => {
    expect(
      driverFleetMarkerInitials(
        baseDriver({
          driver_name: "Khamsa",
          first_name: "Khamsa",
          last_name: "Ali",
        })
      )
    ).toBe("KA");
  });

  it("utilise full_name multi-mots si prénom/nom incomplets", () => {
    expect(
      driverFleetMarkerInitials(
        baseDriver({
          driver_name: "Khamsa",
          full_name: "Khamsa Ali",
        })
      )
    ).toBe("KA");
  });

  it("dérive depuis le nom affiché si prénom/nom absents", () => {
    expect(resolveFleetMarkerInitialsFromDisplayName("Jean Dupont")).toBe("JD");
  });
});
