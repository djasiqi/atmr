import {
  formatDeliveryDescriptionDisplay,
  formatMissionTypeLabel,
  getDeliveryDescription,
  isMaterialDeliveryMission,
  MISSING_DELIVERY_DESCRIPTION,
  MISSION_DELIVERY_BADGE,
} from "./missionDisplay";
import type { DriverMission } from "../types";

function mission(partial: Partial<DriverMission> & Record<string, unknown>): DriverMission {
  return { id: 1, status: "ASSIGNED", ...partial } as DriverMission;
}

describe("missionDisplay livraison", () => {
  it("détecte une livraison matériel", () => {
    expect(isMaterialDeliveryMission(mission({ mission_type: "material_delivery" }))).toBe(true);
    expect(isMaterialDeliveryMission(mission({ mission_type: "patient_transport" }))).toBe(false);
    expect(isMaterialDeliveryMission(null)).toBe(false);
  });

  it("extrait la description ou indique l'absence", () => {
    expect(getDeliveryDescription(mission({ delivery_description: "  Oxygène  " }))).toBe("Oxygène");
    expect(getDeliveryDescription(mission({ mission_type: "material_delivery" }))).toBeNull();
  });

  it("libellé FR du type de mission", () => {
    expect(formatMissionTypeLabel("material_delivery")).toBe("Livraison");
    expect(MISSION_DELIVERY_BADGE).toBe("LIVRAISON");
    expect(formatMissionTypeLabel(null)).toBe("Transport patient");
  });

  it("affiche le contrat #39869 sans inventer une description", () => {
    const fixture = mission({
      id: 39869,
      mission_type: "material_delivery",
      delivery_description: "Livraison des effets personnels de M. Basset.",
    });
    expect(formatDeliveryDescriptionDisplay(fixture)).toBe(
      "Livraison des effets personnels de M. Basset."
    );
    expect(formatDeliveryDescriptionDisplay(mission({ mission_type: "material_delivery" }))).toBe(
      MISSING_DELIVERY_DESCRIPTION
    );
  });

  it("n’affiche pas une livraison si seule delivery_description est renseignée", () => {
    const accidental = mission({
      mission_type: "patient_transport",
      delivery_description: "Livraison de documents",
    });
    expect(isMaterialDeliveryMission(accidental)).toBe(false);
    expect(formatMissionTypeLabel("patient_transport")).toBe("Transport patient");
  });
});
