import { describe, expect, it } from "@jest/globals";
import { getDropoffHints, getPickupHints } from "./missionHints";

describe("missionHints", () => {
  it("met en avant les accès domicile au pickup d'un aller", () => {
    const hints = getPickupHints({
      is_return: false,
      pickup_door_code: "7421",
      pickup_floor: "3e",
      pickup_access_notes: "Entrée côté cour",
      client: {
        door_code: "legacy",
        floor: "legacy",
        access_notes: "legacy",
        contact_phone: "+41790000000",
      },
    });

    expect(hints.map((hint) => [hint.label, hint.value])).toEqual([
      ["Code porte", "7421"],
      ["Étage", "3e"],
      ["Notes d'accès", "Entrée côté cour"],
      ["Contact", "+41790000000"],
    ]);
  });

  it("inverse les informations pour le dropoff domicile d'un retour", () => {
    const hints = getDropoffHints({
      is_return: true,
      dropoff_door_code: "8855",
      dropoff_floor: "Rez",
      dropoff_access_notes: "Rampe par le parking",
      client: {
        door_code: "legacy",
        floor: "legacy",
        access_notes: "legacy",
        contact_phone: "+41791111111",
      },
    });

    expect(hints.map((hint) => [hint.label, hint.value])).toEqual([
      ["Code porte", "8855"],
      ["Étage", "Rez"],
      ["Notes d'accès", "Rampe par le parking"],
      ["Contact", "+41791111111"],
    ]);
  });

  it("affiche les informations HUG pour la destination d'un aller", () => {
    const hints = getDropoffHints({
      is_return: false,
      medical_facility: "HUG",
      hospital_service: "Ophtalmologie - secteur B",
      doctor_name: "Dr Martin",
      dropoff_floor: "4e",
      dropoff_access_notes: "Accueil ambulatoire",
    });

    expect(hints.map((hint) => [hint.label, hint.value])).toEqual([
      ["Établissement", "HUG"],
      ["Service / Bâtiment", "Ophtalmologie - secteur B"],
      ["Étage / Secteur", "4e"],
      ["Médecin", "Dr Martin"],
      ["Instructions accès", "Accueil ambulatoire"],
    ]);
  });
});
