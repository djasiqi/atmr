import { describe, expect, it } from "@jest/globals";
import { __parseAddressOptionsForTests, __parseClientDetailForTests } from "./useRideForms";

describe("useRideForms client detail hospitalization parsing", () => {
  it("scénario 1: client domicile sans hospitalisation", () => {
    const detail = __parseClientDetailForTests({
      id: 1,
      home_address: "Rue du Test 1, Genève",
      phone: "+41790000000",
    });
    expect(detail?.hasActiveStay).toBe(false);
    expect(detail?.pickupAddressCandidate?.label).toContain("Rue du Test 1");
    expect(detail?.clinicBillingPartyId).toBeNull();
  });

  it("utilise domicile.* comme pickup candidate quand présent", () => {
    const detail = __parseClientDetailForTests({
      id: 10,
      domicile: {
        address: "Route de Veigy 515",
        zip: "74140",
        city: "Loisin",
        lat: 46.31,
        lon: 6.27,
      },
    });
    expect(detail?.pickupAddressCandidate?.label).toContain("Route de Veigy 515");
    expect(detail?.pickupAddressCandidate?.label).toContain("74140");
    expect(detail?.pickupAddressCandidate?.latitude).toBe(46.31);
    expect(detail?.pickupAddressCandidate?.longitude).toBe(6.27);
  });

  it("assemble domicile_address_line1 + domicile_zip + domicile_city", () => {
    const detail = __parseClientDetailForTests({
      id: 11,
      domicile_address_line1: "Chemin des Fleurs 8",
      domicile_zip: "1201",
      domicile_city: "Genève",
      domicile_lat: 46.205,
      domicile_lon: 6.145,
    });
    expect(detail?.pickupAddressCandidate?.label).toBe("Chemin des Fleurs 8, 1201 Genève");
    expect(detail?.pickupAddressCandidate?.latitude).toBe(46.205);
    expect(detail?.pickupAddressCandidate?.longitude).toBe(6.145);
  });

  it("fallback sur billing_address puis user.address", () => {
    const detailBilling = __parseClientDetailForTests({
      id: 12,
      billing_address: "Avenue de France 10, Genève",
      billing_lat: 46.22,
      billing_lon: 6.13,
    });
    expect(detailBilling?.pickupAddressCandidate?.label).toContain("Avenue de France 10");
    expect(detailBilling?.pickupAddressCandidate?.latitude).toBe(46.22);
    expect(detailBilling?.pickupAddressCandidate?.longitude).toBe(6.13);

    const detailUser = __parseClientDetailForTests({
      id: 13,
      user: { address: "Rue des Alpes 4, Lausanne" },
    });
    expect(detailUser?.pickupAddressCandidate?.label).toContain("Rue des Alpes 4");
  });

  it("fallback profond sur format adresse imbriqué non standard", () => {
    const detail = __parseClientDetailForTests({
      id: 14,
      profile: {
        contact: {
          location_label: "Chemin du Signal 12, 1000 Lausanne",
        },
      },
    });
    expect(detail?.pickupAddressCandidate?.label).toContain("Chemin du Signal 12");
  });

  it("scénario 2: client hospitalisé avec clinique payeuse", () => {
    const detail = __parseClientDetailForTests({
      id: 2,
      active_stay: {
        hospital_service: "Cardiologie",
        room: "203",
        floor: "2",
        billing_party_id: 99,
        clinic: {
          id: 10,
          name: "Clinique Arcades",
          domicile_address_line1: "Avenue du Lac 10",
          domicile_zip: "1200",
          domicile_city: "Genève",
          latitude: 46.2,
          longitude: 6.1,
        },
      },
    });
    expect(detail?.hasActiveStay).toBe(true);
    expect(detail?.clinicName).toBe("Clinique Arcades");
    expect(detail?.clinicAddress?.label).toContain("Avenue du Lac 10");
    expect(detail?.clinicService).toBe("Cardiologie");
    expect(detail?.clinicRoom).toBe("203");
    expect(detail?.clinicBillingPartyId).toBe(99);
  });

  it("scénario 3: client hospitalisé avec override possible patient", () => {
    const detail = __parseClientDetailForTests({
      id: 3,
      stay: {
        clinic: {
          name: "HUG",
          address: "Rue Gabrielle-Perret-Gentil 4",
        },
      },
    });
    expect(detail?.hasActiveStay).toBe(true);
    expect(detail?.clinicAddress?.label).toContain("Rue Gabrielle-Perret-Gentil 4");
    expect(detail?.clinicBillingPartyId).toBeNull();
  });
});

describe("useRideForms address autocomplete parsing", () => {
  it("parse un payload Photon features[]", () => {
    const options = __parseAddressOptionsForTests({
      features: [
        {
          geometry: { coordinates: [6.145, 46.205] },
          properties: {
            osm_id: "node-123",
            label: "Rue de Lausanne 1, 1201 Genève",
          },
        },
      ],
    });
    expect(options).toHaveLength(1);
    expect(options[0]?.label).toContain("Rue de Lausanne 1");
    expect(options[0]?.latitude).toBe(46.205);
    expect(options[0]?.longitude).toBe(6.145);
  });

  it("parse un payload predictions[] avec place_id string", () => {
    const options = __parseAddressOptionsForTests({
      predictions: [
        {
          place_id: "abc-google-place-id",
          description: "HUG, Genève",
          lat: 46.21,
          lon: 6.15,
        },
      ],
    });
    expect(options).toHaveLength(1);
    expect(options[0]?.label).toBe("HUG, Genève");
    expect(options[0]?.placeId).toBe("abc-google-place-id");
  });
});
