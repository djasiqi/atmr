import { describe, expect, it } from "@jest/globals";
import { resolveCockpitConnectivityBanner } from "./resolveCockpitBanner";

describe("resolveCockpitConnectivityBanner", () => {
  it("0/T hors socket = un seul bloc temps réel", () => {
    expect(
      resolveCockpitConnectivityBanner({
        showNoGps: true,
        socketConnected: false,
        realtimeOffline: true,
      })
    ).toEqual({
      title: "Temps réel indisponible",
      body: "Données issues du dernier chargement. Vérifiez la connexion réseau.",
    });
  });

  it("0/T socket sain = Aucun GPS récent (sémantique inchangée)", () => {
    expect(
      resolveCockpitConnectivityBanner({
        showNoGps: true,
        socketConnected: true,
        realtimeOffline: false,
      })?.title
    ).toBe("Aucun GPS récent");
  });

  it("flotte vide + hors ligne = bandeau réseau", () => {
    expect(
      resolveCockpitConnectivityBanner({
        showNoGps: false,
        socketConnected: false,
        realtimeOffline: true,
      })?.title
    ).toBe("Temps réel indisponible");
  });

  it("LIVE avec positions = pas de bandeau", () => {
    expect(
      resolveCockpitConnectivityBanner({
        showNoGps: false,
        socketConnected: true,
        realtimeOffline: false,
      })
    ).toBeNull();
  });
});
