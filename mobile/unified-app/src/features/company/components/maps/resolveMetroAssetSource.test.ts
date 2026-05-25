import { resolveMetroAssetSource } from "./resolveMetroAssetSource";
import { resolveLirieDriverMarkerModule } from "./fleetLirieDriverMarkerModules";

describe("resolveMetroAssetSource", () => {
  it("resout un PNG embarque via expo-asset Asset.fromModule", () => {
    const moduleId = resolveLirieDriverMarkerModule("available");
    const resolved = resolveMetroAssetSource(moduleId);
    expect(resolved.uri.length).toBeGreaterThan(0);
  });
});