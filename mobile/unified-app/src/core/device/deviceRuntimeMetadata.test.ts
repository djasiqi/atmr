jest.mock("react-native", () => ({
  Platform: { OS: "ios" },
}));

jest.mock("expo-device", () => ({
  manufacturer: "Apple",
  modelName: "iPhone 15 Pro",
  deviceName: "iPhone de Drin",
  osVersion: "18.6",
  deviceType: 1,
  DeviceType: { UNKNOWN: 0, PHONE: 1, TABLET: 2, DESKTOP: 3, TV: 4 },
}));

jest.mock("expo-application", () => ({
  nativeApplicationVersion: "1.0.11",
  nativeBuildVersion: "69",
  applicationName: "Lirie",
}));

jest.mock("expo-updates", () => ({
  updateId: null,
  runtimeVersion: "1.0.11",
  channel: null,
  isEmbeddedLaunch: true,
}));

describe("deviceRuntimeMetadata", () => {
  it("résout deviceName humain et ignore applicationName", () => {
     
    const {
      resolveDeviceRuntimeMetadata,
      resolveDeviceHumanName,
      resolveDeviceDisplayName,
      buildDeviceMetadataHeaders,
    } = require("./deviceRuntimeMetadata") as typeof import("./deviceRuntimeMetadata");

    const meta = resolveDeviceRuntimeMetadata();
    expect(meta.deviceName).toBe("iPhone de Drin");
    expect(meta.model).toBe("iPhone 15 Pro");
    expect(meta.manufacturer).toBe("Apple");
    expect(meta.platform).toBe("ios");
    expect(meta.osVersion).toBe("18.6");
    expect(meta.appVersion).toBe("1.0.11");
    expect(meta.appBuild).toBe("69");
    expect(meta.deviceType).toBe("phone");

    expect(resolveDeviceHumanName(meta)).toBe("iPhone de Drin");
    expect(resolveDeviceDisplayName(meta)).toBe("iPhone de Drin");
    expect(resolveDeviceDisplayName(meta)).not.toBe("Lirie");

    const headers = buildDeviceMetadataHeaders(meta);
    expect(headers["X-Device-Name"]).toBe("iPhone de Drin");
    expect(headers["X-Client-Platform"]).toBe("ios");
    expect(headers["X-Platform"]).toBe("ios");
    expect(headers["X-Device-Model"]).toBe("iPhone 15 Pro");
    expect(headers["X-App-Version"]).toBe("1.0.11");
    expect(headers["X-App-Build"]).toBe("69");
    expect(headers["X-OS-Version"]).toBe("18.6");
  });

  it("resolveDeviceHumanName retombe à null sans deviceName OS, display utilise le modèle", () => {
     
    const {
      resolveDeviceHumanName,
      resolveDeviceDisplayName,
    } = require("./deviceRuntimeMetadata") as typeof import("./deviceRuntimeMetadata");

    const metaWithoutName = {
      platform: "ios",
      deviceName: null,
      manufacturer: "Apple",
      model: "iPhone 15 Pro",
      deviceType: "phone" as const,
      osVersion: "18.6",
      appVersion: "1.0.11",
      appBuild: "69",
      expoRuntimeVersion: null,
      otaUpdateId: null,
      releaseChannel: null,
      releaseSha: null,
    };

    expect(resolveDeviceHumanName(metaWithoutName)).toBeNull();
    expect(resolveDeviceDisplayName(metaWithoutName)).toBe("iPhone 15 Pro");

    const appLabelAsName = { ...metaWithoutName, deviceName: "Lirie" };
    expect(resolveDeviceHumanName(appLabelAsName)).toBeNull();
    expect(resolveDeviceDisplayName(appLabelAsName)).toBe("iPhone 15 Pro");
  });
});
