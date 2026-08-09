import { afterEach, describe, expect, it, jest } from "@jest/globals";

describe("feature flags registry", () => {
  const originalSocket = process.env.EXPO_PUBLIC_ENABLE_DRIVER_SOCKET;
  const originalBg = process.env.EXPO_PUBLIC_ENABLE_BG_LOCATION;
  const originalPush = process.env.EXPO_PUBLIC_ENABLE_DRIVER_PUSH;
  const originalFcmNative = process.env.EXPO_PUBLIC_ENABLE_DRIVER_FCM_NATIVE;
  const originalCompanyRealtime = process.env.EXPO_PUBLIC_ENABLE_COMPANY_REALTIME;
  const originalCompanyDispatch = process.env.EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH;
  const originalCompanyDispatchScreen = process.env.EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH_SCREEN;

  afterEach(() => {
    process.env.EXPO_PUBLIC_ENABLE_DRIVER_SOCKET = originalSocket;
    process.env.EXPO_PUBLIC_ENABLE_BG_LOCATION = originalBg;
    process.env.EXPO_PUBLIC_ENABLE_DRIVER_PUSH = originalPush;
    process.env.EXPO_PUBLIC_ENABLE_DRIVER_FCM_NATIVE = originalFcmNative;
    process.env.EXPO_PUBLIC_ENABLE_COMPANY_REALTIME = originalCompanyRealtime;
    process.env.EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH = originalCompanyDispatch;
    process.env.EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH_SCREEN = originalCompanyDispatchScreen;
    jest.resetModules();
  });

  it("resolves env-backed flags and keeps company dispatch disabled", () => {
    process.env.EXPO_PUBLIC_ENABLE_DRIVER_SOCKET = "1";
    process.env.EXPO_PUBLIC_ENABLE_BG_LOCATION = "0";
    process.env.EXPO_PUBLIC_ENABLE_DRIVER_PUSH = "1";
    process.env.EXPO_PUBLIC_ENABLE_DRIVER_FCM_NATIVE = "1";
    process.env.EXPO_PUBLIC_ENABLE_COMPANY_REALTIME = "0";
    process.env.EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH = "0";
    process.env.EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH_SCREEN = "0";
    let registry: typeof import("./registry");
    jest.isolateModules(() => {
       
      registry = require("./registry");
    });

    expect(registry!.isFeatureEnabled("realtime_socket_enabled")).toBe(true);
    expect(registry!.isFeatureEnabled("tracking_background_enabled")).toBe(false);
    expect(registry!.isFeatureEnabled("driver_push_enabled")).toBe(true);
    expect(registry!.isFeatureEnabled("driver_fcm_native_enabled")).toBe(true);
    expect(registry!.isFeatureEnabled("company_dispatch_enabled")).toBe(false);
    expect(registry!.isFeatureEnabled("company_realtime_enabled")).toBe(false);
    expect(registry!.isFeatureEnabled("company_dispatch_screen_enabled")).toBe(false);
    expect(registry!.getFeatureFlagSource("driver_unified_enabled")).toBe("external");
  });

  it("enables company dispatch when EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH=1", () => {
    process.env.EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH = "1";
    let registry: typeof import("./registry");
    jest.isolateModules(() => {
       
      registry = require("./registry");
    });
    expect(registry!.isFeatureEnabled("company_dispatch_enabled")).toBe(true);
  });

  it("enables company realtime in __DEV__ when dispatch is enabled", () => {
    process.env.EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH = "1";
    delete process.env.EXPO_PUBLIC_ENABLE_COMPANY_REALTIME;
    let registry: typeof import("./registry");
    jest.isolateModules(() => {
       
      registry = require("./registry");
    });
    expect(registry!.isFeatureEnabled("company_realtime_enabled")).toBe(true);
    expect(registry!.isCompanyRealtimeSocketExpected()).toBe(true);
  });

  it("enables company dispatch screen when EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH_SCREEN=1", () => {
    process.env.EXPO_PUBLIC_ENABLE_COMPANY_DISPATCH_SCREEN = "1";
    let registry: typeof import("./registry");
    jest.isolateModules(() => {
       
      registry = require("./registry");
    });
    expect(registry!.isFeatureEnabled("company_dispatch_screen_enabled")).toBe(true);
  });

  it("enables driver_unified (espace chauffeur) by default in unified app", () => {
    let registry: typeof import("./registry");
    jest.isolateModules(() => {
       
      registry = require("./registry");
    });
    expect(registry!.isFeatureEnabled("driver_unified_enabled")).toBe(true);
  });
});
