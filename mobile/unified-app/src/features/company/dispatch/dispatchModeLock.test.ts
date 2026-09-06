import { describe, expect, it } from "@jest/globals";
import {
  DispatchFeatureDisabledError,
  FULLY_AUTO_DISPATCH_ENABLED,
  OPTIMIZER_ENABLED,
  SEMI_AUTO_DISPATCH_ENABLED,
  assertDispatchModeSwitchAllowed,
  assertFullyAutoDispatchEnabled,
  assertOptimizerEnabled,
  assertSemiAutoDispatchEnabled,
  shouldMountDispatchEngine,
} from "./dispatchModeLock";

describe("dispatchModeLock", () => {
  it("verrouille semi-auto, optimiseur et auto (DEV et PROD)", () => {
    expect(SEMI_AUTO_DISPATCH_ENABLED).toBe(false);
    expect(OPTIMIZER_ENABLED).toBe(false);
    expect(FULLY_AUTO_DISPATCH_ENABLED).toBe(false);
    expect(shouldMountDispatchEngine()).toBe(false);
  });

  it("autorise uniquement le bascule vers manuel", () => {
    expect(() => assertDispatchModeSwitchAllowed("manual")).not.toThrow();
    expect(() => assertDispatchModeSwitchAllowed("semi_auto")).toThrow(DispatchFeatureDisabledError);
    expect(() => assertDispatchModeSwitchAllowed("fully_auto")).toThrow(DispatchFeatureDisabledError);
  });

  it("abort les assertions moteur sans I/O", () => {
    expect(() => assertSemiAutoDispatchEnabled()).toThrow(DispatchFeatureDisabledError);
    expect(() => assertOptimizerEnabled()).toThrow(DispatchFeatureDisabledError);
    expect(() => assertFullyAutoDispatchEnabled()).toThrow(DispatchFeatureDisabledError);
  });
});
