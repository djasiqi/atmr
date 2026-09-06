import { beforeEach, describe, expect, it, jest } from "@jest/globals";

import {
  buildFcmRegistrationKey,
  getFcmRegistrationInFlightCountForTests,
  getLastFcmRegistrationSuccessKeyForTests,
  hasSuccessfulFcmRegistrationForOwner,
  resetFcmRegistrationGuardForTests,
  runFcmRegistrationOnce,
  runFcmTokenAcquisitionOnce,
} from "./fcmRegistrationGuard";

describe("fcmRegistrationGuard (MOB-STARTUP-STORM-FIX-01)", () => {
  beforeEach(() => {
    resetFcmRegistrationGuardForTests();
  });

  it("FCM-01: concurrent callers share one in-flight registration", async () => {
    let resolveRegister: (() => void) | undefined;
    const register = jest.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveRegister = resolve;
        })
    );

    const first = runFcmRegistrationOnce({ ownerKey: "driver:1", token: "tok-a" }, register);
    const second = runFcmRegistrationOnce({ ownerKey: "driver:1", token: "tok-a" }, register);
    const third = runFcmRegistrationOnce({ ownerKey: "driver:1", token: "tok-a" }, register);

    expect(getFcmRegistrationInFlightCountForTests()).toBe(1);
    expect(register).toHaveBeenCalledTimes(1);

    resolveRegister?.();
    const outcomes = await Promise.all([first, second, third]);
    expect(outcomes).toEqual(["registered", "registered", "registered"]);
    expect(register).toHaveBeenCalledTimes(1);
  });

  it("FCM-02: same owner+token does not POST twice", async () => {
    const register = jest.fn(async () => undefined);
    const key = buildFcmRegistrationKey("driver:7", "same-token");

    await expect(
      runFcmRegistrationOnce({ ownerKey: "driver:7", token: "same-token" }, register)
    ).resolves.toBe("registered");
    await expect(
      runFcmRegistrationOnce({ ownerKey: "driver:7", token: "same-token" }, register)
    ).resolves.toBe("skipped");
    await expect(
      runFcmRegistrationOnce({ ownerKey: "driver:7", token: "same-token" }, register)
    ).resolves.toBe("skipped");

    expect(register).toHaveBeenCalledTimes(1);
    expect(getLastFcmRegistrationSuccessKeyForTests()).toBe(key);
  });

  it("FCM-03: failure does not tight-loop retry", async () => {
    const register = jest
      .fn<() => Promise<void>>()
      .mockRejectedValueOnce(new Error("network"))
      .mockResolvedValue(undefined);

    await expect(
      runFcmRegistrationOnce({ ownerKey: "driver:2", token: "tok-b" }, register)
    ).resolves.toBe("failed");
    await expect(
      runFcmRegistrationOnce({ ownerKey: "driver:2", token: "tok-b" }, register)
    ).resolves.toBe("skipped");

    expect(register).toHaveBeenCalledTimes(1);
  });

  it("DRIVER-RUNTIME-01B : getToken concurrent rejoint la même Promise", async () => {
    let resolveAcquire: ((value: boolean) => void) | undefined;
    const acquire = jest.fn(
      () =>
        new Promise<boolean>((resolve) => {
          resolveAcquire = resolve;
        })
    );
    const first = runFcmTokenAcquisitionOnce("driver:4", acquire);
    const second = runFcmTokenAcquisitionOnce("driver:4", acquire);
    expect(acquire).toHaveBeenCalledTimes(1);
    resolveAcquire?.(true);
    await expect(Promise.all([first, second])).resolves.toEqual([true, true]);
    expect(acquire).toHaveBeenCalledTimes(1);
  });

  it("DRIVER-RUNTIME-01 : owner déjà enregistré = skip getToken/POST", async () => {
    const register = jest.fn(async () => undefined);
    await runFcmRegistrationOnce({ ownerKey: "driver:9", token: "tok-z" }, register);
    expect(hasSuccessfulFcmRegistrationForOwner("driver:9")).toBe(true);
    expect(hasSuccessfulFcmRegistrationForOwner("driver:8")).toBe(false);
  });

  it("allows registration when token rotates for same owner", async () => {
    const register = jest.fn(async () => undefined);
    await runFcmRegistrationOnce({ ownerKey: "driver:3", token: "old" }, register);
    await runFcmRegistrationOnce({ ownerKey: "driver:3", token: "new" }, register);
    expect(register).toHaveBeenCalledTimes(2);
  });
});
