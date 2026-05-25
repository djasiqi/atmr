import { describe, expect, it } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import {
  applyContextCachePolicyOnSwitch,
  clearAllContextCache,
  clearContextScopedCache,
  contextScopedKey,
  hasRestorableContextCache,
  parkContextCache,
} from "./contextCache";

describe("context cache policy", () => {
  it("creates context-scoped keys", () => {
    expect(contextScopedKey("client:self", ["bookings"])).toEqual(["ctx", "client:self", "bookings"]);
  });

  it("clears only targeted context cache", () => {
    const queryClient = new QueryClient();
    queryClient.setQueryData(["ctx", "client:self", "bookings"], [1]);
    queryClient.setQueryData(["ctx", "driver:42", "missions"], [2]);

    clearContextScopedCache(queryClient, "client:self");

    expect(queryClient.getQueryData(["ctx", "client:self", "bookings"])).toBeUndefined();
    expect(queryClient.getQueryData(["ctx", "driver:42", "missions"])).toEqual([2]);
    queryClient.clear();
  });

  it("parks context cache without removing queries", () => {
    const queryClient = new QueryClient();
    queryClient.setQueryData(["ctx", "company:1", "dashboard"], { ok: true });

    parkContextCache(queryClient, "company:1");

    expect(queryClient.getQueryData(["ctx", "company:1", "dashboard"])).toEqual({ ok: true });
    expect(hasRestorableContextCache(queryClient, "company:1")).toBe(true);
    queryClient.clear();
  });

  it("applyContextCachePolicyOnSwitch parks when flag enabled", () => {
    const queryClient = new QueryClient();
    queryClient.setQueryData(["ctx", "driver:42", "missions"], [1]);
    applyContextCachePolicyOnSwitch(queryClient, "driver:42");
    expect(queryClient.getQueryData(["ctx", "driver:42", "missions"])).toEqual([1]);
    queryClient.clear();
  });

  it("evicts oldest parked context when LRU exceeds 5", () => {
    const queryClient = new QueryClient();
    for (let i = 1; i <= 6; i += 1) {
      queryClient.setQueryData(["ctx", `company:${i}`, "dashboard"], { v: i });
      parkContextCache(queryClient, `company:${i}`);
    }

    expect(queryClient.getQueryData(["ctx", "company:1", "dashboard"])).toBeUndefined();
    expect(queryClient.getQueryData(["ctx", "company:6", "dashboard"])).toEqual({ v: 6 });
    queryClient.clear();
  });

  it("clears all context cache on logout", () => {
    const queryClient = new QueryClient();
    queryClient.setQueryData(["ctx", "client:self", "bookings"], [1]);
    queryClient.setQueryData(["ctx", "driver:42", "missions"], [2]);

    clearAllContextCache(queryClient);

    expect(queryClient.getQueryData(["ctx", "client:self", "bookings"])).toBeUndefined();
    expect(queryClient.getQueryData(["ctx", "driver:42", "missions"])).toBeUndefined();
    queryClient.clear();
  });
});
