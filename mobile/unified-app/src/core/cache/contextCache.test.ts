import { describe, expect, it } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import { clearAllContextCache, clearContextScopedCache, contextScopedKey } from "./contextCache";

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
