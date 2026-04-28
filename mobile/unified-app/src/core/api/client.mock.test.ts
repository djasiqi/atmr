import { describe, expect, it } from "@jest/globals";
import { bootstrapResponseSchema, switchContextResponseSchema } from "../contracts/auth";
import { buildMockBootstrap, buildMockSwitchContext } from "./mockData";

describe("api client mock bootstrap mode", () => {
  it("produces valid mock bootstrap payload", () => {
    const bootstrap = bootstrapResponseSchema.parse(buildMockBootstrap());
    expect(bootstrap.is_authenticated).toBe(true);
    expect(bootstrap.available_contexts.length).toBeGreaterThan(0);
  });

  it("produces valid mock switch-context payload", () => {
    const switched = switchContextResponseSchema.parse(buildMockSwitchContext("driver:42"));
    expect(switched.success).toBe(true);
    expect(switched.active_context_id).toBe("driver:42");
  });
});
