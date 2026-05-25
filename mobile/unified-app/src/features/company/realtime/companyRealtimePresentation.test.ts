import { describe, expect, it } from "@jest/globals";
import { formatCompanyActivityHint } from "./companyRealtimePresentation";

describe("formatCompanyActivityHint", () => {
  it("returns null when data is fresh", () => {
    expect(formatCompanyActivityHint(new Date().toISOString(), "fresh")).toBeNull();
  });

  it("formats idle and stale hints", () => {
    const twoMinAgo = new Date(Date.now() - 2 * 60_000).toISOString();
    expect(formatCompanyActivityHint(twoMinAgo, "idle")).toContain("2 min");
    expect(formatCompanyActivityHint(twoMinAgo, "stale")).toContain("anciennes");
  });
});
