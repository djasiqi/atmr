import { describe, expect, it } from "@jest/globals";
import { companyContextScope, companyQueryKeys } from "./companyQueryKeys";

describe("company query keys", () => {
  it("embeds explicit context scope metadata", () => {
    expect(companyContextScope("company:42")).toEqual({
      context_type: "company",
      context_id: "company:42",
      company_id: "42",
    });
  });

  it("generates scoped query keys with context scope object", () => {
    const missionsKey = companyQueryKeys.missions("company:42", "2026-01-01");
    expect(missionsKey).toEqual(
      expect.arrayContaining([
        "company",
        "dispatch",
        "missions",
        expect.objectContaining({ context_id: "company:42", context_type: "company" }),
        "2026-01-01",
      ])
    );
    expect(missionsKey).toHaveLength(5);
  });
});
