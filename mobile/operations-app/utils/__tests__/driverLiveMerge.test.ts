import {
  canonicalTimeMs,
  shouldIgnoreObservabilityRegression,
} from "../driverLiveMerge";

describe("canonicalTimeMs", () => {
  it("priorise received_at > recorded_at > timestamp", () => {
    const a = canonicalTimeMs({
      received_at: "2020-01-02T00:00:00.000Z",
      recorded_at: "2020-01-01T00:00:00.000Z",
      timestamp: "2020-01-03T00:00:00.000Z",
    });
    expect(a).toBe(Date.parse("2020-01-02T00:00:00.000Z"));
  });

  it("accepte ts", () => {
    const t = canonicalTimeMs({ ts: "2020-01-04T00:00:00.000Z" });
    expect(t).toBe(Date.parse("2020-01-04T00:00:00.000Z"));
  });
});

describe("shouldIgnoreObservabilityRegression", () => {
  it("ignore si observabilité et point plus vieux", () => {
    const ign = shouldIgnoreObservabilityRegression(
      {
        accept_status: "accepted_observability_only",
        recorded_at: "2020-01-01T00:00:00.000Z",
      },
      { recorded_at: "2020-01-02T00:00:00.000Z" }
    );
    expect(ign).toBe(true);
  });

  it("n'ignore pas si canonique", () => {
    const ign = shouldIgnoreObservabilityRegression(
      {
        accept_status: "accepted_canonical",
        recorded_at: "2020-01-01T00:00:00.000Z",
      },
      { recorded_at: "2020-01-02T00:00:00.000Z" }
    );
    expect(ign).toBe(false);
  });
});
