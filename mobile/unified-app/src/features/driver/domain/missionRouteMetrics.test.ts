import { describe, expect, it } from "@jest/globals";
import { resolveMissionRouteLeg } from "./missionRouteMetrics";

describe("resolveMissionRouteLeg", () => {
  it("retourne planned pour ASSIGNED", () => {
    expect(resolveMissionRouteLeg("ASSIGNED")).toEqual({ mode: "planned" });
  });

  it("retourne live pickup pour EN_ROUTE", () => {
    expect(resolveMissionRouteLeg("EN_ROUTE")).toEqual({
      mode: "live",
      destination: "pickup",
    });
  });

  it("retourne live dropoff pour ARRIVED", () => {
    expect(resolveMissionRouteLeg("ARRIVED")).toEqual({
      mode: "live",
      destination: "dropoff",
    });
    expect(resolveMissionRouteLeg("arrived")).toEqual({
      mode: "live",
      destination: "dropoff",
    });
  });

  it("retourne live dropoff pour IN_PROGRESS", () => {
    expect(resolveMissionRouteLeg("IN_PROGRESS")).toEqual({
      mode: "live",
      destination: "dropoff",
    });
  });
});
