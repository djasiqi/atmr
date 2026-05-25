import { buildTeamQuickReplies } from "./teamQuickReplies";
import type { DriverMission } from "../types";

function mission(overrides: Partial<DriverMission> = {}): DriverMission {
  return {
    id: 42,
    status: "assigned",
    client_name: "Catherine BRONNIMANN",
    client: { gender: "female", last_name: "Bronnimann" },
    ...overrides,
  };
}

describe("buildTeamQuickReplies", () => {
  it("sans mission active : suggestions génériques", () => {
    const items = buildTeamQuickReplies(null);
    expect(items.some((i) => i.content.includes("disponible"))).toBe(true);
    expect(items.some((i) => i.label.includes("Patient"))).toBe(false);
  });

  it("IN_PROGRESS : Madame NOM à bord et course terminée", () => {
    const items = buildTeamQuickReplies(mission({ status: "in_progress" }));
    expect(items[0]?.content).toBe("Madame BRONNIMANN à bord");
    expect(items[0]?.label).toContain("Madame BRONNIMANN");
    expect(items.some((i) => i.content.includes("course terminée"))).toBe(true);
  });

  it("EN_ROUTE : retard avec formule de civilité", () => {
    const items = buildTeamQuickReplies(mission({ status: "en_route" }));
    expect(items.some((i) => i.content === "Retard 5 min — Madame BRONNIMANN")).toBe(true);
    expect(items.some((i) => i.content.includes("J'arrive — Madame BRONNIMANN"))).toBe(true);
  });

  it("genre homme : Monsieur", () => {
    const items = buildTeamQuickReplies(
      mission({ status: "in_progress", client: { gender: "male", last_name: "Dupont" } })
    );
    expect(items[0]?.content).toBe("Monsieur DUPONT à bord");
  });

  it("genre inconnu : nom de famille sans civilité", () => {
    const items = buildTeamQuickReplies(
      mission({ status: "in_progress", client: { last_name: "Bronnimann" } })
    );
    expect(items[0]?.content).toBe("BRONNIMANN à bord");
    expect(items[0]?.content).not.toContain("Madame");
    expect(items[0]?.content).not.toContain("Monsieur");
  });
});
