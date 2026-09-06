import { describe, expect, it } from "@jest/globals";
import { rankAddressSuggestion, sortAddressSuggestions, splitAddressLabel } from "./addressSuggestionRank";
import type { RideAddressOption } from "../../useRideForms";

function opt(partial: Partial<RideAddressOption> & Pick<RideAddressOption, "id" | "label">): RideAddressOption {
  return {
    placeId: null,
    latitude: null,
    longitude: null,
    source: "manual",
    ...partial,
  };
}

describe("addressSuggestionRank", () => {
  it("sépare le libellé adresse", () => {
    expect(splitAddressLabel("Rue A, 1000 Lausanne")).toEqual({
      primary: "Rue A",
      secondary: "1000 Lausanne",
    });
  });

  it("priorise alias puis POI Google", () => {
    const rows = [
      opt({ id: 1, label: "Gare", source: "google_places", types: ["geocode"] }),
      opt({ id: 2, label: "Alias Gare", source: "alias" }),
      opt({ id: 3, label: "Hopital", source: "google_places", types: ["hospital"] }),
    ];
    const sorted = sortAddressSuggestions(rows, "g");
    expect(sorted[0]?.id).toBe(2);
    expect(rankAddressSuggestion(rows[2]!, "h")).toBeGreaterThan(
      rankAddressSuggestion(rows[0]!, "h")
    );
  });
});
