import { describe, expect, it } from "@jest/globals";
import { nextSuggestionFields } from "./suggestionOverlayVisibility";

describe("nextSuggestionFields", () => {
  it("ne réalloue pas si l’état est déjà le bon (évite la boucle setState)", () => {
    const prev = ["client"];
    expect(nextSuggestionFields(prev, "client", true)).toBe(prev);
    expect(nextSuggestionFields(prev, "pickup", false)).toBe(prev);
  });

  it("ajoute et retire un champ", () => {
    expect(nextSuggestionFields([], "client", true)).toEqual(["client"]);
    expect(nextSuggestionFields(["client", "pickup"], "client", false)).toEqual(["pickup"]);
  });
});
