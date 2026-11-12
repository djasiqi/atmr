import {
  END_OF_TRIP_COLLECTION_ENABLED,
  cashReceiptDecision,
  collectionScreenVisible,
  openTapAttempt,
  resolveTapSimulation,
} from "./endOfTripCollection";

describe("encaissement conditionnel chauffeur", () => {
  it("reste invisible tant que l'activation est coupée", () => {
    expect(END_OF_TRIP_COLLECTION_ENABLED).toBe(false);
    expect(collectionScreenVisible(END_OF_TRIP_COLLECTION_ENABLED, true, true)).toBe(false);
  });

  it("n'affiche les options que si le serveur les autorise et que la course est prête", () => {
    expect(collectionScreenVisible(true, true, true)).toBe(true);
    expect(collectionScreenVisible(true, false, true)).toBe(false);
    expect(collectionScreenVisible(true, true, false)).toBe(false);
  });

  it("ne relance pas un débit après une réponse perdue", () => {
    const started = openTapAttempt("ready", null, "tentative-1");
    const lost = resolveTapSimulation(started.state, started.attemptId, "uncertain");
    const again = openTapAttempt(lost.state, lost.attemptId, "tentative-2");
    expect(again.state).toBe("uncertain");
    expect(again.attemptId).toBe("tentative-1");
    expect(again.mayStartAnotherDebit).toBe(false);
  });

  it("ne traite pas le choix espèces comme une réception", () => {
    expect(cashReceiptDecision("45.00", null, false)).toBe("missing");
    expect(cashReceiptDecision("45.00", "45.00", false)).toBe("missing");
    expect(cashReceiptDecision("45.00", "45.00", true)).toBe("declared");
    expect(cashReceiptDecision("90.00", "50.00", true)).toBe("difference");
  });
});
