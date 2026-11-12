import { createElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { consultationFailure, httpRefusal, settlementPanelPhase, useSettlementAttempt, clearSettlementAttempts } from "./settlementReconciliation";

const open = {
  hasData: true,
  openOnServer: true,
  fetching: false,
  fetchedAfterMount: true,
  failed: false,
  hold: null,
  refusal: false,
};

describe("rapprochement d'un dossier financier", () => {
  it("laisse Préparé seulement après une lecture réussie", () => {
    expect(settlementPanelPhase(open)).toBe("server");
    expect(settlementPanelPhase({ ...open, fetchedAfterMount: false })).toBe("reconciling");
    expect(settlementPanelPhase({ ...open, fetching: true })).toBe("reconciling");
  });

  it("garde l'incertitude et bloque l'envoi tant que le serveur n'est pas relu", () => {
    expect(settlementPanelPhase({ ...open, hold: "sending" })).toBe("sending");
    expect(settlementPanelPhase({ ...open, hold: "uncertain" })).toBe("uncertain");
    expect(settlementPanelPhase({ ...open, hold: "reconciling", fetching: true })).toBe("reconciling");
    expect(settlementPanelPhase({ ...open, failed: true })).toBe("uncertain");
    expect(settlementPanelPhase({ ...open, hold: "refused" })).toBe("refused");
  });

  it("affiche l'état confirmé même s'il reste en cache pendant une relecture", () => {
    expect(
      settlementPanelPhase({
        ...open,
        openOnServer: false,
        fetching: true,
        hold: null,
      })
    ).toBe("server");
  });

  it("distingue le refus d'une lecture impossible", () => {
    const refusal = { response: { status: 409, data: { error_code: "institution_facturee" } } };
    expect(httpRefusal(refusal, "perdu")).toContain("institution");
    expect(consultationFailure(new Error("réseau"), "perdu")).toEqual({
      refusal: false,
      text: "Erreur de consultation. Aucune nouvelle opération.",
    });
    expect(settlementPanelPhase({ ...open, hasData: false, failed: true, fetching: false })).toBe(
      "consultation-error"
    );
    expect(
      settlementPanelPhase({ ...open, hasData: false, failed: true, fetching: false, refusal: true })
    ).toBe("refused");
  });

  it("ignore un second appui pendant l'envoi", () => {
    clearSettlementAttempts();
    let send: ((key: string) => boolean) | null = null;
    function Probe() {
      send = useSettlementAttempt("cash:p2l03").beginSend;
      return null;
    }
    act(() => {
      TestRenderer.create(createElement(Probe));
    });
    let first = false;
    let second = false;
    act(() => {
      first = Boolean(send?.("operation-1"));
    });
    act(() => {
      second = Boolean(send?.("operation-1"));
    });
    expect(first).toBe(true);
    expect(second).toBe(false);
  });
});
