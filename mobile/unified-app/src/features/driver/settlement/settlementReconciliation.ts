import { useRef, useState } from "react";
import { settlementHttpNotice } from "./settlementHttpNotice";

/** Déclaration partie sans confirmation, ou lecture en cours. */
export type SettlementHold = "sending" | "uncertain" | "reconciling" | "refused";

export type SettlementPanelPhase =
  | "loading"
  | "reconciling"
  | "sending"
  | "uncertain"
  | "refused"
  | "consultation-error"
  | "server";

const pendingAttempts = new Map<string, string>();

export function rememberSettlementAttempt(scope: string, key: string): void {
  pendingAttempts.set(scope, key);
}

export function forgetSettlementAttempt(scope: string): void {
  pendingAttempts.delete(scope);
}

export function settlementAttempt(scope: string): string | null {
  return pendingAttempts.get(scope) ?? null;
}

export function clearSettlementAttempts(): void {
  pendingAttempts.clear();
}

/**
 * « Préparé » n'est certain que si le serveur vient d'être lu et qu'aucune
 * déclaration n'attend encore son rapprochement.
 */
export function settlementPanelPhase(input: {
  hasData: boolean;
  openOnServer: boolean;
  fetching: boolean;
  fetchedAfterMount: boolean;
  failed: boolean;
  hold: SettlementHold | null;
  refusal: boolean;
}): SettlementPanelPhase {
  if (!input.hasData) {
    if (input.fetching) return "loading";
    if (input.refusal || input.hold === "refused") return "refused";
    if (input.failed || input.hold === "uncertain") return "consultation-error";
    return "loading";
  }
  if (input.hold === "sending") return "sending";
  if (input.hold === "refused" || input.refusal) return "refused";
  if (input.openOnServer && !input.fetchedAfterMount) return "reconciling";
  if (input.fetching && (input.openOnServer || input.hold === "reconciling")) return "reconciling";
  if (input.hold === "uncertain" || input.hold === "reconciling" || input.failed) return "uncertain";
  return "server";
}

export function financialSendAllowed(phase: SettlementPanelPhase): boolean {
  return phase === "server";
}

/** Refus HTTP structuré. Une absence de réponse n'est pas un refus. */
export function httpRefusal(error: unknown, lostMessage: string): string | null {
  const sentence = settlementHttpNotice(error, lostMessage);
  if (sentence.startsWith("Refus") || sentence.startsWith("Accès refusé")) return sentence;
  return null;
}

/** Une lecture ratée n'est pas présentée comme une réponse perdue de déclaration. */
export function consultationFailure(error: unknown, lostMessage: string): {
  refusal: boolean;
  text: string;
} {
  const refusal = httpRefusal(error, lostMessage);
  if (refusal) return { refusal: true, text: refusal };
  return { refusal: false, text: "Erreur de consultation. Aucune nouvelle opération." };
}

/**
 * Conserve la clé d'idempotence si l'écran est démonté avant la confirmation.
 * Un redémarrage de l'application oublie cette mémoire et relit le serveur.
 */
export function useSettlementAttempt(scope: string) {
  const remembered = settlementAttempt(scope);
  const busy = useRef(Boolean(remembered));
  const mountPending = useRef(Boolean(remembered));
  const [trackedScope, setTrackedScope] = useState(scope);
  const [hold, setHold] = useState<SettlementHold | null>(remembered ? "reconciling" : null);
  const [attempt, setAttempt] = useState<string | null>(remembered);

  if (trackedScope !== scope) {
    const again = settlementAttempt(scope);
    setTrackedScope(scope);
    setHold(again ? "reconciling" : null);
    setAttempt(again);
    busy.current = Boolean(again);
    mountPending.current = Boolean(again);
  }

  function beginSend(key: string): boolean {
    if (busy.current) return false;
    busy.current = true;
    rememberSettlementAttempt(scope, key);
    setAttempt(key);
    setHold("sending");
    return true;
  }

  function beginReread(): boolean {
    if (hold === "sending" || hold === "reconciling") return false;
    setHold("reconciling");
    return true;
  }

  function finishLost() {
    busy.current = true;
    setHold("uncertain");
  }

  function finishRefused() {
    busy.current = true;
    forgetSettlementAttempt(scope);
    setAttempt(null);
    setHold("refused");
  }

  function finishRereadFailed() {
    busy.current = true;
    setHold("uncertain");
  }

  /**
   * @param keepAttempt vrai seulement si le serveur confirme que l'opération
   * partie n'est pas encore enregistrée. Un succès HTTP oublie la clé.
   */
  function finishConfirmed(stillOpen: boolean, keepAttempt: boolean) {
    busy.current = false;
    mountPending.current = false;
    if (!stillOpen || !keepAttempt) {
      forgetSettlementAttempt(scope);
      setAttempt(null);
    }
    setHold(null);
  }

  function consumeMountPending(): boolean {
    if (!mountPending.current) return false;
    mountPending.current = false;
    return true;
  }

  return {
    hold,
    attempt,
    beginSend,
    beginReread,
    finishLost,
    finishRefused,
    finishRereadFailed,
    finishConfirmed,
    consumeMountPending,
  };
}
