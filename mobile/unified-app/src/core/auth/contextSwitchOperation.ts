/**
 * Intention de changement de contexte — distincte de sessionGenerationId.
 * Protège : switch A puis B, réponse A tardive.
 */
import {
  getSessionGenerationId,
  type SessionGenerationId,
} from "./authCredentialStore";

export type ContextSwitchOperation = {
  operationId: string;
  sessionGenerationId: SessionGenerationId;
  sourceContextId: string | null;
  targetContextId: string;
};

let activeOperation: ContextSwitchOperation | null = null;

function newOperationId(): string {
  return `ctxsw-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

export function beginContextSwitchOperation(params: {
  sourceContextId: string | null;
  targetContextId: string;
}): ContextSwitchOperation {
  const op: ContextSwitchOperation = {
    operationId: newOperationId(),
    sessionGenerationId: getSessionGenerationId(),
    sourceContextId: params.sourceContextId,
    targetContextId: params.targetContextId,
  };
  activeOperation = op;
  return op;
}

export function getActiveContextSwitchOperation(): ContextSwitchOperation | null {
  return activeOperation;
}

export function isCurrentContextSwitchOperation(operationId: string): boolean {
  return activeOperation?.operationId === operationId;
}

export function clearContextSwitchOperationIfCurrent(operationId: string): void {
  if (activeOperation?.operationId === operationId) {
    activeOperation = null;
  }
}

/** Tests uniquement. */
export function __resetContextSwitchOperationForTests(): void {
  activeOperation = null;
}
