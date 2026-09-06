import type { ReactElement } from "react";
import { shouldMountDispatchEngine } from "../../dispatch/dispatchModeLock";

type MaybeCompanyRidesEngineActionsProps = {
  contextId: string | null | undefined;
  selectedDate: string;
  onRan: () => Promise<void>;
};

/**
 * Gate d’arbre : si le LOCK est OFF, la branche moteur n’est ni requise ni montée.
 * Aucun hook / effet / query semi-auto ne peut s’exécuter depuis Courses.
 */
export function MaybeCompanyRidesEngineActions(
  props: MaybeCompanyRidesEngineActionsProps
): ReactElement | null {
  if (!shouldMountDispatchEngine()) {
    return null;
  }
  // require volontaire : évite d’évaluer le module moteur tant que le LOCK est OFF.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { CompanyRidesEngineActions } = require("./CompanyRidesEngineActions") as typeof import("./CompanyRidesEngineActions");
  return <CompanyRidesEngineActions {...props} />;
}
