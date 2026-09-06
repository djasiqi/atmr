import { useCallback, useState } from "react";
import { StyleSheet, View } from "react-native";
import { useFocusEffect } from "@react-navigation/native";
import { AppButton } from "../../../../design/responsive";
import { isFeatureEnabled } from "../../../../core/featureFlags/registry";
import { useSession } from "../../../../core/sessionProvider";
import { createShadow } from "../../../../styles/shadowStyles";
import { E } from "../../theme/enterpriseOpsTheme";
import {
  OPTIMIZER_ENABLED,
  SEMI_AUTO_DISPATCH_ENABLED,
  shouldMountDispatchEngine,
} from "../../dispatch/dispatchModeLock";
import {
  getCompanyDispatchModes,
  runCompanyDispatch,
  runCompanyOptimizer,
} from "../../api/companyApi";

const cardShadow = createShadow({
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 1 },
  shadowOpacity: 0.03,
  shadowRadius: 4,
  elevation: 1,
});

type CompanyRidesEngineActionsProps = {
  contextId: string | null | undefined;
  selectedDate: string;
  onRan: () => Promise<void>;
};

/**
 * Branche dormante : dispatch semi-auto + optimiseur.
 * Ne doit être montée que si `shouldMountDispatchEngine()` est vrai.
 * Tant que le LOCK est OFF, Courses n’importe ce module que pour le gate,
 * et n’instancie jamais ce composant.
 */
export function CompanyRidesEngineActions({
  contextId,
  selectedDate,
  onRan,
}: CompanyRidesEngineActionsProps) {
  const { activeContext, can } = useSession();
  const [actionPending, setActionPending] = useState<null | "dispatch" | "optimizer">(null);
  const [activeMode, setActiveMode] = useState<"manual" | "semi_auto" | "fully_auto" | null>(null);

  const roleGuardsEnabled = isFeatureEnabled("company_mobile_role_guards_enabled");
  const contextPermissions = activeContext?.permissions ?? [];
  const canDispatchManage = (() => {
    if (!roleGuardsEnabled) return true;
    if (contextPermissions.includes("company:dispatch:manage")) {
      return can("company:dispatch:manage");
    }
    return can("company:rides:read");
  })();

  const loadDispatchMode = useCallback(async () => {
    if (!shouldMountDispatchEngine()) return;
    if (!contextId) return;
    try {
      const payload = await getCompanyDispatchModes({ contextId });
      if (!payload || typeof payload !== "object") return;
      const obj = payload as Record<string, unknown>;
      const nextMode = obj.mode ?? obj.current_mode ?? obj.dispatch_mode ?? null;
      if (nextMode === "manual" || nextMode === "semi_auto" || nextMode === "fully_auto") {
        setActiveMode(nextMode);
      }
    } catch {
      // Mode illisible : pas d’actions moteur.
    }
  }, [contextId]);

  useFocusEffect(
    useCallback(() => {
      if (!shouldMountDispatchEngine()) return undefined;
      void loadDispatchMode();
      return undefined;
    }, [loadDispatchMode])
  );

  const runDispatchNow = useCallback(async () => {
    if (!SEMI_AUTO_DISPATCH_ENABLED || !contextId) return;
    setActionPending("dispatch");
    try {
      await runCompanyDispatch({ contextId, date: selectedDate });
      await onRan();
    } finally {
      setActionPending(null);
    }
  }, [contextId, onRan, selectedDate]);

  const runOptimizerNow = useCallback(async () => {
    if (!OPTIMIZER_ENABLED || !contextId) return;
    setActionPending("optimizer");
    try {
      await runCompanyOptimizer({ contextId, date: selectedDate });
      await onRan();
    } finally {
      setActionPending(null);
    }
  }, [contextId, onRan, selectedDate]);

  if (!shouldMountDispatchEngine()) {
    return null;
  }

  const showDispatch = SEMI_AUTO_DISPATCH_ENABLED && activeMode !== "manual";
  const showOptimizer = OPTIMIZER_ENABLED && activeMode !== "manual";
  if (!showDispatch && !showOptimizer) {
    return null;
  }

  return (
    <View style={styles.dispatchActionsCard}>
      {showDispatch ? (
        <AppButton
          title={actionPending === "dispatch" ? "Exécution…" : "Lancer le dispatch"}
          variant="primary"
          onPress={() => void runDispatchNow()}
          disabled={!contextId || actionPending !== null || !canDispatchManage}
          style={styles.actionBtn}
        />
      ) : null}
      {showOptimizer ? (
        <AppButton
          title={actionPending === "optimizer" ? "Optimiseur…" : "Lancer l’optimiseur"}
          variant="secondary"
          onPress={() => void runOptimizerNow()}
          disabled={!contextId || actionPending !== null || !canDispatchManage}
          style={styles.actionBtn}
        />
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  dispatchActionsCard: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    backgroundColor: E.CARD,
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.22)",
    ...cardShadow,
  },
  actionBtn: { flexGrow: 1, minWidth: 108 },
});
