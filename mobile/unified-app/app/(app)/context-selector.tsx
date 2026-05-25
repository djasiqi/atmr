import { StyleSheet, Text, TouchableOpacity } from "react-native";
import { useRouter } from "expo-router";
import { type AuthContext } from "../../src/core/contracts/auth";
import { useSession } from "../../src/core/sessionProvider";
import {
  companyDriverSwitchBlockedReason,
  isCompanyDriverSwitchAllowedForRequest,
} from "../../src/core/contextSwitchPolicy";
import {
  brandPrimary,
  brandSurfaceSoft,
  ResponsiveContainer,
  Screen,
} from "../../src/design/responsive";
import { FONT_SIZE } from "../../src/design/responsive/typographyTokens";
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ReactRuntime: any = require("react");

export default function ContextSelectorScreen() {
  const router = useRouter();
  const { bootstrap, activeContext, changeContext, error } = useSession();
  const [pendingContextId, setPendingContextId] = ReactRuntime.useState(
    null as string | null
  );
  const contexts = ReactRuntime.useMemo(
    () => (bootstrap?.available_contexts ?? []) as AuthContext[],
    [bootstrap]
  ) as AuthContext[];

  ReactRuntime.useEffect(() => {
    if (!pendingContextId || !activeContext) return;
    if (activeContext.context_id !== pendingContextId) return;
    setPendingContextId(null);
    switch (activeContext.context_type) {
      case "company":
        router.replace("/(app)/(company)" as any);
        return;
      case "driver":
        router.replace("/(app)/(driver)" as any);
        return;
      case "client":
        router.replace("/(app)/(client)" as any);
        return;
      case "institution":
        router.replace("/(app)/(institution)" as any);
        return;
      default:
        router.replace("/(app)/context-selector" as any);
    }
  }, [activeContext, pendingContextId, router]);

  const handleContextPress = ReactRuntime.useCallback(
    async (ctx: AuthContext) => {
      if (pendingContextId) return;
      if (ctx.context_id === activeContext?.context_id) return;
      if (!isCompanyDriverSwitchAllowedForRequest(activeContext, ctx, bootstrap?.user?.role)) {
        return;
      }
      setPendingContextId(ctx.context_id);
      await changeContext(ctx.context_id);
    },
    [activeContext, changeContext, pendingContextId]
  );

  return (
    <Screen scroll backgroundColor={brandSurfaceSoft} contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <Text style={styles.heading}>Choisir un espace</Text>
        <Text style={styles.activeLabel}>
          Contexte actif : {activeContext?.label ?? "Aucun"}
        </Text>
        {contexts.map((ctx: AuthContext) => {
          const companyDriverBlocked = !isCompanyDriverSwitchAllowedForRequest(
            activeContext,
            ctx,
            bootstrap?.user?.role
          );
          const blockReason = companyDriverSwitchBlockedReason(
            activeContext,
            ctx,
            bootstrap?.user?.role
          );
          const rowDisabled = Boolean(pendingContextId) || companyDriverBlocked;
          const selected = ctx.context_id === activeContext?.context_id;
          return (
            <TouchableOpacity
              key={ctx.context_id}
              onPress={() => void handleContextPress(ctx)}
              disabled={rowDisabled}
              style={[
                styles.row,
                selected && styles.rowSelected,
                rowDisabled && styles.rowDisabled,
              ]}
              accessibilityRole="button"
            >
              <Text style={styles.rowTitle}>{ctx.label}</Text>
              <Text style={styles.rowMeta}>
                {ctx.context_type} · {ctx.context_id}
              </Text>
              {companyDriverBlocked && ctx.context_id !== activeContext?.context_id ? (
                <Text style={styles.rowHint}>
                  {blockReason === "not_company_account"
                    ? "Réservé au compte entreprise (un chauffeur seul n'accède pas à la gestion)."
                    : "Bascule entreprise / chauffeur : app mobile, compte entreprise, dispatch actif."}
                </Text>
              ) : null}
            </TouchableOpacity>
          );
        })}
        {error ? <Text style={styles.error}>{error}</Text> : null}
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    paddingVertical: 28,
    gap: 12,
  },
  heading: {
    fontSize: FONT_SIZE.px22,
    fontWeight: "700",
    color: "#163A34",
    marginBottom: 4,
  },
  activeLabel: {
    fontSize: FONT_SIZE.px15,
    color: "#5F7369",
    marginBottom: 8,
  },
  row: {
    padding: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#FFFFFF",
    marginBottom: 10,
  },
  rowSelected: {
    borderColor: brandPrimary,
    backgroundColor: "rgba(10,143,122,0.06)",
  },
  rowDisabled: {
    opacity: 0.5,
  },
  rowTitle: {
    fontWeight: "600",
    fontSize: FONT_SIZE.px16,
    color: "#163A34",
  },
  rowMeta: {
    marginTop: 4,
    fontSize: FONT_SIZE.px13,
    color: "#64748b",
  },
  rowHint: {
    color: "#64748b",
    marginTop: 8,
    fontSize: FONT_SIZE.px12,
    lineHeight: 17,
  },
  error: {
    color: "#B42318",
    marginTop: 8,
    fontSize: FONT_SIZE.px14,
  },
});
