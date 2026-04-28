import { Text, TouchableOpacity, View } from "react-native";
import { useRouter } from "expo-router";
import { type AuthContext } from "../../src/core/contracts/auth";
import { useSession } from "../../src/core/sessionProvider";
import {
  companyDriverSwitchBlockedReason,
  isCompanyDriverSwitchAllowedForRequest,
} from "../../src/core/contextSwitchPolicy";
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
    <View style={{ flex: 1, justifyContent: "center", padding: 24, gap: 12 }}>
      <Text style={{ fontSize: 22, fontWeight: "700" }}>Choisir un espace</Text>
      <Text>Contexte actif: {activeContext?.label ?? "Aucun"}</Text>
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
        return (
        <TouchableOpacity
          key={ctx.context_id}
          onPress={() => void handleContextPress(ctx)}
          disabled={rowDisabled}
          style={{
            padding: 12,
            borderRadius: 10,
            borderWidth: 1,
            borderColor: "#ddd",
            opacity: rowDisabled ? 0.5 : 1,
          }}
        >
          <Text style={{ fontWeight: "600" }}>{ctx.label}</Text>
          <Text>
            {ctx.context_type} - {ctx.context_id}
          </Text>
          {companyDriverBlocked && ctx.context_id !== activeContext?.context_id ? (
            <Text style={{ color: "#666", marginTop: 4, fontSize: 12 }}>
              {blockReason === "not_company_account"
                ? "Réservé au compte entreprise (un chauffeur seul n'accède pas à la gestion)."
                : "Bascule entreprise / chauffeur : app mobile, compte entreprise, dispatch actif."}
            </Text>
          ) : null}
        </TouchableOpacity>
        );
      })}
      {error ? <Text style={{ color: "red" }}>{error}</Text> : null}
    </View>
  );
}
