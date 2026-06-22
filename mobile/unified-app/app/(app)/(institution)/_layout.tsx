import { Redirect, Stack } from "expo-router";
import { useSession } from "../../../src/core/sessionProvider";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import { resolveInstitutionUnifiedGate } from "../../../src/core/guardDecisions";

export default function InstitutionLayout() {
  const { activeContext } = useSession();
  const gate = resolveInstitutionUnifiedGate(isFeatureEnabled("institution_unified_enabled"));

  if (!gate.allowed) {
    return <Redirect href={gate.redirectTo as never} />;
  }

  if (!activeContext) {
    return <Redirect href="/(app)/context-selector" />;
  }

  if (activeContext.context_type === "company") {
    return <Redirect href="/(app)/(company)" />;
  }
  if (activeContext.context_type === "driver") {
    return <Redirect href="/(app)/(driver)" />;
  }
  if (activeContext.context_type === "client") {
    return <Redirect href="/(app)/(client)" />;
  }
  if (activeContext.context_type !== "institution") {
    return <Redirect href="/(app)/unauthorized" />;
  }

  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="index" />
    </Stack>
  );
}
