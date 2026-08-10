import { Stack } from "expo-router";
import { AppVersionGuard, AuthGuard, ContextGuard, OnboardingGuard } from "../../src/core/guards";
import { stackFadeOptions } from "../../src/design/navigation/stackScreenOptions";

/**
 * Couche authentifiée — changement de « monde » (driver / company / client / institution)
 * = fade contexte LIRIE (280 ms), pas de slide latéral pour signaler la bascule de rôle.
 */
export default function AuthenticatedLayout() {
  return (
    <AppVersionGuard>
      <AuthGuard>
        <OnboardingGuard>
          <ContextGuard>
            <Stack screenOptions={{ headerShown: false, ...stackFadeOptions }}>
              <Stack.Screen name="context-selector" />
              <Stack.Screen name="onboarding" />
              <Stack.Screen name="unauthorized" />
              <Stack.Screen name="maintenance" />
              <Stack.Screen name="blocked" />
              <Stack.Screen name="device-sessions" />
              <Stack.Screen name="(client)" />
              <Stack.Screen name="(driver)" />
              <Stack.Screen name="(company)" />
              <Stack.Screen name="(institution)" />
            </Stack>
          </ContextGuard>
        </OnboardingGuard>
      </AuthGuard>
    </AppVersionGuard>
  );
}
