import { Stack } from "expo-router";
import { AppVersionGuard, AuthGuard, ContextGuard, OnboardingGuard } from "../../src/core/guards";

export default function AuthenticatedLayout() {
  return (
    <AppVersionGuard>
      <AuthGuard>
        <OnboardingGuard>
          <ContextGuard>
            <Stack screenOptions={{ headerShown: false }}>
              <Stack.Screen name="context-selector" />
              <Stack.Screen name="onboarding" />
              <Stack.Screen name="unauthorized" />
              <Stack.Screen name="maintenance" />
              <Stack.Screen name="blocked" />
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
