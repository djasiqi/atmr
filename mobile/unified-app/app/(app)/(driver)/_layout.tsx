import { Stack } from "expo-router";
import { DriverUnifiedGateGuard } from "../../../src/core/guards";

export default function DriverLayout() {
  return (
    <DriverUnifiedGateGuard>
      <Stack screenOptions={{ headerShown: false }}>
        <Stack.Screen name="index" />
        <Stack.Screen name="missions" />
        <Stack.Screen name="trips" />
        <Stack.Screen name="chat" />
        <Stack.Screen name="schedule" />
        <Stack.Screen name="profile" />
        <Stack.Screen name="trips/[tripId]" />
        <Stack.Screen name="missions/[missionId]" />
      </Stack>
    </DriverUnifiedGateGuard>
  );
}
