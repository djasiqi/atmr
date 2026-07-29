import { Stack } from "expo-router";
import { View } from "react-native";
import {
  ClientFloatingAppBar,
  useClientFloatingBarVisible,
} from "../../../src/features/client/navigation/ClientFloatingAppBar";
import { AppFloatingBarMetricsProvider } from "../../../src/design/navigation/AppFloatingBarMetricsProvider";

export default function ClientLayout() {
  const showFloatingBar = useClientFloatingBarVisible();

  return (
    <AppFloatingBarMetricsProvider preset="client">
      <View style={{ flex: 1 }}>
        <Stack screenOptions={{ headerShown: false }}>
          <Stack.Screen name="index" />
          <Stack.Screen name="bookings" />
          <Stack.Screen name="booking/[bookingId]" />
          <Stack.Screen name="booking/new" />
          <Stack.Screen name="payment" />
          <Stack.Screen name="account" />
        </Stack>
        {showFloatingBar ? <ClientFloatingAppBar /> : null}
      </View>
    </AppFloatingBarMetricsProvider>
  );
}
