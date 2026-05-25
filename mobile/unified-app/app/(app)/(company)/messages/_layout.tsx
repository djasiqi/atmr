import { Stack } from "expo-router";
import {
  stackNoneOptions,
  stackPushOptions,
} from "../../../../src/design/navigation/stackScreenOptions";

/**
 * Stack messages entreprise — push hiérarchique LIRIE (220 ms aller / 180 ms retour).
 * L'écran racine `index` n'anime pas (porté par les tabs).
 */
export default function CompanyMessagesLayout() {
  return (
    <Stack
      screenOptions={{
        headerShown: true,
        headerBackTitle: "Retour",
        headerTintColor: "#0A8F7A",
        ...stackPushOptions,
      }}
    >
      <Stack.Screen name="index" options={{ title: "Messages", headerShown: false, ...stackNoneOptions }} />
      <Stack.Screen name="[threadId]/manage" options={{ headerShown: false, ...stackPushOptions }} />
      <Stack.Screen name="[threadId]" options={{ headerShown: false, ...stackPushOptions }} />
    </Stack>
  );
}
