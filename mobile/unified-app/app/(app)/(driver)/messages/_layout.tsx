import { Stack } from "expo-router";
import {
  stackNoneOptions,
  stackPushOptions,
} from "../../../../src/design/navigation/stackScreenOptions";

/**
 * Stack messages chauffeur — push hiérarchique LIRIE (identique entreprise).
 * Racine `index` non animée (geste tab dédié).
 */
export default function DriverMessagesLayout() {
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
      <Stack.Screen name="[threadId]" options={{ headerShown: false, ...stackPushOptions }} />
      <Stack.Screen name="[threadId]/details" options={{ title: "Détails mission", ...stackPushOptions }} />
      <Stack.Screen name="[threadId]/files" options={{ title: "Fichiers & position", ...stackPushOptions }} />
      <Stack.Screen name="colleagues" options={{ title: "Collègues", ...stackPushOptions }} />
      <Stack.Screen name="search" options={{ title: "Recherche", ...stackPushOptions }} />
      <Stack.Screen name="settings" options={{ title: "Paramètres messagerie", ...stackPushOptions }} />
    </Stack>
  );
}
