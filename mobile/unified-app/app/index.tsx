import { Redirect } from "expo-router";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { useSession } from "../src/core/sessionProvider";
import { resolveInitialRoute } from "../src/core/navigation/resolveInitialRoute";
import { brandSurfaceSoft, ResponsiveContainer, Screen } from "../src/design/responsive";
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ReactRuntime: any = require("react");

export default function IndexScreen() {
  const { status, bootstrap, error, bootstrapSession } = useSession();

  ReactRuntime.useEffect(() => {
    if (status === "idle") {
      void bootstrapSession();
    }
  }, [status, bootstrapSession]);

  if (status === "error") {
    return (
      <Screen scroll backgroundColor={brandSurfaceSoft} contentContainerStyle={styles.centerContent}>
        <ResponsiveContainer>
          <View style={styles.panel}>
            <Text style={styles.title}>Échec du démarrage</Text>
            <Text style={styles.body}>
              {error ?? "Impossible de charger la session depuis le backend."}
            </Text>
            <Pressable onPress={() => void bootstrapSession()} style={styles.retry}>
              <Text style={styles.retryText}>Réessayer</Text>
            </Pressable>
          </View>
        </ResponsiveContainer>
      </Screen>
    );
  }

  if (!bootstrap || status === "bootstrapping") {
    return (
      <Screen backgroundColor={brandSurfaceSoft}>
        <ResponsiveContainer>
          <View style={styles.loadingWrap}>
            <Text style={styles.loadingText}>Chargement de la session…</Text>
          </View>
        </ResponsiveContainer>
      </Screen>
    );
  }

  return <Redirect href={resolveInitialRoute(bootstrap) as any} />;
}

const styles = StyleSheet.create({
  centerContent: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 32,
  },
  panel: {
    gap: 14,
    alignItems: "stretch",
  },
  title: {
    fontSize: 20,
    fontWeight: "700",
    color: "#163A34",
    textAlign: "center",
  },
  body: {
    fontSize: 15,
    lineHeight: 22,
    color: "#475569",
    textAlign: "center",
  },
  retry: {
    alignSelf: "center",
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: "#0A8F7A",
    backgroundColor: "#FFFFFF",
  },
  retryText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#0A8F7A",
  },
  loadingWrap: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    minHeight: 120,
  },
  loadingText: {
    fontSize: 15,
    color: "#5F7369",
  },
});
