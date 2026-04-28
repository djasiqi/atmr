import { Redirect } from "expo-router";
import { Pressable, Text, View } from "react-native";
import { useSession } from "../src/core/sessionProvider";
import { resolveInitialRoute } from "../src/core/navigation/resolveInitialRoute";
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
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center", padding: 24, gap: 12 }}>
        <Text style={{ fontSize: 18, fontWeight: "700" }}>Echec du bootstrap</Text>
        <Text style={{ textAlign: "center" }}>
          {error ?? "Impossible de charger la session depuis le backend."}
        </Text>
        <Pressable
          onPress={() => void bootstrapSession()}
          style={{
            paddingHorizontal: 16,
            paddingVertical: 10,
            borderRadius: 8,
            borderWidth: 1,
            borderColor: "#555",
          }}
        >
          <Text>Reessayer</Text>
        </Pressable>
      </View>
    );
  }

  if (!bootstrap || status === "bootstrapping") {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
        <Text>Bootstrap session en cours...</Text>
      </View>
    );
  }

  return <Redirect href={resolveInitialRoute(bootstrap) as any} />;
}
