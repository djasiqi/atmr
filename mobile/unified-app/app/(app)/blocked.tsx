import { Text, View, StyleSheet } from "react-native";
import { useLocalSearchParams } from "expo-router";

const MESSAGES: Record<string, { title: string; body: string }> = {
  driver_gate: {
    title: "Déploiement en cours",
    body: "Cette version est en cours de déploiement progressif. Utilisez l'application Chauffeur habituelle en attendant votre activation.",
  },
};

const DEFAULT_MESSAGE = {
  title: "Accès restreint",
  body: "Votre accès à cette section est restreint. Contactez le support si vous pensez qu'il s'agit d'une erreur.",
};

export default function BlockedScreen() {
  const { reason } = useLocalSearchParams<{ reason?: string }>();
  const message = (reason ? MESSAGES[reason] : undefined) ?? DEFAULT_MESSAGE;

  return (
    <View style={styles.container}>
      <Text style={styles.title}>{message.title}</Text>
      <Text style={styles.body}>{message.body}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 32,
    gap: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: "700",
    textAlign: "center",
  },
  body: {
    fontSize: 15,
    textAlign: "center",
    opacity: 0.7,
    lineHeight: 22,
  },
});
