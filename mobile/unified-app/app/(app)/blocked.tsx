import { StyleSheet, View } from "react-native";
import { useLocalSearchParams } from "expo-router";
import { AppText, brandSurfaceSoft, ResponsiveContainer, Screen } from "../../src/design/responsive";

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
    <Screen scroll backgroundColor={brandSurfaceSoft} contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.block}>
          <AppText variant="sectionTitle" style={styles.title}>
            {message.title}
          </AppText>
          <AppText variant="body" style={styles.body}>
            {message.body}
          </AppText>
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 32,
  },
  block: {
    gap: 16,
    alignItems: "center",
  },
  title: {
    textAlign: "center",
    color: "#163A34",
  },
  body: {
    textAlign: "center",
    lineHeight: 22,
  },
});
