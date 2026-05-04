import { useMemo } from "react";
import { Platform, Pressable, StyleSheet, Text, View } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { ResponsiveContainer, Screen, useAppViewport } from "../../../src/design/responsive";

export default function PreRequestAuthGateScreen() {
  const router = useRouter();
  const { topInset } = useAppViewport();
  const params = useLocalSearchParams<{ draftId?: string }>();

  const nextRoute = useMemo(() => {
    const draftId = typeof params.draftId === "string" ? params.draftId.trim() : "";
    if (!draftId) return "/(app)/(client)/booking/new";
    return `/(app)/(client)/booking/new?publicDraftId=${encodeURIComponent(draftId)}`;
  }, [params.draftId]);

  return (
    <Screen
      scroll
      backgroundColor="#EAF3F1"
      keyboardVerticalOffset={Platform.OS === "ios" ? topInset : 0}
      contentContainerStyle={styles.scroll}
    >
      <ResponsiveContainer>
        <View style={styles.card}>
          <Text style={styles.title}>Paiement et confirmation</Text>
          <Text style={styles.body}>
            Votre trajet est deja enregistre. Pour regler et confirmer la reservation, connectez-vous ou creez un
            compte : le parcours reprendra sans ressaisir les adresses.
          </Text>

          <Pressable
            onPress={() =>
              router.replace({
                pathname: "/(public)/login",
                params: { next: nextRoute },
              } as any)
            }
            style={styles.primaryBtn}
          >
            <Text style={styles.primaryBtnText}>Se connecter pour finaliser</Text>
          </Pressable>
          <Pressable
            onPress={() =>
              router.replace({
                pathname: "/(public)/signup",
                params: { next: nextRoute },
              } as any)
            }
            style={styles.outlineBtn}
          >
            <Text style={styles.outlineBtnText}>Creer un compte pour finaliser</Text>
          </Pressable>
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 24,
  },
  card: {
    gap: 14,
    borderRadius: 26,
    padding: 24,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#FFFFFF",
  },
  title: {
    fontSize: 26,
    fontWeight: "800",
    color: "#163A34",
  },
  body: {
    color: "#45655D",
    lineHeight: 22,
  },
  primaryBtn: {
    backgroundColor: "#0A8F7A",
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: "center",
  },
  primaryBtnText: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: 16,
  },
  outlineBtn: {
    borderWidth: 1.5,
    borderColor: "#0A8F7A",
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: "center",
    backgroundColor: "#FFFFFF",
  },
  outlineBtnText: {
    color: "#0A8F7A",
    fontWeight: "700",
    fontSize: 16,
  },
});
