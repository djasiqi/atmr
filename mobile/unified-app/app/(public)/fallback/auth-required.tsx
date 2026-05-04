import { Pressable, StyleSheet, Text, View } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { ResponsiveContainer, Screen } from "../../../src/design/responsive";

export default function AuthRequiredScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ next?: string }>();
  return (
    <Screen scroll backgroundColor="#F7FBFA" contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.block}>
          <Text style={styles.title}>Connexion requise</Text>
          <Text style={styles.body}>
            Connectez-vous pour poursuivre cette action en toute sécurité.
          </Text>
          <Pressable
            onPress={() =>
              router.replace({
                pathname: "/(public)/login",
                params: params.next ? { next: params.next } : {},
              } as any)
            }
            style={styles.linkWrap}
          >
            <Text style={styles.link}>Se connecter</Text>
          </Pressable>
          <Pressable onPress={() => router.replace("/(public)/signup" as any)} style={styles.linkWrap}>
            <Text style={styles.link}>Créer un compte</Text>
          </Pressable>
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    paddingVertical: 32,
    justifyContent: "center",
  },
  block: {
    gap: 14,
  },
  title: {
    fontSize: 24,
    fontWeight: "800",
    color: "#163A34",
  },
  body: {
    fontSize: 15,
    lineHeight: 22,
    color: "#475569",
  },
  linkWrap: {
    alignSelf: "flex-start",
    paddingVertical: 4,
  },
  link: {
    color: "#0A8F7A",
    fontWeight: "700",
    fontSize: 15,
    textDecorationLine: "underline",
  },
});
