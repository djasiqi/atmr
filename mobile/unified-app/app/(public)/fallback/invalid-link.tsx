import { Pressable, StyleSheet, Text, View } from "react-native";
import { useRouter } from "expo-router";
import { ResponsiveContainer, Screen } from "../../../src/design/responsive";

export default function InvalidLinkScreen() {
  const router = useRouter();
  return (
    <Screen scroll backgroundColor="#F7FBFA" contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.block}>
          <Text style={styles.title}>Lien invalide</Text>
          <Text style={styles.body}>
            Le format du lien est incorrect ou incomplet. Réessayez depuis la source d&apos;origine.
          </Text>
          <Pressable onPress={() => router.replace("/(public)" as any)} style={styles.linkWrap}>
            <Text style={styles.link}>Retour à l&apos;accueil</Text>
          </Pressable>
          <Pressable onPress={() => router.replace("/(public)/contact" as any)} style={styles.linkWrap}>
            <Text style={styles.link}>Contacter le support</Text>
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
    color: "#7f1d1d",
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
