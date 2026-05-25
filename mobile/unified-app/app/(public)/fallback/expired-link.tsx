import { Pressable, StyleSheet, Text, View } from "react-native";
import { useRouter } from "expo-router";
import { ResponsiveContainer, Screen } from "../../../src/design/responsive";
import { FONT_SIZE } from "../../../src/design/responsive/typographyTokens";

export default function ExpiredLinkScreen() {
  const router = useRouter();
  return (
    <Screen scroll backgroundColor="#F7FBFA" contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.block}>
          <Text style={styles.title}>Lien expiré</Text>
          <Text style={styles.body}>
            Ce lien n&apos;est plus valide. Demandez un nouveau lien depuis l&apos;application.
          </Text>
          <Pressable onPress={() => router.replace("/(public)/login" as any)} style={styles.linkWrap}>
            <Text style={styles.link}>Se connecter</Text>
          </Pressable>
          <Pressable onPress={() => router.replace("/(public)/help" as any)} style={styles.linkWrap}>
            <Text style={styles.link}>Voir l&apos;aide</Text>
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
    fontSize: FONT_SIZE.px24,
    fontWeight: "800",
    color: "#7f1d1d",
  },
  body: {
    fontSize: FONT_SIZE.px15,
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
    fontSize: FONT_SIZE.px15,
    textDecorationLine: "underline",
  },
});
