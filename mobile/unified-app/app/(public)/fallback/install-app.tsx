import { Pressable, StyleSheet, Text, View } from "react-native";
import * as Linking from "expo-linking";
import { ResponsiveContainer, Screen } from "../../../src/design/responsive";

const IOS_STORE_URL = "https://apps.apple.com";
const ANDROID_STORE_URL = "https://play.google.com/store";

export default function InstallAppScreen() {
  return (
    <Screen scroll backgroundColor="#F7FBFA" contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.block}>
          <Text style={styles.title}>Installer l&apos;application</Text>
          <Text style={styles.body}>Cette action requiert l&apos;application mobile Lirie.</Text>
          <Pressable onPress={() => void Linking.openURL(IOS_STORE_URL)} style={styles.linkWrap}>
            <Text style={styles.link}>App Store</Text>
          </Pressable>
          <Pressable onPress={() => void Linking.openURL(ANDROID_STORE_URL)} style={styles.linkWrap}>
            <Text style={styles.link}>Google Play</Text>
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
